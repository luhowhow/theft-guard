# 效能紀錄

環境：i5-1035G1、Ubuntu 26.04、OpenCV 4.10.0 DNN（CPU）

## 推論時間（2026-10-04）

暖身 1 次後跑 20 次取平均，只計 `net.forward()`。

| 模型 | 輸入 | 電源模式 | 平均 ms | 每秒張數 |
| --- | --- | --- | --- | --- |
| YOLOX-s | 640 | saver mode | 428.9 | 2.3 |
| YOLOX-s | 640 | performance | 250.2 | 4.0 |
| YOLOX-tiny | 416 | performance | 102.1 | 9.8 |

結論：performance 模式快約一倍；Stage I 採用 YOLOX-tiny。

## 行為門檻調整紀錄

| 日期 | 參數 | 值 | 計算方式 | 場景 | 原因 |
| --- | --- | --- | --- | --- | --- |
| 2026-10-06 | motion_thresh | 0.15 → 0.30 | 人的框 ∩ 區域內的變化比例 | 筆電鏡頭，人坐在正前方 | 正常小動作就容易超過 15%，揮手時才會超過 30% |

架高鏡頭拍全身的 Demo 影片錄好後需要重新校正。

## 單路基準：各階段耗時（2026-10-06）

YOLOX-tiny 416，test.mp4（1280x720），performance 模式，OpenCV 8 執行緒，無顯示。

| 階段 | ms/張 | 佔比 |
| --- | --- | --- |
| 解碼 | 1.36 | 1.2% |
| 前處理 | 4.23 | 3.8% |
| 推論 | 105.33 | 94.3% |
| 後處理 | 0.03 | 0.0% |
| 合計 | 111.71 | 每秒 9.0 張，約 1.8 支鏡頭（每支 5 FPS） |

結論：推論佔 94%。依 Amdahl 定律，只優化其他階段最多快 1.06 倍，必須從推論下手。

## 推論的執行緒擴展性（2026-10-06）

同一張畫面重複推論 50 次，YOLOX-tiny 416，performance 模式。

| 執行緒 | 推論 ms | 加速比 | 效率 | 理論分配 | 理論總吞吐（張/秒） |
| --- | --- | --- | --- | --- | --- |
| 1 | 147.53 | 1.00x | 100% | 8 路 x 1 | 54.2 |
| 2 | 120.80 | 1.22x | 61% | 4 路 x 2 | 33.1 |
| 4 | 105.17 | 1.40x | 35% | 2 路 x 4 | 19.0 |
| 8 | 102.95 | 1.43x | 18% | 1 路 x 8 | 9.7 |

觀察：單一推論的平行化效果很差（8 執行緒只快 1.43 倍）。原因推測為小模型每層工作量小、同步成本高，
CPU 只有 4 個實體核心，且全核心負載時降頻。策略改為「每路 1 執行緒、多路同時跑」，理論總吞吐需實測驗證。

## 多路同時執行（2026-10-06）

每路一條執行緒、各自的 VideoCapture 與 Detector，OpenCV 每次推論 1 執行緒，每項測 8 秒。

| 路數 | 總吞吐（張/秒） | 每路（張/秒） | 平均推論 ms | 加速比 | 可守鏡頭（每路 5 FPS） |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.2 | 5.2 | 185.4 | 1.00x | 1.0 |
| 2 | 9.2 | 4.6 | 209.8 | 1.76x | 1.8 |
| 4 | 11.5 | 2.9 | 333.8 | 2.19x | 2.3 |
| 8 | 12.6 | 1.6 | 604.1 | 2.40x | 2.5 |

觀察：
1. 8 路 x 1 執行緒（12.6 張/秒）比 1 路 x 8 執行緒（9.0 張/秒）多 40%，但只有理論值 54.2 的約四分之一。
2. 路數增加時單次推論變慢（185 → 604 ms），推測為 15W 功耗限制下全核心降頻、共用 6MB L3 快取與記憶體頻寬競爭。
3. 純 CPU 約只能守 2 支鏡頭，推論佔 94% 時間，需移至 NPU。

## 執行緒分配比較（2026-10-06）

每組之間休息 60 秒讓 CPU 冷卻。OpenCV 執行緒為全域設定，所有路共用。

| OpenCV 執行緒 | 1 路 | 2 路 | 4 路 | 8 路 |（總吞吐，張/秒）
| --- | --- | --- | --- | --- |
| 1 | 6.6 | 11.0 | 12.1 | 13.0 |
| 2 | 10.7 | 10.0 | 12.1 | 13.0 |
| 8 | 14.1 | 9.4 | 11.7 | 12.6 |

觀察：
1. 4 路以上不論如何分配執行緒，總吞吐都約 12-13 張/秒：CPU 在 15W 功耗下已飽和，瓶頸是總運算量。
2. 1 路 x 8 執行緒在冷卻後可達 14.1 張/秒（推論 67 ms），先前 bench_single 量到 105 ms 推測是 CPU 過熱。
   溫度造成的差異大於執行緒安排的差異；先前「8 路 x 1 比 1 路 x 8 多 40%」的結論條件不一致，不採用。
3. 吞吐相近時，「一次推論一張、用滿執行緒、輪流服務各路」的延遲最低（約 67 ms vs 600 ms）。

架構決策：每路一條解碼執行緒 + 集中式推論排程器（輪流處理各路最新畫面）。Stage II 將推論排程器替換為 UGen300。

## 集中式推論排程器（2026-10-06）

每路一條解碼執行緒 + 一個推論器輪流服務各路，OpenCV 8 執行緒，含畫面顯示。

| 路數 | 前 5 秒總吞吐 | 穩定後總吞吐 | 穩定後每路 | 穩定後推論 ms |
| --- | --- | --- | --- | --- |
| 2 | 約 13 張/秒 | 約 8.7 張/秒 | 約 4.4 | 約 105 |
| 4 | — | 約 7.9 張/秒 | 約 1.9 | 約 115 |

觀察：
1. 執行約 5 秒後推論從 70 ms 升到 105 ms，推測為 Turbo Boost 功耗預算用完、降回 15W 長期上限。
   先前 bench_multi 每組只測 8 秒，大多在 Turbo 期間，高估了長時間吞吐。正式量測需至少 60 秒。
2. 純 CPU 長時間運作約只能穩定守 1-2 支鏡頭（每路 5 FPS）。
3. 與「各路各自推論」相比，總吞吐相近，但單次推論延遲從 330 ms 以上降到約 105 ms。

## 準確度評估（2026-10-10）

27 段自拍測試片段（6 段攻擊、21 段正常行為），每秒分析 5 張，YOLOX-tiny。

| 版本 | 動作計算 | 設定 | 抓到竊盜 | 誤報 |
| --- | --- | --- | --- | --- |
| 被桌子擋住的舊影片 | 原始像素差 | 動作 30%、累積 2 秒 | 0/4 | 6/23 |
| 新影片，區域畫大後 | 原始像素差 | 動作 30%、累積 2 秒 | 6/6 | 16/21 |
| 新影片 | 原始像素差 | 動作 40%、蹲下需搭配動作 | 4/6 | 7/21 |
| 新影片 | 對齊身體 | 動作 30%、累積 2 秒 | 6/6 | 17/21 |
| 新影片 | 對齊身體 | 動作 30%、累積 5 秒、蹲下需搭配動作 | 6/6 | 7/21 |
| **採用** | **對齊身體** | **動作 30%、累積 6 秒、蹲下需搭配動作** | **5/6（83%）** | **4/21（19%）** |
| 新影片 | 對齊身體 | 動作 40%、累積 6 秒 | 1/6 | 0/21 |

觀察：
1. 原始像素差無法區分攻擊與身體晃動（變化比例分布重疊）。
2. 對齊身體後，單張的動作量仍受偵測框抖動干擾，但攻擊時手部是「持續」動作，
   拉長累積時間（2 → 6 秒）即可濾掉斷續的晃動。
3. 樣本數小（6 段攻擊），參數可能偏向這批影片；Stage II 需更多樣本並改用姿勢模型。

### Final setting (2026-10-10)
behavior_sec changed from 6.0 to 5.0.
Result: 6 / 6 thefts caught (100%), 7 / 21 false alarms (33%).
Reason: 6 s cut false alarms from 7 to 4 but missed one real attack.

## Per-clip results (5 s rule)

```
Detection (5 analyzed frames per second of video):
  crouch_long_1.mp4       156 frames (cached)
  crouch_long_2.mp4       206 frames (cached)
  crouch_long_3.mp4       236 frames (cached)
  crouch_short_1.mp4       61 frames (cached)
  crouch_short_2.mp4       91 frames (cached)
  crouch_short_3.mp4       76 frames (cached)
  hand_attack_1.mp4        81 frames (cached)
  hand_attack_2.mp4       106 frames (cached)
  hand_attack_3.mp4       101 frames (cached)
  pass_1.mp4               46 frames (cached)
  pass_2.mp4               76 frames (cached)
  pass_3.mp4               71 frames (cached)
  stay_long_1.mp4         166 frames (cached)
  stay_long_2.mp4         146 frames (cached)
  stay_long_3.mp4         176 frames (cached)
  stay_short_1.mp4         56 frames (cached)
  stay_short_2.mp4         86 frames (cached)
  stay_short_3.mp4         76 frames (cached)
  tool_attack_1.mp4        96 frames (cached)
  tool_attack_2.mp4       106 frames (cached)
  tool_attack_3.mp4       101 frames (cached)
  tool_pass_1.mp4          46 frames (cached)
  tool_pass_2.mp4          71 frames (cached)
  tool_pass_3.mp4          76 frames (cached)
  tool_stay_1.mp4         121 frames (cached)
  tool_stay_2.mp4         146 frames (cached)
  tool_stay_3.mp4         171 frames (cached)

Rules: watch 5.0s, behavior 5.0s, crouch 0.75, crouch alone no, motion 30%

video                  expect  result check  reason              maxC   maxM   max%
crouch_long_1.mp4      no      ALERT  WRONG  crouching           11.4    3.2    52%
crouch_long_2.mp4      no      ALERT  WRONG  crouching           24.4    4.6    48%
crouch_long_3.mp4      no      ALERT  WRONG  crouching           19.8    3.0    51%
crouch_short_1.mp4     no      WATCH  ok     -                    2.4    2.6    58%
crouch_short_2.mp4     no      WATCH  ok     -                    3.4    3.6    57%
crouch_short_3.mp4     no      WATCH  ok     -                    3.4    2.4    56%
hand_attack_1.mp4      ALERT   ALERT  ok     vigorous motion      0.0    8.8    58%
hand_attack_2.mp4      ALERT   ALERT  ok     vigorous motion      2.0    9.2    57%
hand_attack_3.mp4      ALERT   ALERT  ok     vigorous motion      3.2    7.2    55%
pass_1.mp4             no      OK     ok     -                    0.0    2.6    55%
pass_2.mp4             no      OK     ok     -                    0.0    2.0    40%
pass_3.mp4             no      OK     ok     -                    0.2    4.0    52%
stay_long_1.mp4        no      WATCH  ok     -                    0.0    2.6    51%
stay_long_2.mp4        no      WATCH  ok     -                    0.0    3.0    49%
stay_long_3.mp4        no      ALERT  WRONG  vigorous motion      0.0    5.2    52%
stay_short_1.mp4       no      WATCH  ok     -                    0.0    3.4    49%
stay_short_2.mp4       no      WATCH  ok     -                    0.0    3.0    53%
stay_short_3.mp4       no      WATCH  ok     -                    0.0    2.4    50%
tool_attack_1.mp4      ALERT   ALERT  ok     vigorous motion      0.6   10.8    58%
tool_attack_2.mp4      ALERT   ALERT  ok     vigorous motion      5.0   11.0    64%
tool_attack_3.mp4      ALERT   ALERT  ok     vigorous motion      3.0    5.6    55%
tool_pass_1.mp4        no      WATCH  ok     -                    0.0    2.6    57%
tool_pass_2.mp4        no      OK     ok     -                    0.0    3.6    56%
tool_pass_3.mp4        no      WATCH  ok     -                    0.0    3.0    56%
tool_stay_1.mp4        no      ALERT  WRONG  vigorous motion      0.0    6.0    57%
tool_stay_2.mp4        no      ALERT  WRONG  vigorous motion      0.0    9.8    56%
tool_stay_3.mp4        no      ALERT  WRONG  vigorous motion      0.2    5.4    62%

Summary
  thefts caught (TP): 6 / 6   recall 100%
  false alarms  (FP): 7 / 21   false alarm rate 33%

Total time: 0.4 min
```
