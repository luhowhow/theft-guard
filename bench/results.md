# 效能紀錄

環境：i5-1035G1、Ubuntu 26.04、OpenCV 4.10.0 DNN（CPU）

## 推論時間（2026-10-04）

暖身 1 次後跑 20 次取平均，只計 `net.forward()`。

| 模型 | 輸入 | 電源模式 | 平均 ms | 每秒張數 |
| --- | --- | --- | --- | --- |
| YOLOX-s | 640 | balanced | 428.9 | 2.3 |
| YOLOX-s | 640 | performance | 250.2 | 4.0 |
| YOLOX-tiny | 416 | performance | 102.1 | 9.8 |

結論：performance 模式快約一倍；Stage I 採用 YOLOX-tiny。
