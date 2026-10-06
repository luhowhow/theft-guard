// 第 8 步：用 YOLOX 找出畫面中的人，並畫出框
//
// 用法：
//   ./detect                                   # 鏡頭，預設 YOLOX-tiny
//   ./detect test.mp4                          # 影片
//   ./detect test.mp4 models/yolox_s.onnx 640  # 指定模型與輸入大小

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 一個偵測結果：框的位置 + 信心分數
struct Detection {
    cv::Rect box;
    float score;
};

// 等比例縮放到 size x size，多出來的地方補灰色（114）
// ratio 回傳縮放比例，之後要用它把座標換回原圖
cv::Mat letterbox(const cv::Mat& img, int size, float& ratio) {
    ratio = std::min((float)size / img.cols, (float)size / img.rows);
    int w = (int)std::round(img.cols * ratio);
    int h = (int)std::round(img.rows * ratio);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(w, h));

    cv::Mat out(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(0, 0, w, h)));  // 貼在左上角
    return out;
}

// 把模型輸出的 N x 85 個數字，整理成「人」的框
std::vector<Detection> decode(const cv::Mat& output, int size, float ratio,
                              float score_thresh, float nms_thresh) {
    const float* data = (const float*)output.data;
    const int num_candidates = output.size[1];  // 416 輸入時是 3549
    const int num_values = output.size[2];      // 85

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    // YOLOX 用三種格子大小（8、16、32 像素）切割畫面
    // 每個格子負責預測一個候選框
    int idx = 0;
    for (int stride : {8, 16, 32}) {
        int grid = size / stride;
        for (int gy = 0; gy < grid; ++gy) {
            for (int gx = 0; gx < grid; ++gx, ++idx) {
                const float* p = data + idx * num_values;

                // p[4]：這裡有東西的信心；p[5]：這個東西是「人」的機率
                float score = p[4] * p[5];
                if (score < score_thresh) continue;

                // 模型給的是「相對於格子」的值，要換算成像素座標
                float cx = (p[0] + gx) * stride;
                float cy = (p[1] + gy) * stride;
                float w = std::exp(p[2]) * stride;
                float h = std::exp(p[3]) * stride;

                // 除以縮放比例，換回原圖座標
                int x = (int)((cx - w / 2) / ratio);
                int y = (int)((cy - h / 2) / ratio);
                boxes.emplace_back(x, y, (int)(w / ratio), (int)(h / ratio));
                scores.push_back(score);
            }
        }
    }

    if (idx != num_candidates) {
        std::cerr << "候選框數量不符：算出 " << idx << "，模型輸出 "
                  << num_candidates << "（輸入大小可能設錯）" << std::endl;
        return {};
    }

    // NMS：同一個人常被好幾個框重複框到，只保留分數最高的那個
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, score_thresh, nms_thresh, keep);

    std::vector<Detection> result;
    for (int i : keep) result.push_back({boxes[i], scores[i]});
    return result;
}

int main(int argc, char** argv) {
    std::string source = (argc >= 2) ? argv[1] : "0";
    std::string model  = (argc >= 3) ? argv[2] : "models/yolox_tiny.onnx";
    int size           = (argc >= 4) ? std::stoi(argv[3]) : 416;

    const float score_thresh = 0.3f;  // 分數低於這個就不算
    const float nms_thresh = 0.45f;   // 兩個框重疊超過這個比例就視為重複

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model);

    cv::VideoCapture cap;
    if (source == "0") cap.open(0);
    else cap.open(source);
    if (!cap.isOpened()) {
        std::cerr << "打不開來源：" << source << std::endl;
        return 1;
    }

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    cv::Mat frame;
    while (cap.read(frame) && !frame.empty()) {
        auto t0 = clock::now();

        // 1. 前處理
        float ratio;
        cv::Mat input = letterbox(frame, size, ratio);
        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0, cv::Size(),
                                              cv::Scalar(), false, false);
        auto t1 = clock::now();

        // 2. 推論
        net.setInput(blob);
        cv::Mat output = net.forward();
        auto t2 = clock::now();

        // 3. 後處理
        std::vector<Detection> people = decode(output, size, ratio,
                                               score_thresh, nms_thresh);
        auto t3 = clock::now();

        // 4. 畫框
        cv::Rect bounds(0, 0, frame.cols, frame.rows);
        for (const auto& d : people) {
            cv::Rect box = d.box & bounds;  // 裁掉超出畫面的部分
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            char label[32];
            std::snprintf(label, sizeof(label), "person %.2f", d.score);
            cv::putText(frame, label, box.tl() + cv::Point(0, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        // 左上角顯示人數與各階段耗時
        char info[128];
        std::snprintf(info, sizeof(info),
                      "people: %zu  pre %.1f  infer %.1f  post %.1f ms",
                      people.size(), ms(t0, t1), ms(t1, t2), ms(t2, t3));
        cv::putText(frame, info, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::imshow("detect", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}
