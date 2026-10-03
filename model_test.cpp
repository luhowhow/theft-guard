#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX("models/yolox_s.onnx");
    std::cout << "模型載入成功" << std::endl;

    cv::VideoCapture cap;
    if (argc >= 2) cap.open(argv[1]);
    else cap.open(0);

    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) {
        std::cerr << "讀不到畫面" << std::endl;
        return 1;
    }

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(640, 640),
                                          cv::Scalar(), false, false);
    net.setInput(blob);

    // 暖身：第一次很慢，不計時
    auto w0 = std::chrono::steady_clock::now();
    net.forward();
    auto w1 = std::chrono::steady_clock::now();
    std::cout << "暖身（第 1 次）："
              << std::chrono::duration<double, std::milli>(w1 - w0).count()
              << " ms" << std::endl;

    // 正式量測：跑 20 次取平均
    const int runs = 20;
    double total = 0, best = 1e9, worst = 0;
    for (int i = 0; i < runs; ++i) {
        net.setInput(blob);
        auto t0 = std::chrono::steady_clock::now();
        net.forward();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total += ms;
        best = std::min(best, ms);
        worst = std::max(worst, ms);
    }

    double avg = total / runs;
    std::cout << "平均：" << avg << " ms" << std::endl;
    std::cout << "最快：" << best << " ms，最慢：" << worst << " ms" << std::endl;
    std::cout << "換算每秒可處理：" << 1000.0 / avg << " 張" << std::endl;

    return 0;
}
