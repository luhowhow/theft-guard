#include <chrono>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

// 用法：./model_test 影片 模型 輸入大小
// 例如：./model_test test.mp4 models/yolox_tiny.onnx 416
int main(int argc, char** argv) {
    std::string video = (argc >= 2) ? argv[1] : "test.mp4";
    std::string model = (argc >= 3) ? argv[2] : "models/yolox_s.onnx";
    int size          = (argc >= 4) ? std::stoi(argv[3]) : 640;

    std::cout << "模型：" << model << "，輸入：" << size << "x" << size << std::endl;
    std::cout << "OpenCV 執行緒數：" << cv::getNumThreads() << std::endl;

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model);

    cv::VideoCapture cap(video);
    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) {
        std::cerr << "讀不到畫面" << std::endl;
        return 1;
    }

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(size, size),
                                          cv::Scalar(), false, false);

    // 暖身
    net.setInput(blob);
    cv::Mat output = net.forward();
    std::cout << "輸出形狀：";
    for (int i = 0; i < output.dims; ++i)
        std::cout << output.size[i] << (i + 1 < output.dims ? " x " : "\n");

    // 正式量測
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
    std::cout << "平均：" << avg << " ms（最快 " << best << "，最慢 " << worst << "）" << std::endl;
    std::cout << "每秒可處理：" << 1000.0 / avg << " 張" << std::endl;
    return 0;
}
