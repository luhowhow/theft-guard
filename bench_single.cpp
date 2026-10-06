// Baseline benchmark: one camera, one thread, no display.
// Measures how long each stage takes per frame.
//
// Usage:
//   ./bench_single test.mp4          # 300 frames
//   ./bench_single test.mp4 500      # 500 frames

#include <chrono>
#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>
#include "detector.hpp"

int main(int argc, char** argv) {
    std::string source = (argc >= 2) ? argv[1] : "test.mp4";
    int frames = (argc >= 3) ? std::stoi(argv[2]) : 300;

    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        std::printf("Cannot open %s\n", source.c_str());
        return 1;
    }
    Detector detector("models/yolox_tiny.onnx", 416);

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Read the next frame; loop back to the start when the video ends.
    cv::Mat frame;
    auto read_frame = [&]() {
        if (!cap.read(frame) || frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            cap.read(frame);
        }
    };

    // Warm-up: the first few inferences are slow, so do not time them.
    for (int i = 0; i < 10; ++i) {
        read_frame();
        detector.detect(frame);
    }

    double t_read = 0, t_pre = 0, t_infer = 0, t_post = 0;
    auto start = clock::now();
    for (int i = 0; i < frames; ++i) {
        auto t0 = clock::now();
        read_frame();
        auto t1 = clock::now();
        detector.detect(frame);

        t_read  += ms(t0, t1);
        t_pre   += detector.pre_ms;
        t_infer += detector.infer_ms;
        t_post  += detector.post_ms;
    }
    double total = ms(start, clock::now());

    double per_frame = total / frames;
    double fps = 1000.0 / per_frame;
    std::printf("Source: %s, %d frames, OpenCV threads: %d\n",
                source.c_str(), frames, cv::getNumThreads());
    std::printf("  read (decode) : %7.2f ms  (%4.1f%%)\n", t_read / frames,  100 * t_read / total);
    std::printf("  pre-process   : %7.2f ms  (%4.1f%%)\n", t_pre / frames,   100 * t_pre / total);
    std::printf("  inference     : %7.2f ms  (%4.1f%%)\n", t_infer / frames, 100 * t_infer / total);
    std::printf("  post-process  : %7.2f ms  (%4.1f%%)\n", t_post / frames,  100 * t_post / total);
    std::printf("  total         : %7.2f ms per frame -> %.1f frames/s\n", per_frame, fps);
    std::printf("  cameras at 5 FPS each: %.1f\n", fps / 5.0);
    return 0;
}
