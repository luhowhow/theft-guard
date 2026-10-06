// Thread-scaling benchmark for inference.
// Runs the same frame through the model with 1, 2, 4 and 8 OpenCV threads
// and reports latency, speedup, efficiency and a theoretical throughput.
//
// Usage:
//   ./bench_threads test.mp4        # 50 runs per setting
//   ./bench_threads test.mp4 100    # 100 runs per setting

#include <chrono>
#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>
#include "detector.hpp"

int main(int argc, char** argv) {
    std::string source = (argc >= 2) ? argv[1] : "test.mp4";
    int runs = (argc >= 3) ? std::stoi(argv[2]) : 50;

    cv::VideoCapture cap(source);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::printf("Cannot read a frame from %s\n", source.c_str());
        return 1;
    }

    Detector detector("models/yolox_tiny.onnx", 416);
    const int max_threads = 8;  // i5-1035G1: 4 cores, 8 hardware threads

    std::printf("threads | infer ms | speedup | efficiency | streams x threads | total frames/s\n");
    std::printf("--------+----------+---------+------------+-------------------+---------------\n");

    double base_ms = 0;
    for (int t : {1, 2, 4, 8}) {
        cv::setNumThreads(t);

        // Warm-up after changing the thread count
        for (int i = 0; i < 5; ++i) detector.detect(frame);

        double sum = 0;
        for (int i = 0; i < runs; ++i) {
            detector.detect(frame);
            sum += detector.infer_ms;
        }
        double ms = sum / runs;
        if (t == 1) base_ms = ms;

        double speedup = base_ms / ms;
        double efficiency = speedup / t;
        // If the 8 hardware threads were split into (8 / t) independent streams,
        // each using t threads, this would be the ideal combined throughput
        // (ignores contention between streams).
        int streams = max_threads / t;
        double throughput = streams * 1000.0 / ms;

        std::printf("   %d    | %8.2f | %6.2fx | %9.0f%% | %8d x %-7d | %10.1f\n",
                    t, ms, speedup, efficiency * 100, streams, t, throughput);
    }
    return 0;
}
