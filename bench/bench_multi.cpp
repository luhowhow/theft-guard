// Multi-stream benchmark: N cameras run at the same time, one thread each.
// Each stream has its own VideoCapture and its own Detector (a cv::dnn::Net
// must not be used by several threads at once).
//
// Usage:
//   ./bench_multi test.mp4           # 8 s per test, OpenCV threads = 1
//   ./bench_multi test.mp4 8 8       # 8 s per test, OpenCV threads = 8

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "detector.hpp"

struct Result {
    long frames = 0;          // frames fully processed during the timed window
    double infer_ms_sum = 0;  // sum of inference times, for the average latency
};

int main(int argc, char** argv) {
    std::string source = (argc >= 2) ? argv[1] : "test.mp4";
    double seconds     = (argc >= 3) ? std::stod(argv[2]) : 8.0;
    int cv_threads     = (argc >= 4) ? std::stoi(argv[3]) : 1;

    // Threads used *inside* each inference. 1 = each stream runs on a single thread.
    cv::setNumThreads(cv_threads);

    std::printf("Source: %s, %.0f s per test, OpenCV threads per inference: %d\n\n",
                source.c_str(), seconds, cv_threads);
    std::printf("streams | total frames/s | per stream | avg infer ms | speedup | cameras @5 FPS\n");
    std::printf("--------+----------------+------------+--------------+---------+---------------\n");

    double base_fps = 0;
    for (int n : {1, 2, 4, 8}) {
        // One capture and one detector per stream, created before timing starts.
        std::vector<cv::VideoCapture> caps(n);
        std::vector<std::unique_ptr<Detector>> dets;
        for (int i = 0; i < n; ++i) {
            if (!caps[i].open(source)) {
                std::printf("Cannot open %s\n", source.c_str());
                return 1;
            }
            dets.push_back(std::make_unique<Detector>("models/yolox_tiny.onnx", 416));
        }

        std::atomic<int> ready{0};
        std::atomic<bool> go{false}, stop{false};
        std::vector<Result> results(n);
        std::vector<std::thread> workers;

        for (int i = 0; i < n; ++i) {
            workers.emplace_back([&, i]() {
                cv::Mat frame;
                // Read the next frame; loop back to the start when the video ends.
                auto read_frame = [&]() {
                    if (!caps[i].read(frame) || frame.empty()) {
                        caps[i].set(cv::CAP_PROP_POS_FRAMES, 0);
                        caps[i].read(frame);
                    }
                };

                // Warm-up, not timed.
                for (int w = 0; w < 3; ++w) {
                    read_frame();
                    dets[i]->detect(frame);
                }

                // Wait until every stream is ready, then start together.
                ++ready;
                while (!go) std::this_thread::yield();

                while (!stop) {
                    read_frame();
                    dets[i]->detect(frame);
                    if (stop) break;  // finished after the window closed: do not count
                    results[i].frames++;
                    results[i].infer_ms_sum += dets[i]->infer_ms;
                }
            });
        }

        while (ready < n) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto t0 = std::chrono::steady_clock::now();
        go = true;
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop = true;
        auto t1 = std::chrono::steady_clock::now();
        for (auto& w : workers) w.join();

        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        long total_frames = 0;
        double infer_sum = 0;
        for (const auto& r : results) {
            total_frames += r.frames;
            infer_sum += r.infer_ms_sum;
        }
        double total_fps = total_frames / elapsed;
        double avg_infer = total_frames ? infer_sum / total_frames : 0;
        if (n == 1) base_fps = total_fps;

        std::printf("   %d    | %14.1f | %10.1f | %12.1f | %6.2fx | %13.1f\n",
                    n, total_fps, total_fps / n, avg_infer, total_fps / base_fps, total_fps / 5.0);
    }
    return 0;
}
