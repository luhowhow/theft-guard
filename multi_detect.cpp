// Step 6: one central detector serves all cameras in turn (round-robin).
// Each camera has its own decode thread (FrameGrabber); inference runs on the
// main thread, one frame at a time, using all OpenCV threads.
//
// Usage:
//   ./multi_detect test.mp4 test.mp4 test.mp4 test.mp4
//   ./multi_detect 0 test.mp4
// Press q to quit.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "detector.hpp"
#include "frame_grabber.hpp"

int main(int argc, char** argv) {
    std::vector<std::string> sources;
    for (int i = 1; i < argc; ++i) sources.push_back(argv[i]);
    if (sources.empty()) sources = {"test.mp4", "test.mp4", "test.mp4", "test.mp4"};
    const int n = (int)sources.size();

    // One decode thread per camera.
    std::vector<std::unique_ptr<FrameGrabber>> cams;
    for (const auto& s : sources) {
        cams.push_back(std::make_unique<FrameGrabber>(s));
        if (!cams.back()->start()) return 1;
    }

    // One shared detector for all cameras.
    Detector detector("models/yolox_tiny.onnx", 416);

    // Grid layout
    const int cols = (int)std::ceil(std::sqrt((double)n));
    const int rows = (n + cols - 1) / cols;
    const cv::Size tile(640, 360);
    cv::Mat grid(rows * tile.height, cols * tile.width, CV_8UC3, cv::Scalar(0, 0, 0));

    std::vector<cv::Mat> frames(n);
    std::vector<double> times(n, 0);

    // Statistics, reset every second
    std::vector<int> analyzed(n, 0);       // frames analyzed per camera in this second
    std::vector<double> analyzed_fps(n, 0);
    double infer_sum = 0;
    int infer_count = 0;
    double avg_infer = 0, total_fps = 0;
    auto last_report = std::chrono::steady_clock::now();

    int next = 0;  // the camera whose turn it is
    while (true) {
        // --- Round-robin: find the next camera that has a new frame ---
        int cam = -1;
        for (int k = 0; k < n; ++k) {
            int i = (next + k) % n;
            if (cams[i]->latest(frames[i], times[i])) {
                cam = i;
                break;
            }
        }

        if (cam < 0) {
            // No camera has a new frame yet: wait a little instead of spinning.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            next = (cam + 1) % n;  // next turn starts after this camera

            // --- Inference on this camera's latest frame ---
            std::vector<Detection> people = detector.detect(frames[cam]);
            infer_sum += detector.infer_ms;
            ++infer_count;
            ++analyzed[cam];

            // --- Draw and place into the grid ---
            for (const auto& d : people)
                cv::rectangle(frames[cam], d.box, cv::Scalar(0, 255, 0), 3);
            cv::Rect r((cam % cols) * tile.width, (cam / cols) * tile.height, tile.width, tile.height);
            cv::resize(frames[cam], grid(r), tile);
        }

        // --- Once per second: update statistics ---
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_report).count();
        if (dt >= 1.0) {
            int total = 0;
            for (int i = 0; i < n; ++i) {
                analyzed_fps[i] = analyzed[i] / dt;
                total += analyzed[i];
                analyzed[i] = 0;
            }
            total_fps = total / dt;
            avg_infer = infer_count ? infer_sum / infer_count : 0;
            infer_sum = 0;
            infer_count = 0;
            last_report = now;
            std::printf("total %.1f frames/s, infer %.1f ms, per camera:", total_fps, avg_infer);
            for (int i = 0; i < n; ++i) std::printf(" %.1f", analyzed_fps[i]);
            std::printf("\n");
        }

        // --- Display with labels ---
        cv::Mat view = grid.clone();
        for (int i = 0; i < n; ++i) {
            char label[64];
            std::snprintf(label, sizeof(label), "CAM %d  analyzed %.1f fps", i + 1, analyzed_fps[i]);
            cv::Scalar color = analyzed_fps[i] >= 5.0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(view, label,
                        cv::Point((i % cols) * tile.width + 10, (i / cols) * tile.height + 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        }
        char info[96];
        std::snprintf(info, sizeof(info), "total %.1f frames/s  infer %.1f ms", total_fps, avg_infer);
        cv::putText(view, info, cv::Point(10, view.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        cv::imshow("multi_detect", view);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}
