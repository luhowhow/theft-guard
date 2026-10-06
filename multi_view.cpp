// Step 5: open several cameras at once, one decode thread each, and show them in a grid.
// No detection yet.
//
// Usage:
//   ./multi_view test.mp4 test.mp4 test.mp4 test.mp4   # four "cameras"
//   ./multi_view 0 test.mp4                            # webcam + one video
// Press q to quit.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "frame_grabber.hpp"

int main(int argc, char** argv) {
    std::vector<std::string> sources;
    for (int i = 1; i < argc; ++i) sources.push_back(argv[i]);
    if (sources.empty()) sources = {"test.mp4", "test.mp4", "test.mp4", "test.mp4"};
    const int n = (int)sources.size();

    // One FrameGrabber (and therefore one decode thread) per camera.
    std::vector<std::unique_ptr<FrameGrabber>> cams;
    for (const auto& s : sources) {
        cams.push_back(std::make_unique<FrameGrabber>(s));
        if (!cams.back()->start()) return 1;
    }

    // Grid layout: as square as possible, each tile 640x360.
    const int cols = (int)std::ceil(std::sqrt((double)n));
    const int rows = (n + cols - 1) / cols;
    const cv::Size tile(640, 360);
    cv::Mat grid(rows * tile.height, cols * tile.width, CV_8UC3, cv::Scalar(0, 0, 0));

    std::vector<cv::Mat> frames(n);
    std::vector<double> times(n, 0);
    std::vector<uint64_t> last_count(n, 0);
    std::vector<double> decode_fps(n, 0);
    auto last_report = std::chrono::steady_clock::now();

    while (true) {
        // Copy the latest frame of each camera into its tile.
        for (int i = 0; i < n; ++i) {
            if (cams[i]->latest(frames[i], times[i])) {
                cv::Rect r((i % cols) * tile.width, (i / cols) * tile.height, tile.width, tile.height);
                cv::resize(frames[i], grid(r), tile);
            }
        }

        // Once per second, compute how many frames each camera decoded.
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_report).count();
        if (dt >= 1.0) {
            for (int i = 0; i < n; ++i) {
                uint64_t c = cams[i]->frames_decoded();
                decode_fps[i] = (c - last_count[i]) / dt;
                last_count[i] = c;
            }
            last_report = now;
        }

        // Label each tile (drawn on a copy so labels do not pile up on old tiles).
        cv::Mat view = grid.clone();
        for (int i = 0; i < n; ++i) {
            char label[64];
            std::snprintf(label, sizeof(label), "CAM %d  t=%.1fs  decode %.0f fps",
                          i + 1, times[i], decode_fps[i]);
            cv::putText(view, label,
                        cv::Point((i % cols) * tile.width + 10, (i / cols) * tile.height + 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow("multi_view", view);
        if (cv::waitKey(10) == 'q') break;
    }
    return 0;
}
