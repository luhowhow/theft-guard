// Theft Guard main program.
//   - One decode thread per camera (FrameGrabber), keeping only the latest frame.
//   - One shared detector serving the cameras in turn (round-robin).
//   - One CameraPipeline per camera: tracking, zones, behavior, alerts.
//   - All cameras shown in a grid; zones can be drawn on any tile.
//
// Usage:
//   ./guard test.mp4 test.mp4 test.mp4 test.mp4
//   ./guard 0 test.mp4
//
// Controls:
//   Click a tile or press 1-9 : choose the camera to edit (yellow border)
//   Left click   : add a zone vertex          Enter : finish the zone
//   Right click  : cancel / delete a zone     c     : clear zones of this camera
//   s            : save zones of all cameras (cam1.yml, cam2.yml, ...)
//   q            : quit

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "detector.hpp"
#include "frame_grabber.hpp"

// State shared with the mouse callback.
// (HighGUI calls the mouse callback from inside cv::waitKey on the main thread,
//  so it never runs at the same time as CameraPipeline::process.)
struct Ui {
    std::vector<std::unique_ptr<CameraPipeline>>* cams = nullptr;
    int cols = 1;
    cv::Size tile{640, 360};
    int active = 0;  // camera currently being edited
};

// Convert a point on the grid into (camera index, point in that camera's original frame).
static bool grid_to_camera(const Ui& ui, int x, int y, int& cam, cv::Point& p) {
    int c = x / ui.tile.width, r = y / ui.tile.height;
    if (c >= ui.cols) return false;
    cam = r * ui.cols + c;
    if (cam < 0 || cam >= (int)ui.cams->size()) return false;
    cv::Size fs = (*ui.cams)[cam]->frame_size();
    if (fs.width == 0) return false;  // this camera has not been processed yet
    p.x = (x - c * ui.tile.width) * fs.width / ui.tile.width;
    p.y = (y - r * ui.tile.height) * fs.height / ui.tile.height;
    return true;
}

static void on_mouse_grid(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_RBUTTONDOWN) return;
    auto* ui = static_cast<Ui*>(userdata);
    int cam;
    cv::Point p;
    if (!grid_to_camera(*ui, x, y, cam, p)) return;
    if (cam != ui->active) {
        // Switching cameras cancels a half-drawn zone on the previous one.
        (*ui->cams)[ui->active]->editor.pending.clear();
        ui->active = cam;
    }
    on_mouse_zone(event, p.x, p.y, flags, &(*ui->cams)[cam]->editor);
}

int main(int argc, char** argv) {
    std::vector<std::string> sources;
    for (int i = 1; i < argc; ++i) sources.push_back(argv[i]);
    if (sources.empty()) sources = {"test.mp4", "test.mp4", "test.mp4", "test.mp4"};
    const int n = (int)sources.size();

    Rules rules;  // thresholds: see camera.hpp

    // One decode thread and one pipeline per camera.
    std::vector<std::unique_ptr<FrameGrabber>> grabbers;
    std::vector<std::unique_ptr<CameraPipeline>> cams;
    for (int i = 0; i < n; ++i) {
        grabbers.push_back(std::make_unique<FrameGrabber>(sources[i]));
        if (!grabbers.back()->start()) return 1;
        cams.push_back(std::make_unique<CameraPipeline>(i, rules));
        std::printf("CAM %d: %s\n", i + 1, sources[i].c_str());
    }

    // One detector shared by all cameras.
    Detector detector("models/yolox_tiny.onnx", 416);

    // Grid layout
    const int cols = (int)std::ceil(std::sqrt((double)n));
    const int rows = (n + cols - 1) / cols;
    const cv::Size tile(640, 360);
    cv::Mat grid(rows * tile.height, cols * tile.width, CV_8UC3, cv::Scalar(0, 0, 0));

    Ui ui;
    ui.cams = &cams;
    ui.cols = cols;
    ui.tile = tile;
    cv::namedWindow("guard");
    cv::setMouseCallback("guard", on_mouse_grid, &ui);

    std::vector<cv::Mat> frames(n);
    std::vector<double> times(n, 0);

    // Statistics, updated once per second
    std::vector<int> analyzed(n, 0);
    std::vector<double> analyzed_fps(n, 0);
    double infer_sum = 0, avg_infer = 0, total_fps = 0;
    int infer_count = 0;
    auto last_report = std::chrono::steady_clock::now();

    int next = 0;  // the camera whose turn it is
    while (true) {
        // --- Round-robin: the next camera that has a new frame ---
        int cam = -1;
        for (int k = 0; k < n; ++k) {
            int i = (next + k) % n;
            if (grabbers[i]->latest(frames[i], times[i])) {
                cam = i;
                break;
            }
        }

        if (cam < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            next = (cam + 1) % n;

            // Detection (shared) + everything else (per camera)
            std::vector<Detection> dets = detector.detect(frames[cam]);
            cams[cam]->process(frames[cam], times[cam], dets);

            infer_sum += detector.infer_ms;
            ++infer_count;
            ++analyzed[cam];

            cv::Rect r((cam % cols) * tile.width, (cam / cols) * tile.height, tile.width, tile.height);
            cv::resize(frames[cam], grid(r), tile);
        }

        // --- Once per second: statistics ---
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
        }

        // --- Display ---
        cv::Mat view = grid.clone();
        for (int i = 0; i < n; ++i) {
            cv::Point origin((i % cols) * tile.width, (i / cols) * tile.height);
            char label[96];
            std::snprintf(label, sizeof(label), "%s  %.1f fps  people %d",
                          cams[i]->name().c_str(), analyzed_fps[i], cams[i]->people());
            cv::Scalar color = analyzed_fps[i] >= 5.0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(view, label, origin + cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        // Yellow border around the camera being edited, plus its half-drawn zone.
        {
            int c = ui.active % cols, r = ui.active / cols;
            cv::Rect border(c * tile.width, r * tile.height, tile.width, tile.height);
            cv::rectangle(view, border, cv::Scalar(0, 255, 255), 3);

            const CameraPipeline& a = *cams[ui.active];
            cv::Size fs = a.frame_size();
            if (!a.editor.pending.empty() && fs.width > 0) {
                std::vector<cv::Point> pts;
                for (const auto& p : a.editor.pending)
                    pts.emplace_back(border.x + p.x * tile.width / fs.width,
                                     border.y + p.y * tile.height / fs.height);
                cv::polylines(view, pts, false, cv::Scalar(255, 255, 255), 2);
                for (const auto& p : pts) cv::circle(view, p, 4, cv::Scalar(255, 255, 255), -1);
            }
        }

        char info[160];
        std::snprintf(info, sizeof(info),
                      "total %.1f frames/s  infer %.1f ms  | editing CAM %d  "
                      "(1-9 / click: select, Enter: finish, s: save, c: clear, q: quit)",
                      total_fps, avg_infer, ui.active + 1);
        cv::putText(view, info, cv::Point(10, view.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        cv::imshow("guard", view);
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key >= '1' && key <= '9' && key - '1' < n) {
            cams[ui.active]->editor.pending.clear();
            ui.active = key - '1';
        }
        if (key == 13 || key == 10) cams[ui.active]->editor.finish();  // Enter
        if (key == 'c') cams[ui.active]->editor.clear();
        if (key == 's') {
            for (auto& c : cams) {
                if (c->save_zone_file())
                    std::printf("%s: saved %zu zone(s) to %s\n", c->name().c_str(),
                                c->editor.zones.size(), c->zone_file().c_str());
            }
        }
    }

    for (auto& c : cams) c->finish();  // close any events still open
    return 0;
}
