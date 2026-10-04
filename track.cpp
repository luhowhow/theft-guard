// Step 10: track people and check dwell time in multiple alert zones.
//
// Usage:
//   ./track                          # webcam, alert after 5 s, zones.yml
//   ./track test.mp4                 # video file
//   ./track test.mp4 10              # alert after 10 s
//   ./track test.mp4 10 cam1.yml     # use a specific zone file (one per camera)
//
// Controls:
//   Left click : add a vertex to the zone being drawn
//   Enter      : finish the zone being drawn
//   Right click: cancel the zone being drawn, or delete the zone under the cursor
//   s          : save zones to the zone file
//   c          : clear all zones
//   q          : quit

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "detector.hpp"
#include "tracker.hpp"
#include "zones.hpp"

int main(int argc, char** argv) {
    std::string source    = (argc >= 2) ? argv[1] : "0";
    double dwell_limit    = (argc >= 3) ? std::stod(argv[2]) : 5.0;
    std::string zone_file = (argc >= 4) ? argv[3] : "zones.yml";
    bool is_file = (source != "0");

    Detector detector("models/yolox_tiny.onnx", 416);
    Tracker tracker;
    ZoneEditor editor;

    cv::VideoCapture cap;
    if (is_file) cap.open(source);
    else cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open source: " << source << std::endl;
        return 1;
    }

    cv::namedWindow("track");
    cv::setMouseCallback("track", on_mouse_zone, &editor);

    auto start = std::chrono::steady_clock::now();
    cv::Mat frame;
    bool first = true;

    while (cap.read(frame) && !frame.empty()) {
        cv::Size frame_size(frame.cols, frame.rows);

        // Load saved zones on the first frame (we need the frame size first).
        if (first) {
            if (load_zones(zone_file, editor.zones, frame_size)) {
                editor.next_index = (int)editor.zones.size() + 1;
                std::printf("Loaded %zu zone(s) from %s\n", editor.zones.size(), zone_file.c_str());
            } else {
                std::printf("No zone file found. Left-click to draw a zone, Enter to finish, s to save.\n");
            }
            first = false;
        }

        // Current time in seconds: video files use their own timestamps,
        // the webcam uses wall-clock time.
        double now = is_file
            ? cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0
            : std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        std::vector<Detection> dets = detector.detect(frame);
        std::vector<Track>& tracks = tracker.update(dets, now);

        std::vector<bool> zone_alert(editor.zones.size(), false);
        bool any_alert = false;

        for (Track& tr : tracks) {
            if (tr.last_seen != now) continue;  // not detected in this frame, skip

            // A zone may have been deleted while this person was inside it.
            if (tr.zone_id >= (int)editor.zones.size()) {
                tr.zone_id = -1;
                tr.alerted = false;
            }

            // Use the foot point (bottom-center of the box) to decide the position.
            cv::Point foot(tr.box.x + tr.box.width / 2, tr.box.y + tr.box.height);
            int z = which_zone(editor.zones, foot);

            if (z >= 0) {
                if (tr.zone_id != z) {          // entered a zone (or moved to another one)
                    tr.zone_id = z;
                    tr.zone_enter = now;
                    tr.alerted = false;
                }
                tr.last_in_zone = now;
            } else if (tr.zone_id >= 0 && now - tr.last_in_zone > 1.0) {
                tr.zone_id = -1;                // out of the zone for over 1 second: reset
                tr.alerted = false;
            }

            double dwell = (tr.zone_id >= 0) ? now - tr.zone_enter : 0;

            // Color: green = normal, yellow = in a zone, red = stayed too long
            cv::Scalar color(0, 255, 0);
            if (tr.zone_id >= 0) color = cv::Scalar(0, 255, 255);
            if (tr.zone_id >= 0 && dwell >= dwell_limit) {
                color = cv::Scalar(0, 0, 255);
                zone_alert[tr.zone_id] = true;
                any_alert = true;
                if (!tr.alerted) {
                    std::printf("[ALERT] t=%.1fs  ID %d stayed at %s for %.1f s\n",
                                now, tr.id, editor.zones[tr.zone_id].name.c_str(), dwell);
                    tr.alerted = true;
                }
            }

            cv::rectangle(frame, tr.box, color, 2);
            cv::circle(frame, foot, 5, color, -1);
            char label[64];
            if (tr.zone_id >= 0)
                std::snprintf(label, sizeof(label), "ID %d  %.1fs", tr.id, dwell);
            else
                std::snprintf(label, sizeof(label), "ID %d", tr.id);
            cv::putText(frame, label, tr.box.tl() + cv::Point(0, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        }

        draw_zones(frame, editor, zone_alert);

        if (any_alert) {
            cv::putText(frame, "ALERT", cv::Point(frame.cols - 200, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 4);
        }

        char info[128];
        std::snprintf(info, sizeof(info), "t=%.1fs  people %zu  zones %zu  infer %.1f ms",
                      now, dets.size(), editor.zones.size(), detector.infer_ms);
        cv::putText(frame, info, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "L-click: add point  Enter: finish  R-click: delete  s: save  c: clear",
                    cv::Point(10, frame.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        cv::imshow("track", frame);
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 13 || key == 10) editor.finish();  // Enter
        if (key == 'c') editor.clear();
        if (key == 's') {
            if (save_zones(zone_file, editor.zones, frame_size))
                std::printf("Saved %zu zone(s) to %s\n", editor.zones.size(), zone_file.c_str());
            else
                std::printf("Failed to save %s\n", zone_file.c_str());
        }
    }
    return 0;
}
