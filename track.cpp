// Step 14: behavior-based alerts; one photo per second until the person leaves.
//
// States:
//   OK    (green)  : not in any zone
//   WATCH (yellow) : stayed in a zone longer than dwell_watch
//   ALERT (red)    : WATCH + at least behavior_sec of crouching or vigorous motion
//                    -> plays sounds/alert.wav and records photos under events/
//
// Usage:
//   ./track                          # webcam, zones.yml
//   ./track test.mp4                 # video file
//   ./track test.mp4 5               # WATCH after 5 s in a zone
//   ./track test.mp4 5 cam1.yml      # use a specific zone file (one per camera)
//
// Controls:
//   Left click : add a vertex to the zone being drawn
//   Enter      : finish the zone being drawn
//   Right click: cancel the zone being drawn, or delete the zone under the cursor
//   s          : save zones to the zone file
//   c          : clear all zones
//   q          : quit

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "alert.hpp"
#include "behavior.hpp"
#include "detector.hpp"
#include "tracker.hpp"
#include "zones.hpp"

// All tunable thresholds in one place.
struct Rules {
    double dwell_watch = 5.0;    // seconds in a zone before WATCH
    double behavior_sec = 2.0;   // accumulated suspicious-behavior seconds needed for ALERT
    float crouch_ratio = 0.75f;  // box height below 75% of usual height = crouching
    float motion_thresh = 0.30f; // fraction of (person box ∩ zone) pixels changing = vigorous motion
};

int main(int argc, char** argv) {
    std::string source    = (argc >= 2) ? argv[1] : "0";
    Rules rules;
    if (argc >= 3) rules.dwell_watch = std::stod(argv[2]);
    std::string zone_file = (argc >= 4) ? argv[3] : "zones.yml";
    bool is_file = (source != "0");

    Detector detector("models/yolox_tiny.onnx", 416);
    Tracker tracker;
    ZoneEditor editor;
    MotionMeter motion_meter;
    AlertAction alert_action;

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
    double last_now = 0;  // time of the latest frame, used to close events at exit

    while (cap.read(frame) && !frame.empty()) {
        cv::Size frame_size(frame.cols, frame.rows);
        cv::Mat raw = frame.clone();  // untouched copy for evidence (we draw on `frame`)

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
        last_now = now;

        std::vector<Detection> dets = detector.detect(frame);
        std::vector<Track>& tracks = tracker.update(dets, now);

        // Frame difference and zone masks (computed before drawing anything on the frame).
        motion_meter.update(frame, editor.zones);

        std::vector<bool> zone_alert(editor.zones.size(), false);
        bool any_alert = false;
        std::vector<AlertEvent> new_events;  // alerts that fired in this frame

        for (Track& tr : tracks) {
            if (tr.last_seen != now) continue;  // not detected in this frame, skip

            // Time since this track was last updated (capped to avoid big jumps).
            double dt = (tr.prev_t >= 0) ? std::min(now - tr.prev_t, 0.5) : 0.0;
            tr.prev_t = now;

            // A zone may have been deleted while this person was inside it.
            if (tr.zone_id >= (int)editor.zones.size()) tr.zone_id = -1;

            // --- Zone state (foot point = bottom-center of the box) ---
            cv::Point foot(tr.box.x + tr.box.width / 2, tr.box.y + tr.box.height);
            int z = which_zone(editor.zones, foot);

            bool reset = false;
            if (z >= 0) {
                if (tr.zone_id != z) {          // entered a zone (or moved to another one)
                    tr.zone_id = z;
                    tr.zone_enter = now;
                    reset = true;
                }
                tr.last_in_zone = now;
            } else if (tr.zone_id >= 0 && now - tr.last_in_zone > 1.0) {
                tr.zone_id = -1;                // out of the zone for over 1 second
                reset = true;
            }
            if (reset) {
                tr.alerted = false;
                tr.crouch_sec = 0;
                tr.motion_sec = 0;
            }

            // --- Behavior cues ---
            bool crouching = update_crouch(tr, dt, rules.crouch_ratio);
            float person_motion = (tr.zone_id >= 0) ? motion_meter.motion_in(tr.box, tr.zone_id) : 0.f;
            bool vigorous = person_motion >= rules.motion_thresh;
            if (tr.zone_id >= 0) {
                if (crouching) tr.crouch_sec += dt;
                if (vigorous) tr.motion_sec += dt;
            }

            // --- Decide state ---
            double dwell = (tr.zone_id >= 0) ? now - tr.zone_enter : 0;
            bool watch = tr.zone_id >= 0 && dwell >= rules.dwell_watch;
            bool alert = watch && (tr.crouch_sec >= rules.behavior_sec ||
                                   tr.motion_sec >= rules.behavior_sec);

            cv::Scalar color(0, 255, 0);                    // OK: green
            if (tr.zone_id >= 0) color = cv::Scalar(0, 200, 200);  // in zone: dim yellow
            if (watch) color = cv::Scalar(0, 255, 255);      // WATCH: yellow
            if (alert) {
                color = cv::Scalar(0, 0, 255);               // ALERT: red
                zone_alert[tr.zone_id] = true;
                any_alert = true;
                if (!tr.alerted) {
                    const char* reason = (tr.crouch_sec >= rules.behavior_sec) ? "crouching"
                                                                               : "vigorous motion";
                    std::printf("[ALERT] t=%.1fs  ID %d at %s  dwell %.1fs  reason: %s\n",
                                now, tr.id, editor.zones[tr.zone_id].name.c_str(), dwell, reason);
                    new_events.push_back({tr.id, editor.zones[tr.zone_id].name, reason, now, dwell});
                    tr.alerted = true;
                }
            }

            // --- Draw ---
            cv::rectangle(frame, tr.box, color, 2);
            cv::circle(frame, foot, 5, color, -1);
            char label[96];
            if (tr.zone_id >= 0)
                std::snprintf(label, sizeof(label), "ID %d %.1fs C%.1f M%.1f (%.0f%%)%s",
                              tr.id, dwell, tr.crouch_sec, tr.motion_sec,
                              person_motion * 100, crouching ? " [crouch]" : "");
            else
                std::snprintf(label, sizeof(label), "ID %d%s", tr.id, crouching ? " [crouch]" : "");
            cv::putText(frame, label, tr.box.tl() + cv::Point(0, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
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

        // Alert actions run after drawing, so annotated photos include boxes and zones.
        // 1. Start a new event for each alert that fired in this frame.
        for (const AlertEvent& ev : new_events) alert_action.start(ev, raw, frame);

        // 2. Keep taking photos (one per second) of everyone who is still alerting.
        std::vector<int> alerting_ids;
        for (const Track& tr : tracks) {
            if (!tr.alerted || tr.zone_id < 0) continue;
            alerting_ids.push_back(tr.id);
            if (tr.last_seen == now) alert_action.tick(tr.id, now, raw, frame);
        }

        // 3. Close events whose person left the zone or disappeared.
        alert_action.finish_missing(alerting_ids, now);

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

    alert_action.finish_all(last_now);  // close any events still open
    return 0;
}
