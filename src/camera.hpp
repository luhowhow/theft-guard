#pragma once
// CameraPipeline: everything that happens for ONE camera after detection:
// tracking, zones, behavior cues, state (OK / WATCH / ALERT), drawing, and alert actions.
// Each camera owns its own tracker, zones (camN.yml), motion meter and event folder
// (events/camN/), so cameras never interfere with each other.
//
// Usage:
//   CameraPipeline cam(0, rules);                        // CAM 1, actions on
//   CameraPipeline cam(0, rules, false, "cam1.yml");     // evaluation: no sound/photos/notify
//   cam.process(frame, now, detections);                 // draws on `frame`
//   cam.finish();                                        // at program exit

#include <algorithm>
#include <cstdio>
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
    double dwell_watch = 5.0;      // seconds in a zone before WATCH
    double behavior_sec = 5.0;     // accumulated suspicious-behavior seconds needed for ALERT
    float crouch_ratio = 0.75f;    // box height below 75% of usual height = crouching
    float motion_thresh = 0.30f;   // fraction of (person box ∩ zone) pixels changing = vigorous motion
    double zone_exit_grace = 2.0;  // seconds outside a zone before the stay resets
    double track_missing = 2.0;    // seconds unseen before a track is removed
    bool crouch_alone = false;     // false: crouching must come with some vigorous motion
    float max_shift = 0.10f;       // box center moving more than this x height = walking
};

class CameraPipeline {
public:
    CameraPipeline(int index, const Rules& rules,
                   bool enable_actions = true, const std::string& zone_file = "")
        : name_("CAM " + std::to_string(index + 1)),
          zone_file_(zone_file.empty() ? "cam" + std::to_string(index + 1) + ".yml" : zone_file),
          rules_(rules),
          actions_(enable_actions),
          tracker_(0.3f, rules.track_missing),
          alert_("sounds/alert.wav", "events/cam" + std::to_string(index + 1)) {}

    // Process one analyzed frame. Draws boxes, zones and labels onto `frame`.
    void process(cv::Mat& frame, double now, const std::vector<Detection>& dets) {
        // First frame: remember the size and load this camera's saved zones.
        if (frame_size_.width == 0) {
            frame_size_ = frame.size();
            if (load_zones(zone_file_, editor.zones, frame_size_)) {
                editor.next_index = (int)editor.zones.size() + 1;
                if (actions_)
                    std::printf("%s: loaded %zu zone(s) from %s\n",
                                name_.c_str(), editor.zones.size(), zone_file_.c_str());
            }
        }
        last_now_ = now;
        people_ = (int)dets.size();

        cv::Mat raw = actions_ ? frame.clone() : cv::Mat();  // untouched copy for evidence
        std::vector<Track>& tracks = tracker_.update(dets, now);
        motion_.update(frame, editor.zones);  // before drawing on the frame

        std::vector<bool> zone_alert(editor.zones.size(), false);
        bool any_alert = false;
        std::vector<AlertEvent> new_events;

        for (Track& tr : tracks) {
            if (tr.last_seen != now) continue;  // not detected in this frame

            // Time since this track was last updated (capped to avoid big jumps).
            double dt = (tr.prev_t >= 0) ? std::min(now - tr.prev_t, 1.0) : 0.0;
            tr.prev_t = now;

            if (tr.zone_id >= (int)editor.zones.size()) tr.zone_id = -1;  // zone was deleted

            // --- Zone state (foot point = bottom-center of the box) ---
            cv::Point foot(tr.box.x + tr.box.width / 2, tr.box.y + tr.box.height);
            int z = which_zone(editor.zones, foot);

            bool reset = false;
            if (z >= 0) {
                if (tr.zone_id != z) {
                    tr.zone_id = z;
                    tr.zone_enter = now;
                    reset = true;
                }
                tr.last_in_zone = now;
            } else if (tr.zone_id >= 0 && now - tr.last_in_zone > rules_.zone_exit_grace) {
                tr.zone_id = -1;
                reset = true;
            }
            if (reset) {
                tr.alerted = false;
                tr.crouch_sec = 0;
                tr.motion_sec = 0;
            }

            // --- Behavior cues ---
            bool crouching = update_crouch(tr, dt, rules_.crouch_ratio);
            // Vigorous motion = limbs moving while the body stays in place.
            float person_motion = 0.f;
            if (tr.zone_id >= 0 && tr.last_box.area() > 0) {
                cv::Point2f c0(tr.last_box.x + tr.last_box.width / 2.f, tr.last_box.y + tr.last_box.height / 2.f);
                cv::Point2f c1(tr.box.x + tr.box.width / 2.f, tr.box.y + tr.box.height / 2.f);
                float shift = (float)cv::norm(c1 - c0) / std::max(1, tr.box.height);
                if (shift < rules_.max_shift)
                    person_motion = motion_.aligned_motion(tr.last_box, tr.box, tr.zone_id);
            }
            bool vigorous = person_motion >= rules_.motion_thresh;
            if (tr.zone_id >= 0) {
                if (crouching) tr.crouch_sec += dt;
                if (vigorous) tr.motion_sec += dt;
                // Statistics for tuning thresholds
                max_crouch_sec_ = std::max(max_crouch_sec_, tr.crouch_sec);
                max_motion_sec_ = std::max(max_motion_sec_, tr.motion_sec);
                max_motion_pct_ = std::max(max_motion_pct_, person_motion);
            }

            // --- Decide state ---
            double dwell = (tr.zone_id >= 0) ? now - tr.zone_enter : 0;
            bool watch = tr.zone_id >= 0 && dwell >= rules_.dwell_watch;
            // Vigorous motion alone is enough. Crouching alone is enough only if crouch_alone
            // is set; otherwise it must come with at least half the required motion time.
            bool motion_ok = tr.motion_sec >= rules_.behavior_sec;
            bool crouch_ok = tr.crouch_sec >= rules_.behavior_sec &&
                             (rules_.crouch_alone || tr.motion_sec >= rules_.behavior_sec * 0.5);
            bool alert = watch && (motion_ok || crouch_ok);
            if (watch) saw_watch_ = true;

            cv::Scalar color(0, 255, 0);                           // OK: green
            // in zone but before WATCH: stay green
            if (watch) color = cv::Scalar(0, 255, 255);             // WATCH: yellow
            if (alert) {
                color = cv::Scalar(0, 0, 255);                      // ALERT: red
                zone_alert[tr.zone_id] = true;
                any_alert = true;
                if (!tr.alerted) {
                    ++alerts_;
                    const char* reason = crouch_ok ? (motion_ok ? "crouch + motion" : "crouching")
                                                   : "vigorous motion";
                    last_reason_ = reason;
                    std::string where = name_ + " / " + editor.zones[tr.zone_id].name;
                    if (actions_)
                        std::printf("[ALERT] %s  t=%.1fs  ID %d  dwell %.1fs  reason: %s\n",
                                    where.c_str(), now, tr.id, dwell, reason);
                    new_events.push_back({tr.id, where, reason, now, dwell});
                    tr.alerted = true;
                }
            }

            // --- Draw ---
            cv::rectangle(frame, tr.box, color, 3);
            cv::circle(frame, foot, 6, color, -1);
            char label[96];
            if (tr.zone_id >= 0)
                std::snprintf(label, sizeof(label), "ID %d %.1fs C%.1f M%.1f (%.0f%%)%s",
                              tr.id, dwell, tr.crouch_sec, tr.motion_sec,
                              person_motion * 100, crouching ? " [crouch]" : "");
            else
                std::snprintf(label, sizeof(label), "ID %d%s", tr.id, crouching ? " [crouch]" : "");
            cv::putText(frame, label, tr.box.tl() + cv::Point(0, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
            tr.last_box = tr.box;  // for the next frame's aligned motion
        }

        draw_zones(frame, editor, zone_alert);
        if (any_alert) {
            cv::putText(frame, "ALERT", cv::Point(frame.cols - 260, 70),
                        cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 5);
        }

        if (!actions_) return;  // evaluation mode: no sound, photos or notifications

        // --- Alert actions (after drawing, so annotated photos include boxes and zones) ---
        for (const AlertEvent& ev : new_events) alert_.start(ev, raw, frame);
        std::vector<int> alerting_ids;
        for (const Track& tr : tracks) {
            if (!tr.alerted || tr.zone_id < 0) continue;
            alerting_ids.push_back(tr.id);
            if (tr.last_seen == now) alert_.tick(tr.id, now, raw, frame);
        }
        alert_.finish_missing(alerting_ids, now);
    }

    // Close any events still open (call at program exit).
    void finish() {
        if (actions_) alert_.finish_all(last_now_);
    }

    bool save_zone_file() {
        if (frame_size_.width == 0) return false;
        return save_zones(zone_file_, editor.zones, frame_size_);
    }

    const std::string& name() const { return name_; }
    const std::string& zone_file() const { return zone_file_; }
    cv::Size frame_size() const { return frame_size_; }
    int people() const { return people_; }

    // Statistics (used by the evaluation tool)
    int alerts() const { return alerts_; }
    bool saw_watch() const { return saw_watch_; }
    const std::string& last_reason() const { return last_reason_; }
    double max_crouch_sec() const { return max_crouch_sec_; }
    double max_motion_sec() const { return max_motion_sec_; }
    float max_motion_pct() const { return max_motion_pct_; }

    ZoneEditor editor;  // public: the mouse and keyboard handlers edit it directly

private:
    std::string name_;
    std::string zone_file_;
    Rules rules_;
    bool actions_;
    Tracker tracker_;
    MotionMeter motion_;
    AlertAction alert_;
    cv::Size frame_size_{0, 0};
    double last_now_ = 0;
    int people_ = 0;

    int alerts_ = 0;
    bool saw_watch_ = false;
    std::string last_reason_;
    double max_crouch_sec_ = 0, max_motion_sec_ = 0;
    float max_motion_pct_ = 0;
};
