#pragma once
// Actions taken when an ALERT fires:
//   1. Play an on-site warning sound (non-blocking, with a cooldown).
//   2. Record the event: one photo per second until the person leaves the zone.
//
// Each event is saved to its own folder, e.g.
//   events/20261006_143022_ID7/
//     raw/0001.jpg, 0002.jpg, ...        original frames, no drawings (evidence)
//     annotated/0001.jpg, 0002.jpg, ...  frames with boxes and zones (for review)
//     event.json                         when, where, who, why, how long
//
// Usage per frame:
//   action.start(ev, raw, annotated);           // when a new alert fires
//   action.tick(track_id, now, raw, annotated); // for every person still alerting
//   action.finish_missing(alerting_ids, now);   // close events whose person left
//   action.finish_all(now);                     // at program exit

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct AlertEvent {
    int track_id;        // which person
    std::string zone;    // which zone, e.g. "Kiosk 1"
    std::string reason;  // "crouching" or "vigorous motion"
    double stream_time;  // seconds since start (video timestamp for files)
    double dwell;        // seconds in the zone when the alert fired
};

class AlertAction {
public:
    explicit AlertAction(std::string sound_path = "sounds/alert.wav",
                         std::string event_dir = "events",
                         double capture_interval_sec = 1.0,
                         int max_photos = 600,
                         double sound_cooldown_sec = 8.0)
        : sound_path_(std::move(sound_path)),
          event_dir_(std::move(event_dir)),
          interval_(capture_interval_sec),
          max_photos_(max_photos),
          cooldown_(sound_cooldown_sec) {}

    // A new alert fired: play the warning, create the event folder, take the first photo.
    void start(const AlertEvent& ev, const cv::Mat& raw, const cv::Mat& annotated) {
        play_sound();

        Active a;
        a.info = ev;
        a.start_iso = local_time("%Y-%m-%dT%H:%M:%S%z");
        a.dir = event_dir_ + "/" + local_time("%Y%m%d_%H%M%S") + "_ID" + std::to_string(ev.track_id);
        std::filesystem::create_directories(a.dir + "/raw");
        std::filesystem::create_directories(a.dir + "/annotated");

        capture(a, ev.stream_time, raw, annotated);
        write_json(a, false, ev.stream_time);
        std::printf("        event started: %s\n", a.dir.c_str());

        active_[ev.track_id] = a;
    }

    // Take another photo if at least `interval_` seconds have passed.
    void tick(int track_id, double now, const cv::Mat& raw, const cv::Mat& annotated) {
        auto it = active_.find(track_id);
        if (it == active_.end()) return;
        Active& a = it->second;
        if (a.count >= max_photos_) return;
        if (now - a.last_capture >= interval_) capture(a, now, raw, annotated);
    }

    // Close every active event whose person is no longer alerting
    // (left the zone, or disappeared from view).
    void finish_missing(const std::vector<int>& alerting_ids, double now) {
        for (auto it = active_.begin(); it != active_.end();) {
            bool still = std::find(alerting_ids.begin(), alerting_ids.end(), it->first) != alerting_ids.end();
            if (still) {
                ++it;
            } else {
                finish(it->second, now);
                it = active_.erase(it);
            }
        }
    }

    // Close all remaining events (call at program exit).
    void finish_all(double now) {
        for (auto& [id, a] : active_) finish(a, now);
        active_.clear();
    }

private:
    struct Active {
        AlertEvent info;
        std::string dir;
        std::string start_iso;
        double last_capture = -1;
        int count = 0;
    };

    static std::string local_time(const char* fmt) {
        std::time_t t = std::time(nullptr);
        std::tm local = *std::localtime(&t);
        char buf[64];
        std::strftime(buf, sizeof(buf), fmt, &local);
        return buf;
    }

    void capture(Active& a, double now, const cv::Mat& raw, const cv::Mat& annotated) {
        ++a.count;
        char name[16];
        std::snprintf(name, sizeof(name), "%04d.jpg", a.count);
        cv::imwrite(a.dir + "/raw/" + name, raw);
        cv::imwrite(a.dir + "/annotated/" + name, annotated);
        a.last_capture = now;
    }

    void finish(const Active& a, double now) {
        write_json(a, true, now);
        std::printf("        event finished: %s  (%d photos, %.1f s)\n",
                    a.dir.c_str(), a.count, now - a.info.stream_time);
    }

    void write_json(const Active& a, bool finished, double now) {
        FILE* f = std::fopen((a.dir + "/event.json").c_str(), "w");
        if (!f) return;
        std::fprintf(f,
                     "{\n"
                     "  \"status\": \"%s\",\n"
                     "  \"start_time\": \"%s\",\n"
                     "  \"end_time\": \"%s\",\n"
                     "  \"start_stream_sec\": %.2f,\n"
                     "  \"duration_sec\": %.1f,\n"
                     "  \"track_id\": %d,\n"
                     "  \"zone\": \"%s\",\n"
                     "  \"reason\": \"%s\",\n"
                     "  \"dwell_at_alert_sec\": %.1f,\n"
                     "  \"photos\": %d,\n"
                     "  \"photo_interval_sec\": %.1f\n"
                     "}\n",
                     finished ? "finished" : "ongoing",
                     a.start_iso.c_str(),
                     finished ? local_time("%Y-%m-%dT%H:%M:%S%z").c_str() : "",
                     a.info.stream_time,
                     finished ? now - a.info.stream_time : 0.0,
                     a.info.track_id, a.info.zone.c_str(), a.info.reason.c_str(),
                     a.info.dwell, a.count, interval_);
        std::fclose(f);
    }

    void play_sound() {
        auto now = std::chrono::steady_clock::now();
        // Do not replay the sound while the previous one may still be playing.
        if (played_ && std::chrono::duration<double>(now - last_sound_).count() < cooldown_)
            return;
        if (!std::filesystem::exists(sound_path_)) {
            std::printf("Sound file not found: %s\n", sound_path_.c_str());
            return;
        }
        // The trailing '&' runs aplay in the background, so the video loop does not wait.
        std::string cmd = "aplay -q \"" + sound_path_ + "\" &";
        int ret = std::system(cmd.c_str());
        (void)ret;
        played_ = true;
        last_sound_ = now;
    }

    std::string sound_path_;
    std::string event_dir_;
    double interval_;
    int max_photos_;
    double cooldown_;
    bool played_ = false;
    std::chrono::steady_clock::time_point last_sound_;
    std::map<int, Active> active_;  // track ID -> its ongoing event
};
