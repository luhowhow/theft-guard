#pragma once
// Actions taken when an ALERT fires:
//   1. Play an on-site warning sound (non-blocking, with a cooldown).
//   2. Save an event record: raw snapshot, annotated snapshot, and event.json.
//
// Each event is saved to its own folder, e.g.
//   events/20261005_143022_ID7/
//     raw.jpg         original frame, no drawings (evidence)
//     annotated.jpg   frame with boxes and zones (for notifications)
//     event.json      when, where, who, and why

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <string>
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
                         double sound_cooldown_sec = 8.0)
        : sound_path_(std::move(sound_path)),
          event_dir_(std::move(event_dir)),
          cooldown_(sound_cooldown_sec) {}

    // Play the warning and save the event. Returns the event folder path.
    std::string fire(const AlertEvent& ev, const cv::Mat& raw, const cv::Mat& annotated) {
        play_sound();
        return save_event(ev, raw, annotated);
    }

private:
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

    std::string save_event(const AlertEvent& ev, const cv::Mat& raw, const cv::Mat& annotated) {
        // Folder name from the local date/time and track ID.
        std::time_t t = std::time(nullptr);
        std::tm local = *std::localtime(&t);
        char stamp[32], iso[40];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &local);
        std::strftime(iso, sizeof(iso), "%Y-%m-%dT%H:%M:%S%z", &local);

        std::string dir = event_dir_ + "/" + stamp + "_ID" + std::to_string(ev.track_id);
        std::filesystem::create_directories(dir);

        cv::imwrite(dir + "/raw.jpg", raw);
        cv::imwrite(dir + "/annotated.jpg", annotated);

        FILE* f = std::fopen((dir + "/event.json").c_str(), "w");
        if (f) {
            std::fprintf(f,
                         "{\n"
                         "  \"time\": \"%s\",\n"
                         "  \"stream_time_sec\": %.2f,\n"
                         "  \"track_id\": %d,\n"
                         "  \"zone\": \"%s\",\n"
                         "  \"reason\": \"%s\",\n"
                         "  \"dwell_sec\": %.1f\n"
                         "}\n",
                         iso, ev.stream_time, ev.track_id, ev.zone.c_str(),
                         ev.reason.c_str(), ev.dwell);
            std::fclose(f);
        }
        return dir;
    }

    std::string sound_path_;
    std::string event_dir_;
    double cooldown_;
    bool played_ = false;
    std::chrono::steady_clock::time_point last_sound_;
};
