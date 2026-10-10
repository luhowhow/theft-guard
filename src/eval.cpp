// Batch evaluation: run each video through the full pipeline (detection, tracking,
// zones, behavior) WITHOUT sound, photos or notifications, then compare the result
// with the expected label taken from the file name.
//
// Expected label: files starting with "tool_attack", "hand_attack" or "screw"
// should ALERT; all other files should not.
//
// Usage:
//   ./build/eval data/clips/*.mp4                          # current rules, per-video table
//   ./build/eval data/clips/*.mp4 --motion 0.5 --crouch-alone 0
//   ./build/eval data/clips/*.mp4 --sweep                  # compare several rule settings
//   ./build/eval data/clips/*.mp4 --no-cache               # force detection again
//
// Detection takes ~90% of the time and does not depend on the rules, so its results
// are cached in .eval_cache/. A cache entry is reused only if the video file, model
// and analysis rate are unchanged.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "detector.hpp"

namespace fs = std::filesystem;

static const char* kModel = "models/yolox_tiny.onnx";
static const int kInputSize = 416;
static const char* kCacheDir = ".eval_cache";

struct Video {
    std::string path, name;
    int step = 1;                                // analyze every `step`-th frame
    bool expect = false;                         // should this video raise an ALERT?
    std::vector<std::vector<Detection>> dets;    // detections of each analyzed frame
};

struct Outcome {
    bool alerted = false, watch = false;
    std::string reason;
    double max_c = 0, max_m = 0;
    float max_pct = 0;
};

struct Score {
    int tp = 0, fn = 0, fp = 0, tn = 0;
    double recall() const { return tp + fn ? 100.0 * tp / (tp + fn) : 0; }
    double false_alarm() const { return fp + tn ? 100.0 * fp / (fp + tn) : 0; }
};

static bool should_alert(const std::string& name) {
    for (const char* p : {"tool_attack", "hand_attack", "screw"})
        if (name.rfind(p, 0) == 0) return true;
    return false;
}

// A string that changes whenever the video file, model or analysis rate changes.
static std::string cache_key(const std::string& path, double analysis_fps) {
    auto size = fs::file_size(path);
    auto mtime = fs::last_write_time(path).time_since_epoch().count();
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s|%d|%.2f|%llu|%lld", kModel, kInputSize, analysis_fps,
                  (unsigned long long)size, (long long)mtime);
    return buf;
}

static bool load_cache(Video& v, const std::string& key) {
    cv::FileStorage f(std::string(kCacheDir) + "/" + v.name + ".yml", cv::FileStorage::READ);
    if (!f.isOpened()) return false;
    std::string stored;
    f["key"] >> stored;
    if (stored != key) return false;
    f["step"] >> v.step;
    v.dets.clear();
    for (const auto& node : f["frames"]) {
        std::vector<float> flat;  // x, y, w, h, score for each person
        node >> flat;
        std::vector<Detection> frame_dets;
        for (size_t i = 0; i + 4 < flat.size(); i += 5)
            frame_dets.push_back({cv::Rect((int)flat[i], (int)flat[i + 1], (int)flat[i + 2], (int)flat[i + 3]),
                                  flat[i + 4]});
        v.dets.push_back(frame_dets);
    }
    return true;
}

static void save_cache(const Video& v, const std::string& key) {
    fs::create_directories(kCacheDir);
    cv::FileStorage f(std::string(kCacheDir) + "/" + v.name + ".yml", cv::FileStorage::WRITE);
    f << "key" << key << "step" << v.step << "frames" << "[";
    for (const auto& frame_dets : v.dets) {
        std::vector<float> flat;
        for (const auto& d : frame_dets) {
            flat.insert(flat.end(), {(float)d.box.x, (float)d.box.y, (float)d.box.width,
                                     (float)d.box.height, d.score});
        }
        f << flat;
    }
    f << "]";
}

// Replay one video with cached detections under the given rules.
static Outcome replay(const Video& v, const Rules& rules, const std::string& zone_file) {
    cv::VideoCapture cap(v.path);
    CameraPipeline cam(0, rules, /*enable_actions=*/false, zone_file);
    cv::Mat frame;
    long idx = 0;
    size_t k = 0;
    while (cap.grab()) {
        if (idx++ % v.step != 0) continue;
        if (k >= v.dets.size() || !cap.retrieve(frame) || frame.empty()) break;
        double now = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        cam.process(frame, now, v.dets[k++]);
    }
    Outcome o;
    o.alerted = cam.alerts() > 0;
    o.watch = cam.saw_watch();
    o.reason = o.alerted ? cam.last_reason() : "-";
    o.max_c = cam.max_crouch_sec();
    o.max_m = cam.max_motion_sec();
    o.max_pct = cam.max_motion_pct();
    return o;
}

static Score score(const std::vector<Video>& videos, const std::vector<Outcome>& out) {
    Score s;
    for (size_t i = 0; i < videos.size(); ++i) {
        bool e = videos[i].expect, a = out[i].alerted;
        if (e && a) ++s.tp; else if (e) ++s.fn; else if (a) ++s.fp; else ++s.tn;
    }
    return s;
}

static void print_table(const std::vector<Video>& videos, const std::vector<Outcome>& out) {
    std::printf("%-22s %-7s %-6s %-6s %-17s %6s %6s %6s\n",
                "video", "expect", "result", "check", "reason", "maxC", "maxM", "max%");
    for (size_t i = 0; i < videos.size(); ++i) {
        const Outcome& o = out[i];
        const char* result = o.alerted ? "ALERT" : (o.watch ? "WATCH" : "OK");
        std::printf("%-22s %-7s %-6s %-6s %-17s %6.1f %6.1f %5.0f%%\n",
                    videos[i].name.c_str(), videos[i].expect ? "ALERT" : "no", result,
                    videos[i].expect == o.alerted ? "ok" : "WRONG", o.reason.c_str(),
                    o.max_c, o.max_m, o.max_pct * 100);
    }
}

static void print_summary(const Score& s) {
    std::printf("  thefts caught (TP): %d / %d   recall %.0f%%\n", s.tp, s.tp + s.fn, s.recall());
    std::printf("  false alarms  (FP): %d / %d   false alarm rate %.0f%%\n",
                s.fp, s.fp + s.tn, s.false_alarm());
}

int main(int argc, char** argv) {
    std::vector<std::string> files;
    double analysis_fps = 5.0;
    std::string zone_file = "cam1.yml";
    bool sweep = false, use_cache = true;
    Rules base;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--fps" && i + 1 < argc) analysis_fps = std::stod(argv[++i]);
        else if (a == "--zones" && i + 1 < argc) zone_file = argv[++i];
        else if (a == "--motion" && i + 1 < argc) base.motion_thresh = std::stof(argv[++i]);
        else if (a == "--behavior" && i + 1 < argc) base.behavior_sec = std::stod(argv[++i]);
        else if (a == "--crouch-alone" && i + 1 < argc) base.crouch_alone = std::stoi(argv[++i]) != 0;
        else if (a == "--sweep") sweep = true;
        else if (a == "--no-cache") use_cache = false;
        else if (a.rfind("--", 0) == 0) {
            std::printf("Unknown option: %s\n", a.c_str());
            return 1;
        } else files.push_back(a);
    }
    if (files.empty()) {
        std::printf("Usage: ./build/eval data/clips/*.mp4 [--sweep] [--motion 0.3] [--behavior 2] "
                    "[--crouch-alone 1] [--fps 5] [--zones cam1.yml] [--no-cache]\n");
        return 1;
    }
    if (!fs::exists(zone_file)) {
        std::printf("Zone file %s not found. Draw zones with ./build/guard <video> and press s.\n",
                    zone_file.c_str());
        return 1;
    }
    auto t0 = std::chrono::steady_clock::now();

    // --- Pass 1: detection, once per video (or loaded from the cache) ---
    std::unique_ptr<Detector> detector;  // loaded only if some video is not cached
    std::vector<Video> videos;
    std::printf("Detection (%.0f analyzed frames per second of video):\n", analysis_fps);
    for (const auto& path : files) {
        Video v;
        v.path = path;
        v.name = fs::path(path).filename().string();
        v.expect = should_alert(v.name);
        if (!fs::exists(path)) {
            std::printf("  %-22s not found, skipped\n", v.name.c_str());
            continue;
        }
        std::string key = cache_key(path, analysis_fps);
        if (use_cache && load_cache(v, key)) {
            std::printf("  %-22s %4zu frames (cached)\n", v.name.c_str(), v.dets.size());
            videos.push_back(std::move(v));
            continue;
        }

        cv::VideoCapture cap(path);
        if (!cap.isOpened()) {
            std::printf("  %-22s cannot open, skipped\n", v.name.c_str());
            continue;
        }
        if (!detector) detector = std::make_unique<Detector>(kModel, kInputSize);
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (!(fps > 0 && fps < 240)) fps = 30.0;
        v.step = std::max(1, (int)std::lround(fps / analysis_fps));
        cv::Mat frame;
        long idx = 0;
        while (cap.grab()) {
            if (idx++ % v.step != 0) continue;
            if (!cap.retrieve(frame) || frame.empty()) break;
            v.dets.push_back(detector->detect(frame));
        }
        save_cache(v, key);
        std::printf("  %-22s %4zu frames (detected)\n", v.name.c_str(), v.dets.size());
        std::fflush(stdout);
        videos.push_back(std::move(v));
    }

    if (!sweep) {
        // --- Single rule setting: per-video table ---
        std::vector<Outcome> out;
        for (const auto& v : videos) out.push_back(replay(v, base, zone_file));
        std::printf("\nRules: watch %.1fs, behavior %.1fs, crouch %.2f, crouch alone %s, motion %.0f%%\n\n",
                    base.dwell_watch, base.behavior_sec, base.crouch_ratio,
                    base.crouch_alone ? "yes" : "no", base.motion_thresh * 100);
        print_table(videos, out);
        std::printf("\nSummary\n");
        print_summary(score(videos, out));
    } else {
        // --- Sweep: try several rule settings in parallel threads ---
        std::vector<Rules> configs;
        for (float m : {0.10f, 0.15f, 0.20f, 0.30f})
            for (bool ca : {true, false}) {
                Rules r = base;
                r.motion_thresh = m;
                r.crouch_alone = ca;
                configs.push_back(r);
            }

        cv::setNumThreads(1);  // one thread per configuration instead
        std::vector<std::vector<Outcome>> results(configs.size());
        std::vector<std::thread> workers;
        for (size_t c = 0; c < configs.size(); ++c)
            workers.emplace_back([&, c]() {
                for (const auto& v : videos) results[c].push_back(replay(v, configs[c], zone_file));
            });
        for (auto& w : workers) w.join();

        std::printf("\n%-8s %-13s %-9s %-9s %-8s %-12s\n",
                    "motion", "crouch alone", "caught", "false", "recall", "false alarm");
        size_t best = 0;
        double best_score = -1e9;
        for (size_t c = 0; c < configs.size(); ++c) {
            Score s = score(videos, results[c]);
            std::printf("%5.0f%%   %-13s %2d / %-4d %2d / %-4d %5.0f%%   %6.0f%%\n",
                        configs[c].motion_thresh * 100, configs[c].crouch_alone ? "yes" : "no",
                        s.tp, s.tp + s.fn, s.fp, s.fp + s.tn, s.recall(), s.false_alarm());
            double sc = s.recall() - s.false_alarm();  // simple balance of the two
            if (sc > best_score) { best_score = sc; best = c; }
        }

        std::printf("\nBest balance: motion %.0f%%, crouch alone %s\n\n",
                    configs[best].motion_thresh * 100, configs[best].crouch_alone ? "yes" : "no");
        print_table(videos, results[best]);
        std::printf("\nSummary (best setting)\n");
        print_summary(score(videos, results[best]));
    }

    double minutes = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 60;
    std::printf("\nTotal time: %.1f min\n", minutes);
    return 0;
}
