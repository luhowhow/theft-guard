#pragma once
// FrameGrabber: a background thread reads and decodes frames from one camera,
// keeping only the latest frame (older frames are dropped, so there is no backlog).
// Supports a webcam index ("0"), a video file, or an RTSP URL.
//
// Usage:
//   FrameGrabber cam("test.mp4");
//   cam.start();
//   cv::Mat frame; double t;
//   if (cam.latest(frame, t)) { ...new frame, captured at time t (seconds)... }
//
// Note: latest() swaps buffers instead of copying. If you need to keep a frame
// after the next call to latest(), clone it.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <opencv2/opencv.hpp>

class FrameGrabber {
public:
    explicit FrameGrabber(std::string source) : source_(std::move(source)) {}
    ~FrameGrabber() { stop(); }
    FrameGrabber(const FrameGrabber&) = delete;
    FrameGrabber& operator=(const FrameGrabber&) = delete;

    bool start() {
        if (!open()) return false;
        start_time_ = std::chrono::steady_clock::now();
        running_ = true;
        thread_ = std::thread(&FrameGrabber::loop, this);
        return true;
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        cap_.release();
    }

    // If a new frame arrived since the last call, swap it into `out`, set its
    // timestamp in seconds, and return true. Otherwise return false.
    bool latest(cv::Mat& out, double& timestamp) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_new_) return false;
        cv::swap(out, shared_);
        timestamp = shared_time_;
        has_new_ = false;
        return true;
    }

    const std::string& source() const { return source_; }
    bool running() const { return running_; }
    uint64_t frames_decoded() const { return decoded_; }

private:
    bool is_camera() const {
        return !source_.empty() &&
               std::all_of(source_.begin(), source_.end(),
                           [](unsigned char c) { return std::isdigit(c); });
    }
    bool is_stream() const { return source_.rfind("rtsp://", 0) == 0; }
    bool is_file() const { return !is_camera() && !is_stream(); }

    bool open() {
        if (is_camera()) {
            cap_.open(std::stoi(source_));
        } else {
            cap_.open(source_);
        }
        if (!cap_.isOpened()) {
            std::printf("[FrameGrabber] cannot open %s\n", source_.c_str());
            return false;
        }
        double fps = cap_.get(cv::CAP_PROP_FPS);
        fps_ = (fps > 0 && fps < 240) ? fps : 30.0;
        return true;
    }

    void loop() {
        using clock = std::chrono::steady_clock;
        const auto period = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / fps_));
        auto next_due = clock::now();
        double loop_offset = 0;  // adds up the video length each time a file loops

        while (running_) {
            if (!cap_.read(frame_) || frame_.empty()) {
                if (is_file()) {                         // video ended: loop from the start
                    loop_offset += last_file_time_ + 1.0 / fps_;
                    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                    continue;
                }
                if (is_stream()) {                       // RTSP dropped: reconnect after 1 s
                    std::printf("[FrameGrabber] reconnecting %s\n", source_.c_str());
                    cap_.release();
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    open();
                    continue;
                }
                running_ = false;                        // webcam unplugged
                break;
            }

            // Timestamp: video files use their own clock (keeps increasing across loops),
            // cameras use wall-clock time since start().
            double t;
            if (is_file()) {
                last_file_time_ = cap_.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
                t = loop_offset + last_file_time_;
            } else {
                t = std::chrono::duration<double>(clock::now() - start_time_).count();
            }
            ++decoded_;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                cv::swap(frame_, shared_);
                shared_time_ = t;
                has_new_ = true;
            }

            // Play video files at their real speed, like a live camera.
            if (is_file()) {
                next_due += period;
                std::this_thread::sleep_until(next_due);
                if (clock::now() - next_due > std::chrono::seconds(1))
                    next_due = clock::now();             // fell far behind: resync
            }
        }
    }

    std::string source_;
    double fps_ = 30.0;
    double last_file_time_ = 0;
    std::chrono::steady_clock::time_point start_time_;

    cv::VideoCapture cap_;
    cv::Mat frame_;        // decoded by the background thread
    cv::Mat shared_;       // latest frame handed to the main thread
    double shared_time_ = 0;
    bool has_new_ = false;
    std::mutex mutex_;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> decoded_{0};
};
