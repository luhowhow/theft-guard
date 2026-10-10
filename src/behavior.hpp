#pragma once
// Behavior cues that need no extra model:
//   1. MotionMeter: how much the area of (person box ∩ zone) changes between frames.
//      Vigorous actions such as prying or cutting cause large, sustained change.
//      Measuring inside the person's own box makes the ratio relative to the person,
//      so it depends less on how large the zone is drawn or how far the camera is.
//   2. update_crouch: whether a person's box is much shorter than their usual height.

#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tracker.hpp"
#include "zones.hpp"

class MotionMeter {
public:
    // Call once per frame, before motion_in().
    void update(const cv::Mat& frame, const std::vector<Zone>& zones) {
        // Work on a half-size, blurred grayscale image: cheaper and less noisy.
        cv::Mat small, gray;
        cv::resize(frame, small, cv::Size(), scale_, scale_);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        // Pixels whose brightness changed by more than pixel_thresh_ are marked 255.
        if (!prev_.empty() && prev_.size() == gray.size()) {
            cv::absdiff(gray, prev_, diff_);
            cv::threshold(diff_, diff_, pixel_thresh_, 255, cv::THRESH_BINARY);
        } else {
            diff_ = cv::Mat::zeros(gray.size(), CV_8U);  // first frame: no change yet
        }
        last_ = prev_;  // previous frame, kept for aligned_motion()
        cur_ = gray;
        prev_ = gray;

        // One mask per zone, at the reduced resolution.
        masks_.clear();
        for (const auto& z : zones) {
            std::vector<cv::Point> pts;
            for (const auto& p : z.points)
                pts.emplace_back(cvRound(p.x * scale_), cvRound(p.y * scale_));
            cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
            cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255));
            masks_.push_back(mask);
        }
    }

    // Fraction (0..1) of changed pixels between the person's crop in the previous
    // frame (at prev_box) and in the current frame (at box), restricted to the zone.
    // Aligning the two crops cancels whole-body movement, so only limbs moving
    // relative to the body are counted.
    float aligned_motion(const cv::Rect& prev_box, const cv::Rect& box, int zone) const {
        if (zone < 0 || zone >= (int)masks_.size() || last_.empty() || cur_.empty()) return 0.f;
        auto scaled = [&](const cv::Rect& r) {
            cv::Rect s(cvRound(r.x * scale_), cvRound(r.y * scale_),
                       cvRound(r.width * scale_), cvRound(r.height * scale_));
            return s & cv::Rect(0, 0, cur_.cols, cur_.rows);
        };
        cv::Rect a = scaled(prev_box), b = scaled(box);
        if (a.area() < min_area_ || b.area() < min_area_) return 0.f;

        cv::Mat region = masks_[zone](b);  // part of the current box inside the zone
        int area = cv::countNonZero(region);
        if (area < min_area_) return 0.f;

        cv::Mat prev_crop, diff;
        cv::resize(last_(a), prev_crop, b.size());
        cv::absdiff(cur_(b), prev_crop, diff);
        cv::threshold(diff, diff, pixel_thresh_, 255, cv::THRESH_BINARY);
        cv::bitwise_and(diff, region, diff);
        return (float)cv::countNonZero(diff) / area;
    }

    // Fraction (0..1) of changed pixels inside (person box ∩ zone).
    // Returns 0 if the overlap is too small to be meaningful.
    float motion_in(const cv::Rect& box, int zone) const {
        if (zone < 0 || zone >= (int)masks_.size() || diff_.empty()) return 0.f;

        // Scale the box to the reduced resolution and clip it to the image.
        cv::Rect r(cvRound(box.x * scale_), cvRound(box.y * scale_),
                   cvRound(box.width * scale_), cvRound(box.height * scale_));
        r &= cv::Rect(0, 0, diff_.cols, diff_.rows);
        if (r.area() == 0) return 0.f;

        cv::Mat region = masks_[zone](r);       // part of the box that is inside the zone
        int area = cv::countNonZero(region);
        if (area < min_area_) return 0.f;

        cv::Mat changed;
        cv::bitwise_and(diff_(r), region, changed);
        return (float)cv::countNonZero(changed) / area;
    }

private:
    cv::Mat prev_;               // previous grayscale frame
    cv::Mat last_, cur_;         // previous and current frames, for aligned_motion()
    cv::Mat diff_;               // changed-pixel mask of the latest frame
    std::vector<cv::Mat> masks_; // zone masks
    double scale_ = 0.5;         // downscale factor before comparing
    double pixel_thresh_ = 25;   // brightness change (0..255) to count a pixel as changed
    int min_area_ = 200;         // minimum overlap (in reduced pixels) to measure motion
};

// Update the person's reference (standing) height and report whether they are crouching.
// dt: seconds since this track was last updated.
inline bool update_crouch(Track& tr, double dt, float crouch_ratio) {
    float h = (float)tr.box.height;
    if (tr.ref_height <= 0) {
        tr.ref_height = h;  // first observation
        return false;
    }
    // The reference follows the tallest recent height and slowly decays (2% per second),
    // so it adapts if the person walks closer to or farther from the camera.
    tr.ref_height = std::max(tr.ref_height * (1.0f - 0.02f * (float)dt), h);
    return h < crouch_ratio * tr.ref_height;
}
