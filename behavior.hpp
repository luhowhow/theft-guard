#pragma once
// Behavior cues that need no extra model:
//   1. MotionMeter: how much each zone changes between consecutive frames
//      (vigorous actions such as prying or cutting cause large, sustained change).
//   2. update_crouch: whether a person's box is much shorter than their usual height.

#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tracker.hpp"
#include "zones.hpp"

class MotionMeter {
public:
    // Returns, for each zone, the fraction (0..1) of its pixels that changed
    // since the previous call.
    std::vector<float> update(const cv::Mat& frame, const std::vector<Zone>& zones) {
        // Work on a half-size, blurred grayscale image: cheaper and less noisy.
        cv::Mat small, gray;
        cv::resize(frame, small, cv::Size(), scale_, scale_);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        std::vector<float> result(zones.size(), 0.f);
        if (!prev_.empty() && prev_.size() == gray.size()) {
            // Pixels whose brightness changed by more than pixel_thresh_ count as "changed".
            cv::Mat diff;
            cv::absdiff(gray, prev_, diff);
            cv::threshold(diff, diff, pixel_thresh_, 255, cv::THRESH_BINARY);

            for (size_t i = 0; i < zones.size(); ++i) {
                // Build a mask of the zone at the reduced resolution.
                std::vector<cv::Point> pts;
                for (const auto& p : zones[i].points)
                    pts.emplace_back(cvRound(p.x * scale_), cvRound(p.y * scale_));
                cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
                cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255));

                int area = cv::countNonZero(mask);
                if (area == 0) continue;
                cv::Mat changed;
                cv::bitwise_and(diff, mask, changed);
                result[i] = (float)cv::countNonZero(changed) / area;
            }
        }
        prev_ = gray;
        return result;
    }

private:
    cv::Mat prev_;               // previous grayscale frame
    double scale_ = 0.5;         // downscale factor before comparing
    double pixel_thresh_ = 25;   // brightness change (0..255) to count a pixel as changed
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
