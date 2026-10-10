#pragma once
// Simple IoU tracker: the more two boxes in consecutive frames overlap,
// the more likely they belong to the same person.
// Usage:
//   Tracker tracker;                  // defaults: 30% overlap, drop after 1 s unseen
//   Tracker tracker(0.3f, 2.0);       // drop after 2 s (for low per-camera frame rates)
//   std::vector<Track>& tracks = tracker.update(detections, now_sec);

#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include "detector.hpp"

struct Track {
    int id;                    // track ID of this person
    cv::Rect box;              // current position
    double last_seen;          // last time this person was detected (seconds)

    // Zone state
    int zone_id = -1;          // index of the zone this person is in, -1 = none
    double zone_enter = -1;    // time of entering that zone
    double last_in_zone = -1;  // last time this person was inside that zone
    bool alerted = false;      // whether an alert has been raised for this stay

    // Behavior state
    double prev_t = -1;        // time of the previous behavior update
    float ref_height = 0;      // usual (standing) box height, used to detect crouching
    double crouch_sec = 0;     // accumulated crouching time during this stay
    double motion_sec = 0;     // accumulated vigorous-motion time during this stay
    cv::Rect last_box;         // box at the previous analyzed frame (for aligned motion)
};

// IoU = intersection area / union area. 0 = no overlap, 1 = identical boxes.
inline float iou(const cv::Rect& a, const cv::Rect& b) {
    int inter = (a & b).area();
    int uni = a.area() + b.area() - inter;
    return uni > 0 ? (float)inter / uni : 0.f;
}

class Tracker {
public:
    explicit Tracker(float iou_thresh = 0.3f, double max_missing_sec = 1.0)
        : iou_thresh_(iou_thresh), max_missing_sec_(max_missing_sec) {}

    std::vector<Track>& update(const std::vector<Detection>& dets, double now) {
        // 1. Compute the overlap between every existing track and every new detection.
        struct Pair { float iou; int t; int d; };
        std::vector<Pair> pairs;
        for (int t = 0; t < (int)tracks_.size(); ++t)
            for (int d = 0; d < (int)dets.size(); ++d) {
                float v = iou(tracks_[t].box, dets[d].box);
                if (v >= iou_thresh_) pairs.push_back({v, t, d});
            }

        // 2. Greedy matching from the highest overlap down;
        //    each track and each detection can be matched only once.
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b) { return a.iou > b.iou; });
        std::vector<bool> t_used(tracks_.size(), false), d_used(dets.size(), false);
        for (const auto& p : pairs) {
            if (t_used[p.t] || d_used[p.d]) continue;
            tracks_[p.t].box = dets[p.d].box;
            tracks_[p.t].last_seen = now;
            t_used[p.t] = d_used[p.d] = true;
        }

        // 3. Unmatched detections are new people: assign new IDs.
        for (int d = 0; d < (int)dets.size(); ++d)
            if (!d_used[d]) tracks_.push_back(Track{next_id_++, dets[d].box, now});

        // 4. Tracks not seen for too long are treated as gone: remove them.
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [&](const Track& tr) { return now - tr.last_seen > max_missing_sec_; }),
                      tracks_.end());
        return tracks_;
    }

private:
    std::vector<Track> tracks_;
    int next_id_ = 1;
    float iou_thresh_;         // minimum overlap to be the same person
    double max_missing_sec_;   // remove a track after this many seconds without detection
};
