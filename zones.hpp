#pragma once
// Alert zones: draw multiple polygons with the mouse, save them to a file,
// and load them on the next start.
//
// Zones are saved in normalized coordinates (0..1), so they still fit
// if the camera resolution changes.

#include <cstdio>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct Zone {
    std::string name;
    std::vector<cv::Point> points;  // polygon vertices in pixels
};

struct ZoneEditor {
    std::vector<Zone> zones;
    std::vector<cv::Point> pending;  // the zone currently being drawn
    int next_index = 1;              // used to name new zones: Kiosk 1, Kiosk 2, ...

    // Close the pending polygon and add it as a new zone.
    void finish() {
        if (pending.size() < 3) {
            std::printf("A zone needs at least 3 points (currently %zu)\n", pending.size());
            return;
        }
        zones.push_back({"Kiosk " + std::to_string(next_index++), pending});
        std::printf("Added %s (%zu points)\n", zones.back().name.c_str(), pending.size());
        pending.clear();
    }

    // Right click: cancel the pending zone, or delete the zone under the cursor.
    void remove_at(cv::Point p) {
        if (!pending.empty()) {
            pending.clear();
            std::printf("Cancelled the zone being drawn\n");
            return;
        }
        for (size_t i = 0; i < zones.size(); ++i) {
            if (cv::pointPolygonTest(zones[i].points, cv::Point2f(p), false) >= 0) {
                std::printf("Deleted %s\n", zones[i].name.c_str());
                zones.erase(zones.begin() + i);
                return;
            }
        }
    }

    void clear() {
        zones.clear();
        pending.clear();
        next_index = 1;
        std::printf("Cleared all zones\n");
    }
};

// Return the index of the zone containing point p, or -1 if none.
inline int which_zone(const std::vector<Zone>& zones, cv::Point p) {
    for (size_t i = 0; i < zones.size(); ++i)
        if (cv::pointPolygonTest(zones[i].points, cv::Point2f(p), false) >= 0)
            return (int)i;
    return -1;
}

// Mouse callback for cv::setMouseCallback. userdata must point to a ZoneEditor.
inline void on_mouse_zone(int event, int x, int y, int, void* userdata) {
    auto* ed = static_cast<ZoneEditor*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
        ed->pending.push_back({x, y});
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        ed->remove_at({x, y});
    }
}

// Save zones as normalized coordinates (0..1).
inline bool save_zones(const std::string& path, const std::vector<Zone>& zones, cv::Size frame) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "zones" << "[";
    for (const auto& z : zones) {
        std::vector<cv::Point2f> norm;
        for (const auto& p : z.points)
            norm.emplace_back((float)p.x / frame.width, (float)p.y / frame.height);
        fs << "{" << "name" << z.name << "points" << norm << "}";
    }
    fs << "]";
    return true;
}

// Load zones and convert them back to pixels for the current frame size.
inline bool load_zones(const std::string& path, std::vector<Zone>& zones, cv::Size frame) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    zones.clear();
    for (const auto& node : fs["zones"]) {
        Zone z;
        std::vector<cv::Point2f> norm;
        node["name"] >> z.name;
        node["points"] >> norm;
        for (const auto& p : norm)
            z.points.emplace_back(cvRound(p.x * frame.width), cvRound(p.y * frame.height));
        if (z.points.size() >= 3) zones.push_back(z);
    }
    return true;
}

// Draw all zones. alert[i] == true draws zone i in red.
inline void draw_zones(cv::Mat& img, const ZoneEditor& ed, const std::vector<bool>& alert) {
    for (size_t i = 0; i < ed.zones.size(); ++i) {
        const Zone& z = ed.zones[i];
        bool on = i < alert.size() && alert[i];
        cv::Scalar color = on ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);
        cv::polylines(img, z.points, true, color, on ? 3 : 2);
        cv::putText(img, z.name, z.points[0] + cv::Point(5, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    }
    // The zone being drawn: white outline with vertex markers
    if (!ed.pending.empty()) {
        cv::polylines(img, ed.pending, false, cv::Scalar(255, 255, 255), 2);
        for (const auto& p : ed.pending)
            cv::circle(img, p, 4, cv::Scalar(255, 255, 255), -1);
    }
}
