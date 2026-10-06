#pragma once
// Person detection: YOLOX pre-processing, inference, and post-processing.
// Usage:
//   Detector det("models/yolox_tiny.onnx", 416);
//   std::vector<Detection> people = det.detect(frame);

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float score;
};

// Resize while keeping the aspect ratio, then pad to size x size with gray (114).
// `ratio` returns the scale factor, used later to map boxes back to the original frame.
inline cv::Mat letterbox(const cv::Mat& img, int size, float& ratio) {
    ratio = std::min((float)size / img.cols, (float)size / img.rows);
    int w = (int)std::round(img.cols * ratio);
    int h = (int)std::round(img.rows * ratio);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(w, h));
    cv::Mat out(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(0, 0, w, h)));  // place at the top-left corner
    return out;
}

// Convert raw model output (N x 85) into person boxes.
inline std::vector<Detection> decode(const cv::Mat& output, int size, float ratio,
                                     float score_thresh, float nms_thresh) {
    const float* data = (const float*)output.data;
    const int num_candidates = output.size[1];  // 3549 for a 416 input
    const int num_values = output.size[2];      // 85 = 4 box + 1 objectness + 80 classes

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    // YOLOX splits the image into grids with strides 8, 16 and 32 pixels.
    // Each grid cell predicts one candidate box.
    int idx = 0;
    for (int stride : {8, 16, 32}) {
        int grid = size / stride;
        for (int gy = 0; gy < grid; ++gy) {
            for (int gx = 0; gx < grid; ++gx, ++idx) {
                const float* p = data + idx * num_values;

                // p[4]: objectness, p[5]: probability of class "person"
                float score = p[4] * p[5];
                if (score < score_thresh) continue;

                // Outputs are relative to the grid cell; convert to pixels.
                float cx = (p[0] + gx) * stride;
                float cy = (p[1] + gy) * stride;
                float w = std::exp(p[2]) * stride;
                float h = std::exp(p[3]) * stride;

                // Divide by the scale factor to map back to the original frame.
                boxes.emplace_back((int)((cx - w / 2) / ratio), (int)((cy - h / 2) / ratio),
                                   (int)(w / ratio), (int)(h / ratio));
                scores.push_back(score);
            }
        }
    }

    if (idx != num_candidates) {
        std::cerr << "Candidate count mismatch: computed " << idx
                  << ", model output " << num_candidates
                  << " (input size may be wrong)" << std::endl;
        return {};
    }

    // NMS: one person is often covered by several overlapping boxes;
    // keep only the highest-scoring one.
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, score_thresh, nms_thresh, keep);

    std::vector<Detection> result;
    for (int i : keep) result.push_back({boxes[i], scores[i]});
    return result;
}

class Detector {
public:
    Detector(const std::string& model_path, int input_size)
        : net_(cv::dnn::readNetFromONNX(model_path)), size_(input_size) {}

    std::vector<Detection> detect(const cv::Mat& frame,
                                  float score_thresh = 0.3f, float nms_thresh = 0.45f) {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();

        // 1. Pre-processing
        float ratio;
        cv::Mat input = letterbox(frame, size_, ratio);
        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0, cv::Size(), cv::Scalar(), false, false);
        auto t1 = clock::now();

        // 2. Inference
        net_.setInput(blob);
        cv::Mat output = net_.forward();
        auto t2 = clock::now();

        // 3. Post-processing
        std::vector<Detection> result = decode(output, size_, ratio, score_thresh, nms_thresh);
        cv::Rect bounds(0, 0, frame.cols, frame.rows);
        for (auto& d : result) d.box &= bounds;  // clip boxes to the frame
        auto t3 = clock::now();

        pre_ms = ms(t0, t1);
        infer_ms = ms(t1, t2);
        post_ms = ms(t2, t3);
        return result;
    }

    // Timing of each stage in the most recent detect() call
    double pre_ms = 0, infer_ms = 0, post_ms = 0;

private:
    static double ms(std::chrono::steady_clock::time_point a,
                     std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }
    cv::dnn::Net net_;
    int size_;
};
