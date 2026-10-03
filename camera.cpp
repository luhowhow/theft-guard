#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::VideoCapture cap;

	// 如果傳入兩個 代表是用影片檔 一個則代表直接連接鏡頭
    if (argc >= 2) {
        cap.open(argv[1]);   
    } else {
        cap.open(0);       
    }

    if (!cap.isOpened()) {
        std::cerr << "打不開來源" << std::endl;
        return 1;
    }

    cv::Mat frame;
    while (true) {
        cap.read(frame);
        if (frame.empty()) break;     // 影片播完或鏡頭斷了就結束

        cv::imshow("camera", frame);

        int key = cv::waitKey(30);    // 等 30 毫秒，大約每秒 30 張
        if (key == 'q') break;
    }

    return 0;
}
