#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // open the camera; 0 means the first camera on the laptop
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "打不開鏡頭" << std::endl;
        return 1;
    }

    cv::Mat frame;

    //start to read and display
    while (true) {
        cap.read(frame);              
        if (frame.empty()) break;     

        cv::imshow("camera", frame);  

        int key = cv::waitKey(1);     // wait the instruction 
        if (key == 'q') break;        // q means leaving 
    }

    return 0;
}
