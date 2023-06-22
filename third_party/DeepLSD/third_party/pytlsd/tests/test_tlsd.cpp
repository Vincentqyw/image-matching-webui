#include <iostream>
#include <opencv2/opencv.hpp>

#include "lsd.h"

int main(int argc, char **argv){
    std::cout << "**********************************************" << std::endl;
    std::cout << "****************** TLSD Test *****************" << std::endl;
    std::cout << "**********************************************" << std::endl;

    cv::Mat gray = cv::imread("../resources/ai_001_001.frame.0000.color.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img_flt;
    gray.convertTo(img_flt, CV_64FC1);

    double *imagePtr = reinterpret_cast<double *>(img_flt.data);

    // LSD call. Returns [x1,y1,x2,y2,width,p,-log10(NFA)] for each segment
    int N;
    double *out = lsd(&N, imagePtr, img_flt.cols, img_flt.rows);

    cv::Mat color;
    cv::cvtColor(gray,color, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < N; i++) {
        cv::line(color,
                 cv::Point(out[7 * i + 0], out[7 * i + 1]),
                 cv::Point(out[7 * i + 2], out[7 * i + 3]), CV_RGB(0, 255, 0));
    }
    free((void *) out);


    cv::imshow("segments", color);
    cv::waitKey();
}