// TrackerDBM.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include "Mosse.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cuda;

int main()
{
	cv::Rect ROI;
	//std::vector<cv::Rect> boundingBOX;
	cv::Rect boundingBOX;
	cv::Mat frame;
	MOSSETracker mossetracker;
	cv::VideoCapture cap("E://test_video/004_1.wmv");
	cap >> frame;
	//resize(frame, frame, cv::Size(800, 600));
	ROI = selectROI(frame);
	mossetracker.initTracker(frame,ROI);
	while (cap.read(frame))
	{
		mossetracker.updateTracker(frame, boundingBOX, 0.5);
		rectangle(frame, boundingBOX, cv::Scalar(0, 0, 0), 5);
		cv::imshow("frame", frame);
		if (waitKey(1) == 27) break;
	}
	return 0;
}
