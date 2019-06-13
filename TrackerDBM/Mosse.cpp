#include "pch.h"
#include "Mosse.h"
#include <time.h>


void MOSSETracker::randWarp(const cv::Mat & InputArray, cv::Mat& OutputArray) const
{
	std::srand((int)time(0));
	double c1 = rand()% 6 - 3;
	//cv::RNG rng((int)time(0));
	//double c1 = rng.uniform(-3.14, 3.14);
	double c = cos(c1);
	double s = sin(c1);
	cv::Mat_<double> M(2, 3);
	M << c, -s, 0,
		s, c, 0;
	
	cv::Mat_<double> center_warp(2, 1);
	center_warp << InputArray.cols / 2,
				InputArray.rows / 2;
	M.col(2) = center_warp - (M.colRange(0, 2))*center_warp;//
	warpAffine(InputArray, OutputArray, M, InputArray.size());//随机仿射变换

}

void MOSSETracker::preProcess(Mat & ROI) const
{
	try {
	ROI.convertTo(ROI, CV_32F);//转换成同类型矩阵运算条件
	cv::log(ROI + 1.0f, ROI);
	//无量纲化
	cv::Scalar mean,StdDev;
	cv::meanStdDev(ROI, mean,StdDev);
	ROI = (ROI - mean[0]) / (StdDev[0] + eps);//标准化归一化

		ROI = ROI.mul(this->hanWin);
	}
	catch (Exception& e)
	{
		e.what();
	}
}

bool MOSSETracker::initTracker(const cv::Mat & inputArray,const cv::Rect & boundingBox)
{
	try
	{
		cv::Mat img;
		cv::Mat paddedImg;
		if (inputArray.channels() == 1)
			img = inputArray;
		else
			cv::cvtColor(inputArray, img, COLOR_BGR2GRAY);
		int h = cv::getOptimalDFTSize(boundingBox.height);
		int w = cv::getOptimalDFTSize(boundingBox.width);
		this->size.height = h;
		this->size.width = w;
		this->center.x = int(floor((boundingBox.x + (boundingBox.width - w) / 2 + w / 2)));
		this->center.y = int(floor((boundingBox.y + (boundingBox.height - h) / 2 + h / 2)));
		cv::Mat roiImg;
		getRectSubPix(img, this->size, this->center, roiImg);//获取rect区域
		createHanningWindow(this->hanWin, this->size, CV_32F);//汉宁窗
		//cv::copyMakeBorder(roiImg, paddedImg, (h - boundingBox.height) / 2, (h - boundingBox.height) / 2, (w - boundingBox.width) / 2, (w - boundingBox.width) / 2, BORDER_REFLECT, cv::Scalar::all(0));//延拓原图像
		Mat g = cv::Mat::zeros(this->size, CV_32F);
		g.at<float>(h / 2, w / 2) = 1;
		double maxVal;
		cv::GaussianBlur(g, g, cv::Size(-1, -1), 2.0);
		minMaxLoc(g, 0, &maxVal);
		g = g / maxVal;//创建高斯响应矩阵
//-----------------------------------------------------
		dft(g, this->G, DFT_COMPLEX_OUTPUT);//dft求出G
		this->A = cv::Mat::zeros(this->G.size(), this->G.type());
		this->B = cv::Mat::zeros(this->G.size(), this->G.type());//初始化
		for (int i = 0; i < 32; i++)
		{	
			cv::Mat img_rand,dft_warped, A_i, B_i;
			this->randWarp(roiImg, img_rand);
			this->preProcess(img_rand);
			dft(img_rand, dft_warped, DFT_COMPLEX_OUTPUT);
			mulSpectrums(this->G, dft_warped, A_i, 0, true);
			mulSpectrums(dft_warped, dft_warped, B_i, 0, true);
			A += A_i;
			B += B_i;
		}//训练初始H
		this->divDFTs(A, B,this->H);
	}
	catch (Exception& e)
	{
		std::cout << e.what();
		return false;
	}
	
	return true;
}

bool MOSSETracker::updateTracker(const cv::Mat & inputImage, cv::Rect& opBoundingbox,double rate)
{
	try {
		cv::Mat img_roi, IMG_ROI, RPO, rpo,IMG;
		cv::Point deltaL;
		if (inputImage.channels() != 1) cvtColor(inputImage, IMG, COLOR_BGR2GRAY);
		cv::getRectSubPix(IMG, this->size, this->center, img_roi);
//-----------------------------
		this->preProcess(img_roi);
		cv::dft(img_roi, IMG_ROI, DFT_COMPLEX_OUTPUT);
		cv::mulSpectrums(IMG_ROI, this->H, RPO, 0, true);
		cv::idft(RPO, rpo, DFT_SCALE | DFT_REAL_OUTPUT);
		double maxVal;
		cv::Point maxLoc;
	
		cv::minMaxLoc(rpo, 0, &maxVal, 0, &maxLoc);
		deltaL.x = maxLoc.x - (int)(img_roi.cols / 2);
		deltaL.y = maxLoc.y - (int)(img_roi.rows / 2);
		this->center.x += deltaL.x;
		this->center.y += deltaL.y;
		opBoundingbox.x = this->center.x - (int)(img_roi.cols / 2);
		opBoundingbox.y = this->center.y - (int)(img_roi.rows / 2);
		opBoundingbox.width = img_roi.cols;
		opBoundingbox.height = img_roi.rows;

		cv::getRectSubPix(IMG, this->size, this->center, img_roi);
		//-----------------------------
		this->preProcess(img_roi);
		
		cv::dft(img_roi, IMG_ROI, DFT_COMPLEX_OUTPUT);
		cv::Mat A_new, B_new;
		cv::mulSpectrums(this->G, IMG_ROI, A_new, 0, true);
		cv::mulSpectrums(IMG_ROI, IMG_ROI, B_new, 0, true);
		A = A * (1 - rate) + A_new * rate;
		B = B * (1 - rate) + B_new * rate;
		this->divDFTs(A, B, this->H);

	}
	catch (Exception& e)
	{
		std::cout << e.what();
		return false;
	}
	return true;
}



void MOSSETracker::divDFTs(const cv::Mat & src1, const cv::Mat & src2,cv::Mat& opArray) const
{
	cv::Mat c1[2], c2[2],a,b,c,d,e,f,re,im;
	split(src1, c1);
	split(src2, c2);
	cv::multiply(c2[0], c2[0], a);
	cv::multiply(c2[1], c2[1], b);
	cv::multiply(c1[0], c2[0], c);
	cv::multiply(c1[1], c2[1], d);
	cv::multiply(c1[1], c2[0], e);
	cv::multiply(c1[0], c2[1], f);
	divide(c + d, a + b, re);
	divide(e - f, a + b, im);
	cv::Mat u[] = { re,im };
	cv::merge(u, 2, opArray);//复矩阵的除法
}

MOSSETracker::~MOSSETracker()
{
	std::cout << "deleted";
}
