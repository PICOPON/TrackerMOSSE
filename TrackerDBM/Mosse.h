
#include <opencv2/opencv.hpp>
#include <stdexcept>
using namespace cv;
class MOSSETracker
{
private:
	const double eps = 0.00001;//-------常量
	cv::Size size;
	cv::Point center;
	cv::Mat G, H, A, B;
	cv::Mat hanWin;
protected:
	virtual void preProcess(Mat &ROI) const;
	virtual void randWarp(const cv::Mat& InputArray, cv::Mat& OutputArray)const;
	void divDFTs(const cv::Mat & src1, const cv::Mat & src2, cv::Mat& opArray) const;//复矩阵商值
public:
	bool initTracker(const cv::Mat & inputArray, const cv::Rect & roi);
	bool updateTracker(const cv::Mat& inputImage,cv::Rect& opBoundingbox, double rate);
	~MOSSETracker();
};