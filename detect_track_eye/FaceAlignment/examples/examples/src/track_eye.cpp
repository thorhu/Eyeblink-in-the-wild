#include "Tracker.h"
#include "cf.h"
#include "Config.h"
#include "rect.h"
#include "drawRect.h"
#include "VOT.hpp"
#include "math.h"

#include <windows.h>
#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "core.h"
using namespace std;
using namespace cv;

static Mat  rgbImg;
struct res_pro
{
	bool blink = false;
	IplImage *img_grayscale;
	IplImage *res;
	IplImage *res_zuo;
	//int face_num=0;
	//CvSVM svm;
	//CvSVM svm_zuo;
	//seeta::FaceDetection *detector;
	//seeta::FaceAlignment *point_detector;
	float score;
	float score_zuo;
	cv::Rect bbox;
	cv::Rect bbox_zuo;
};

//返回原始RGB图像
Mat GKCgetInputRGBImage(void)
{
	return rgbImg;
}
//在图像上画结果矩形框
void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour, 4);
}
//在图像上画跟总结果的十字中心
void GKCrectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	//rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);

	vector<Point2f> corners1;

	float cen_x = r.XCentre();
	float cen_y = r.YCentre();
	//画十字星
	corners1.push_back(Point2f(cen_x - 18, cen_y));
	corners1.push_back(Point2f(cen_x, cen_y - 18));
	corners1.push_back(Point2f(cen_x + 18, cen_y));
	corners1.push_back(Point2f(cen_x, cen_y + 18));

	line(rMat, corners1[0], corners1[2], rColour, 2);
	line(rMat, corners1[1], corners1[3], rColour, 2);
}
/*#define BUILD_DLL
#ifdef BUILD_DLL
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
extern "C" EXPORT void track_eye(void *tracker, void *tracker_zuo, IplImage *frame, res_pro* reut, float &scale);*/
//主函数
void track_eye(void *tracker, void *tracker_zuo, IplImage *frame, res_pro* reut, float &scale)
{
	Mat tmp = cvarrToMat(frame);
	float scaleW = scale;
	float scaleH = scale;
	//Mat result(tmp.rows, tmp.cols, CV_8UC3);
	Mat frame1;
	Mat frame_tmp;
	//frametoshow = tmp.clone();
	frame1 = tmp.clone();
	resize(tmp, rgbImg, Size(tmp.cols * scaleW, tmp.rows * scaleH));
	//转换为灰度图
	if (tmp.channels() == 3)
	{
		cv::cvtColor(tmp, frame1, CV_RGB2GRAY);
		resize(frame1, frame_tmp, Size(tmp.cols * scaleW, tmp.rows * scaleH));
		//cv::cvtColor(frame_tmp, result, CV_GRAY2RGB);
	}
	else
	{
		frame1 = tmp.clone();
		resize(frame1, frame_tmp, Size(tmp.cols * scaleW, tmp.rows * scaleH));
		//cv::cvtColor(frame_tmp, result, CV_GRAY2RGB);
	}
	//if (tracker->IsInitialised())
	//{
		//跟踪主体函数
		//double t0 = (double)cvGetTickCount();
	Tracker *tracker_tmp = (Tracker *)tracker;
	tracker_tmp->Track(frame_tmp);
		//double score = tracker->CFtracker.getCFScore();
		//cout << score << endl;
		//float txmin = tracker->GetBB().XMin();
		//float tymin = tracker->GetBB().YMin();
		//float twidth = tracker->GetBB().Width();
		//float theight = tracker->GetBB().Height();
		//FloatRect lastBB(txmin / scaleW, tymin / scaleH, twidth / scaleW, theight / scaleH);
		//cv::Rect grect;
		//GKCrectangle(result, tracker->GetBB(), CV_RGB(0, 255, 0));
		//GKCrectangle(frametoshow, lastBB, CV_RGB(255, 0, 0));
	Tracker *tracker_zuo_tmp = (Tracker *)tracker_zuo;
	tracker_zuo_tmp->Track(frame_tmp);
	reut->score = tracker_tmp->CFtracker.getCFScore();
	reut->bbox.x = tracker_tmp->GetBB().XMin() / scale;
	reut->bbox.y = tracker_tmp->GetBB().YMin() / scale;
	reut->bbox.width = tracker_tmp->GetBB().Width() / scale;
	reut->bbox.height = tracker_tmp->GetBB().Height() / scale;
	reut->score_zuo = tracker_zuo_tmp->CFtracker.getCFScore() / scale;
	reut->bbox_zuo.x = tracker_zuo_tmp->GetBB().XMin() / scale;
	reut->bbox_zuo.y = tracker_zuo_tmp->GetBB().YMin() / scale;
	reut->bbox_zuo.width = tracker_zuo_tmp->GetBB().Width() / scale;
	reut->bbox_zuo.height = tracker_zuo_tmp->GetBB().Height() / scale;
		//score = tracker_zuo->CFtracker.getCFScore();
		//cout << score << endl;
		//txmin = tracker_zuo->GetBB().XMin();
		//tymin = tracker_zuo->GetBB().YMin();
		//twidth = tracker_zuo->GetBB().Width();
		//theight = tracker_zuo->GetBB().Height();
		//FloatRect lastBB_zuo(txmin / scaleW, tymin / scaleH, twidth / scaleW, theight / scaleH);
		//cv::Rect grect;
		//GKCrectangle(result, tracker_zuo->GetBB(), CV_RGB(0, 255, 0));
		//GKCrectangle(frametoshow, lastBB_zuo, CV_RGB(255, 0, 0));
		//rectangle(result, tracker.GetBB(), CV_RGB(0, 255, 0));
		//rectangle(frametoshow, lastBB, CV_RGB(255, 0, 0));
	//}
	//显示预测结果图像
	//imshow("result", result);
	//waitKey(1);
	//imshow("orig", frametoshow);
	//waitKey(1);
	//return track_res;
}
