#include "Tracker.h"
#include "cf.h"
#include "Config.h"
#include "rect.h"
#include "math.h"

#include <windows.h>
#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

struct hu_Rect
{
	int x;
	int y;
	int width;
	int height;
};
//返回原始RGB图像
/*#define BUILD_DLL
#ifdef BUILD_DLL
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
extern "C" EXPORT void *track_initial(cv::Rect eyefield, Mat init_frame, float &scale);*/
//主函数
void *track_initial(cv::Rect eyefield, cv::Mat init_frame,float &scale)
{
	// read config file
	string configPath = "KCFconfig.txt";
	Config conf(configPath);
	FloatRect initBB;
	float scaleW = 0.5;
	float scaleH = 0.5;
	double dWidth = conf.frameWidth;
	double dHeight = conf.frameHeight;
	//cout << "帧尺寸：" << dWidth << "x" << dHeight << endl;
	Size frameSize(init_frame.cols, init_frame.rows);		
	float xmin, ymin, width, height;
	xmin = eyefield.x;
	ymin = eyefield.y;
	width = eyefield.width;
	height = eyefield.height;
	bool scaleChange = 0;
	//如果处理图像过大则对图像降低分辨率
	if (width > 30 || height > 20)
	{
		scaleW = 0.5;// (float)conf.frameWidth / tmp.cols;
		scaleH = 0.5;// (float)conf.frameHeight / tmp.rows;
		scaleChange = 1;
		initBB = FloatRect((xmin*scaleW>2) ? (xmin*scaleW) : (xmin*scaleW), (ymin*scaleH>2) ? (ymin*scaleH) : (ymin*scaleH), width*scaleW, height*scaleH);
	}
	else
	{
		scaleW = 1.0;
		scaleH = 1.0;
		conf.frameWidth = init_frame.cols;
		conf.frameHeight = init_frame.rows;
		scaleChange = 0;
		initBB = FloatRect((xmin*scaleW>2) ? (xmin*scaleW) : (xmin*scaleW), (ymin*scaleH>2) ? (ymin*scaleH) : (ymin*scaleH), width*scaleW, height*scaleH);
	}
	scale = scaleW;
	//跟踪器类初始化
	//Tracker tracker_tmp(conf);
	//Tracker *trackerTemp = new(Tracker);
	//trackerTemp->Reset(configPath);
	
	cv::Mat frame1, framevideo;
	//序列处理循环
	cv::Mat frame;
	//resize(init_frame, rgbImg, Size(init_frame.cols * scaleW, init_frame.rows * scaleH));
	//转换为灰度图
	if (init_frame.channels() == 3)
	{
		cv::cvtColor(init_frame, frame1, CV_RGB2GRAY);
		resize(frame1, frame, Size(init_frame.cols * scaleW, init_frame.rows * scaleH));
				//cv::cvtColor(frame, result, CV_GRAY2RGB);
	}
	else
	{
		frame1 = init_frame.clone();
		resize(frame1, frame, Size(init_frame.cols * scaleW, init_frame.rows * scaleH));
			//cv::cvtColor(frame, result, CV_GRAY2RGB);
	}
	Tracker * tracker = new Tracker(conf);
	tracker->Initialise(frame, initBB);
	//trackerTemp->Initialise(frame, initBB);
	//*tracker = tracker_tmp;
	//tracker = trackerTemp;
	//初始化左眼追踪器
	//xmin = eyefield_zuo.x;
	//ymin = eyefield_zuo.y;
	//width = eyefield_zuo.width;
	//height = eyefield_zuo.height;
	//initBB = FloatRect((xmin*scaleW>2) ? (xmin*scaleW) : (xmin*scaleW), (ymin*scaleH>2) ? (ymin*scaleH) : (ymin*scaleH), width*scaleW, height*scaleH);
	//tracker_tmp.Initialise(frame, initBB);
	//tracker_zuo = &tracker_tmp;
	return (void *)tracker;
}
