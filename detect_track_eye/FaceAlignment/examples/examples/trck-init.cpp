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

void track_init(cv::Rect eyefield, cv::Rect eyefield_zuo, Mat init_frame,Tracker *ini_tracker)
{
	// read config file
	string configPath = "KCFconfig.txt";
	Config conf(configPath);
	//统计结果 平均像素误差 平均算法处理时间
	FloatRect initBB;
	float scaleW = 1.f;
	float scaleH = 1.f;

	double dWidth = conf.frameWidth;
	double dHeight = conf.frameHeight;
	Size frameSize(init_frame.cols, init_frame.rows);
	cout << "帧尺寸：" << dWidth << "x" << dHeight << endl;
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
		initBB = FloatRect((xmin*scaleW > 2) ? (xmin*scaleW) : (xmin*scaleW), (ymin*scaleH > 2) ? (ymin*scaleH) : (ymin*scaleH), width*scaleW, height*scaleH);
	}
	else
	{
		scaleW = 1.0;
		scaleH = 1.0;
		conf.frameWidth = init_frame.cols;
		conf.frameHeight = init_frame.rows;
		scaleChange = 0;
		initBB = FloatRect((xmin*scaleW > 2) ? (xmin*scaleW) : (xmin*scaleW), (ymin*scaleH > 2) ? (ymin*scaleH) : (ymin*scaleH), width*scaleW, height*scaleH);
	}
	//跟踪器类初始化
	Tracker tracker(conf);
	ini_tracker = &tracker;
}