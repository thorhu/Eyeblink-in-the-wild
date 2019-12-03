#include "cv.h"
#include <iostream>
#include "highgui.h"
//#include "face_detection.h"
#include "face_alignment.h"
using namespace std;
void alive(IplImage *img_color, seeta::FacialLandmark points[5],int *a)
{
//	cvShowImage("test", img_color);
	//cvWaitKey(0);
	IplImage *dst;
	IplImage src = *img_color;
	//a = 0;
	int basis = abs(points[3].y + points[4].y) / 2;
	int bias = abs(basis - points[2].y) / 6;
	CvRect rect;
	//ut << bias<<","<<basis<< " "<<points[3].x<<endl;
	rect.width = abs(points[0].x - points[1].x) / 6;
	rect.height = bias;
	rect.y = points[0].y-bias/2;
	rect.x = points[0].x-rect.width/2;
	
	
	cvRectangle(img_color, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width - 1, rect.y + rect.height - 1), CV_RGB(255, 0, 0));
//	alive(img_color, points, &gray_value);
	//ut << rect.x << rect.y<<rect.width<<rect.height<< endl;
	dst = cvCreateImage(cvSize(rect.width, rect.height), 8, 3);//´´½¨Í¼Ïñ¿Õ¼ä  
	cvSetImageROI(&src, rect);
	cvCvtColor(&src, dst, CV_BGR2HSV);
	//cvCvtColor(floatsrc, floathsv, CV_BGR2HSV);

	//cvShowImage("dst", dst);
	
	uchar* data = (uchar *)dst->imageData;
	int step = dst->widthStep / sizeof(uchar);
	int channels = dst->nChannels;
	for (int i = 0; i<dst->height; i++)
		for (int j = 0; j<dst->width; j++){
			*a = *a + data[i*step + j*channels + 0]+data[i*step + j*channels + 1] + data[i*step + j*channels + 2];
		}
	//int a = data[5];
	*a = *a / (dst->height*dst->width*channels);

}