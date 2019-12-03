#ifndef DRAWRECT_H
#define DRAWRECT_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void on_MouseHandle(int event, int x,int y,int flag,void* param);
//void DrawRectangle(cv::Mat& img,cv::Rect box);
cv::Rect drawmain(cv::Mat srcImage);

#endif