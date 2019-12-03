#include "drawRect.h"
#include <opencv2/opencv.hpp>
#include "highgui.hpp"
using namespace cv;

#define WINDOWNAME "³õÊ¼¿ò"

static Rect g_rectangle;
static bool g_bDrawingbox = false;
static bool  mouseFlag[4] = { 0 };

//Êó±êÏìÓ¦º¯Êý
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
	Mat& image = *(cv::Mat*)param;
	switch (event)
	{
	case EVENT_MOUSEMOVE:
		if (g_bDrawingbox)
		{
			g_rectangle.width = x - g_rectangle.x;
			g_rectangle.height = y - g_rectangle.y;
			mouseFlag[1] = 1;
		}
		break;
	case EVENT_LBUTTONDOWN:
		g_bDrawingbox = true;
		g_rectangle = Rect(x,y,0,0);
		mouseFlag[0] = 1;
		mouseFlag[1] = 0;
		mouseFlag[2] = 0;
		break;
	case EVENT_LBUTTONUP:
		g_bDrawingbox = false;
		if (g_rectangle.width < 0)
		{
			g_rectangle.x += g_rectangle.width;
			g_rectangle.width *= -1;
		}
		if (g_rectangle.height < 0)
		{
			g_rectangle.y += g_rectangle.height;
			g_rectangle.height *= -1;
		}
		mouseFlag[2] = 1;
		mouseFlag[1] = 0;
		mouseFlag[0] = 0;
		DrawRectangle(image,g_rectangle);
		break;
	}
}

//»­¿òº¯Êý
void DrawRectangle(cv::Mat& img, cv::Rect box)
{
	rectangle(img, box.tl(), box.br(), CV_RGB(0, 0, 255));
}

//»­¿òº¯Êý
Rect drawmain(Mat srcImage)
{
	g_rectangle = Rect(-1,-1,0,0);
	Mat tempImage;
	srcImage.copyTo(tempImage);
	g_rectangle = Rect(-1, -1, 0, 0);
	namedWindow(WINDOWNAME);
	setMouseCallback(WINDOWNAME,on_MouseHandle,(void*) &tempImage);

	while (1)
	{
		if (mouseFlag[0] == 1)
		{
			srcImage.copyTo(tempImage);
		}
		if (mouseFlag[2] == 1)
		{
			DrawRectangle(tempImage, g_rectangle);
		}
		if (g_bDrawingbox)
			DrawRectangle(tempImage,g_rectangle);
		imshow(WINDOWNAME, tempImage);
		if (waitKey(10) == 27) break;
	}
	return g_rectangle;
}