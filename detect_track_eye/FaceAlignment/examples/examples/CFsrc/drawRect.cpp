#include "drawRect.h"
#include "stdafx.h"
#include <opencv2/opencv.hpp>

//using namespace cv;

#define WINDOWNAME "³õÊ¼¿ò"

static cv::Rect g_rectangle;
static bool g_bDrawingbox = false;
static bool  mouseFlag[4] = { 0 };

static void DrawRectangle(cv::Mat& img, cv::Rect box);

void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
	cv::Mat& image = *(cv::Mat*)param;
	switch (event)
	{
	case cv::EVENT_MOUSEMOVE:
		if (g_bDrawingbox)
		{
			g_rectangle.width = x - g_rectangle.x;
			g_rectangle.height = y - g_rectangle.y;
			mouseFlag[1] = 1;
		}
		break;
	case cv::EVENT_LBUTTONDOWN:
		g_bDrawingbox = true;
		g_rectangle = cv::Rect(x, y, 0, 0);
		mouseFlag[0] = 1;
		mouseFlag[1] = 0;
		mouseFlag[2] = 0;
		break;
	case cv::EVENT_LBUTTONUP:
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

void DrawRectangle(cv::Mat& img, cv::Rect box)
{
	rectangle(img, box.tl(), box.br(), CV_RGB(0, 0, 255));
}

cv::Rect drawmain(cv::Mat srcImage)
{
	g_rectangle = cv::Rect(-1, -1, 0, 0);
	cv::Mat tempImage;
	srcImage.copyTo(tempImage);
	g_rectangle = cv::Rect(-1, -1, 0, 0);
	cv::namedWindow(WINDOWNAME);
	cv::setMouseCallback(WINDOWNAME, on_MouseHandle, (void*)&tempImage);

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
		if (cv::waitKey(10) == 27) break;
	}
	return g_rectangle;
}