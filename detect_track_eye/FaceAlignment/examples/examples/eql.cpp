#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace std;
using namespace cv;

IplImage* processimg(IplImage* img)
{
	//import the image
	int row = img->height;
	int col = img->width;
	IplImage* image_process = cvCreateImage(cvSize(col, row), IPL_DEPTH_8U, 1);
	int num[256] = { 0 };
	CvScalar s;
	CvScalar ss;
	int position;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			s = cvGet2D(img, i, j);
			position = s.val[0];
			num[position]++;
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			int count = 0;
			s = cvGet2D(img, i, j);
			int r = s.val[0];
			for (int k = 0; k <= r; k++)
				count = count + num[k];
			ss.val[0] = round(255 * count / row / col);

			cvSet2D(image_process, i, j, ss);
		}
	}
	return image_process;
}