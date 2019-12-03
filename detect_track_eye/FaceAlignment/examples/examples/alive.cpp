//====================================================================  
// 作者   : quarryman  
// 邮箱   : quarrying{at}qq.com  
// 主页   : http://blog.csdn.net/quarryman  
// 日期   : 2013年08月11日  
// 描述   : Uniform Pattern的LBP  
//==================================================================== 
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <opencv2/ml/ml.hpp>
#include <math.h>
using namespace std;
using namespace cv;
int getHopCount(uchar i)
{
	int a[8] = { 0 };
	int k = 7;
	int cnt = 0;
	while (i)
	{
		a[k] = i & 1;
		i >>= 1;
		--k;
	}
	for (int k = 0; k<8; ++k)
	{
		if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
		{
			++cnt;
		}
	}
	return cnt;
}

void lbp59table(uchar* table)
{
	memset(table, 0, 256);
	uchar temp = 1;
	for (int i = 0; i<256; ++i)
	{
		if (getHopCount(i) <= 2)
		{
			table[i] = temp;
			temp++;
		}
		//printf("%d\n",table[i]);
	}
}

void LBP(IplImage* src, IplImage* dst)
{
	int width = src->width;
	int height = src->height;
	uchar table[256];
	lbp59table(table);
	for (int j = 1; j<width - 1; j++)
	{
		for (int i = 1; i<height - 1; i++)
		{
			uchar neighborhood[8] = { 0 };
			neighborhood[7] = CV_IMAGE_ELEM(src, uchar, i - 1, j - 1);
			neighborhood[6] = CV_IMAGE_ELEM(src, uchar, i - 1, j);
			neighborhood[5] = CV_IMAGE_ELEM(src, uchar, i - 1, j + 1);
			neighborhood[4] = CV_IMAGE_ELEM(src, uchar, i, j + 1);
			neighborhood[3] = CV_IMAGE_ELEM(src, uchar, i + 1, j + 1);
			neighborhood[2] = CV_IMAGE_ELEM(src, uchar, i + 1, j);
			neighborhood[1] = CV_IMAGE_ELEM(src, uchar, i + 1, j - 1);
			neighborhood[0] = CV_IMAGE_ELEM(src, uchar, i, j - 1);
			uchar center = CV_IMAGE_ELEM(src, uchar, i, j);
			uchar temp = 0;

			for (int k = 0; k<8; k++)
			{
				temp += (neighborhood[k] >= center)*(1 << k);
			}
			//CV_IMAGE_ELEM( dst, uchar, i, j)=temp;
			CV_IMAGE_ELEM(dst, uchar, i, j) = table[temp];
		}
	}
}

int alivee(IplImage* img, CvSVM *svm)
{
	int i = 1;
	float sample[1][59] = { 0 };
	//接入测试样本
	int temp = 0;
	int sum = 0;
	IplImage* dst = cvCreateImage(cvGetSize(img), 8, 1);
	LBP(img, dst);
	Mat result(dst);
	for (int j = 0; j < result.cols; j++)
		{
			for (int k = 0; k < result.rows; k++)
			{
				temp = int(result.at<uchar>(k, j));
				if (temp <= 58 && temp > 0)
				{
					sample[0][temp]++;
				}
				if (temp == 0)
				{
					sample[0][temp]++;
				}
			}

		}
	for (int j = 0; j < 59; j++)
		{
			sum = sum + sample[0][j] * sample[0][j];
		}
	for (int j = 0; j < 59; j++)
		{
			sample[0][j] = sample[0][j] / sqrt(sum);
		}
	CvMat testDataMat = cvMat(1, 59, CV_32FC1, sample);
	float response = (float)svm->predict(&testDataMat);
	cout << response;
	if (response == 1)
	{
		cout << "眨眼" << endl;
	}
	return 0;
}