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
#include <math.h>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;
struct ret
{
	float out[1][118];
};
IplImage* processimg(IplImage* img);
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
		//'printf("%d\n",table[i]);
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

ret LBP_59(IplImage *res, IplImage *reut_res,int sig)
{
	//clock_t start, finish;
	ret res_ = {0};
	int i = 0;
	//ofstream fin;
	float feature[2][59] = { 0 };
	float feature_cha[1][59] = { 0 };
	float feature_blink[1][118] = { 0 };
	for (i=0; i <= 1; i++)
	{
		IplImage* img ;
		int temp = 0;
		int sum = 0;
		String name;
		if (i==0)
		{
			img = cvCreateImage(cvGetSize(res), res->depth, 1);
			cvCvtColor(res, img, CV_BGR2GRAY);
			//cvSaveImage("eye.jpg", img);
		}	
		else
		{
			img = cvCreateImage(cvGetSize(reut_res), reut_res->depth, 1);
			cvCvtColor(reut_res, img, CV_BGR2GRAY);
			//cvSaveImage("eyezuo.jpg", img);
		}	
		//start = clock();
		IplImage* img_=processimg(img);
		//finish = clock();
		//cout << finish - start << endl;
		//if (sig == 1)
			
		//	cvSaveImage("eyeprocess.jpg", img);
		//else
		//	cvSaveImage("eyeprocesszuo.jpg", img);
		//IplImage* img = cvLoadImage(name.c_str(), 0);
		//cvWaitKey(1);
		IplImage* dst = cvCreateImage(cvGetSize(img_), 8, 1);
		LBP(img_, dst);
		Mat result=cvarrToMat(dst);
		for (int j = 1; j < result.cols-1; j++)
		{
			for (int k = 1; k < result.rows-1; k++)
			{
				temp = int(result.at<uchar>(k, j));
				if (temp <= 58 && temp > 0)
				{
					feature[i][temp]++;
				}
				if (temp == 0)
				{
					feature[i][temp]++;
				}
			}
		}
		for (int j = 0; j < 59; j++)
		{
			//fin << feature[i - 1][j] << ' ';
			
			sum = sum + feature[i][j] * feature[i][j];
		}
		//fin << sum << ' ';
		for (int j = 0; j < 59; j++)
		{
			feature[i][j] = feature[i][j] / sqrt(sum);
		}
		cvReleaseImage(&img);
		cvReleaseImage(&dst);
		cvReleaseImage(&img_);
	}
	for (int j = 0; j < 59; j++)
	{
		feature_cha[0][j] = feature[0][j] - feature[1][j];
		feature_blink[0][j + 59] = feature_cha[0][j];
		feature_blink[0][j] = feature[0][j];
	}
	//fin << "feature_blink:" << '\n';
	for (int j = 0; j < 118; j++)
	{
		res_.out[0][j] = feature_blink[0][j];
		//fin << feature_blink[0][j];
		//fin << " ";
	}
		//fin << '\n';
	//fin.close();
	return res_;
}