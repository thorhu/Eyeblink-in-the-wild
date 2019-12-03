/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "cv.h"
#include "highgui.h"

#include "face_detection.h"
#include "face_alignment.h"
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;


IplConvKernel*  kernel;
IplImage *diff;
struct ret
{
	float out[1][118];
};

struct res_pro
{
	bool blink= false;
	IplImage *img_grayscale;
	IplImage *res;
	IplImage *res_zuo;
	//int face_num;
	//CvSVM svm;
	//CvSVM svm_zuo;
	//seeta::FaceDetection *detector;
	//seeta::FaceAlignment *point_detector;
	float score;
	float score_zuo;
	cv::Rect bbox;
	cv::Rect bbox_zuo;
};
struct fea
{
	float out[1][118];
	float out_zuo[1][118];
};
//ret alivedetector(String str);
ret LBP_59(IplImage *res, IplImage *reut_res,int sig);
//int track_eye();
/*#define BUILD_DLL
#ifdef BUILD_DLL
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
extern "C" EXPORT fea aliverdet(IplImage *frame, int count, res_pro* reut);*/
fea aliverdet(IplImage *frame, int count, res_pro* reut)
{
	
	fea feature;
	ret result;
	
	// Initialize face detection model
	//clock_t start, finish;
	/*IplImage **eye_template, *template_match, *gray, *prev, *nose_template, *nose_template_match;
	
	//start = clock();
	//finish = clock();
	//cout << finish - start << endl;
	//system("pause");
	IplImage *img_grayscale = cvCreateImage(cvGetSize(frame), frame->depth, 1);
	cvCvtColor(frame, img_grayscale, CV_BGR2GRAY);
	/* Always check if frame exists */
	/*int pts_num = 5;
	int im_width = img_grayscale->width;
	int im_height = img_grayscale->height;
	unsigned char* data = new unsigned char[im_width * im_height];
	unsigned char* data_ptr = data;
	unsigned char* image_data_ptr = (unsigned char*)img_grayscale->imageData;
	int h = 0;
	for (h = 0; h < im_height; h++) {
		memcpy(data_ptr, image_data_ptr, im_width);
		data_ptr += im_width;
		image_data_ptr += img_grayscale->widthStep;
	}
	seeta::ImageData image_data;
	image_data.data = data;
	image_data.width = im_width;
	image_data.height = im_height;
	image_data.num_channels = 1;
	seeta::FacialLandmark points[5];
	// Detect faces
	
	std::vector<seeta::FaceInfo> faces = reut->detector->Detect(image_data);
	
	int32_t face_num = static_cast<int32_t>(faces.size());*/
	//CvScalar p;
	//track_eye();
	
		//reut->point_detector->PointDetectLandmarks(image_data, faces[0], points);
		
	//	int width = (points[1].x - points[0].x) / 3;
	//	int height = width / 2;	
	
		Rect eyefield=reut->bbox;
		IplImage *res= cvCreateImage(cvSize(reut->bbox.width, reut->bbox.height), IPL_DEPTH_8U, 3);
		cvSetImageROI(frame, eyefield);
		//提取ROI  
		cvCopy(frame, res);
		//取消设置  
		cvResetImageROI(frame);
		cvRectangle(frame, cvPoint(eyefield.x, eyefield.y), cvPoint(eyefield.x + eyefield.width, eyefield.y + eyefield.height  ), Scalar(0, 0, 255), 1, 1, 0);
		//String name = to_string(count) + ".bmp";
		//cvSaveImage(name.c_str(), res);	
		cvSaveImage("res.jpg", res);
		//zuoyan
		Rect eyefield_zuo = reut->bbox_zuo;// (points[0].x - width / 2, points[0].y - height / 2, width, height);
		IplImage *res_zuo = cvCreateImage(cvSize(reut->bbox_zuo.width, reut->bbox_zuo.height), IPL_DEPTH_8U, 3);
		cvSetImageROI(frame, eyefield_zuo);
		//提取ROI  
		cvCopy(frame, res_zuo);
		//取消设置  
		cvResetImageROI(frame);
		cvRectangle(frame, cvPoint(eyefield_zuo.x, eyefield_zuo.y), cvPoint(eyefield_zuo.x + eyefield_zuo.width, eyefield_zuo.y + eyefield_zuo.height), Scalar(0, 0, 255), 1, 1, 0);
		//String name_zuo = to_string(count) + "zuo.bmp";
		//cvSaveImage(name_zuo.c_str(), res_zuo);
		cvSaveImage("res_zuo.jpg", res_zuo);
		if (count >= 2)
		{		
			result = LBP_59(res, reut->res,1);			
			for (int j = 0; j<118; j ++)
			{
				feature.out[0][j] = result.out[0][j];
			}
			result = LBP_59(res_zuo,reut->res_zuo,0);
			for (int j = 0; j<118; j++)
			{
				feature.out_zuo[0][j] = result.out[0][j];
			}
		}
	reut->img_grayscale = frame;
	cvReleaseImage(&res_zuo);
	cvReleaseImage(&res);
	return feature;
}
