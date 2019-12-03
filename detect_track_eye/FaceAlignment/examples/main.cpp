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

#include <fstream>
#include <iostream>
#include <string>

#include "cv.h"
#include "highgui.h"
#include <opencv2/ml/ml.hpp>

#include "face_detection.h"
#include "face_alignment.h"
#include "videoio.hpp"
#include "imgproc.hpp"
#include "opencv.hpp"
//#include "cvaux.h"
//#include "cxcore.h
//#include "afxdialogex.h"
using namespace std;
using namespace cv;

#define initial 1
#define track 2
struct res_pro
{
	bool blink = false;
	IplImage *img_grayscale;
	IplImage *res;
	IplImage *res_zuo;
	//int face_num=0;
	//CvSVM svm;
	//CvSVM svm_zuo;
	//seeta::FaceDetection *detector;
	//seeta::FaceAlignment *point_detector;
	float score;
	float score_zuo;
	cv::Rect bbox;
	cv::Rect bbox_zuo;
	//float scale;
};
struct fea
{
	float out[1][118];
	float out_zuo[1][118];
};
void* track_initial(cv::Rect eyefield,  Mat init_frame,float &scale);
fea aliverdet(IplImage *frame, int count, res_pro* reut);
void track_eye(void *tracker, void *tracker_zuo, IplImage *frame, res_pro* reut, float &scale);
void destory(void *tracker);
/*#define BUILD_DLL
#ifdef BUILD_DLL
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
extern "C"{
	EXPORT Tracker track_initial(cv::Rect eyefield, Mat init_frame);
	EXPORT void aliverdet(IplImage *frame, int count, res_pro* reut);
	EXPORT void track_eye(Tracker *tracker, Tracker *tracker_zuo, IplImage *frame);
}*/
//void track_init(cv::Rect eyefield, cv::Rect eyefield_zuo, Mat init_frame, Tracker *ini_tracker);
int main(int argc, char** argv)
{
	int count_all = 0;
	fea feature;
	float scale = 1;
	int stage = initial;
	Mat init_frame;
	void *tracker = NULL;
	void *tracker_zuo = NULL;
	cv::Rect eyefield;
	cv::Rect eyefield_zuo;
	res_pro reut;
	seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");
	seeta::FaceAlignment point_detector("seeta_fa_v1.1.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);
	char key='a';
	IplImage *frame;
	int count = 1;
	int count_Frame = 0;
	bool info_sig = true;
	string configPath = "KCFconfig.txt";
	double totalTime = 0;
	seeta::FacialLandmark points_raw[5] = {0};
	char *fName = "eyeblink3.mp4";//视频的名字
	cv::VideoCapture capture(fName);// 打开视频fName 视频路径
	cv::Mat imageTemp;
	int countFrame = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout << countFrame << endl;
	while (1)
	{	
		imageTemp=imread("00052.bmp");
		if (imageTemp.empty())
		{
			cerr << "ERROR1" << endl;
			system("pause");
			exit(1);
		}
		IplImage *frame = &(IplImage(imageTemp));		
		if (stage == initial)
		{
			
			if (info_sig == true)
			{
				std::cout << "开始定位眼睛，请坐正" << endl;
				info_sig = false;
			}
			Mat img_grayscale;
			cvtColor(imageTemp, img_grayscale, CV_RGB2GRAY);
			IplImage *img_gray = &(IplImage(img_grayscale));
			int pts_num = 5;
			int im_width = img_gray->width;
			int im_height = img_gray->height;	
			unsigned char* data = new unsigned char[im_width * im_height];
			unsigned char* data_ptr = data;
			unsigned char* image_data_ptr = (unsigned char*)img_gray->imageData;		
			int h = 0;
			for (h = 0; h < im_height; h++) {
				memcpy(data_ptr, image_data_ptr, im_width);
				data_ptr += im_width;
				image_data_ptr += img_gray->widthStep;
			}
			seeta::ImageData image_data;
			image_data.data = data;
			image_data.width = im_width;
			image_data.height = im_height;
			image_data.num_channels = 1;
			seeta::FacialLandmark points[5];
			std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
			int32_t face_num = static_cast<int32_t>(faces.size());
			if (face_num > 0)//modefy 0 1-face_num
			{
				point_detector.PointDetectLandmarks(image_data, faces[0], points);
				int width = (points[1].x - points[0].x)*0.7 + abs(points[1].y - points[0].y)*0.7;
				int height = width / 2;
				eyefield = cv::Rect(points[1].x - width / 2, points[1].y - height / 2, width, height);
				cvRectangle(img_gray, cvPoint(points[1].x - width / 2, points[1].y - height / 2), cvPoint(points[1].x + width / 2, points[1].y + height / 2), Scalar(0, 0, 255), 1, 1, 0);
				eyefield_zuo = cv::Rect(points[0].x - width / 2, points[0].y - height / 2, width, height);
				cvRectangle(img_gray, cvPoint(points[0].x - width / 2, points[0].y - height / 2), cvPoint(points[0].x + width / 2, points[1].y + height / 2), Scalar(0, 0, 255), 1, 1, 0);
				cv::imshow("video", img_grayscale);
				key = cvWaitKey(1);
				stage = track;
				count = 1;
				init_frame = cvarrToMat(frame);
				tracker = track_initial(eyefield, init_frame, scale);
				tracker_zuo = track_initial(eyefield_zuo, init_frame,scale);
				Mat roi_you = init_frame(eyefield);
				Mat roi_zuo = init_frame(eyefield_zuo);	
				string filename_zuo = "./photo_zuo/" + to_string(count) + "zuo.bmp";//左眼存储目录
				string filename_you = "./photo_you/" + to_string(count) + "you.bmp";//右眼存储目录
				imwrite(filename_zuo.c_str(), roi_zuo);
				imwrite(filename_you.c_str(), roi_you);
				std::cout << "初始化完成，开始追踪,请开始眨眼" << endl;			
			}
			delete data;
		}
		else if (stage = track)
		{	
			count++;
			double t0 = (double)cvGetTickCount();
			track_eye(tracker, tracker_zuo, frame, &reut, scale);
			Mat roi_you = imageTemp(reut.bbox);
			Mat roi_zuo = imageTemp(reut.bbox_zuo);
			imshow("eye",roi_zuo);
			imshow("eye_you", roi_you);
			string filename_zuo = "./photo_zuo/" + to_string(count) + "zuo.bmp";//左眼存储目录
			string filename_you = "./photo_you/" + to_string(count) + "you.bmp";//右眼存储目录
			imwrite(filename_zuo.c_str(), roi_zuo);
			imwrite(filename_you.c_str(), roi_you);
			cvRectangle(frame, cvPoint(reut.bbox_zuo.x, reut.bbox_zuo.y), cvPoint(reut.bbox_zuo.x + reut.bbox_zuo.width, reut.bbox_zuo.y + reut.bbox_zuo.height), Scalar(0, 0, 255), 1, 1, 0);
			cvRectangle(frame, cvPoint(reut.bbox.x, reut.bbox.y), cvPoint(reut.bbox.x + reut.bbox.width, reut.bbox.y + reut.bbox.height), Scalar(0, 0, 255), 1, 1, 0);
			cv::imshow("video", imageTemp);
			cvWaitKey(1);
		}
		//cvReleaseImage(&frame);
	}

}
//cv::line