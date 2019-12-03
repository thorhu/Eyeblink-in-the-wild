#include <iostream>
#include <fstream>  
#include <stdlib.h>
#include <opencv/cv.h>
#include<opencv2\opencv.hpp>
#include <opencv/highgui.h>
#include<opencv2/imgproc/imgproc.hpp>

#include "vot.hpp"

using namespace std;
using namespace cv;

static string imgFormat = "F:/VSopencv/dataset/OTB50/BlurFace/BlurFace/img/%4d.jpg"; 
static char imgPath[256] = "";	// "F:/VSopencv/dataset/OTB50/Basketball/img/1.jpg";
static int frameNum = 899;

int main()
{
	VOT vot_io("F:/VSopencv/dataset/OTB50/BlurFace/BlurFace/groundtruth_rect.txt");

	Mat image;
	int frames = 1;
	sprintf(imgPath, imgFormat.c_str(), frames);
	Mat frameOrig = cv::imread(imgPath, CV_LOAD_IMAGE_COLOR);//0  CV_LOAD_IMAGE_COLOR
	if (frameOrig.empty())
	{
		cout << "error: could not read frame: " << imgPath << endl;
		return EXIT_FAILURE;
	}

	Size frameSize(static_cast<int>(frameOrig.cols), static_cast<int>(frameOrig.rows));
	VideoWriter oVideoWriter("F:\\VSopencv\\sequence\\MyVideo.avi", CV_FOURCC('D', 'I', 'V', 'X'), 20, frameSize, true);
	if (!oVideoWriter.isOpened())
	{
		cout << "³õÊ¼»¯VideoWriterÊ§°Ü£¡" << endl;
		return -1;
	}
	ofstream ofs;
	int testoutfile = 1;
	if (testoutfile)
	{
		ofs.open("F:/VSopencv/dataset/OTB50/BlurFace/BlurFace/result.txt", ios::out);
		if (!ofs)
		{
			cout << "error: could not open results file: " << "result.txt" << endl;
			return EXIT_FAILURE;
		}
	}

	int startFrame = 1;
	image = frameOrig;

	while (frames <= frameNum)
	{
		sprintf(imgPath, imgFormat.c_str(), frames);
		frameOrig = cv::imread(imgPath, CV_LOAD_IMAGE_COLOR);
		if (frameOrig.empty())
		{
			cout << "error: could not read frame: " << imgPath << endl;
			return EXIT_FAILURE;
		}

		frames++;
		image = frameOrig;
		cv::Rect rect = vot_io.getInitRectangle();
		cv::rectangle(image, rect, CV_RGB(0, 255, 0), 2);


		if (testoutfile)
		{
			ofs << rect.x << "\t" << rect.y << "\t" << rect.width << "\t" << rect.height<< endl;
		}
		cv::imshow("output", image);
		oVideoWriter.write(image);
		cv::waitKey(1);
	}

}
