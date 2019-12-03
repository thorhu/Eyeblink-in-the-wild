#include "cv.h"
#include <iostream>
#include "highgui.h"
//#include "face_detection.h"
#include "face_alignment.h"
using namespace std;
void evaluate(seeta::FacialLandmark points[5], float *b)
{
	float anpower = 0.5;
	float dispower = 0.5;
	//≈–∂œ «∑Ò◊Û”“£ª
	float left_dis = points[2].x - points[0].x;
	float right_dis = points[1].x - points[2].x;
    //≈–∂œÕ∑ «∑ÒÕ·¡À
	float para = (points[0].y - points[1].y) / (points[1].x - points[0].x);
	float angle = atan(para)*180/3.1415926/90;
	float score = 1 - anpower*abs(angle) - dispower*abs((left_dis - right_dis) / (points[3].x - points[4].x) * 2);
//	cout << score<<endl;
	*b = score;

}