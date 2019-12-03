#ifndef RAW_FEATURES_H
#define RAW_FEATURES_H

#include "Features.h"

#include <opencv/cv.h>

class Config;

class RawFeatures : public Features
{
public:
	RawFeatures(const Config& conf);
	
private:
	cv::Mat m_patchImage;
	
	virtual void UpdateFeatureVector(const Sample& s);
};

#endif
