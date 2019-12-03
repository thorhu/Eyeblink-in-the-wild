#ifndef HAAR_FEATURES_H
#define HAAR_FEATURES_H

#include "Features.h"
#include "HaarFeature.h"

#include <vector>

class Config;

class HaarFeatures : public Features
{
public:
	HaarFeatures(const Config& conf);
	
private:
	std::vector<HaarFeature> m_features;
	
	virtual void UpdateFeatureVector(const Sample& s);
	
	void GenerateSystematic();
};

#endif
