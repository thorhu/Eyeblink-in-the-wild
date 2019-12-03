#ifndef HAAR_FEATURE_H
#define HAAR_FEATURE_H

#include "Rect.h"
#include "ImageRep.h"

#include <vector>

class Sample;

class HaarFeature
{
public:
	HaarFeature(const FloatRect& bb, int type);
	~HaarFeature();
	
	float Eval(const Sample& s) const;
	
private:
	FloatRect m_bb;
	std::vector<FloatRect> m_rects;
	std::vector<float> m_weights;
	float m_factor;
};

#endif
