#ifndef SAMPLER_H
#define SAMPLER_H

#include "Rect.h"

#include <vector>

class Config;

class Sampler
{
public:	
	static std::vector<FloatRect> RadialSamples(FloatRect centre, int radius, int nr, int nt);
	static std::vector<FloatRect> PixelSamples(FloatRect centre, FloatRect centreTemp, double scale_now, int radius, bool scale = false, bool halfSample = false);
	static std::vector<FloatRect> Sampler::SimplePixelSamples(FloatRect centreTemp, double scale_now, int radius, bool scale, bool halfSample);
	static std::vector<FloatRect> GuassSamples(FloatRect centre, FloatRect centreTemp, int radius, bool scale);
};

#endif
