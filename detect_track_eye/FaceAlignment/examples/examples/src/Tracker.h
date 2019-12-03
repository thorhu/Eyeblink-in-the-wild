#ifndef TRACKER_H
#define TRACKER_H

#include "Rect.h"
#include "cf.h"

#include <vector>
#include <Eigen/Core>
#include <opencv/cv.h>

class Config;
class Features;
class Kernel;
class LaRank;
class ImageRep;



class Tracker
{
public:
	double scalesNow;
	//Tracker();
	Tracker(Config& conf);
	~Tracker();
	
	void Initialise(const cv::Mat& frame, FloatRect bb);
	void Reset();
	//void Reset(const string& conf);
	void Track(const cv::Mat& frame);
	void Debug();
	
	inline const FloatRect& GetBB() const { return m_bb; }
	inline bool IsInitialised() const { return m_initialised; }
	CF_Tracker CFtracker;

	Tracker operator=(Tracker temp);

private:
	
	Config& m_config;
	bool m_initialised;
	//std::vector<Features*> m_features;
	//std::vector<Kernel*> m_kernels;
	//LaRank* m_pLearner;
	FloatRect m_bb;
	cv::Mat m_debugImage;
	bool m_needsIntegralImage;
	bool m_needsIntegralHist;
	
	void UpdateLearner(const ImageRep& image);
	void UpdateDebugImage(const std::vector<FloatRect>& samples, const FloatRect& centre, const std::vector<double>& scores);
};

#endif
