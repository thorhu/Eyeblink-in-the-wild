#ifndef IMAGE_REP_H
#define IMAGE_REP_H

#include "Rect.h"

#include <opencv/cv.h>
#include <vector>

#include <Eigen/Core>

class ImageRep
{
public:
	ImageRep(const cv::Mat& rImage, bool computeIntegral, bool computeIntegralHists, bool colour = false);
	
	int Sum(const IntRect& rRect, int channel = 0) const;
	void Hist(const IntRect& rRect, Eigen::VectorXd& h) const;
	
	inline const cv::Mat& GetImage(int channel = 0) const { return m_images[channel]; }
	inline const IntRect& GetRect() const { return m_rect; }

private:
	std::vector<cv::Mat> m_images;
	std::vector<cv::Mat> m_integralImages;
	std::vector<cv::Mat> m_integralHistImages;
	int m_channels;
	IntRect m_rect;
};

#endif
