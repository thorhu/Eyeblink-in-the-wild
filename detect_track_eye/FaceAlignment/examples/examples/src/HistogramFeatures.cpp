#include "HistogramFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"

#include <iostream>

using namespace Eigen;
using namespace cv;
using namespace std;

static const int kNumBins = 16;
static const int kNumLevels = 4;
static const int kNumCellsX = 3;
static const int kNumCellsY = 3;

HistogramFeatures::HistogramFeatures(const Config& conf)
{
	int nc = 0;
	for (int i = 0; i < kNumLevels; ++i)
	{
		//nc += 1 << 2*i;
		nc += (i+1)*(i+1);
	}
	SetCount(kNumBins*nc);
	cout << "histogram bins: " << GetCount() << endl;
}

void HistogramFeatures::UpdateFeatureVector(const Sample& s)
{
	IntRect rect = s.GetROI(); // note this truncates to integers
	//cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
	//cv::resize(s.GetImage().GetImage(0)(roi), m_patchImage, m_patchImage.size());
	
	m_featVec.setZero();
	VectorXd hist(kNumBins);
	
	int histind = 0;
	for (int il = 0; il < kNumLevels; ++il)
	{
		int nc = il+1;
		float w = s.GetROI().Width()/nc;
		float h = s.GetROI().Height()/nc;
		FloatRect cell(0.f, 0.f, w, h);
		for (int iy = 0; iy < nc; ++iy)
		{
			cell.SetYMin(s.GetROI().YMin()+iy*h);
			for (int ix = 0; ix < nc; ++ix)
			{
				cell.SetXMin(s.GetROI().XMin()+ix*w);
				s.GetImage().Hist(cell, hist);
				m_featVec.segment(histind*kNumBins, kNumBins) = hist;
				++histind;
			}
		}
	}
	m_featVec /= histind;
}
