#include "Tracker.h"
#include "Config.h"
#include "ImageRep.h"
#include "Sampler.h"
#include "Sample.h"
#include "GraphUtils/GraphUtils.h"

#include "HaarFeatures.h"
#include "RawFeatures.h"
#include "HistogramFeatures.h"
#include "MultiFeatures.h"

#include "Kernels.h"

#include "LaRank.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>

#include <vector>
#include <algorithm>

#include "cf.h"

using namespace cv;
using namespace std;
using namespace Eigen;

static CF_Tracker CFtracker;
static bool useCF = 1;
static bool ifOcclusion = 0;

typedef struct centerPosition
{
	double cx, cy, dw, dh;
}centerPosition;

void track_rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);
}

vector<IntRect> GKCRadialSamples(IntRect centre)//, int radius, int nr, int nt)
{
	vector<IntRect> samples;

	IntRect s(centre);

	int ud = 30;
	int lf = 30;

	//samples.push_back(centre);

	for (int ir = -1; ir <= 1; ++ir)
	{
		for (int it = -1; it < 2; ++it)
		{
			int dy = ir*ud;
			int dx = it*lf;
			s.SetXMin(centre.XMin() + dx);
			s.SetYMin(centre.YMin() + dy);
			samples.push_back(s);
		}
	}

	return samples;
}

void GKCDebugImage(Mat totalImage, IntRect c_bb)
{
	Mat debugImage;
	debugImage.setTo(0);
	vector<IntRect> rects = GKCRadialSamples(c_bb); //在目标区域周围采样
	int n = 9;// (int)m_svs.size();

	if (n == 0) return;

	const int kCanvasSize = 600;
	int gridSize = (int)sqrtf((float)(n - 1)) + 1;
	int tileSize = (int)((float)kCanvasSize / gridSize);

	if (tileSize < 5)
	{
		cout << "too many support vectors to display" << endl;
		return;
	}

	Mat temp(tileSize, tileSize, CV_8UC1);
	Mat roi;
	Mat iii = totalImage;
	int x = 0;
	int y = 0;
	
	for (int i = 0; i < n; ++i)
	{
		Mat I = debugImage(cv::Rect(x, y, tileSize, tileSize));
		roi = iii(Range(rects[i].YMin(),rects[i].Height()),Range(rects[i].XMin(),rects[i].Width()));
		resize(roi, temp, temp.size());
		cvtColor(temp, I, CV_GRAY2RGB);
		double w = 1.0;
		rectangle(I, Point(0, 0), Point(tileSize - 1, tileSize - 1),  CV_RGB(0, (uchar)(255 * w), 0) , 3);
		x += tileSize;
		if ((x + tileSize) > kCanvasSize)
		{
				y += tileSize;
				x = 0;
		}
		imshow("GKClearner", debugImage);
		waitKey(1);
	}
	
}

Tracker::Tracker(const Config& conf) :
	m_config(conf),
	m_initialised(false),
	m_pLearner(0),
	m_debugImage(2*conf.searchRadius+1, 2*conf.searchRadius+1, CV_32FC1),
	m_needsIntegralImage(false)
{
	Reset();
}

Tracker::~Tracker()
{
	delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
}

void Tracker::Reset()
{
	m_initialised = false;
	m_debugImage.setTo(0);
	if (m_pLearner) delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
	m_features.clear();
	m_kernels.clear();
	
	m_needsIntegralImage = false;
	m_needsIntegralHist = false;
	
	int numFeatures = m_config.features.size();
	vector<int> featureCounts;
	for (int i = 0; i < numFeatures; ++i)
	{
		switch (m_config.features[i].feature)
		{
		case Config::kFeatureTypeHaar:
			m_features.push_back(new HaarFeatures(m_config));
			m_needsIntegralImage = true;
			break;			
		case Config::kFeatureTypeRaw:
			m_features.push_back(new RawFeatures(m_config));
			break;
		case Config::kFeatureTypeHistogram:
			m_features.push_back(new HistogramFeatures(m_config));
			m_needsIntegralHist = true;
			break;
		}
		featureCounts.push_back(m_features.back()->GetCount());
		
		switch (m_config.features[i].kernel)
		{
		case Config::kKernelTypeLinear:
			m_kernels.push_back(new LinearKernel());
			break;
		case Config::kKernelTypeGaussian:
			m_kernels.push_back(new GaussianKernel(m_config.features[i].params[0]));
			break;
		case Config::kKernelTypeIntersection:
			m_kernels.push_back(new IntersectionKernel());
			break;
		case Config::kKernelTypeChi2:
			m_kernels.push_back(new Chi2Kernel());
			break;
		}
	}
	
	if (numFeatures > 1)
	{
		MultiFeatures* f = new MultiFeatures(m_features);
		m_features.push_back(f);
		
		MultiKernel* k = new MultiKernel(m_kernels, featureCounts);
		m_kernels.push_back(k);		
	}
	
	m_pLearner = new LaRank(m_config, *m_features.back(), *m_kernels.back());
}
	

void Tracker::Initialise(const cv::Mat& frame, FloatRect bb)
{
	m_bb = IntRect(bb);
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	for (int i = 0; i < 1; ++i)
	{
		UpdateLearner(image);
	}

	Mat CFimage = frame.clone();
	BBox_c CFbb;
	CFbb.cx = bb.XCentre() ;
	CFbb.cy = bb.YCentre();
	CFbb.w = bb.Width();
	CFbb.h = bb.Height();
	CFtracker.init(CFimage, CFbb);
	//GKCDebugImage(frame,m_bb);

	m_initialised = true;
}

void Tracker::Track(const cv::Mat& frame)
{
	int stateFlag = -1;
	static int frameNum = 0;
	static int occlusionNum = 0;
	static bool lastOccuFlag = 0;
	static double globalBestScore = 0;
	static int noCFnum = 0;
	bool expendSearch = 0;
	frameNum++;
	if (ifOcclusion == 1)
	{
		occlusionNum++;
		if (occlusionNum > 30)
		{
			expendSearch = 1;
		}
	}
	else if ((lastOccuFlag == 1))// && (ifOcclusion == 0))
	{
		occlusionNum = 0;
		noCFnum = 15;
	}
	if (noCFnum > 0)
	{
		noCFnum--;
	}
	lastOccuFlag = ifOcclusion;
	cout << "occlusionNum___" << occlusionNum << endl;
	cout << "noCFnum___" << noCFnum << endl;
	assert(m_initialised);
	int intstyMean = 0, intstyVar = 0;
	const int intenstyNum = 10;
	static int lightIntsty[intenstyNum] = {0};
	Mat illuIntsty,GKCroi;
	cv::Rect roi_bb;
	roi_bb.x = ((m_bb.XCentre() - m_bb.Width() * 3) > 0) ? (m_bb.XCentre() - m_bb.Width() * 3) : 0;
	roi_bb.y = ((m_bb.YCentre() - m_bb.Height() * 3) > 0) ? (m_bb.YCentre() - m_bb.Height() * 3) : 0;
	roi_bb.width = ((roi_bb.x + m_bb.Width() * 3) < frame.cols) ? (m_bb.Width() * 3) : (frame.cols - roi_bb.x);
	roi_bb.height = ((roi_bb.y + m_bb.Height() * 3) < frame.rows) ? (m_bb.Height() * 3) : (frame.rows - roi_bb.y);
	GKCroi = frame(roi_bb);
	int lightIntensity = 0;
	resize(GKCroi, illuIntsty, Size(30, 30));
	for (int i = 0; i < illuIntsty.rows; i++)
	{
		for (int j = 0; j < illuIntsty.cols; j++)
		{
			lightIntensity += illuIntsty.at<char>(i, j);
		}
	}
	lightIntensity /= 200; //300
	cout << "intensty: ";
	for (int i = 0; i < intenstyNum-1; i++)
	{
		lightIntsty[i] = lightIntsty[i + 1];
		cout << lightIntsty[i] << "___";
	}
	lightIntsty[intenstyNum - 1] = lightIntensity;
	cout << lightIntsty[intenstyNum - 1] << endl;
	for (int i = 0; i < intenstyNum; i++)
	{
		intstyMean += lightIntsty[i];
	}
	intstyMean /= intenstyNum;
	for (int i = 0; i < intenstyNum; i++)
	{
		intstyVar += (lightIntsty[i] - intstyMean)*(lightIntsty[i] - intstyMean);
	}
	intstyVar /= intenstyNum;
	cout << "intstyVar" << intstyVar << endl;

	FloatRect m_bbTemp;
	if ((useCF) && (ifOcclusion == 0))// && (noCFnum == 0))
	{
		double tCF = (double)cvGetTickCount();
		Mat CFimage = frame.clone();
		BBox_c CFbb;
		CFbb.cx = 0 ;
		CFbb.cy = 0;
		CFbb.w = 0;
		CFbb.h = 0;
		CFtracker.track(CFimage, 0, CFbb);
		CFbb = CFtracker.getBBox();

		cv::Rect temprect;
		float xmin = ((CFbb.cx - CFbb.w / 2)>0) ? (CFbb.cx - CFbb.w / 2) : 0;
		float ymin = ((CFbb.cy - CFbb.h / 2)>0) ? (CFbb.cy - CFbb.h / 2) : 0;
		float width = CFbb.w;
		float height = CFbb.h;

		m_bbTemp = FloatRect(xmin, ymin, width, height);

		track_rectangle(CFimage, m_bbTemp, CV_RGB(0, 255, 0));

		tCF = (double)cvGetTickCount() - tCF;
		printf("CF time = %gms\n", tCF / (cvGetTickFrequency() * 1000));
	}
	else
	{
		m_bbTemp = m_bb;
	}

	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist); //求积分图
	int radiusGKC = m_config.searchRadius;
	if (ifOcclusion == 1)
	{
		radiusGKC = 2.5*radiusGKC;//2.5
		if (expendSearch == 1)
		{
			radiusGKC = 3*radiusGKC;
		}
	}
	vector<FloatRect> rects = Sampler::PixelSamples(m_bb, m_bbTemp, radiusGKC, 0, 0); //在目标区域周围采样
	//vector<FloatRect> rects = Sampler::GuassSamples(m_bb, m_bbTemp, radiusGKC, 1); //在目标区域周围采样
	vector<FloatRect> keptRects;

	Mat Sampleimage = frame.clone();
	keptRects.reserve(rects.size());
	for (int i = 0; i < (int)rects.size(); i++)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);			//判断采样框是不是都在图像内部
		track_rectangle(Sampleimage, keptRects[i], CV_RGB(0, 255, 0));
	}
	
	MultiSample sample(image, keptRects);
	
	vector<double> scores;
	m_pLearner->Eval(sample, scores);

	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];
			bestInd = i;
		}
	}
	//UpdateDebugImage(keptRects, m_bb, scores);
	Mat updateimage = frame.clone();
	float bestThreshold = 1.3;// 1.4;
	if (bestInd != -1)
	{
		if ((bestScore > (globalBestScore - 0.65*(((globalBestScore>bestThreshold) ? globalBestScore : bestThreshold) / bestThreshold))) || (frameNum<10) || ((intstyVar > 100) && (bestScore>0.1)))
		{
			stateFlag = 1;
			m_bb = keptRects[bestInd];

			BBox_c newPose;
			newPose.cx = m_bb.XCentre();
			newPose.cy = m_bb.YCentre();
			if ((bestScore >(globalBestScore - 0.35*(((globalBestScore>bestThreshold) ? globalBestScore : bestThreshold) / bestThreshold))) || (frameNum<10) || ((intstyVar > 100) && (bestScore>0.15)))
			{
				stateFlag = 22;
				if (noCFnum == 0)
				{
 					vector<FloatRect> scaleRects = Sampler::PixelSamples(m_bb, m_bb, 5, 1); //8 1
					keptRects.clear();
					keptRects.reserve(scaleRects.size());
					for (int i = 0; i < (int)scaleRects.size(); i++)
					{
						if (!scaleRects[i].IsInside(image.GetRect())) continue;
						keptRects.push_back(scaleRects[i]);			//判断采样框是不是都在图像内部
					}
					MultiSample sample2(image, keptRects);
					vector<double> scores2;
					m_pLearner->Eval(sample2, scores2);
					bestScore = -DBL_MAX;
					bestInd = -1;
					for (int i = 0; i < (int)keptRects.size(); ++i)
					{
						if (scores2[i] > bestScore)
						{
							bestScore = scores2[i];
							bestInd = i;
						}
					}
					//bestInd = -1;
					if (bestInd != -1)
					{
						m_bb = keptRects[bestInd];
					}
				}
				if (globalBestScore < bestScore)
				{
					globalBestScore = bestScore;
				}

				UpdateLearner(image);
				if (useCF)
				{
					newPose.cx = m_bb.XCentre();
					newPose.cy = m_bb.YCentre();
					CFtracker.track(updateimage, 1, newPose);
				}
			}
			else
			{
				if (useCF)
				{
					CFtracker.track(updateimage, 2, newPose);
				}
			}
			ifOcclusion = 0;
		}
		else
		{
			ifOcclusion = 1;
			stateFlag = 333;
			if (useCF)
			{
				BBox_c newPose;
				newPose.cx = m_bb.XCentre();
				newPose.cy = m_bb.YCentre();
				CFtracker.track(updateimage, 2, newPose);
			}
		}
		
#if VERBOSE		
		cout << "track score: " << bestScore << endl;
#endif
	}
	cout << "oclusion---" << ifOcclusion << "---stateFlag---"<< stateFlag <<endl;
	std::cout <<"globalBest="<<globalBestScore << "  ""frame" << frameNum << "  " << "track score: " << bestScore << endl;
}

void Tracker::UpdateDebugImage(const vector<FloatRect>& samples, const FloatRect& centre, const vector<double>& scores)
{
	double mn = VectorXd::Map(&scores[0], scores.size()).minCoeff();
	double mx = VectorXd::Map(&scores[0], scores.size()).maxCoeff();
	m_debugImage.setTo(0);
	for (int i = 0; i < (int)samples.size(); ++i)
	{
		int x = (int)(samples[i].XMin() - centre.XMin());
		int y = (int)(samples[i].YMin() - centre.YMin());
		m_debugImage.at<float>(m_config.searchRadius+y, m_config.searchRadius+x) = (float)((scores[i]-mn)/(mx-mn));
	}
}

void Tracker::Debug()
{
	//imshow("tracker", m_debugImage);
	//m_pLearner->Debug();
}

void Tracker::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0
	vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 3*m_config.searchRadius, 5, 16);
	//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);
	
	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
	}
		
#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif
		
	MultiSample sample(image, keptRects);
	m_pLearner->Update(sample, 0);
}

