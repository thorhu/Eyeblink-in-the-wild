#include "Tracker.h"
#include "Config.h"
#include <opencv2/core/core.hpp>//修改
//#include "ImageRep.h"
//#include "Sampler.h"
//#include "Sample.h"
//#include "HaarFeatures.h"
//#include "RawFeatures.h"
//#include "HistogramFeatures.h"
//#include "MultiFeatures.h"
//#include "Kernels.h"
//#include "LaRank.h"
//#include <Eigen/Core>
//#include "GraphUtils/GraphUtils.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <algorithm>

#include "cf.h"

using namespace cv;
using namespace std;
using namespace Eigen;

//static CF_Tracker CFtracker;
const static bool useCF = 1;
const static bool useStruck = 0;
const static bool ifDebug = 0;
static Mat debugImg;

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
	/*vector<IntRect> samples;

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

	return samples;*/
}

void GKCDebugImage(Mat totalImage, IntRect c_bb)
{
	/*Mat debugImage;
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
	}*/
	
}

//跟踪器类构造函数
Tracker::Tracker(Config& conf) :
	m_config(conf),
	m_initialised(false)
	//m_pLearner(0),
	//m_debugImage(2*conf.searchRadius+1, 2*conf.searchRadius+1, CV_32FC1),
	//m_needsIntegralImage(false)
{
	;//Reset();
}


Tracker::~Tracker()
{
	/*delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}*/
}
//void Tracker::Reset(const string& conf) 
//{
	//m_config.setConfig(conf);
//}
void Tracker::Reset()
{
	/*m_initialised = false;
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
	
	m_pLearner = new LaRank(m_config, *m_features.back(), *m_kernels.back());*/
}

Tracker Tracker::operator=(Tracker temp)
{
	m_initialised = temp.IsInitialised();
	m_bb = temp.GetBB();
	CFtracker = temp.CFtracker;
	return *this;
}
	
//跟踪器参数初始化
void Tracker::Initialise(const cv::Mat& frame, FloatRect bb)
{
	//scalesNow = 1.0;
	m_bb = IntRect(bb);
	/*ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	for (int i = 0; i < 1; ++i)
	{
		UpdateLearner(image);
	}*/

	Mat CFimage = frame.clone();
	BBox_c CFbb;
	CFbb.cx = bb.XCentre() ;
	CFbb.cy = bb.YCentre();
	CFbb.w = bb.Width();
	CFbb.h = bb.Height();
	//相关滤波器初始化
	CFtracker.init(CFimage, CFbb);
	//GKCDebugImage(frame,m_bb);
	//初始化标志位
	m_initialised = true;
}

//跟踪主体函数
void Tracker::Track(const cv::Mat& frame)
{
	//判断跟踪器已经初始化
	assert(m_initialised);
	
	Mat CFimage = frame.clone();
	FloatRect m_bbTemp;
	BBox_c CFbb;
	if(useCF)
	{
		double tCF = (double)cvGetTickCount();
		
		CFbb.cx = 0, CFbb.cy = 0, CFbb.w = 0, CFbb.h = 0;
		//算法主体函数
		CFtracker.track(CFimage, 0, CFbb);
		//得到预测结果
		CFbb = CFtracker.getBBox();
		//得到尺度变化
		scalesNow = CFtracker.scale_now;
		//cout << "scaleNow： " << scalesNow << endl;
		cv::Rect temprect;
		float xmin = ((CFbb.cx - CFbb.w* scalesNow / 2)>0) ? (CFbb.cx - CFbb.w* scalesNow / 2) : 0;
		float ymin = ((CFbb.cy - CFbb.h* scalesNow / 2)>0) ? (CFbb.cy - CFbb.h* scalesNow / 2) : 0;
		float width = CFbb.w * scalesNow;
		float height = CFbb.h * scalesNow;

		m_bbTemp = FloatRect(xmin, ymin, width, height);

		//track_rectangle(CFimage, m_bbTemp, CV_RGB(0, 255, 0)); 
		//imshow("CF:", CFimage);
		//waitKey(1);
		//tCF = (double)cvGetTickCount() - tCF;
		//printf("CF time = %gms\n", tCF / (cvGetTickFrequency() * 1000));
	}
	else
	{
		m_bbTemp = m_bb;
	}
	/*if(useStruck)
	{
		CFScore = CFtracker.getCFScore();
 		ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist); //求积分图
		if (CFScore < CFScoreThro)
		{
			int radiusGKC = 1.5*m_config.searchRadius; //1.5
			if (ifOcclusion == 1)
			{
				radiusGKC = 1.5*radiusGKC;
				if (expendSearch == 1)
				{
					//radiusGKC = 3 * radiusGKC;//3
				}
			}
			vector<FloatRect> rects = Sampler::SimplePixelSamples(m_bb, scalesNow, radiusGKC, 0, 0); //在目标区域周围采样
			//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, m_bbTemp, scalesNow, radiusGKC, 0, 0); //在目标区域周围采样
			//vector<FloatRect> rects = Sampler::GuassSamples(m_bb, m_bbTemp, radiusGKC, 1); //在目标区域周围采样
			vector<FloatRect> keptRects;
			if (ifDebug == 1)
			{
				debugImg = GKCgetInputRGBImage();
			}
			Mat Sampleimage = frame.clone();
			keptRects.reserve(rects.size());
			for (int i = 0; i < (int)rects.size(); i++)
			{
				if (!rects[i].IsInside(image.GetRect())) continue;
				keptRects.push_back(rects[i]);			//判断采样框是不是都在图像内部
				if (ifDebug == 1)
				{
					//track_rectangle(debugImg, keptRects[0], CV_RGB(255, 0, 0));
					track_rectangle(debugImg, keptRects[i], CV_RGB(0, 255, 0));
				}
			}
			if (ifDebug == 1)
			{
				imshow("debug", debugImg);
				waitKey(1);
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
				BBox_c StruckPose;
				StruckPose.cx = keptRects[bestInd].XCentre();
				StruckPose.cy = keptRects[bestInd].YCentre();
				StruckPose.w = keptRects[bestInd].Width();
				StruckPose.h = keptRects[bestInd].Height();
				CFtracker.track(CFimage, 3, StruckPose);
				if (CFtracker.getCFScore() > updateThros)
				{
					CFbb = CFtracker.getBBox();
					scalesNow = CFtracker.scale_now;
					float xmin = ((CFbb.cx - CFbb.w / 2)>0) ? (CFbb.cx - CFbb.w / 2) : 0;
					float ymin = ((CFbb.cy - CFbb.h / 2)>0) ? (CFbb.cy - CFbb.h / 2) : 0;
					float width = CFbb.w * scalesNow;
					float height = CFbb.h * scalesNow;

					m_bb = FloatRect(xmin, ymin, width, height);
					if (CFtracker.getCFScore() > (0.3)) //0.3
					{
						UpdateLearner(image);
					}
					CFtracker.track(CFimage, 1, CFbb);
				}
				else
				{
					StruckPose.cx = m_bb.XCentre();
					StruckPose.cy = m_bb.YCentre();
					StruckPose.w = m_bb.Width();
					StruckPose.h = m_bb.Height();
					CFtracker.track(CFimage, 2, StruckPose);
				}

#if VERBOSE		
				cout << "track score: " << bestScore << endl;
#endif
			}
			if (ifDebug == 1)
			{
				Debug();
			}
			//std::cout << "globalBest=" << globalBestScore << "  ""frame" << frameNum << "  " << "track score: " << bestScore << endl;
			std::cout << "globalBest=" << globalBestScore << "  ""frame" << frameNum << "  " << "track score: " << CFtracker.getCFScore() << endl;
		}
		else
		{
			m_bb = m_bbTemp;
			struckUpdateFlag--;
			if ((struckUpdateFlag <= 0) && (CFScore>0.3))// 
			{
				UpdateLearner(image);
				struckUpdateFlag = updateNum;
			}
			Mat updateimage = frame.clone();
			BBox_c newPose = CFbb;
			CFtracker.track(updateimage, 1, newPose);
		}
	}
	else
	{*/
		Mat updateimage = frame.clone();
		m_bb = m_bbTemp;
		BBox_c newPose = CFbb;
		//跟踪器参数在线学习
		CFtracker.track(updateimage, 1, newPose);
	//}
	//cout <<"CFScore---"<< CFScore << "---oclusion---" << ifOcclusion << "---stateFlag---"<< stateFlag <<endl;
}

//调试函数
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
//调试函数
void Tracker::Debug()
{
	if (ifDebug == 1)
	{
		imshow("tracker", m_debugImage);
		//m_pLearner->Debug();
	}
}

void Tracker::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0
	/*if (m_bb.XMin() < 0)
	{
		m_bb.SetXMin(0);
	}
	if (m_bb.YMin() < 0)
	{
		m_bb.SetYMin(0);
	}
	if (m_bb.XMax()>GKCgetInputRGBImage().cols)
	{
		m_bb.SetWidth(GKCgetInputRGBImage().cols-m_bb.XMin());
	}
	if (m_bb.YMax()>GKCgetInputRGBImage().rows)
	{
		m_bb.SetHeight(GKCgetInputRGBImage().rows - m_bb.YMin());
	}
	vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 3*m_config.searchRadius, 5, 16);
	//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);
	if (ifDebug == 1)
	{
		debugImg = GKCgetInputRGBImage();
	}
	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
		if (ifDebug == 1)
		{
			track_rectangle(debugImg, keptRects[0], CV_RGB(255, 0, 0));
			track_rectangle(debugImg, keptRects[i], CV_RGB(0, 255, 0));
		}
	}

	if (ifDebug == 1)
	{
		imshow("update Debug", debugImg);
		waitKey(1);
	}
		
#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif
		
	MultiSample sample(image, keptRects);
	m_pLearner->Update(sample, 0);*/
}

