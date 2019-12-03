#include "Sampler.h"
#include "Config.h"
#include "guassRandNum.h"
#include <fstream> 

#define _USE_MATH_DEFINES
#include <cmath>

///add by fzw
#ifndef M_PI
#define M_PI 3.1415926
#endif

using namespace std;


vector<FloatRect> Sampler::RadialSamples(FloatRect centre, int radius, int nr, int nt)
{
	vector<FloatRect> samples;
	
	FloatRect s(centre);
	float rstep = (float)radius/nr;
	float tstep = 2*(float)M_PI/nt;
	samples.push_back(centre);
	
	for (int ir = 1; ir <= nr; ++ir)
	{
		float phase = (ir % 2)*tstep/2;
		for (int it = 0; it < nt; ++it)
		{
			float dx = ir*rstep*cosf(it*tstep+phase);
			float dy = ir*rstep*sinf(it*tstep+phase);
			s.SetXMin(((centre.XMin() + dx)>0) ? (centre.XMin() + dx) : 0);
			s.SetYMin(((centre.YMin() + dy)>0) ? (centre.YMin() + dy) : 0);
			samples.push_back(s);
		}
	}
	
	return samples;
}

vector<FloatRect> Sampler::PixelSamples(FloatRect centre, FloatRect centreTemp,double scale_now, int radius, bool scale, bool halfSample)
{
	vector<FloatRect> samples;
	float ScaleFactor[5] = {1.0, 0.9, 1.05, 1.1, 1.1};//{ 1.0 };//
	int scaleNum = (scale == 0) ? (1) : (3);
	int smallThreshold = 10;
	vector<IntRect> s;
	for (int i = 0; i < scaleNum; i++)
	{
		IntRect tempS(centreTemp);
		tempS.SetWidth(ScaleFactor[i] * centre.Width());
		tempS.SetHeight(ScaleFactor[i] * centre.Height());
		IntRect scaletempS(centreTemp);
		scaletempS.SetWidth(ScaleFactor[i] * centre.Width() * scale_now);
		scaletempS.SetHeight(ScaleFactor[i] * centre.Height() * scale_now);
		float dertaScale = (ScaleFactor[i]>1) ? (ScaleFactor[i] - 1) : (1 - ScaleFactor[i]);
		//float changeMAX = (centre.Width() > centre.Height()) ? (dertaScale*centre.Width()) : (dertaScale*centre.Height());
		
		if ((ScaleFactor[i] > (1.0 + 0.01)) || (ScaleFactor[i]<(1.0 - 0.01)))
		{
			if ((ScaleFactor[i] * centre.Width()) < (smallThreshold*(centre.Width()/centre.Height())))
			{
				tempS.SetWidth((smallThreshold*(centre.Width() / centre.Height())));
			}
			else if ((dertaScale * centre.Width()) < 2)
			{
				tempS.SetWidth((ScaleFactor[i]>1) ? (centre.Width() + 2) : (centre.Width() - 1));
			}
			if ((ScaleFactor[i] * centre.Height()) < smallThreshold)
			{
				tempS.SetHeight(smallThreshold);
			}
			else if ((dertaScale * centre.Height()) < 2)
			{
				tempS.SetHeight((ScaleFactor[i]>1) ? (centre.Height() + 2) : (centre.Height() - 1));
			}
			
		}
		double width_now = centre.Width() * scale_now;
		double height_now = centre.Height() * scale_now;
		if ((ScaleFactor[i] > (1.0 + 0.01)) || (ScaleFactor[i]<(1.0 - 0.01)))
		{
			if ((ScaleFactor[i] * width_now) < (smallThreshold*(width_now / height_now)))
			{
				scaletempS.SetWidth((smallThreshold*(width_now / height_now)));
			}
			else if ((dertaScale * width_now) < 2)
			{
				scaletempS.SetWidth((ScaleFactor[i]>1) ? (width_now + 2) : (width_now - 1));
			}
			if ((ScaleFactor[i] * height_now) < smallThreshold)
			{
				scaletempS.SetHeight(smallThreshold);
			}
			else if ((dertaScale * height_now) < 2)
			{
				scaletempS.SetHeight((ScaleFactor[i]>1) ? (height_now + 2) : (height_now - 1));
			}

		}
		s.push_back(tempS);
		samples.push_back(tempS);
		s.push_back(scaletempS);
		samples.push_back(scaletempS);
	}
	/*IntRect s(centre);
	samples.push_back(s);
	IntRect s2(centre);
	s2.SetWidth(1.3*centre.Width());
	s2.SetHeight(1.3*centre.Height());
	samples.push_back(s2);*/
	
	int r2 = radius*radius;
	int GKCthroshold = 3;
	for (int iy = -radius; iy <= radius; iy=iy+1)
	{
		for (int ix = -radius; ix <= radius; ix=ix+1)
		{
			if (ix*ix+iy*iy > r2) continue;
			if (iy == 0 && ix == 0) continue; // already put this one at the start
			//if (halfSample && (ix % 2 != 0 || iy % 2 != 0)) continue;
			if ((ix > GKCthroshold) || (ix<-GKCthroshold) || (scale==1))
			{
				ix = ix + 1;
			}
			int x = (((int)centreTemp.XMin() + ix)>0) ? ((int)centreTemp.XMin() + ix) : 0;
			int y = (((int)centreTemp.YMin() + iy)>0) ? ((int)centreTemp.YMin() + iy) : 0;
			for (int i = 0; i < scaleNum * 2; i++)
			{
				s[i].SetXMin(x);
				s[i].SetYMin(y);
				samples.push_back(s[i]);
			}
			if (scale == 0)
			{
				x = (((int)centre.XMin() + ix)>0) ? ((int)centre.XMin() + ix) : 0;
				y = (((int)centre.YMin() + iy)>0) ? ((int)centre.YMin() + iy) : 0;
				//if (halfSample && (ix % 2 != 0 || iy % 2 != 0)) continue;
				if ((ix > GKCthroshold) || (ix<-GKCthroshold))
				{
					ix = ix + 1;
				}
				for (int i = 0; i < scaleNum * 2; i++)
				{
					s[i].SetXMin(x);
					s[i].SetYMin(y);
					samples.push_back(s[i]);
				}
			}
		}
		if ((iy>GKCthroshold) || (iy < -GKCthroshold) || (scale == 1))
		{
			iy = iy + 1;
		}
	}
	
	return samples;
}

vector<FloatRect> Sampler::SimplePixelSamples(FloatRect centreTemp, double scale_now, int radius, bool scale, bool halfSample)
{
	vector<FloatRect> samples;
	float ScaleFactor[5] = { 1.0, 0.9, 1.05, 1.1, 1.1 };//{ 1.0 };//
	int scaleNum = (scale == 0) ? (1) : (3);
	int smallThreshold = 10;
	vector<IntRect> s;
	for (int i = 0; i < scaleNum; i++)
	{
		IntRect scaletempS(centreTemp);
		scaletempS.SetWidth(ScaleFactor[i] * centreTemp.Width() * scale_now);
		scaletempS.SetHeight(ScaleFactor[i] * centreTemp.Height() * scale_now);
		float dertaScale = (ScaleFactor[i]>1) ? (ScaleFactor[i] - 1) : (1 - ScaleFactor[i]);
		//float changeMAX = (centre.Width() > centre.Height()) ? (dertaScale*centre.Width()) : (dertaScale*centre.Height());

		double width_now = centreTemp.Width() * scale_now;
		double height_now = centreTemp.Height() * scale_now;
		if ((ScaleFactor[i] > (1.0 + 0.01)) || (ScaleFactor[i]<(1.0 - 0.01)))
		{
			if ((ScaleFactor[i] * width_now) < (smallThreshold*(width_now / height_now)))
			{
				scaletempS.SetWidth((smallThreshold*(width_now / height_now)));
			}
			else if ((dertaScale * width_now) < 2)
			{
				scaletempS.SetWidth((ScaleFactor[i]>1) ? (width_now + 2) : (width_now - 1));
			}
			if ((ScaleFactor[i] * height_now) < smallThreshold)
			{
				scaletempS.SetHeight(smallThreshold);
			}
			else if ((dertaScale * height_now) < 2)
			{
				scaletempS.SetHeight((ScaleFactor[i]>1) ? (height_now + 2) : (height_now - 1));
			}

		}
		s.push_back(scaletempS);
		samples.push_back(scaletempS);
	}

	int r2 = radius*radius;
	int GKCthroshold = 4;
	for (int iy = -radius; iy <= radius; iy = iy + 1)
	{
		for (int ix = -radius; ix <= radius; ix = ix + 1)
		{
			if (ix*ix + iy*iy > r2) continue;
			if (iy == 0 && ix == 0) continue; // already put this one at the start
			//if (halfSample && (ix % 2 != 0 || iy % 2 != 0)) continue;
			if ((ix > GKCthroshold) || (ix<-GKCthroshold) || (scale == 1))
			{
				ix = ix + 1;
			}
			int x = (((int)centreTemp.XMin() + ix)>0) ? ((int)centreTemp.XMin() + ix) : 0;
			int y = (((int)centreTemp.YMin() + iy)>0) ? ((int)centreTemp.YMin() + iy) : 0;
			for (int i = 0; i < scaleNum; i++)
			{
				s[i].SetXMin(x);
				s[i].SetYMin(y);
				samples.push_back(s[i]);
			}
		}
		if ((iy>GKCthroshold) || (iy < -GKCthroshold) || (scale == 1))
		{
			iy = iy + 1;
		}
	}

	return samples;
}

vector<FloatRect> Sampler::GuassSamples(FloatRect centre, FloatRect centreTemp, int radius, bool scale)
{
	vector<FloatRect> samples;
	FloatRect s(centre);
	samples.push_back(s);
	int smallThreshold = 10;
	std::ofstream outfile("Gauss.txt", std::ios::out);  //定义文件对象
	const int GuassNum = 100;// 200;
	const int AffineNum = 3;
	double affinePara[AffineNum] = { radius / 2, radius / 2, 1.0 }; //3 1.0
	double GuassRand[GuassNum][3] = { { 0 } };
	guassRandNum(0, 1, GuassRand, GuassNum, AffineNum);
	for (int i = 0; i < GuassNum; i++)
	{
		for (int j = 0; j < AffineNum; j++)
		{
			outfile << GuassRand[i][j] << "   ";  //将数据C输出到被定义的文件中
			GuassRand[i][j] *= affinePara[j];
		}
		int ix = GuassRand[i][0];
		int iy = GuassRand[i][1];
		int x = (((int)centreTemp.XMin() + ix)>0) ? ((int)centreTemp.XMin() + ix) : 0;
		int y = (((int)centreTemp.YMin() + iy)>0) ? ((int)centreTemp.YMin() + iy) : 0;
		s.SetXMin(x);
		s.SetYMin(y);
		
		if (scale==1)
		{
			float dertaScale = (GuassRand[i][3]>1) ? (GuassRand[i][3] - 1) : (1 - GuassRand[i][3]);
			if ((GuassRand[i][3] * centre.Width()) < (smallThreshold*(centre.Width() / centre.Height())))
			{
				s.SetWidth((smallThreshold*(centre.Width() / centre.Height())));
			}
			else if ((dertaScale * centre.Width()) < 2)
			{
				s.SetWidth((GuassRand[i][3]>1) ? (centre.Width() + 2) : (centre.Width() - 1));
			}
			if ((GuassRand[i][3] * centre.Height()) < smallThreshold)
			{
				s.SetHeight(smallThreshold);
			}
			else if ((dertaScale * centre.Height()) < 2)
			{
				s.SetHeight((GuassRand[i][3]>1) ? (centre.Height() + 2) : (centre.Height() - 1));
			}
		}
		samples.push_back(s);
	}
	return samples;
}