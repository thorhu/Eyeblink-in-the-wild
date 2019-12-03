#include "HaarFeatures.h"
#include "Config.h"

static const int kSystematicFeatureCount = 192;

HaarFeatures::HaarFeatures(const Config& conf)
{
	SetCount(kSystematicFeatureCount);
	GenerateSystematic();
}

void HaarFeatures::GenerateSystematic()
{
	float x[] = {0.2f, 0.4f, 0.6f, 0.8f};
	float y[] = {0.2f, 0.4f, 0.6f, 0.8f};
	float s[] = {0.2f, 0.4f};
	for (int iy = 0; iy < 4; ++iy)
	{
		for (int ix = 0; ix < 4; ++ix)
		{
			for (int is = 0; is < 2; ++is)
			{
				FloatRect r(x[ix]-s[is]/2, y[iy]-s[is]/2, s[is], s[is]);
				for (int it = 0; it < 6; ++it)
				{
					m_features.push_back(HaarFeature(r, it));
				}
			}
		}
	}
}

void HaarFeatures::UpdateFeatureVector(const Sample& s)
{
	for (int i = 0; i < m_featureCount; ++i)
	{
		m_featVec[i] = m_features[i].Eval(s);
	}
}
