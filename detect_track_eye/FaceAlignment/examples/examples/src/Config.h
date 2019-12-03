#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>
#include <ostream>

#define VERBOSE (0)

class Config
{
public:
	Config() { SetDefaults(); }
	Config(const std::string& path);
    void setConfig(const std::string& path);
	
	enum FeatureType
	{
		kFeatureTypeHaar,
		kFeatureTypeRaw,
		kFeatureTypeHistogram
	};

	enum KernelType
	{
		kKernelTypeLinear,
		kKernelTypeGaussian,
		kKernelTypeIntersection,
		kKernelTypeChi2
	};

	struct FeatureKernelPair
	{
		FeatureType feature;
		KernelType kernel;
		std::vector<double> params;
	};
	
	bool							quietMode;
	bool							debugMode;
	bool                            withGround;
	bool							outputResultFlag;

	int                             videoNum;
	std::vector<std::string>        videos;
	
	std::string						sequenceBasePath;
	std::string						sequenceName;
	std::string						resultsPath;
	std::string                     resultsImagePath;

	int								track_x;
	int								track_y;
	int								track_h;
	int								track_w;

	int                             frameNum;
	
	int								frameWidth;
	int								frameHeight;
	
	int								seed;
	int								searchRadius;
	double							svmC;
	int								svmBudgetSize;
	std::vector<FeatureKernelPair>	features;
	
	friend std::ostream& operator<< (std::ostream& out, const Config& conf);
	
private:
	void SetDefaults();
	static std::string FeatureName(FeatureType f);
	static std::string KernelName(KernelType k);
};

#endif