#include "Config.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

//从config.txt文件中读取跟踪器预设参数
Config::Config(const std::string& path)
{
	SetDefaults();
	
	ifstream f(path.c_str());
	if (!f)
	{
		cout << "error: could not load config file: " << path << endl;
		return;
	}
	
	string line, name, tmp, word;
	while (getline(f, line))
	{
		istringstream iss(line);
		iss >> name >> tmp;
		
		// skip invalid lines and comments
		if (iss.fail() || tmp != "=" || name[0] == '#') continue;
		
		if      (name == "seed") iss >> seed;
		else if (name == "quietMode") iss >> quietMode;
		else if (name == "withGround") iss >> withGround;
		else if (name == "debugMode") iss >> debugMode;
		else if (name == "sequenceBasePath") iss >> sequenceBasePath;
		else if (name == "sequenceName") iss >> sequenceName;
		else if (name == "resultsPath") iss >> resultsPath;
		else if (name == "frameWidth") iss >> frameWidth;
		else if (name == "frameHeight") iss >> frameHeight;
		else if (name == "seed") iss >> seed;
		else if (name == "searchRadius") iss >> searchRadius;
		else if (name == "svmC") iss >> svmC;
		else if (name == "svmBudgetSize") iss >> svmBudgetSize;
		else if (name == "resultsImagePath") iss >> resultsImagePath;
		else if (name == "outputResultFlag") iss >> outputResultFlag;
		else if (name == "track_x") iss >> track_x;
		else if (name == "track_y") iss >> track_y;
		else if (name == "track_w") iss >> track_w;
		else if (name == "track_h") iss >> track_h;
		else if (name == "frameNum") iss >> frameNum;
		else if (name == "videos")
		{
			videoNum = 0;
			while (iss >> word)
			{
				videos.push_back(word);
				videoNum++;
			}
		}
		else if (name == "feature")
		{
			string featureName, kernelName;
			double param;
			iss >> featureName >> kernelName >> param;
			
			FeatureKernelPair fkp;
			
			if      (featureName == FeatureName(kFeatureTypeHaar)) fkp.feature = kFeatureTypeHaar;
			else if (featureName == FeatureName(kFeatureTypeRaw)) fkp.feature = kFeatureTypeRaw;
			else if (featureName == FeatureName(kFeatureTypeHistogram)) fkp.feature = kFeatureTypeHistogram;
			else
			{
				cout << "error: unrecognised feature: " << featureName << endl;
				continue;
			}
			
			if      (kernelName == KernelName(kKernelTypeLinear)) fkp.kernel = kKernelTypeLinear;
			else if (kernelName == KernelName(kKernelTypeIntersection)) fkp.kernel = kKernelTypeIntersection;
			else if (kernelName == KernelName(kKernelTypeChi2)) fkp.kernel = kKernelTypeChi2;
			else if (kernelName == KernelName(kKernelTypeGaussian))
			{
				if (iss.fail())
				{
					cout << "error: gaussian kernel requires a parameter (sigma)" << endl;
					continue;
				}
				fkp.kernel = kKernelTypeGaussian;
				fkp.params.push_back(param);
			}
			else
			{
				cout << "error: unrecognised kernel: " << kernelName << endl;
				continue;
			}
			
			features.push_back(fkp);
		}
	}
}

void Config::SetDefaults()
{

	quietMode = false;
	debugMode = false;
	withGround = false;
	
	sequenceBasePath = "";
	sequenceName = "";
	resultsPath = "";
	
	frameWidth = 320;
	frameHeight = 240;
	
	seed = 0;
	searchRadius = 30;
	svmC = 1.0;
	svmBudgetSize = 0;

	videoNum = 1;
	
	features.clear();
}
void Config::setConfig(const std::string& path)
{
	SetDefaults();

	ifstream f(path.c_str());
	if (!f)
	{
		cout << "error: could not load config file: " << path << endl;
		return;
	}

	string line, name, tmp, word;
	while (getline(f, line))
	{
		istringstream iss(line);
		iss >> name >> tmp;

		// skip invalid lines and comments
		if (iss.fail() || tmp != "=" || name[0] == '#') continue;

		if (name == "seed") iss >> seed;
		else if (name == "quietMode") iss >> quietMode;
		else if (name == "withGround") iss >> withGround;
		else if (name == "debugMode") iss >> debugMode;
		else if (name == "sequenceBasePath") iss >> sequenceBasePath;
		else if (name == "sequenceName") iss >> sequenceName;
		else if (name == "resultsPath") iss >> resultsPath;
		else if (name == "frameWidth") iss >> frameWidth;
		else if (name == "frameHeight") iss >> frameHeight;
		else if (name == "seed") iss >> seed;
		else if (name == "searchRadius") iss >> searchRadius;
		else if (name == "svmC") iss >> svmC;
		else if (name == "svmBudgetSize") iss >> svmBudgetSize;
		else if (name == "resultsImagePath") iss >> resultsImagePath;
		else if (name == "outputResultFlag") iss >> outputResultFlag;
		else if (name == "track_x") iss >> track_x;
		else if (name == "track_y") iss >> track_y;
		else if (name == "track_w") iss >> track_w;
		else if (name == "track_h") iss >> track_h;
		else if (name == "frameNum") iss >> frameNum;
		else if (name == "videos")
		{
			videoNum = 0;
			while (iss >> word)
			{
				videos.push_back(word);
				videoNum++;
			}
		}
		else if (name == "feature")
		{
			string featureName, kernelName;
			double param;
			iss >> featureName >> kernelName >> param;

			FeatureKernelPair fkp;

			if (featureName == FeatureName(kFeatureTypeHaar)) fkp.feature = kFeatureTypeHaar;
			else if (featureName == FeatureName(kFeatureTypeRaw)) fkp.feature = kFeatureTypeRaw;
			else if (featureName == FeatureName(kFeatureTypeHistogram)) fkp.feature = kFeatureTypeHistogram;
			else
			{
				cout << "error: unrecognised feature: " << featureName << endl;
				continue;
			}

			if (kernelName == KernelName(kKernelTypeLinear)) fkp.kernel = kKernelTypeLinear;
			else if (kernelName == KernelName(kKernelTypeIntersection)) fkp.kernel = kKernelTypeIntersection;
			else if (kernelName == KernelName(kKernelTypeChi2)) fkp.kernel = kKernelTypeChi2;
			else if (kernelName == KernelName(kKernelTypeGaussian))
			{
				if (iss.fail())
				{
					cout << "error: gaussian kernel requires a parameter (sigma)" << endl;
					continue;
				}
				fkp.kernel = kKernelTypeGaussian;
				fkp.params.push_back(param);
			}
			else
			{
				cout << "error: unrecognised kernel: " << kernelName << endl;
				continue;
			}

			features.push_back(fkp);
		}
	}
}
std::string Config::FeatureName(FeatureType f)
{
	switch (f)
	{
	case kFeatureTypeRaw:
		return "raw";
	case kFeatureTypeHaar:
		return "haar";
	case kFeatureTypeHistogram:
		return "histogram";
	default:
		return "";
	}
}

std::string Config::KernelName(KernelType k)
{
	switch (k)
	{
	case kKernelTypeLinear:
		return "linear";
	case kKernelTypeGaussian:
		return "gaussian";
	case kKernelTypeIntersection:
		return "intersection";
	case kKernelTypeChi2:
		return "chi2";
	default:
		return "";
	}
}

ostream& operator<< (ostream& out, const Config& conf)
{
	out << "config:" << endl;
	out << "  quietMode          = " << conf.quietMode << endl;
	out << "  debugMode          = " << conf.debugMode << endl;
	out << "  sequenceBasePath   = " << conf.sequenceBasePath << endl;
	out << "  sequenceName       = " << conf.sequenceName << endl;
	out << "  resultsPath        = " << conf.resultsPath << endl;
	out << "  frameWidth         = " << conf.frameWidth << endl;
	out << "  frameHeight        = " << conf.frameHeight << endl;
	out << "  seed               = " << conf.seed << endl;
	out << "  searchRadius       = " << conf.searchRadius << endl;
	out << "  svmC               = " << conf.svmC << endl;
	out << "  svmBudgetSize      = " << conf.svmBudgetSize << endl;
	
	for (int i = 0; i < (int)conf.features.size(); ++i)
	{
		out << "  feature " << i << endl;
		out << "    feature: " << Config::FeatureName(conf.features[i].feature) << endl;
		out << "    kernel:  " << Config::KernelName(conf.features[i].kernel) <<endl;
		if (conf.features[i].params.size() > 0)
		{
			out << "    params: ";
			for (int j = 0; j < (int)conf.features[i].params.size(); ++j)
			{
				out << " " << conf.features[i].params[j];
			}
			out << endl;
		}
	}
	
	return out;
}