#include "stdafx.h"
#include <numeric>
#include "cf.h"
using namespace std;

static bool ifDebug = 0;
static bool ifUseScale = 1;
static bool ifUseGray = 1;
static bool ifUseRGB = 0;
static int  scaleStep = 20;
static int featureType = 2;
static float resize_factor = 0.8;// 0.33;
static int PCA_dim = nScales-2;
static double scale_min = 0.5;
static double scale_max = 3.0;
static int scale_IndMax = nScales / 2 + 3;// 10;
static int scale_IndMin = nScales/2-3;//7
int CF_Tracker::init(cv::Mat &img, BBox_c &bbox)
{
    p_pose = bbox;
	p_scale = bbox;
	cv::Mat RGBImg = GKCgetInputRGBImage();
    cv::Mat input;
    if (img.channels() == 3){
        cv::cvtColor(img, input, CV_BGR2GRAY);
        input.convertTo(input, CV_32FC1);
    }else
        img.convertTo(input, CV_32FC1);

    // don't need too large image
    if (bbox.w*bbox.h > 20.*15.) {
		std::cout << "resizing image by factor of " << resize_factor << std::endl;
        p_resize_image = true;
		p_pose.scale(resize_factor);
		p_scale.scale(resize_factor);
		cv::resize(input, input, cv::Size(0, 0), resize_factor, resize_factor, cv::INTER_CUBIC);
		cv::resize(RGBImg, RGBImg, cv::Size(0, 0), resize_factor, resize_factor, cv::INTER_CUBIC);
    }
	scale_min = 100.0 / (double)(p_pose.w * p_pose.h);
	p_init_width = p_pose.w;
	p_init_height = p_pose.h;
    p_windows_size[0] = floor(p_pose.w * (1. + p_padding));
    p_windows_size[1] = floor(p_pose.h * (1. + p_padding));
	if ((p_windows_size[0] > 120) || (p_windows_size[1] > 120))
	{
		if (p_windows_size[0] > p_windows_size[1])
		{
			p_padding = 120 / p_pose.w - 1;
		}
		else
		{
			p_padding = 120 / p_pose.h - 1;
		}
	}
	p_windows_size[0] = floor(p_pose.w * (1. + p_padding));
	p_windows_size[1] = floor(p_pose.h * (1. + p_padding));

    p_output_sigma = std::sqrt(p_pose.w*p_pose.h) * p_output_sigma_factor / static_cast<double>(p_cell_size);

    //window weights, i.e. labels
    p_yf = fft2(gaussian_shaped_labels(p_output_sigma, p_windows_size[0]/p_cell_size, p_windows_size[1]/p_cell_size));

    p_cos_window = cosine_window_function(p_yf.cols, p_yf.rows);

    //obtain a sub-window for training initial model
	cv::Mat patch = get_subwindow_resize(input, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], p_windows_size[0], p_windows_size[1]);
	if (ifUseRGB)
	{
		cv::Mat BGRpatch = get_subwindow_resize(RGBImg, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], p_windows_size[0], p_windows_size[1]);
		p_model_xf = fft2(getMultiFeature(patch, BGRpatch), p_cos_window);
	}
	else if (ifUseGray)
	{
		p_model_xf = fft2(getMultiFeature(patch), p_cos_window);
	}
	else
	{
		p_model_xf = fft2(p_fhog.extract(patch, featureType, p_cell_size, 9), p_cos_window);//getMultiFeature(patch)
	}
	//p_model_xf = fft2(p_fhog.extract(patch, featureType, p_cell_size, 9), p_cos_window);
    //Kernel Ridge Regression, calculate alphas (in Fourier domain)
    ComplexMat kf = gaussian_correlation(p_model_xf, p_model_xf, p_kernel_sigma, true);

    p_model_alphaf = p_yf / (kf + p_lambda);   //equation for fast training

//    p_model_alphaf_num = p_yf * kf;
//    p_model_alphaf_den = kf * (kf + p_lambda);
//    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;

	//scale_init
	scale_now = 1.0;
	if (ifUseScale == 1)
	{
		if (nScales % 2 == 0)
		{
			cout << "error: nScale must be an odd integer!";
			return EXIT_FAILURE;
		}
		int middle_num = (nScales + 1) / 2;
		for (int i = 0; i < nScales; i++)
		{
			scales_array[i] = 1.0;
			if (i < middle_num - 1)
			{
				for (int j = 0; j < (middle_num - 1 - i); j++)
				{
					scales_array[i] *= scale_step;
				}
				scales_array[i] = 1.0 / scales_array[i];
			}
			else if (i > middle_num - 1)
			{
				for (int j = 0; j < (i + 1 - middle_num); j++)
				{
					scales_array[i] *= scale_step;
				}
			}
			else
			{
				scales_array[i] = 1.0;
			}
			scales_index[i] = i + 1 - middle_num;
		}
		for (int i = 0; i < (nScales + 1) / 2; i++)
		{
			double temp = scales_array[i];
			scales_array[i] = scales_array[nScales - 1 - i];
			scales_array[nScales - 1 - i] = temp;
		}
		scale_sigma = nScales / sqrt((double)nScales) * scale_sigma_factor;

		double scales_Gaussian_lable[nScales] = { 0.0 };
		cv::Mat s_yf_temp(1, nScales, CV_32FC1);
		float *temp_ptr = s_yf_temp.ptr<float>(0);
		for (int i = 0; i <= nScales / 2; i++)
		{
			scales_Gaussian_lable[i] = std::exp(-0.5 * (scales_index[i] * scales_index[i]) / (scale_sigma*scale_sigma));
			temp_ptr[i] = scales_Gaussian_lable[i];
			temp_ptr[nScales - 1 - i] = scales_Gaussian_lable[i];
		}
		//cout << "scales_array: " << scales_array << endl;
		//cout << "sqrt(33) = " << sqrt(nScales) << endl;
		//cout << "scales_Gaussian_lable: " << scales_Gaussian_lable << endl;
		//cout << "s_yf_temp: " << s_yf_temp << endl;
		//s_yf = fft2(s_yf_temp);
		cv::Mat s_yf_complex;
		cv::dft(s_yf_temp, s_yf_complex, cv::DFT_COMPLEX_OUTPUT);
		s_yf = ComplexMat(s_yf_complex, 1);
		cout << s_yf << endl;
		scale_window = scale_cosine_window_function(nScales);
		//cout << "scale_window: " << scale_window << endl;
		scale_model_factor = 1.0;
		if (p_init_width*p_init_height > scale_model_area_max)
		{
			scale_model_factor = sqrt(scale_model_area_max / (double)(p_init_width*p_init_height));
		}
		p_scale.scale(scale_model_factor);
		s_init_width = p_scale.w;
		s_init_height = p_scale.h;
		std::vector<cv::Mat> scaleSample;
		int fetureNum = 0;
		for (int i = 0; i < nScales; i++)
		{
			s_windows_size[0] = scales_array[i] * scale_now * p_init_width;
			s_windows_size[1] = scales_array[i] * scale_now * p_init_height;
			cv::Mat scalePatch = get_subwindow_resize(input, p_pose.cx, p_pose.cy, s_windows_size[0], s_windows_size[1], s_init_width, s_init_height);
			cv::imshow("hi", scalePatch);
			cvWaitKey(1);
			std::vector<cv::Mat> scaleSample_temp = p_fhog.extract(scalePatch, featureType, p_cell_size, 9);
			fetureNum = scaleSample_temp.size() * scaleSample_temp[0].rows * scaleSample_temp[0].cols;
			cv::Mat sampleFeature(1, fetureNum, CV_32FC1);
			for (int colj = 0, indexi = 0; colj < scaleSample_temp[0].cols; colj++)
			{
				for (int rowj = 0; rowj < scaleSample_temp[0].rows; rowj++)
				{
					for (int tensorj = 0; tensorj < scaleSample_temp.size(); tensorj++)
					{
						sampleFeature.at<float>(indexi) = scaleSample_temp[tensorj].at<float>(rowj, colj) * scale_window.at<float>(i);
						indexi++;
					}
				}
			}
			//cout << sampleFeature << endl;
			//cv::Mat complex_result;
			//cv::dft(sampleFeature, complex_result, cv::DFT_COMPLEX_OUTPUT);
			//cout << complex_result << endl;
			scaleSample.push_back(sampleFeature.clone());
		}
		std::vector<cv::Mat> scaleSampleNew = PCA_decrease(1, scaleSample, PCA_dim);
		//PCA_dim = fetureNum;
		std::vector<cv::Mat> scaleFeature;// (nScales, fetureNum, CV_32FC1);
		for (int rowi = 0; rowi < PCA_dim; rowi++)
		{
			cv::Mat featureTemp(1, nScales, CV_32FC1);
			for (int coli = 0; coli < nScales; coli++)
			{
				featureTemp.at<float>(coli) = scaleSampleNew[coli].at<float>(rowi);
			}
			cv::Mat complex_result;
			cv::dft(featureTemp, complex_result, cv::DFT_COMPLEX_OUTPUT);
			scaleFeature.push_back(complex_result.clone());
		}
		ComplexMat result(nScales, PCA_dim, 1);
		result.set_channel(scaleFeature);
		s_model_xf = result;
		s_model_alphaf_num = s_model_xf.conj() * s_yf;
		s_model_alphaf_den = (s_model_xf * s_model_xf.conj()).comSum();
	}
	
}

void CF_Tracker::setTrackerPose(BBox_c &bbox, cv::Mat & img)
{
    init(img, bbox);
}

void CF_Tracker::updateTrackerPosition(BBox_c &bbox)
{
    if (p_resize_image) {
        BBox_c tmp = bbox;
		tmp.scale(resize_factor);
        p_pose.cx = tmp.cx;
        p_pose.cy = tmp.cy;
    } else {
        p_pose.cx = bbox.cx;
        p_pose.cy = bbox.cy;
    }
}

BBox_c CF_Tracker::getBBox()
{
    if (p_resize_image) {
        BBox_c tmp = p_pose;
		tmp.scale(1.0 / resize_factor);
        return tmp;
    } else
        return p_pose;
}

void CF_Tracker::track(cv::Mat &img, int updateFlag, BBox_c newPose)
{
	cv::Mat RGBImg = GKCgetInputRGBImage();
	static int scaleTimes = 0;
	if ((ifUseScale) && (updateFlag==0))
	{
		scaleTimes++;
	}
    cv::Mat input;
	cv::Mat debugImg = img.clone();
    if (img.channels() == 3){
        cv::cvtColor(img, input, CV_BGR2GRAY);
        input.convertTo(input, CV_32FC1);
    }else
        img.convertTo(input, CV_32FC1);

    // don't need too large image
	if (p_resize_image)
	{
		cv::resize(input, input, cv::Size(0, 0), resize_factor, resize_factor, cv::INTER_CUBIC);
		if (ifUseRGB)
		{
			cv::resize(RGBImg, RGBImg, cv::Size(0, 0), resize_factor, resize_factor, cv::INTER_CUBIC);
		}
		//cv::resize(img, debugImg, cv::Size(0, 0), resize_factor, resize_factor, cv::INTER_CUBIC);
	}
		
	int ideal_size[2] = { 0 };
	ideal_size[0] = floor(p_init_width * (1. + p_padding));
	ideal_size[1] = floor(p_init_height * (1. + p_padding));
	if (updateFlag == 0)
	{
		p_windows_size[0] = floor(p_pose.w * scale_now * (1. + p_padding));
		p_windows_size[1] = floor(p_pose.h * scale_now * (1. + p_padding));
		cv::Mat patch = get_subwindow_resize(input, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], ideal_size[0], ideal_size[1]);
		ComplexMat zf;// = fft2(p_fhog.extract(patch, featureType, p_cell_size, 9), p_cos_window);
		if (ifUseRGB)
		{
			cv::Mat BGRpatch = get_subwindow_resize(RGBImg, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], ideal_size[0], ideal_size[1]);
			zf = fft2(getMultiFeature(patch, BGRpatch), p_cos_window);
		}
		else if (ifUseGray)
		{
			zf = fft2(getMultiFeature(patch), p_cos_window);
		}
		else
		{
			zf = fft2(p_fhog.extract(patch, featureType, p_cell_size, 9), p_cos_window);//getMultiFeature(patch)
		}
		ComplexMat kzf = gaussian_correlation(zf, p_model_xf, p_kernel_sigma);
		cv::Mat response = ifft2(p_model_alphaf*kzf);
		//std::cout << response << std::endl;

		double min_val, max_val;
		cv::Point2i min_loc, max_loc;
		cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);
		CFscore = max_val;

		if (max_loc.y > zf.rows / 2) //wrap around to negative half-space of vertical axis
			max_loc.y = max_loc.y - zf.rows;
		if (max_loc.x > zf.cols / 2) //same for horizontal axis
			max_loc.x = max_loc.x - zf.cols;

		//shift bbox, no scale change
		//p_pose.cx += p_cell_size * max_loc.x;
		//p_pose.cy += p_cell_size * max_loc.y;
		p_pose.cx += p_cell_size * max_loc.x * ((double)p_windows_size[0] / (double)ideal_size[0]);
		p_pose.cy += p_cell_size * max_loc.y * ((double)p_windows_size[1] / (double)ideal_size[1]);

		//scale detection
		if ((ifUseScale == 1) && (scaleTimes>scaleStep))
		{
			std::vector<cv::Mat> scaleSample;
			int fetureNum = 0;
			for (int i = 0; i < nScales; i++)
			{
				s_windows_size[0] = scales_array[i] * scale_now * p_init_width;
				s_windows_size[1] = scales_array[i] * scale_now * p_init_height;
				cv::Mat scalePatch = get_subwindow_resize(input, p_pose.cx, p_pose.cy, s_windows_size[0], s_windows_size[1], s_init_width, s_init_height);
				std::vector<cv::Mat> scaleSample_temp = p_fhog.extract(scalePatch, featureType, p_cell_size, 9);
				fetureNum = scaleSample_temp.size() * scaleSample_temp[0].rows * scaleSample_temp[0].cols;
				cv::Mat sampleFeature(1, fetureNum, CV_32FC1);
				for (int colj = 0, indexi = 0; colj < scaleSample_temp[0].cols; colj++)
				{
					for (int rowj = 0; rowj < scaleSample_temp[0].rows; rowj++)
					{
						for (int tensorj = 0; tensorj < scaleSample_temp.size(); tensorj++)
						{
							sampleFeature.at<float>(indexi) = scaleSample_temp[tensorj].at<float>(rowj, colj) * scale_window.at<float>(i);
							indexi++;
						}
					}
				}
				//cv::Mat complex_result;
				//cv::dft(sampleFeature, complex_result, cv::DFT_COMPLEX_OUTPUT);
				//cout << complex_result << endl;
				scaleSample.push_back(sampleFeature.clone());
			}
			std::vector<cv::Mat> scaleSampleNew = PCA_decrease(0, scaleSample, PCA_dim);
			std::vector<cv::Mat> scaleFeature;// (nScales, fetureNum, CV_32FC1);
			for (int rowi = 0; rowi < PCA_dim; rowi++)
			{
				cv::Mat featureTemp(1, nScales, CV_32FC1);
				for (int coli = 0; coli < nScales; coli++)
				{
					featureTemp.at<float>(coli) = scaleSampleNew[coli].at<float>(rowi);
				}
				cv::Mat complex_result;
				cv::dft(featureTemp, complex_result, cv::DFT_COMPLEX_OUTPUT);
				scaleFeature.push_back(complex_result.clone());
			}
			ComplexMat result(nScales, PCA_dim, 1);
			result.set_channel(scaleFeature);
			s_model_xf = result;
			cv::Mat scaleResponse = ifft2((s_model_xf * s_model_alphaf_num).comSum() / (s_model_alphaf_den + s_lambda));
			//cout << scaleResponse << endl;
			double scaleResMax = -100;
			int scaleResMaxIndex = -1;
			for (int i = 0; i < nScales; i++)
			{
				if (scaleResponse.at<float>(i) > scaleResMax)
				{
					scaleResMax = scaleResponse.at<float>(i);
					scaleResMaxIndex = i;
				}
			}
			if (scaleResMaxIndex > -1)
			{
				if(scaleResMaxIndex>scale_IndMax)
				{
					scaleResMaxIndex = scale_IndMax;
				}
				if(scaleResMaxIndex<scale_IndMin)
				{
					scaleResMaxIndex = scale_IndMin;
				}

				scale_now *= scales_array[scaleResMaxIndex];
				if (scale_now < scale_min)
				{
					scale_now = scale_min;
				}
				if (scale_now > scale_max)
				{
					scale_now = scale_max;
				}
			}
		}
	}

	if (updateFlag == 1)
	{
		BBox_c temp = newPose;
		if (p_resize_image == true)
			temp.scale(resize_factor);
		p_pose.cx = temp.cx;
		p_pose.cy = temp.cy;
		scale_now = 0.5*((double)temp.w / (double)p_pose.w + (double)temp.h / (double)p_pose.h);
		if (scale_now < scale_min)
		{
			scale_now = scale_min;
		}
		//obtain a subwindow for training at newly estimated target position
		p_windows_size[0] = floor(p_pose.w * scale_now * (1. + p_padding));
		p_windows_size[1] = floor(p_pose.h * scale_now * (1. + p_padding));
		cv::Mat patch = get_subwindow_resize(input, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], ideal_size[0], ideal_size[1]);
		//cv::Mat UpdatePatch = get_subwindow_resize(debugImg, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], ideal_size[0], ideal_size[1]);
		//cv::imshow("updatePatch", UpdatePatch);
		//cv::waitKey(1);
		ComplexMat xf;// = fft2(p_fhog.extract(patch, featureType, p_cell_size, 9), p_cos_window);
		if (ifUseRGB)
		{
			cv::Mat BGRpatch = get_subwindow_resize(RGBImg, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], ideal_size[0], ideal_size[1]);
			xf = fft2(getMultiFeature(patch, BGRpatch), p_cos_window);
		}
		else if (ifUseGray)
		{
			xf = fft2(getMultiFeature(patch), p_cos_window);
		}
		else
		{
			xf = fft2(p_fhog.extract(patch, featureType, p_cell_size, 9), p_cos_window);//getMultiFeature(patch)
		}
		//Kernel Ridge Regression, calculate alphas (in Fourier domain)
		ComplexMat kf = gaussian_correlation(xf, xf, p_kernel_sigma, true);

		ComplexMat alphaf = p_yf / (kf + p_lambda); //equation for fast training

		//subsequent frames, interpolate model
		p_model_xf = p_model_xf * (1. - p_interp_factor) + xf * p_interp_factor;
		p_model_alphaf = p_model_alphaf * (1. - p_interp_factor) + alphaf * p_interp_factor;
		if (ifDebug == 1)
		{
			cv::Mat DebugModel = ifft2(p_model_alphaf);
			cv::imshow("CF_Debug", DebugModel);
			cvWaitKey(1);
		}
		//    ComplexMat alphaf_num = p_yf * kf;
		//    ComplexMat alphaf_den = kf * (kf + p_lambda);
		//    p_model_alphaf_num = p_model_alphaf_num * (1. - p_interp_factor) + (p_yf * kf) * p_interp_factor;
		//    p_model_alphaf_den = p_model_alphaf_den * (1. - p_interp_factor) + kf * (kf + p_lambda) * p_interp_factor;
		//    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
		
		//scale Update
		if ((ifUseScale == 1) && (scaleTimes>scaleStep))
		{
			int fetureNum = 0;
			std::vector<cv::Mat> UpdateScaleSample;
			for (int i = 0; i < nScales; i++)
			{
				s_windows_size[0] = scales_array[i] * scale_now * p_init_width;
				s_windows_size[1] = scales_array[i] * scale_now * p_init_height;
				cv::Mat scalePatch = get_subwindow_resize(input, p_pose.cx, p_pose.cy, s_windows_size[0], s_windows_size[1], s_init_width, s_init_height);
				std::vector<cv::Mat> scaleSample_temp = p_fhog.extract(scalePatch, featureType, p_cell_size, 9);
				fetureNum = scaleSample_temp.size() * scaleSample_temp[0].rows * scaleSample_temp[0].cols;
				cv::Mat sampleFeature(1, fetureNum, CV_32FC1);
				for (int colj = 0, indexi = 0; colj < scaleSample_temp[0].cols; colj++)
				{
					for (int rowj = 0; rowj < scaleSample_temp[0].rows; rowj++)
					{
						for (int tensorj = 0; tensorj < scaleSample_temp.size(); tensorj++)
						{
							sampleFeature.at<float>(indexi) = scaleSample_temp[tensorj].at<float>(rowj, colj) * scale_window.at<float>(i);
							indexi++;
						}
					}
				}
				//cv::Mat complex_result;
				//cv::dft(sampleFeature, complex_result, cv::DFT_COMPLEX_OUTPUT);
				//cout << complex_result << endl;
				UpdateScaleSample.push_back(sampleFeature.clone());
			}
			std::vector<cv::Mat> UPscaleSampleNew = PCA_decrease(0, UpdateScaleSample, PCA_dim);
			std::vector<cv::Mat> UPscaleFeature;// (nScales, fetureNum, CV_32FC1);
			for (int rowi = 0; rowi < PCA_dim; rowi++)
			{
				cv::Mat featureTemp(1, nScales, CV_32FC1);
				for (int coli = 0; coli < nScales; coli++)
				{
					featureTemp.at<float>(coli) = UPscaleSampleNew[coli].at<float>(rowi);
				}
				cv::Mat complex_result;
				cv::dft(featureTemp, complex_result, cv::DFT_COMPLEX_OUTPUT);
				UPscaleFeature.push_back(complex_result.clone());
			}
			ComplexMat updateResult(nScales, PCA_dim, 1);
			updateResult.set_channel(UPscaleFeature);
			s_model_xf = updateResult;
			s_model_alphaf_num = s_model_alphaf_num * (1. - s_learning_rate) + (s_model_xf.conj() * s_yf) * s_learning_rate;
			s_model_alphaf_den = s_model_alphaf_den * (1. - s_learning_rate) + ((s_model_xf * s_model_xf.conj()).comSum()) * s_learning_rate;

		}
	}
	else if (updateFlag == 2)
	{
		BBox_c temp = newPose;
		if (p_resize_image == true)
			temp.scale(resize_factor);
		p_pose.cx = temp.cx;
		p_pose.cy = temp.cy;
	}
	else if (updateFlag == 3)
	{
		BBox_c temp = newPose;
		if (p_resize_image == true)
			temp.scale(resize_factor);
		p_pose.cx = temp.cx;
		p_pose.cy = temp.cy;
		scale_now = 0.5*((double)temp.w / (double)p_pose.w + (double)temp.h / (double)p_pose.h);
		if (scale_now < scale_min)
		{
			scale_now = scale_min;
		}
		p_windows_size[0] = floor(p_pose.w * scale_now * (1. + p_padding));
		p_windows_size[1] = floor(p_pose.h * scale_now * (1. + p_padding));
		cv::Mat patch = get_subwindow_resize(input, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], ideal_size[0], ideal_size[1]);
		ComplexMat zf;
		if (ifUseRGB)
		{
			cv::Mat BGRpatch = get_subwindow_resize(RGBImg, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], ideal_size[0], ideal_size[1]);
			zf = fft2(getMultiFeature(patch, BGRpatch), p_cos_window);
		}
		else if (ifUseGray)
		{
			zf = fft2(getMultiFeature(patch), p_cos_window);
		}
		else
		{
			zf = fft2(p_fhog.extract(patch, featureType, p_cell_size, 9), p_cos_window);//getMultiFeature(patch)
		}
		ComplexMat kzf = gaussian_correlation(zf, p_model_xf, p_kernel_sigma);
		cv::Mat response = ifft2(p_model_alphaf*kzf);
		//std::cout << response << std::endl;

		double min_val, max_val;
		cv::Point2i min_loc, max_loc;
		cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);
		CFscore = max_val;
		if (max_loc.y > zf.rows / 2) //wrap around to negative half-space of vertical axis
			max_loc.y = max_loc.y - zf.rows;
		if (max_loc.x > zf.cols / 2) //same for horizontal axis
			max_loc.x = max_loc.x - zf.cols;

		//shift bbox, no scale change
		//p_pose.cx += p_cell_size * max_loc.x;
		//p_pose.cy += p_cell_size * max_loc.y;
		p_pose.cx += p_cell_size * max_loc.x *((double)p_windows_size[0] / (double)ideal_size[0]);
		p_pose.cy += p_cell_size * max_loc.y *((double)p_windows_size[1] / (double)ideal_size[1]);
	}
}

// ****************************************************************************
std::vector<cv::Mat> CF_Tracker::getMultiFeature(cv::Mat patch)
{
	std::vector<cv::Mat> feature_1 = p_fhog.extract(patch, featureType, p_cell_size, 9);
	cv::Mat featurePatch;
	cv::resize(patch, featurePatch, cv::Size(feature_1[0].cols, feature_1[0].rows));
	cv::Mat featureGray(feature_1[0].rows, feature_1[0].cols, CV_32FC1);
	for (int i = 0; i < featurePatch.rows; i++)
	{
		float *data = featurePatch.ptr<float>(i);
		//#pragma omp parallel for 
		for (int j = 0; j < featurePatch.cols; j++)
		{
			//featureGray.at<float>(i, j) = featurePatch.at<float>(i, j) / 255.0 - 0.5;
			data[j] = data[j] / 255.0 - 0.5;
		}
	}
	feature_1.push_back(featureGray);
	return feature_1;
}
std::vector<cv::Mat> CF_Tracker::getMultiFeature(cv::Mat patch, cv::Mat BGRpatch)
{
	std::vector<cv::Mat> feature_1 = p_fhog.extract(patch, featureType, p_cell_size, 9);
	cv::Mat featurePatch;
	cv::resize(patch, featurePatch, cv::Size(feature_1[0].cols, feature_1[0].rows));
	cv::Mat featureGray(feature_1[0].rows, feature_1[0].cols, CV_32FC1);
	for (int i = 0; i < featurePatch.rows; i++)
	{
		float *data = featurePatch.ptr<float>(i);
		for (int j = 0; j < featurePatch.cols; j++)
		{
			//featureGray.at<float>(i, j) = featurePatch.at<float>(i, j) / 255.0 - 0.5;
			data[j] = data[j] / 255.0 - 0.5;
		}
	}
	feature_1.push_back(featureGray);

	cv::Mat RGBFeaturePatch;
	cv::resize(BGRpatch, RGBFeaturePatch, cv::Size(feature_1[0].cols, feature_1[0].rows));
	cv::Mat feature_B(feature_1[0].rows, feature_1[0].cols, CV_32FC1);
	cv::Mat feature_G(feature_1[0].rows, feature_1[0].cols, CV_32FC1);
	cv::Mat feature_R(feature_1[0].rows, feature_1[0].cols, CV_32FC1);
	for (int i = 0; i < RGBFeaturePatch.rows; i++)
	{
		//float *data = RGBFeaturePatch.ptr<float>(i);
		for (int j = 0; j < RGBFeaturePatch.cols; j++)
		{
			feature_B.at<float>(i, j) = (float)RGBFeaturePatch.at<cv::Vec3b>(i, j)[0] / 255.0 - 0.5;
			//data[j] = data[j] / 255.0 - 0.5;
		}
	}
	for (int i = 0; i < RGBFeaturePatch.rows; i++)
	{
		//float *data = RGBFeaturePatch.ptr<float>(i);
		for (int j = 0; j < RGBFeaturePatch.cols; j++)
		{
			feature_G.at<float>(i, j) = (float)RGBFeaturePatch.at<cv::Vec3b>(i, j)[1] / 255.0 - 0.5;
			//data[j] = data[j] / 255.0 - 0.5;
		}
	}
	for (int i = 0; i < RGBFeaturePatch.rows; i++)
	{
		//float *data = RGBFeaturePatch.ptr<float>(i);
		for (int j = 0; j < RGBFeaturePatch.cols; j++)
		{
			feature_R.at<float>(i, j) = (float)RGBFeaturePatch.at<cv::Vec3b>(i, j)[2] / 255.0 - 0.5;
			//data[j] = data[j] / 255.0 - 0.5;
		}
	}
	feature_1.push_back(feature_B);
	feature_1.push_back(feature_G);
	feature_1.push_back(feature_R);
	return feature_1;
}

std::vector<cv::Mat> CF_Tracker::PCA_decrease(int init_flag, std::vector<cv::Mat> &input, int newDim)
{
	std::vector<cv::Mat> temp;

	if (init_flag == 1)
	{
		cv::Mat featureMat(input.size(), input[0].cols, CV_32FC1);
		for (int i = 0; i < input.size(); i++)
		{
			for (int j = 0; j < input[0].cols; j++)
			{
				featureMat.at<float>(i, j) = input[i].at<float>(j);
			}
		}
		cv::PCA *pca = new cv::PCA(featureMat, cv::Mat(), CV_PCA_DATA_AS_ROW);
		cv::Mat eigenvetors_d(newDim, pca->eigenvectors.cols, CV_32FC1);
		for (int i = 0; i < newDim; i++)
		{
			pca->eigenvectors.row(i).copyTo(eigenvetors_d.row(i));
		}
		pca_encoding->mean = pca->mean;
		pca_encoding->eigenvectors = eigenvetors_d;
	}
	for (int i = 0; i < input.size(); i++)
	{
		cv::Mat featureTemp(1, pca_encoding->eigenvectors.rows, CV_32FC1);
		pca_encoding->project(input[i], featureTemp);
		temp.push_back(featureTemp);
	}
	return temp;
}

cv::Mat CF_Tracker::gaussian_shaped_labels(double sigma, int dim1, int dim2)
{
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = {-dim2 / 2, dim2 - dim2 / 2};
    int range_x[2] = {-dim1 / 2, dim1 - dim1 / 2};

    double sigma_s = sigma*sigma;

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j){
        float * row_ptr = labels.ptr<float>(j);
        double y_s = y*y;
        for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i){
            row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);
        }
    }

    //rotate so that 1 is at top-left corner
    cv::Mat rot_labels = circshift(labels, range_x[0], range_y[0]);
    //sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0,0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat CF_Tracker::circshift(const cv::Mat &patch, int x_rot, int y_rot)
{
    cv::Mat rot_patch(patch.size(), CV_32FC1);
    cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

    //circular rotate x-axis
    if (x_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-x_rot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-x_rot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(0, -x_rot);
        rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }else if (x_rot > 0){
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - x_rot);
        cv::Range rot_range(x_rot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(patch.cols - x_rot, patch.cols);
        rot_range = cv::Range(0, x_rot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }
    //circular rotate y-axis
    if (y_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-y_rot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-y_rot));
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }else if (y_rot > 0){
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - y_rot);
        cv::Range rot_range(y_rot, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(patch.rows - y_rot, patch.rows);
        rot_range = cv::Range(0, y_rot);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }

    return rot_patch;
}

ComplexMat CF_Tracker::fft2(const cv::Mat &input)
{
    cv::Mat complex_result;
//    cv::Mat padded;                            //expand input image to optimal size
//    int m = cv::getOptimalDFTSize( input.rows );
//    int n = cv::getOptimalDFTSize( input.cols ); // on the border add zero pixels
//    copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//    cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
//    return ComplexMat(complex_result(cv::Range(0, input.rows), cv::Range(0, input.cols)));

    cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
    return ComplexMat(complex_result);
}

ComplexMat CF_Tracker::fft2(const std::vector<cv::Mat> &input, const cv::Mat &cos_window)
{
    int n_channels = input.size();
    ComplexMat result(input[0].rows, input[0].cols, n_channels);
    for (int i = 0; i < n_channels; ++i){
        cv::Mat complex_result;
//        cv::Mat padded;                            //expand input image to optimal size
//        int m = cv::getOptimalDFTSize( input[0].rows );
//        int n = cv::getOptimalDFTSize( input[0].cols ); // on the border add zero pixels

//        copyMakeBorder(input[i].mul(cos_window), padded, 0, m - input[0].rows, 0, n - input[0].cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//        cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
//        result.set_channel(i, complex_result(cv::Range(0, input[0].rows), cv::Range(0, input[0].cols)));

        cv::dft(input[i].mul(cos_window), complex_result, cv::DFT_COMPLEX_OUTPUT);
        result.set_channel(i, complex_result);
    }
    return result;
}

cv::Mat CF_Tracker::ifft2(const ComplexMat &inputf)
{

    cv::Mat real_result;
    if (inputf.n_channels == 1){
        cv::dft(inputf.to_cv_mat(), real_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } else {
        std::vector<cv::Mat> mat_channels = inputf.to_cv_mat_vector();
        std::vector<cv::Mat> ifft_mats(inputf.n_channels);
        for (int i = 0; i < inputf.n_channels; ++i) {
            cv::dft(mat_channels[i], ifft_mats[i], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        }
        cv::merge(ifft_mats, real_result);
    }
    return real_result;
}

//hann window actually (Power-of-cosine windows)
cv::Mat CF_Tracker::cosine_window_function(int dim1, int dim2)
{
    cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
    double N_inv = 1./(static_cast<double>(dim1)-1.);
    for (int i = 0; i < dim1; ++i)
        m1.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    N_inv = 1./(static_cast<double>(dim2)-1.);
    for (int i = 0; i < dim2; ++i)
        m2.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    cv::Mat ret = m2*m1;
    return ret;
}

cv::Mat CF_Tracker::scale_cosine_window_function(int scaleNum)
{
	cv::Mat m1(1, scaleNum, CV_32FC1);
	for (int i = 0; i < (scaleNum / 2 + 1); i++)
	{
		m1.at<float>(i) = 0.5 * (1. - std::cos(CV_PI * (double)i / (double)(scaleNum / 2)));
		m1.at<float>(scaleNum - 1 - i) = m1.at<float>(i);
	}
	return m1;
}

// Returns sub-window of image input centered at [cx, cy] coordinates),
// with size [width, height]. If any pixels are outside of the image,
// they will replicate the values at the borders.
cv::Mat CF_Tracker::get_subwindow(const cv::Mat &input, int cx, int cy, int width, int height)
{
    cv::Mat patch;

    int x1 = cx - width/2;
    int y1 = cy - height/2;
    int x2 = cx + width/2;
    int y2 = cy + height/2;

    //out of image
    if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
        patch.create(height, width, CV_32FC1);
        patch.setTo(0.f);
        return patch;
    }

    int top = 0, bottom = 0, left = 0, right = 0;

    //fit to image coordinates, set border extensions;
    if (x1 < 0) {
        left = -x1;
        x1 = 0;
    }
    if (y1 < 0) {
        top = -y1;
        y1 = 0;
    }
    if (x2 >= input.cols) {
        right = x2 - input.cols + width % 2;
        x2 = input.cols;
    } else
        x2 += width % 2;

    if (y2 >= input.rows) {
        bottom = y2 - input.rows + height % 2;
        y2 = input.rows;
    } else
        y2 += height % 2;

    if (x2 - x1 == 0 || y2 - y1 == 0)
        patch = cv::Mat::zeros(height, width, CV_32FC1);
    else
        cv::copyMakeBorder(input(cv::Range(y1, y2), cv::Range(x1, x2)), patch, top, bottom, left, right, cv::BORDER_REPLICATE);

    //sanity check
    assert(patch.cols == width && patch.rows == height);

    return patch;
}

cv::Mat CF_Tracker::get_subwindow_resize(const cv::Mat & input, int cx, int cy, int width, int height, int ideal_x, int ideal_y)
{
	cv::Mat patch;

	int x1 = cx - width / 2;
	int y1 = cy - height / 2;
	int x2 = cx + width / 2;
	int y2 = cy + height / 2;

	//out of image
	if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
		patch.create(ideal_y, ideal_x, CV_32FC1);
		patch.setTo(0.f);
		return patch;
	}

	int top = 0, bottom = 0, left = 0, right = 0;

	//fit to image coordinates, set border extensions;
	if (x1 < 0) {
		left = -x1;
		x1 = 0;
	}
	if (y1 < 0) {
		top = -y1;
		y1 = 0;
	}
	if (x2 >= input.cols) {
		right = x2 - input.cols + width % 2;
		x2 = input.cols;
	}
	else
		x2 += width % 2;

	if (y2 >= input.rows) {
		bottom = y2 - input.rows + height % 2;
		y2 = input.rows;
	}
	else
		y2 += height % 2;

	if (x2 - x1 == 0 || y2 - y1 == 0)
		patch = cv::Mat::zeros(height, width, CV_32FC1);
	else
		cv::copyMakeBorder(input(cv::Range(y1, y2), cv::Range(x1, x2)), patch, top, bottom, left, right, cv::BORDER_REPLICATE);

	//sanity check
	assert(patch.cols == width && patch.rows == height);
	double scale_factor_x = (double)ideal_x / (double)patch.cols;
	double scale_factor_y = (double)ideal_y / (double)patch.rows;
	cv::resize(patch, patch, cv::Size(0, 0), scale_factor_x, scale_factor_y, cv::INTER_CUBIC);

	return patch;
}

ComplexMat CF_Tracker::gaussian_correlation(const ComplexMat &xf, const ComplexMat &yf, double sigma, bool auto_correlation)
{
    float xf_sqr_norm = xf.sqr_norm();
    float yf_sqr_norm = auto_correlation ? xf_sqr_norm : yf.sqr_norm();

    ComplexMat xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj();

    //ifft2 and sum over 3rd dimension, we dont care about individual channels
    cv::Mat xy_sum(xf.rows, xf.cols, CV_32FC1);
    xy_sum.setTo(0);
    cv::Mat ifft2_res = ifft2(xyf);
    for (int y = 0; y < xf.rows; ++y) {
        float * row_ptr = ifft2_res.ptr<float>(y);
        float * row_ptr_sum = xy_sum.ptr<float>(y);
        for (int x = 0; x < xf.cols; ++x){
            row_ptr_sum[x] = std::accumulate((row_ptr + x*ifft2_res.channels()), (row_ptr + x*ifft2_res.channels() + ifft2_res.channels()), 0.f);
        }
    }

    float numel_xf_inv = 1.f/(xf.cols * xf.rows * xf.n_channels);
    cv::Mat tmp;
    cv::exp(- 1.f / (sigma * sigma) * cv::max((xf_sqr_norm + yf_sqr_norm - 2 * xy_sum) * numel_xf_inv, 0), tmp);

    return fft2(tmp);
}