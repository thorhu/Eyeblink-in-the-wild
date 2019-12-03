#ifndef CF_HEADER_6565467831231
#define CF_HEADER_6565467831231

#include <opencv2/opencv.hpp>
#include <vector>
#include "piotr_fhog\fhog.hpp"
#include "complexmat.hpp"

static const int nScales = 33; //must be odd integer

cv::Mat GKCgetInputRGBImage(void);

struct BBox_c
{
	double cx, cy, w, h;

	inline void scale(double factor){
		cx *= factor;
		cy *= factor;
		w *= factor;
		h *= factor;
	}
};

class CF_Tracker
{
public:
	double scale_now;
	/*
	padding             ... extra area surrounding the target           (1.5)
	kernel_sigma        ... gaussian kernel bandwidth                   (0.5)
	lambda              ... regularization                              (1e-4)
	interp_factor       ... linear interpolation factor for adaptation  (0.02)
	output_sigma_factor ... spatial bandwidth (proportional to target)  (0.1)
	cell_size           ... hog cell size                               (4)
	*/
	CF_Tracker(double padding, double kernel_sigma, double lambda, double interp_factor, double output_sigma_factor, int cell_size) :
		p_padding(padding), p_output_sigma_factor(output_sigma_factor), p_kernel_sigma(kernel_sigma),
		p_lambda(lambda), p_interp_factor(interp_factor), p_cell_size(cell_size) {}
	CF_Tracker() 
	{
		pca_encoding = new cv::PCA();
		p_resize_image = false;

		p_padding = 1.5;// 1.0;
		p_output_sigma_factor = 0.1;//0.1
		p_output_sigma;
		p_kernel_sigma = 0.5;    //def = 0.5
		p_lambda = 1e-4;         //regularization in learning step
		p_interp_factor = 0.01;  //def = 0.02, linear interpolation factor for adaptation
		p_cell_size = 2;            //4 for hog (= bin_size)

		scale_step = 1.02;
		scale_sigma_factor = 0.25;
		scale_sigma;
		scale_model_area_max = 512;
		s_learning_rate = 0.025;
		s_lambda = 1e-2;
	}

	// Init/re-init methods
	int init(cv::Mat & img, BBox_c & bbox);
	void setTrackerPose(BBox_c & bbox, cv::Mat & img);
	void updateTrackerPosition(BBox_c & bbox);

	// frame-to-frame object tracking
	void track(cv::Mat &img, int updateFlag, BBox_c newPose);
	BBox_c getBBox();
	double getCFScore(){ return CFscore; }

	CF_Tracker operator=(CF_Tracker temp);


public:
	double CFscore;
	cv::PCA *pca_encoding;// = new cv::PCA();
	BBox_c p_pose;
	BBox_c p_scale;
	bool p_resize_image;

	double p_padding;
	double p_output_sigma_factor;
	double p_output_sigma;
	double p_kernel_sigma;    //def = 0.5
	double p_lambda;         //regularization in learning step
	double p_interp_factor;  //def = 0.02, linear interpolation factor for adaptation
	int p_cell_size;            //4 for hog (= bin_size)
	double p_init_width;
	double p_init_height;
	int p_windows_size[2];
	int s_windows_size[2];
	cv::Mat p_cos_window;
	cv::Mat scale_window;

	double scale_step;
	double scale_sigma_factor;
	double scale_sigma;
	double scale_model_area_max;
	double s_learning_rate;
	double s_lambda;
	double scales_array[nScales];
	int scales_index[nScales];
	double s_init_width;
	double s_init_height;
	double scale_model_factor;

	FHoG p_fhog;                    //class encapsulating hog feature extraction

	//model
	ComplexMat p_yf;
	ComplexMat p_model_alphaf;
	ComplexMat p_model_alphaf_num;
	ComplexMat p_model_alphaf_den;
	ComplexMat p_model_xf;
	ComplexMat s_yf;
	ComplexMat s_model_alphaf;
	ComplexMat s_model_alphaf_num;
	ComplexMat s_model_alphaf_den;
	ComplexMat s_model_xf;

	//helping functions
	std::vector<cv::Mat> CF_Tracker::getMultiFeature(cv::Mat patch);
	std::vector<cv::Mat> CF_Tracker::getMultiFeature(cv::Mat patch, cv::Mat BGRpatch);
	std::vector<cv::Mat> PCA_decrease(int init_flag, std::vector<cv::Mat> &input, int newDim);
	cv::Mat get_subwindow(const cv::Mat & input, int cx, int cy, int width, int height);
	cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
	ComplexMat gaussian_correlation(const ComplexMat & xf, const ComplexMat & yf, double sigma, bool auto_correlation = false);
	cv::Mat circshift(const cv::Mat & patch, int x_rot, int y_rot);
	cv::Mat cosine_window_function(int dim1, int dim2);
	ComplexMat fft2(const cv::Mat & input);
	ComplexMat fft2(const std::vector<cv::Mat> & input, const cv::Mat & cos_window);
	cv::Mat ifft2(const ComplexMat & inputf);

	//scale
	cv::Mat get_subwindow_resize(const cv::Mat & input, int cx, int cy, int width, int height, int ideal_x, int ideal_y);
	cv::Mat scale_cosine_window_function(int scaleNum);
	//tests
	friend void run_tests(CF_Tracker & tracker, const std::vector<bool> & tests);
};

#endif //CF_HEADER_6565467831231