#ifndef COMPLEX_MAT_HPP_213123048309482094
#define COMPLEX_MAT_HPP_213123048309482094

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <functional>

template<typename T> class ComplexMat_
{
public:
    int cols;
    int rows;
    int n_channels;

    ComplexMat_() : cols(0), rows(0), n_channels(0) {}
    ComplexMat_(int _rows, int _cols, int _n_channels) : cols(_cols), rows(_rows), n_channels(_n_channels)
    {
        p_data.resize(n_channels);
    }

    //assuming that mat has 2 channels (real, img)
    ComplexMat_(const cv::Mat & mat) : cols(mat.cols), rows(mat.rows), n_channels(1)
    {
        p_data.push_back(convert(mat));
    }
	ComplexMat_(const cv::Mat & mat, int GKCflag) : cols(mat.cols), rows(mat.rows), n_channels(1)
	{
		p_data.push_back(GKCconvert(mat));
	}
	ComplexMat_(const std::vector<cv::Mat> & mat) : cols(mat[0].rows), rows(mat[0].cols), n_channels(1)
	{
		p_data.push_back(convertVec(mat));
	}
	/*ComplexMat_(ComplexMat_<T> comMat, int times) : cols(times), rows(comMat.rows), n_channels(1)
	{
		std::vector<std::complex<T>> result;
		result.reserve(comMat.cols*comMat.rows*times);
		for (int t = 0; t < times; t++)
		{
			for (int y = 0; y < comMat.rows; ++y) {
				const T * row_ptr = comMat.p_data[0].begin();
				for (int x = 0; x < comMat.cols; x += 1){
					result.push_back(row_ptr[x]);
				}
			}
		}
		p_data.push_back(result);
	}*/

    //assuming that mat has 2 channels (real, imag)
    void set_channel(int idx, const cv::Mat & mat)
    {
        assert(idx >= 0 && idx < n_channels);
        p_data[idx] = convert(mat);
    }
	void set_channel(const std::vector<cv::Mat> & mat)
	{
		//assert(idx >= 0 && idx < n_channels);
		p_data[0] = convertVec(mat);
	}

	ComplexMat_<T> comSum()
	{
		std::vector<std::complex<T>> result;
		result.reserve(rows);
		ComplexMat_<T> thisComMat = *this;
		auto lhs = thisComMat.p_data[0].begin();
		T realSum=0,imgSum=0;
		
		for (int i = 0; i < rows; i++)
		{
			realSum = 0, imgSum = 0;
			for (int j = 0; j < cols; j++)
			{
				realSum += lhs->real();
				imgSum += lhs->imag();
				lhs++;
			}
			result.push_back(std::complex<T>(realSum, imgSum));
		}
		ComplexMat_<T> SumResult;
		SumResult.p_data.push_back(result);
		SumResult.rows = rows;
		SumResult.cols = 1;
		SumResult.n_channels = 1;
		return SumResult;
	}

    T sqr_norm() const
    {
        T sum_sqr_norm = 0;
        for (int i = 0; i < n_channels; ++i)
            for (auto lhs = p_data[i].begin(); lhs != p_data[i].end(); ++lhs)
                sum_sqr_norm += lhs->real()*lhs->real() + lhs->imag()*lhs->imag();
            //std::for_each(p_data[i].begin(), p_data[i].end(), [&sum_sqr_norm](const std::complex<T> & c) { sum_sqr_norm += c.real()*c.real() + c.imag()*c.imag(); } );
        return sum_sqr_norm / static_cast<T>(cols*rows);
    }

    ComplexMat_<T> sqr_mag() const
    {
        return mat_const_operator( [](std::complex<T> & c) { c = c.real()*c.real() + c.imag()*c.imag(); } );
    }

    ComplexMat_<T> conj() const
    {
        return mat_const_operator( [](std::complex<T> & c) { c = std::complex<T>(c.real(), -c.imag()); } );
    }

    //return 2 channels (real, imag) for first complex channel
    cv::Mat to_cv_mat() const
    {
        assert(p_data.size() >= 1);
        return channel_to_cv_mat(0);
    }
    //return a vector of 2 channels (real, imag) per one complex channel
    std::vector<cv::Mat> to_cv_mat_vector() const
    {
        std::vector<cv::Mat> result;
        result.reserve(n_channels);

        for (int i = 0; i < n_channels; ++i)
            result.push_back(channel_to_cv_mat(i));

        return result;
    }

    //element-wise per channel multiplication, division and addition
    /*ComplexMat_<T> operator*(const ComplexMat_<T> & rhs) const
    {
        return mat_mat_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs) { c_lhs *= c_rhs; }, rhs);
    }
    ComplexMat_<T> operator/(const ComplexMat_<T> & rhs) const
    {
        return mat_mat_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs) { c_lhs /= c_rhs; }, rhs);
    }
    ComplexMat_<T> operator+(const ComplexMat_<T> & rhs) const
    {
        return mat_mat_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs)  { c_lhs += c_rhs; }, rhs);
    }*/
	typedef void (*GKCFUN)(std::complex<T> & c_lhs, const std::complex<T> & c_rhs);//º¯ÊýÖ¸Õë
	
	ComplexMat_<T> operator*(const ComplexMat_<T> & rhs) const
	{
		return mat_mat_operator_1(rhs);
	}
	ComplexMat_<T> operator/(const ComplexMat_<T> & rhs) const
	{
		return mat_mat_operator_2(rhs);
	}
	ComplexMat_<T> operator+(const ComplexMat_<T> & rhs) const
	{
		return mat_mat_operator_3(rhs);
	}

    //multiplying or adding constant
    ComplexMat_<T> operator*(const T & rhs) const
    {
        return mat_const_operator( [&rhs](std::complex<T> & c) { c *= rhs; });
    }
    ComplexMat_<T> operator+(const T & rhs) const
    {
        return mat_const_operator( [&rhs](std::complex<T> & c) { c += rhs; });
    }

    //text output
    friend std::ostream & operator<<(std::ostream & os, const ComplexMat_<T> & mat)
    {
        //for (int i = 0; i < mat.n_channels; ++i){
        for (int i = 0; i < 1; ++i){
            os << "Channel " << i << std::endl;
            for (int j = 0; j < mat.rows; ++j) {
                for (int k = 0; k < mat.cols-1; ++k)
                    os << mat.p_data[i][j*mat.cols + k] << ", ";
                os << mat.p_data[i][j*mat.cols + mat.cols-1] << std::endl;
            }
        }
        return os;
    }


private:
    std::vector<std::vector<std::complex<T>>> p_data;
	void GKCFunc_1(std::complex<T> & c_lhs, const std::complex<T> & c_rhs) const
	{
		c_lhs *= c_rhs;
	}
	void GKCFunc_2(std::complex<T> & c_lhs, const std::complex<T> & c_rhs) const
	{
		c_lhs /= c_rhs;
	}
	void GKCFunc_3(std::complex<T> & c_lhs, const std::complex<T> & c_rhs) const
	{
		c_lhs += c_rhs;
	}
    //convert 2 channel mat (real, imag) to vector row-by-row
    std::vector<std::complex<T>> convert(const cv::Mat & mat)
    {
        std::vector<std::complex<T>> result;
        result.reserve(mat.cols*mat.rows);
        for (int y = 0; y < mat.rows; ++y) {
            const T * row_ptr = mat.ptr<T>(y);
            for (int x = 0; x < 2*mat.cols; x += 2){
                result.push_back(std::complex<T>(row_ptr[x], row_ptr[x+1]));
            }
        }
        return result;
    }
	std::vector<std::complex<T>> GKCconvert(const cv::Mat & mat)
	{
		assert((mat.rows == 1));
		int bigDim = (mat.rows == 1) ? (mat.cols) : (mat.rows);
		std::vector<std::complex<T>> result;
		result.reserve(mat.cols*mat.rows);
		if ((bigDim % 2) == 0)
		{
			for (int y = 0; y < mat.rows; ++y)
			{
				const T * row_ptr = mat.ptr<T>(y);
				int times = 0;
				int x = 0;
				for (; (x < 2 * mat.cols) && (times < mat.cols);)
				{
					if (times < (mat.cols / 2 + 1))
					{
						result.push_back(std::complex<T>(row_ptr[x], row_ptr[x + 1]));
						times++;
						if (times < (mat.cols / 2 + 1))
						{
							x += 2;
						}
					}
					else
					{
						x -= 2;
						result.push_back(std::complex<T>(row_ptr[x], -row_ptr[x + 1]));
						times++;
					}

				}
			}
		}
		else
		{
			for (int y = 0; y < mat.rows; ++y)
			{
				const T * row_ptr = mat.ptr<T>(y);
				int times = 0;
				int x = 0;
				for (; (x < 2 * mat.cols) && (times < mat.cols);)
				{
					if (times < (mat.cols / 2 + 1))
					{
						result.push_back(std::complex<T>(row_ptr[x], row_ptr[x + 1]));
						times++;
						if (times < (mat.cols / 2 + 1))
						{
							x += 2;
						}
					}
					else
					{
						result.push_back(std::complex<T>(row_ptr[x], -row_ptr[x + 1]));
						x -= 2;
						times++;
					}

				}
			}
		}
		return result;
	}
	std::vector<std::complex<T>> convertVec(const std::vector<cv::Mat> & mat)
	{
		std::vector<std::vector<std::complex<T>>> p_data_temp;
		std::vector<std::complex<T>> result;
		result.reserve(mat[0].cols*mat[0].rows*mat.size());
		for (int t = 0; t < mat.size(); t++)
		{
			std::vector<std::complex<T>> result_temp = GKCconvert(mat[t]);
			p_data_temp.push_back(result_temp);
			//for (int i = 0; i < result_temp.size(); i++)
			//{
				//result.push_back(result_temp[i]);
			//}
			/*for (int y = 0; y < mat[t].rows; ++y) {
				const T * row_ptr = mat[t].ptr<T>(y);
				for (int x = 0; x < 2 * mat[t].cols; x += 2){
					result.push_back(std::complex<T>(row_ptr[x], row_ptr[x + 1]));
				}
			}*/
		}
		for (int j = 0; j < mat[0].cols; j++)
		{
			for (int i = 0; i < mat[0].rows; i++)
			{
				for (int t = 0; t < mat.size(); t++)
				{
					result.push_back(p_data_temp[t][j]);
				}
			}
		}
		return result;
	}

    ComplexMat_<T> mat_mat_operator(void (*op)(std::complex<T> & c_lhs, const std::complex<T> & c_rhs), const ComplexMat_<T> & mat_rhs) const
    {
		assert(mat_rhs.n_channels == n_channels);
		ComplexMat_<T> result = *this;
		if (mat_rhs.cols == cols && mat_rhs.rows == rows)
		{
			for (int i = 0; i < n_channels; ++i)
			{
				auto lhs = result.p_data[i].begin();
				auto rhs = mat_rhs.p_data[i].begin();
				for (;
					lhs != result.p_data[i].end(); ++lhs, ++rhs)
					op(*lhs, *rhs);
			}
		}
		else //if (mat_rhs.cols != cols && mat_rhs.rows == rows)
		{
			for (int i = 0; i < n_channels; ++i)
			{
				auto lhs = result.p_data[i].begin();
				auto rhs = mat_rhs.p_data[i].begin();
				//for (;lhs != result.p_data[i].end(); ++lhs, ++rhs)
				for (int i = 0; i < rows; i++)
				{
					for (int j = 0; j < cols; j++)
					{
						op(*lhs, *rhs);
						++lhs;
					}
					++rhs;
				}
					
			}
		}
        return result;
    }
	ComplexMat_<T> mat_mat_operator_1(const ComplexMat_<T> & mat_rhs) const
	{
		/*assert(mat_rhs.n_channels == n_channels && mat_rhs.cols == cols && mat_rhs.rows == rows);

		ComplexMat_<T> result = *this;

		for (int i = 0; i < n_channels; ++i)
		{
			auto lhs = result.p_data[i].begin();
			auto rhs = mat_rhs.p_data[i].begin();
			for (;
				lhs != result.p_data[i].end(); ++lhs, ++rhs)
				GKCFunc_1(*lhs, *rhs);
		}
		return result;*/

		assert(mat_rhs.n_channels == n_channels);
		ComplexMat_<T> result = *this;
		if (mat_rhs.cols == cols && mat_rhs.rows == rows)
		{
			for (int i = 0; i < n_channels; ++i)
			{
				auto lhs = result.p_data[i].begin();
				auto rhs = mat_rhs.p_data[i].begin();
				for (;
					lhs != result.p_data[i].end(); ++lhs, ++rhs)
					GKCFunc_1(*lhs, *rhs);
			}
		}
		else //if (mat_rhs.cols != cols && mat_rhs.rows == rows)
		{
			for (int i = 0; i < n_channels; ++i)
			{
				auto lhs = result.p_data[i].begin();
				auto rhs = mat_rhs.p_data[i].begin();
				//for (;lhs != result.p_data[i].end(); ++lhs, ++rhs)
				for (int i = 0; i < rows; i++)
				{
					for (int j = 0; j < cols; j++)
					{
						GKCFunc_1(*lhs, *rhs);
						++lhs;
					}
					++rhs;
				}
					
			}
		}
        return result;
	}
	ComplexMat_<T> mat_mat_operator_2(const ComplexMat_<T> & mat_rhs) const
	{
		assert(mat_rhs.n_channels == n_channels && mat_rhs.cols == cols && mat_rhs.rows == rows);

		ComplexMat_<T> result = *this;

		for (int i = 0; i < n_channels; ++i)
		{
			auto lhs = result.p_data[i].begin();
			auto rhs = mat_rhs.p_data[i].begin();
			for (;
				lhs != result.p_data[i].end(); ++lhs, ++rhs)
				GKCFunc_2(*lhs, *rhs);
		}
		return result;
	}
	ComplexMat_<T> mat_mat_operator_3(const ComplexMat_<T> & mat_rhs) const
	{
		assert(mat_rhs.n_channels == n_channels && mat_rhs.cols == cols && mat_rhs.rows == rows);

		ComplexMat_<T> result = *this;

		for (int i = 0; i < n_channels; ++i)
		{
			auto lhs = result.p_data[i].begin();
			auto rhs = mat_rhs.p_data[i].begin();
			for (;
				lhs != result.p_data[i].end(); ++lhs, ++rhs)
				GKCFunc_3(*lhs, *rhs);
		}
		return result;
	}
    ComplexMat_<T> mat_const_operator(const std::function<void(std::complex<T> & c_rhs)> & op) const
    {
        ComplexMat_<T> result = *this;
//        std::for_each(result.p_data.begin(), result.p_data.end(),
//                [&op] (std::vector<std::complex<T>> & channel) { std::for_each(channel.begin(), channel.end(), op); });
        for (int i = 0; i < n_channels; ++i)
            for (auto lhs = result.p_data[i].begin(); lhs != result.p_data[i].end(); ++lhs)
                op(*lhs);
        return result;
    }

    cv::Mat channel_to_cv_mat(int channel_id) const
    {
        cv::Mat result(rows, cols, CV_32FC2);
        int data_id = 0;
        for (int y = 0; y < rows; ++y) {
            T * row_ptr = result.ptr<T>(y);
            for (int x = 0; x < 2*cols; x += 2){
                row_ptr[x] = p_data[channel_id][data_id].real();
                row_ptr[x+1] = p_data[channel_id][data_id++].imag();
            }
        }
        return result;
    }

};

typedef ComplexMat_<float> ComplexMat;


#endif //COMPLEX_MAT_HPP_213123048309482094