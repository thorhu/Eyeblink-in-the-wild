#include "Video.h"

#include <iostream>

using namespace std;
using namespace cv;

Video::Video(const std::string& path, bool write, const std::string& format) :
	m_frameIdx(0),
	m_write(write)
{
	m_formatString = path + "/" + format;
	if (write)
	{
		cout << "starting video recording to: " << path << endl;
	}
}


Video::~Video()
{
}


bool Video::WriteFrame(const Mat& frame)
{
	if (!m_write) return false;
	
	char buf[1024];
	sprintf(buf, m_formatString.c_str(), m_frameIdx);
	++m_frameIdx;
	
	return imwrite(buf, frame);
}

bool Video::ReadFrame(Mat& rFrame)
{
	if (m_write) return false;
	
	char buf[1024];
	sprintf(buf, m_formatString.c_str(), m_frameIdx);
	++m_frameIdx;
	
	rFrame = imread(buf, -1);
	
	return !rFrame.empty();
}
	
