#ifndef _PROPOSAL_H_
#define _PROPOSAL_H_

#pragma once
//#pragma warning(disable: 4244)
#pragma warning(disable: 4018)
//#pragma warning(disable: 4305)

#include <vector>
#include <opencv2/core/core.hpp>  
#define PI 3.14159265f

typedef unsigned int   uint32;
typedef unsigned short uint16;
typedef unsigned char  uint8;
typedef unsigned char  uchar;
typedef struct tagBox
{ 
	int c, r, w, h; 
	float s;
	float percent;
}Box;

using namespace std;
//using namespace cv;
typedef vector<Box> Boxes;

/* Random Forest struct */
typedef struct tagOPT_TREE_S
{
	float  *thrs;
	uint32 *fids;
	uint32 *child;
	uint8  *segs;
	uint8  *nSegs;
	uint16 *eBins;
	uint32 *eBnds;
}OPT_TREE_S;

/* EDGEDEC struct */
typedef struct tagOPT_EDGEDEC_S
{
	int imWidth; 
	int gtWidth;   
	int nTrees;  
	int nOrients;       
	int grdSmooth;     
	int chnSmooth;  
	int simSmooth;   
	int normRad;
	int shrink;
	int nCells; 
	int stride;
    int multiscale;
	int sharpen;
	int nTreesEval;
	int nThreads;
	int nms;
	int nChns;
	int nChnFtrs;
	int nSimFtrs;
	int nTotFtrs;
}OPT_EDGEDEC_S;

/* EDGEBOX struct */
typedef struct tagOPT_EDGEBOX_S
{
	int   maxBoxes; 
	int   minBoxArea;   
	int   maxAspectRatio;  
	float alpha;       
	float beta;     
	float minScore;  
	float edgeMinMag;   
	float edgeMergeThr;
	float clusterMinMag;   
	float gamma;
    float kappa;
}OPT_EDGEBOX_S;

//type
//0  double
//1  float
//2  uint8
//3  uint16
//4  uint32
typedef struct tagIMGDATA
{
    uint32  width;
	uint32  height;
	uint32  chans;
	int     imgtype;
	void    *data;     
}IMGDATA;

void convConstMex(IMGDATA *img,IMGDATA *imgout, char *type,int r,int s);
void rgbConvertMex(IMGDATA *img,IMGDATA *imgout, int flag, bool useSingle);
void imResampleMex(IMGDATA *img,IMGDATA *imgout,int height, int width, float norm);
void imPadMex(IMGDATA *img,IMGDATA *imgout, float *pad); 
void GradMagNormMex(IMGDATA *imgm,IMGDATA *imgs,float norm);
void GradMagMex(IMGDATA *img,IMGDATA *imgmag, IMGDATA *imgori,int channel,int full);
void GradHistMex(IMGDATA *imgmag,IMGDATA *imgori, IMGDATA *imghist, int binSize, int nOrients, int softBin);
void Grad2Mex(IMGDATA *img,IMGDATA *imgx,IMGDATA *imgy);
void edgesNmsMex(IMGDATA *imgmagout,IMGDATA *imgmag, IMGDATA *imgori, int r, int s, int m);
void edgesDetectMex(IMGDATA *pstimg,IMGDATA *imgmagout,float *chnsftr, float *chnsSsftr,OPT_TREE_S* pstTree,OPT_EDGEDEC_S* pstedgedec);
void edgeBoxesMex(IMGDATA *imgmag, IMGDATA *imgori, OPT_EDGEBOX_S *pstedgebox, Boxes &vecobj);


void edgesChns(IMGDATA *img, OPT_EDGEDEC_S* pstedgedec, float **chnsReg, float ** chnsSim);
void edgesDetect(IMGDATA *img, IMGDATA *imgmag,IMGDATA *imgori,OPT_TREE_S* pstTree,OPT_EDGEDEC_S* pstedgedec);
void edgeboxes(IMGDATA *img, OPT_EDGEBOX_S *pstedgebox, OPT_TREE_S* pstTree,OPT_EDGEDEC_S* pstedgedec, Boxes &vecobj);

//Boxes eagebox();
Boxes eagebox(cv::Mat frame);

#endif
