#include <math.h>
#include "wrappers.h"
#include "sse.h"
#include "proposal.h"

// normalize gradient magnitude at each location (uses sse)
void gradMagNorm( float *M, float *S, int h, int w, float norm ) 
{
	__m128 *_M, *_S, _norm; int i=0, n=h*w, n4=n/4;
	_S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
	bool sse = !(size_t(M)&15) && !(size_t(S)&15);
	if(sse) 
	{ 
		for(; i<n4; i++) 
			*_M++=MUL(*_M,RCP(ADD(*_S++,_norm))); 
		i*=4; 
	}
	for(; i<n; i++) 
		M[i] /= (S[i] + norm);
}


// gradMagNorm( M, S, norm ) - operates on M - see gradientMag.m
void GradMagNormMex(IMGDATA *imgm,IMGDATA *imgs,float norm) 
{
	int h, w, d; float *M, *S;
	
	h = imgm->height;
	w = imgm->width;
	d = imgm->chans;
	
	M = (float *)imgm->data;
	S = (float *)imgs->data;

	gradMagNorm(M,S,h,w,norm);
}
