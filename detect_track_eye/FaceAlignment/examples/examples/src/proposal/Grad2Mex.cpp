#include <math.h>
#include "wrappers.h"
#include "sse.h"
#include "proposal.h"

// compute x and y gradients for just one column (uses sse)
void grad1( float *I, float *Gx, float *Gy, int h, int w, int x ) 
{
	int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
	// compute column of Gx
	Ip=I-h; In=I+h; r=.5f;
	if(x==0) 
	{ 
		r=1; 
		Ip+=h; 
	} 
	else if(x==w-1) 
	{ 
		r=1; 
		In-=h; 
	}
	if( h<4 || h%4>0 || (size_t(I)&15) || (size_t(Gx)&15) ) 
	{
		for( y=0; y<h; y++ ) 
			*Gx++=(*In++-*Ip++)*r;
	} 
	else 
	{
		_G=(__m128*) Gx; 
		_Ip=(__m128*) Ip; 
		_In=(__m128*) In; 
		_r = SET(r);
		for(y=0; y<h; y+=4) 
			*_G++=MUL(SUB(*_In++,*_Ip++),_r);
	}
	// compute column of Gy
	#define GRADY(r) *Gy++=(*In++-*Ip++)*r;
	Ip=I; In=Ip+1;
	// GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
	y1=((~((size_t) Gy) + 1) & 15)/4; 
	if(y1==0) 
		y1=4; 
	if(y1>h-1) 
		y1=h-1;
	GRADY(1); Ip--; 
	for(y=1; y<y1; y++) 
		GRADY(.5f);
	_r = SET(.5f); 
	_G=(__m128*) Gy;
	for(; y+4<h-1; y+=4, Ip+=4, In+=4, Gy+=4)
		*_G++=MUL(SUB(LDu(*In),LDu(*Ip)),_r);
	for(; y<h-1; y++) 
		GRADY(.5f); 
	In--; 
	GRADY(1);
	#undef GRADY
}

// compute x and y gradients at each location (uses sse)
void grad2( float *I, float *Gx, float *Gy, int h, int w, int d ) 
{
	int o, x, c, a=w*h; 
	for(c=0; c<d; c++) 
		for(x=0; x<w; x++) 
		{
			o=c*a+x*h; 
			grad1( I+o, Gx+o, Gy+o, h, w, x );
		}
}

// [Gx,Gy] = grad2(I) - see gradient2.m
void Grad2Mex(IMGDATA *img,IMGDATA *imgx,IMGDATA *imgy) 
{
	int h, w, d; 
	float *I, *Gx, *Gy;
	
	h = img->height;
	w = img->width;
	d = img->chans;
	I = (float *)img->data;
	
	// create output array (w/o initializing to 0)
	imgx->height  = h;
	imgx->width   = w;
	imgx->chans   = d;
	imgx->imgtype = 1; //float

	imgy->height  = h;
	imgy->width   = w;
	imgy->chans   = d;
	imgy->imgtype = 1; //float
	
	Gx = new float[h * w * d];
	Gy = new float[h * w * d];
	grad2( I, Gx, Gy, h, w, d );
	
	imgx->data = Gx;
	imgy->data = Gy;
}