#include <string.h>
#include "proposal.h"

// pad A by [pt,pb,pl,pr] and store result in B
template<class T> void imPad( T *A, T *B, int h, int w, int d, int pt, int pb,
  int pl, int pr)
{
	int h1=h+pt, hb=h1+pb, w1=w+pl, wb=w1+pr, x, y, z;
	int ct=0, cb=0, cl=0, cr=0;
	
	x=pr>pl?pr:pl; y=pt>pb?pt:pb;
	// helper macro for padding
	#define PAD(XL,XM,XR,YT,YM,YB) \
	for(x=0;  x<pl; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XL+cl)*h+YT+ct]; \
	for(x=0;  x<pl; x++) for(y=pt; y<h1; y++) B[x*hb+y]=A[(XL+cl)*h+YM+ct]; \
	for(x=0;  x<pl; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XL+cl)*h+YB-cb]; \
	for(x=pl; x<w1; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XM+cl)*h+YT+ct]; \
	for(x=pl; x<w1; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XM+cl)*h+YB-cb]; \
	for(x=w1; x<wb; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XR-cr)*h+YT+ct]; \
	for(x=w1; x<wb; x++) for(y=pt; y<h1; y++) B[x*hb+y]=A[(XR-cr)*h+YM+ct]; \
	for(x=w1; x<wb; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XR-cr)*h+YB-cb];
	// pad by appropriate value
	for( z=0; z<d; z++ ) 
	{
		// copy over A to relevant region in B
		for( x=0; x<w-cr-cl; x++ )
			memcpy(B+(x+pl)*hb+pt,A+(x+cl)*h+ct,sizeof(T)*(h-ct-cb));
		// set boundaries of B to appropriate values
		PAD( pl-x-1, x-pl, w+w1-1-x, pt-y-1, y-pt, h+h1-1-y );

		A += h*w;  B += hb*wb;
	}
	#undef PAD
}

// B = imPadMex(A,pad,type); see imPad.m for usage details
void imPadMex(IMGDATA *img,IMGDATA *imgout, float *pad) 
{
	int ns[3], ms[3], nCh, pt, pb, pl, pr; 
	void *A, *B;
	int imgtype;

	ns[0]   = img->height;
	ns[1]   = img->width;
	ns[2]   = img->chans;
	imgtype = img->imgtype;
	nCh = ns[2];

	// extract padding amounts
	pt=int(pad[0]); 
	pb=int(pad[1]); 
	pl=int(pad[2]); 
	pr=int(pad[3]); 

	// create output array
	ms[0]=ns[0]+pt+pb; 
	ms[1]=ns[1]+pl+pr; 
	ms[2]=nCh;

	imgout->height  = ms[0];
	imgout->width   = ms[1];
	imgout->chans   = ms[2];
	imgout->imgtype = imgtype; // uchar
	imgout->data    = new uchar[ms[0] * ms[1] * ms[2]];

	// pad array
	A = img->data; 
	B = imgout->data;
	
	//double
	if(imgtype==0) 
	{
		imPad((double*)A,(double*)B,ns[0],ns[1],nCh,pt,pb,pl,pr );
	}
	//float	
	else if(imgtype==1) 
	{
		imPad((float*)A,(float*)B,ns[0],ns[1],nCh,pt,pb,pl,pr);
	} 
	//uchar
	else if(imgtype==2 ) 
	{
		imPad( (uchar*)A,(uchar*)B,ns[0],ns[1],nCh,pt,pb,pl,pr);
	}
}
