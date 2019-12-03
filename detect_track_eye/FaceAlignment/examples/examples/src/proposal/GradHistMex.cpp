#include <math.h>
#include "wrappers.h"
#include "sse.h"
#include "proposal.h"

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize( float *O, float *M, int *O0, int *O1, float *M0, float *M1,
  int nb, int n, float norm, int nOrients, bool full, bool interpolate )
{
	// assumes all *OUTPUT* matrices are 4-byte aligned
	int i, o0, o1; float o, od, m;
	__m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;
	// define useful constants
	const float oMult=(float)nOrients/(full?2*PI:PI); const int oMax=nOrients*nb;
	const __m128 _norm=SET(norm), _oMult=SET(oMult), _nbf=SET((float)nb);
	const __m128i _oMax=SET(oMax), _nb=SET(nb);
	// perform the majority of the work with sse
	_O0=(__m128i*) O0; _O1=(__m128i*) O1; _M0=(__m128*) M0; _M1=(__m128*) M1;
	if( interpolate ) 
		for( i=0; i<=n-4; i+=4 ) 
		{
			_o=MUL(LDu(O[i]),_oMult); _o0=CVT(_o); _od=SUB(_o,CVT(_o0));
			_o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
			_o1=ADD(_o0,_nb); _o1=AND(CMPGT(_oMax,_o1),_o1); *_O1++=_o1;
			_m=MUL(LDu(M[i]),_norm); *_M1=MUL(_od,_m); *_M0++=SUB(_m,*_M1); _M1++;
		} 
	else 
		for( i=0; i<=n-4; i+=4 ) 
		{
			_o=MUL(LDu(O[i]),_oMult); _o0=CVT(ADD(_o,SET(.5f)));
			_o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
			*_M0++=MUL(LDu(M[i]),_norm); *_M1++=SET(0.f); *_O1++=SET(0);
		}
	// compute trailing locations without sse
	if( interpolate ) 
		for( i; i<n; i++ ) 
		{
			o=O[i]*oMult; o0=(int) o; od=o-o0;
			o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
			o1=o0+nb; if(o1==oMax) o1=0; O1[i]=o1;
			m=M[i]*norm; M1[i]=od*m; M0[i]=m-M1[i];
		} 
	else 
		for( i; i<n; i++ ) 
		{
			o=O[i]*oMult; o0=(int) (o+.5f);
			o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
			M0[i]=M[i]*norm; M1[i]=0; O1[i]=0;
		}
}

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist( float *M, float *O, float *H, int h, int w,
  int bin, int nOrients, int softBin, bool full )
{
	const int hb=h/bin, wb=w/bin, h0=hb*bin, w0=wb*bin, nb=wb*hb;
	const float s=(float)bin, sInv=1/s, sInv2=1/s/s;
	float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1; float xb, init;
	O0=(int*)alMalloc(h*sizeof(int),16); M0=(float*) alMalloc(h*sizeof(float),16);
	O1=(int*)alMalloc(h*sizeof(int),16); M1=(float*) alMalloc(h*sizeof(float),16);
	// main loop
	for( x=0; x<w0; x++ ) 
	{
		// compute target orientation bins for entire column - very fast
		gradQuantize(O+x*h,M+x*h,O0,O1,M0,M1,nb,h0,sInv2,nOrients,full,softBin>=0);

		if( softBin<0 && softBin%2==0 ) 
		{
			// no interpolation w.r.t. either orienation or spatial bin
			H1=H+(x/bin)*hb;
			#define GH H1[O0[y]]+=M0[y]; y++;
			if( bin==1 )      
				for(y=0; y<h0;) 
				{ GH; H1++; }
			else if( bin==2 ) 
				for(y=0; y<h0;) 
				{ GH; GH; H1++; }
			else if( bin==3 ) 
				for(y=0; y<h0;) 
				{ GH; GH; GH; H1++; }
			else if( bin==4 ) 
				for(y=0; y<h0;) 
				{ GH; GH; GH; GH; H1++; }
			else 
				for( y=0; y<h0;) 
				{ for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
			#undef GH
		} 
		else if( softBin%2==0 || bin==1 ) 
		{
			// interpolate w.r.t. orientation only, not spatial bin
			H1=H+(x/bin)*hb;
			#define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
			if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
			else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
			else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
			else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
			else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
			#undef GH
		} 
		else 
		{
			// interpolate using trilinear interpolation
			float ms[4], xyd, yb, xd, yd; __m128 _m, _m0, _m1;
			bool hasLf, hasRt; int xb0, yb0;
			if( x==0 ) { init=(0+.5f)*sInv-0.5f; xb=init; }
			hasLf = xb>=0; xb0 = hasLf?(int)xb:-1; hasRt = xb0 < wb-1;
			xd=xb-xb0; xb+=sInv; yb=init; y=0;
			// macros for code conciseness
			#define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
			ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
			#define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));
			// leading rows, no top bin
			for( ; y<bin/2; y++ ) 
			{
				yb0=-1; GHinit;
				if(hasLf) { H0[O0[y]+1]+=ms[1]*M0[y]; H0[O1[y]+1]+=ms[1]*M1[y]; }
				if(hasRt) { H0[O0[y]+hb+1]+=ms[3]*M0[y]; H0[O1[y]+hb+1]+=ms[3]*M1[y]; }
			}
			// main rows, has top and bottom bins, use SSE for minor speedup
			if( softBin<0 ) 
				for( ; ; y++ ) 
				{
					yb0 = (int) yb; if(yb0>=hb-1) break; GHinit; _m0=SET(M0[y]);
					if(hasLf) { _m=SET(0,0,ms[1],ms[0]); GH(H0+O0[y],_m,_m0); }
					if(hasRt) { _m=SET(0,0,ms[3],ms[2]); GH(H0+O0[y]+hb,_m,_m0); }
				} 
			else 
				for( ; ; y++ ) 
				{
					yb0 = (int) yb; 
					if(yb0>=hb-1) 
						break; 
					GHinit;
					_m0=SET(M0[y]); _m1=SET(M1[y]);
					if(hasLf) 
					{ 
						_m=SET(0,0,ms[1],ms[0]);
						GH(H0+O0[y],_m,_m0); 
						GH(H0+O1[y],_m,_m1); 
					}
					if(hasRt) 
					{ 
						_m=SET(0,0,ms[3],ms[2]);
						GH(H0+O0[y]+hb,_m,_m0); 
						GH(H0+O1[y]+hb,_m,_m1); 
					}
				}
			// final rows, no bottom bin
			for( ; y<h0; y++ ) 
			{
				yb0 = (int) yb; GHinit;
				if(hasLf) 
				{ 
					H0[O0[y]]+=ms[0]*M0[y]; H0[O1[y]]+=ms[0]*M1[y]; 
				}
				if(hasRt) 
				{ 
					H0[O0[y]+hb]+=ms[2]*M0[y]; H0[O1[y]+hb]+=ms[2]*M1[y]; 
				}
			}
			#undef GHinit
			#undef GH
		}
	}
	alFree(O0); alFree(O1); alFree(M0); alFree(M1);
	// normalize boundary bins which only get 7/8 of weight of interior bins
	if( softBin%2!=0 ) 
		for( int o=0; o<nOrients; o++ ) 
		{
			x=0; 
			for( y=0; y<hb; y++ ) 
				H[o*nb+x*hb+y]*=8.f/7.f;
			y=0; 
			for( x=0; x<wb; x++ ) 
				H[o*nb+x*hb+y]*=8.f/7.f;
			x=wb-1; 
			for( y=0; y<hb; y++ ) 
				H[o*nb+x*hb+y]*=8.f/7.f;
			y=hb-1; 
			for( x=0; x<wb; x++ ) 
				H[o*nb+x*hb+y]*=8.f/7.f;
		}
}

// H=gradHist(M,O,[...]) - see gradientHist.m
void GradHistMex(IMGDATA *imgmag,IMGDATA *imgori, IMGDATA *imghist, int binSize, int nOrients, int softBin) 
{
	int h, w, d, hb, wb, nChns;
	bool full; float *M, *O, *H;
	
	h = imgmag->height;
	w = imgmag->width;
	d = imgmag->chans;

	M = (float *)imgmag->data;
	O = (float *)imgori->data;

	full = false;
	hb = h/binSize; 
	wb = w/binSize;
	nChns = nOrients;
	
	// create output array (w/o initializing to 0)
	imghist->height  = hb;
	imghist->width   = wb;
	imghist->chans   = nChns;
	imghist->imgtype = 1; //float
	
	H = new float[hb * wb * nChns]();
	gradHist( M, O, H, h, w, binSize, nOrients, softBin, full);

	imghist->data = H;
}