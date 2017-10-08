// Author: Ce Liu (c) Dec, 2009; celiu@mit.edu
// Modified By: Deepak Pathak (c) 2016; pathak@berkeley.edu

#ifndef _ImageProcessing_h
#define _ImageProcessing_h

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <typeinfo>

//----------------------------------------------------------------------------------
// class to handle basic image processing functions
// this is a collection of template functions. These template functions are
// used in other image classes such as BiImage, IntImage and FImage
//----------------------------------------------------------------------------------

class ImageProcessing
{
public:
	ImageProcessing(void);
	~ImageProcessing(void);
public:
	// basic functions
	template <class T>
	static inline T EnforceRange(const T& x,const int& MaxValue) {return __min(__max(x,0),MaxValue-1);};

	//---------------------------------------------------------------------------------
	// function to interpolate the image plane
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static inline void BilinearInterpolate(const T1* pImage,int width,int height,int nChannels,double x,double y,T2* result);

	template <class T1>
	static inline T1 BilinearInterpolate(const T1* pImage,int width,int height,double x,double y);

	// the transpose of bilinear interpolation
	template <class T1,class T2>
	static inline void BilinearInterpolate_transpose(const T1* pImage,int width,int height,int nChannels,double x,double y,T2* result);

	template <class T1>
	static inline T1 BilinearInterpolate_transpose(const T1* pImage,int width,int height,double x,double y);

	template <class T1,class T2>
	static void ResizeImage(const T1* pSrcImage,T2* pDstImage,int SrcWidth,int SrcHeight,int nChannels,double Ratio);

	template <class T1,class T2>
	static void ResizeImage(const T1* pSrcImage,T2* pDstImage,int SrcWidth,int SrcHeight,int nChannels,int DstWidth,int DstHeight);

	//---------------------------------------------------------------------------------
	// functions for 1D filtering
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void hfiltering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter1D,int fsize);

	template <class T1,class T2>
	static void vfiltering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter1D,int fsize);

	template <class T1,class T2>
	static void hfiltering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter1D,int fsize);

	template <class T1,class T2>
	static void vfiltering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter1D,int fsize);

	//---------------------------------------------------------------------------------
	// functions for 2D filtering
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void filtering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter2D,int fsize);

	template <class T1,class T2>
	static void filtering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter2D,int fsize);

	template <class T1,class T2>
	static void Laplacian(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels);

	//---------------------------------------------------------------------------------
	// functions for sample a patch from the image
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void getPatch(const T1* pSrcImgae,T2* pPatch,int width,int height,int nChannels,double x,double y,int wsize);

	//---------------------------------------------------------------------------------
	// function to warp image
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void warpImage(T1* pWarpIm2,const T1* pIm1,const T1* pIm2,const T2* pVx,const T2* pVy,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImageFlow(T1* pWarpIm2,const T1* pIm1,const T1* pIm2,const T2* pFlow,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImage(T1* pWarpIm2,const T1* pIm2,const T2* pVx,const T2* pVy,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImage_transpose(T1* pWarpIm2,const T1* pIm2,const T2* pVx,const T2* pVy,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImage(T1* pWarpIm2,const T1* pIm2,const T2*flow,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImage_transpose(T1* pWarpIm2,const T1* pIm2,const T2* flow,int width,int height,int nChannels);

	template <class T1,class T2,class T3>
	static void warpImage(T1 *pWarpIm2, T3* pMask,const T1 *pIm1, const T1 *pIm2, const T2 *pVx, const T2 *pVy, int width, int height, int nChannels);


	//---------------------------------------------------------------------------------
	// function to crop an image
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void cropImage(const T1* pSrcImage,int SrcWidth,int SrcHeight,int nChannels,T2* pDstImage,int Left,int Top,int DstWidth,int DstHeight);
	//---------------------------------------------------------------------------------

	//---------------------------------------------------------------------------------
	// function to generate a 2D Gaussian
	//---------------------------------------------------------------------------------
	template <class T>
	static void generate2DGaussian(T*& pImage,int wsize,double sigma=-1);

	template <class T>
	static void generate1DGaussian(T*& pImage,int wsize,double sigma=-1);

};

//--------------------------------------------------------------------------------------------------
// function to interplate multi-channel image plane for (x,y)
// --------------------------------------------------------------------------------------------------
template <class T1,class T2>
inline void ImageProcessing::BilinearInterpolate(const T1* pImage,int width,int height,int nChannels,double x,double y,T2* result)
{
	int xx,yy,m,n,u,v,l,offset;
	xx=x;
	yy=y;
	double dx,dy,s;
	dx=__max(__min(x-xx,1),0);
	dy=__max(__min(y-yy,1),0);

	for(m=0;m<=1;m++)
		for(n=0;n<=1;n++)
		{
			u=EnforceRange(xx+m,width);
			v=EnforceRange(yy+n,height);
			offset=(v*width+u)*nChannels;
			s=fabs(1-m-dx)*fabs(1-n-dy);
			for(l=0;l<nChannels;l++)
				result[l]+=pImage[offset+l]*s;
		}
}

template <class T1>
inline T1 ImageProcessing::BilinearInterpolate(const T1* pImage,int width,int height,double x,double y)
{
	int xx,yy,m,n,u,v,l,offset;
	xx=x;
	yy=y;
	double dx,dy,s;
	dx=__max(__min(x-xx,1),0);
	dy=__max(__min(y-yy,1),0);

	T1 result=0;
	for(m=0;m<=1;m++)
		for(n=0;n<=1;n++)
		{
			u=EnforceRange(xx+m,width);
			v=EnforceRange(yy+n,height);
			offset=v*width+u;
			s=fabs(1-m-dx)*fabs(1-n-dy);
			result+=pImage[offset]*s;
		}
	return result;
}


//--------------------------------------------------------------------------------------------------
// function to interplate multi-channel image plane for (x,y)
// --------------------------------------------------------------------------------------------------
template <class T1,class T2>
inline void ImageProcessing::BilinearInterpolate_transpose(const T1* pInput,int width,int height,int nChannels,double x,double y,T2* pDstImage)
{
	int xx,yy,m,n,u,v,l,offset;
	xx=x;
	yy=y;
	double dx,dy,s;
	dx=__max(__min(x-xx,1),0);
	dy=__max(__min(y-yy,1),0);

	for(m=0;m<=1;m++)
		for(n=0;n<=1;n++)
		{
			u=EnforceRange(xx+m,width);
			v=EnforceRange(yy+n,height);
			offset=(v*width+u)*nChannels;
			s=fabs(1-m-dx)*fabs(1-n-dy);
			for(l=0;l<nChannels;l++)
				pDstImage[offset+l] += pInput[l]*s;
		}
}

//------------------------------------------------------------------------------------------------------------
// this is the most general function for reszing an image with a varying nChannels
// bilinear interpolation is used for now. It might be replaced by other (bicubic) interpolation methods
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::ResizeImage(const T1* pSrcImage,T2* pDstImage,int SrcWidth,int SrcHeight,int nChannels,double Ratio)
{
	int DstWidth,DstHeight;
	DstWidth=(double)SrcWidth*Ratio;
	DstHeight=(double)SrcHeight*Ratio;
	memset(pDstImage,0,sizeof(T2)*DstWidth*DstHeight*nChannels);

	double x,y;

	for(int i=0;i<DstHeight;i++)
		for(int j=0;j<DstWidth;j++)
		{
			x=(double)(j+1)/Ratio-1;
			y=(double)(i+1)/Ratio-1;

			// bilinear interpolation
			BilinearInterpolate(pSrcImage,SrcWidth,SrcHeight,nChannels,x,y,pDstImage+(i*DstWidth+j)*nChannels);
		}
}

template <class T1,class T2>
void ImageProcessing::ResizeImage(const T1 *pSrcImage, T2 *pDstImage, int SrcWidth, int SrcHeight, int nChannels, int DstWidth, int DstHeight)
{
	double xRatio=(double)DstWidth/SrcWidth;
	double yRatio=(double)DstHeight/SrcHeight;
	memset(pDstImage,sizeof(T2)*DstWidth*DstHeight*nChannels,0);

	double x,y;

	for(int i=0;i<DstHeight;i++)
		for(int j=0;j<DstWidth;j++)
		{
			x=(double)(j+1)/xRatio-1;
			y=(double)(i+1)/yRatio-1;

			// bilinear interpolation
			BilinearInterpolate(pSrcImage,SrcWidth,SrcHeight,nChannels,x,y,pDstImage+(i*DstWidth+j)*nChannels);
		}
}

//------------------------------------------------------------------------------------------------------------
//  horizontal direction filtering
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::hfiltering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter1D,int fsize)
{
	memset(pDstImage,0,sizeof(T2)*width*height*nChannels);
	T2* pBuffer;
	double w;
	int i,j,l,k,offset,jj;
	for(i=0;i<height;i++)
		for(j=0;j<width;j++)
		{
			offset=i*width*nChannels;
			pBuffer=pDstImage+offset+j*nChannels;
			for(l=-fsize;l<=fsize;l++)
			{
				w=pfilter1D[l+fsize];
				jj=EnforceRange(j+l,width);
				for(k=0;k<nChannels;k++)
					pBuffer[k]+=pSrcImage[offset+jj*nChannels+k]*w;
			}
		}
}

//------------------------------------------------------------------------------------------------------------
//  horizontal direction filtering transpose
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::hfiltering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter1D,int fsize)
{
	memset(pDstImage,0,sizeof(T2)*width*height*nChannels);
	const T1* pBuffer;
	double w;
	int i,j,l,k,offset,jj;
	for(i=0;i<height;i++)
		for(j=0;j<width;j++)
		{
			int offset0=i*width*nChannels;
			pBuffer=pSrcImage+(i*width+j)*nChannels;
			for(l=-fsize;l<=fsize;l++)
			{
				w=pfilter1D[l+fsize];
				jj=EnforceRange(j+l,width);
				offset = offset0 + jj*nChannels;
				for(k=0;k<nChannels;k++)
					pDstImage[offset+k] += pBuffer[k]*w;
			}
		}
}
//------------------------------------------------------------------------------------------------------------
// fast filtering algorithm for laplacian
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::Laplacian(const T1 *pSrcImage, T2 *pDstImage, int width, int height, int nChannels)
{
	int LineWidth=width*nChannels;
	int nElements=width*height*nChannels;
	// first treat the corners
	for(int k=0;k<nChannels;k++)
	{
		pDstImage[k]=pSrcImage[k]*2-pSrcImage[nChannels+k]-pSrcImage[LineWidth+k];
		pDstImage[LineWidth-nChannels+k]=pSrcImage[LineWidth-nChannels+k]*2-pSrcImage[LineWidth-2*nChannels+k]-pSrcImage[2*LineWidth-nChannels+k];
		pDstImage[nElements-LineWidth+k]=pSrcImage[nElements-LineWidth+k]*2-pSrcImage[nElements-LineWidth+nChannels+k]-pSrcImage[nElements-2*LineWidth+k];
		pDstImage[nElements-nChannels+k]=pSrcImage[nElements-nChannels+k]*2-pSrcImage[nElements-2*nChannels+k]-pSrcImage[nElements-LineWidth-nChannels+k];
	}
	// then treat the borders
	for(int i=1;i<width-1;i++)
		for(int k=0;k<nChannels;k++)
		{
			pDstImage[i*nChannels+k]=pSrcImage[i*nChannels+k]*3-pSrcImage[(i-1)*nChannels+k]-pSrcImage[(i+1)*nChannels+k]-pSrcImage[i*nChannels+LineWidth+k];
			pDstImage[nElements-LineWidth+i*nChannels+k]=pSrcImage[nElements-LineWidth+i*nChannels+k]*3-pSrcImage[nElements-LineWidth+(i-1)*nChannels+k]-pSrcImage[nElements-LineWidth+(i+1)*nChannels+k]-pSrcImage[nElements-2*LineWidth+i*nChannels+k];
		}
	for(int i=1;i<height-1;i++)
		for(int k=0;k<nChannels;k++)
		{
			pDstImage[i*LineWidth+k]=pSrcImage[i*LineWidth+k]*3-pSrcImage[i*LineWidth+nChannels+k]-pSrcImage[(i-1)*LineWidth+k]-pSrcImage[(i+1)*LineWidth+k];
			pDstImage[(i+1)*LineWidth-nChannels+k]=pSrcImage[(i+1)*LineWidth-nChannels+k]*3-pSrcImage[(i+1)*LineWidth-2*nChannels+k]-pSrcImage[i*LineWidth-nChannels+k]-pSrcImage[(i+2)*LineWidth-nChannels+k];
		}
	// now the interior
	for(int i=1;i<height-1;i++)
		for(int j=1;j<width-1;j++)
		{
			int offset=(i*width+j)*nChannels;
			for(int k=0;k<nChannels;k++)
				pDstImage[offset+k]=pSrcImage[offset+k]*4-pSrcImage[offset+nChannels+k]-pSrcImage[offset-nChannels+k]-pSrcImage[offset-LineWidth+k]-pSrcImage[offset+LineWidth+k];
		}
}


//------------------------------------------------------------------------------------------------------------
// vertical direction filtering
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::vfiltering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter1D,int fsize)
{
	memset(pDstImage,0,sizeof(T2)*width*height*nChannels);
	T2* pBuffer;
	double w;
	int i,j,l,k,offset,ii;
	for(i=0;i<height;i++)
		for(j=0;j<width;j++)
		{
			pBuffer=pDstImage+(i*width+j)*nChannels;
			for(l=-fsize;l<=fsize;l++)
			{
				w=pfilter1D[l+fsize];
				ii=EnforceRange(i+l,height);
				for(k=0;k<nChannels;k++)
					pBuffer[k]+=pSrcImage[(ii*width+j)*nChannels+k]*w;
			}
		}
}

//------------------------------------------------------------------------------------------------------------
// vertical direction filtering transpose
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::vfiltering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter1D,int fsize)
{
	memset(pDstImage,0,sizeof(T2)*width*height*nChannels);
	const T1* pBuffer;
	double w;
	int i,j,l,k,offset,ii;
	for(i=0;i<height;i++)
		for(j=0;j<width;j++)
		{
			pBuffer=pSrcImage+(i*width+j)*nChannels;
			for(l=-fsize;l<=fsize;l++)
			{
				w=pfilter1D[l+fsize];
				ii=EnforceRange(i+l,height);
				offset = (ii*width+j)*nChannels;
				for(k=0;k<nChannels;k++)
					//pBuffer[k]+=pSrcImage[(ii*width+j)*nChannels+k]*w;
					pDstImage[offset+k] += pBuffer[k]*w;
			}
		}
}


//------------------------------------------------------------------------------------------------------------
// 2d filtering
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::filtering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter2D,int fsize)
{
	double w;
	int i,j,u,v,k,ii,jj,wsize,offset;
	wsize=fsize*2+1;
	double* pBuffer=new double[nChannels];
	for(i=0;i<height;i++)
		for(j=0;j<width;j++)
		{
			for(k=0;k<nChannels;k++)
				pBuffer[k]=0;
			for(u=-fsize;u<=fsize;u++)
				for(v=-fsize;v<=fsize;v++)
				{
					w=pfilter2D[(u+fsize)*wsize+v+fsize];
					ii=EnforceRange(i+u,height);
					jj=EnforceRange(j+v,width);
					offset=(ii*width+jj)*nChannels;
					for(k=0;k<nChannels;k++)
						pBuffer[k]+=pSrcImage[offset+k]*w;
				}
			offset=(i*width+j)*nChannels;
			for(k=0;k<nChannels;k++)
				pDstImage[offset+k]=pBuffer[k];
		}
	delete pBuffer;
}

//------------------------------------------------------------------------------------------------------------
// 2d filtering transpose
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::filtering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const double* pfilter2D,int fsize)
{
	double w;
	int i,j,u,v,k,ii,jj,wsize,offset;
	wsize=fsize*2+1;
	memset(pDstImage,0,sizeof(T2)*width*height*nChannels);
	for(i=0;i<height;i++)
		for(j=0;j<width;j++)
		{
			int offset0 = (i*width+j)*nChannels;
			for(u=-fsize;u<=fsize;u++)
				for(v=-fsize;v<=fsize;v++)
				{
					w=pfilter2D[(u+fsize)*wsize+v+fsize];
					ii=EnforceRange(i+u,height);
					jj=EnforceRange(j+v,width);
					int offset=(ii*width+jj)*nChannels;
					for(k=0;k<nChannels;k++)
						pDstImage[offset+k]+=pSrcImage[offset0+k]*w;
				}
		}
}


//------------------------------------------------------------------------------------------------------------
// function to sample a patch from the source image
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::getPatch(const T1* pSrcImage,T2* pPatch,int width,int height,int nChannels,double x0,double y0,int wsize)
{
	// suppose pPatch has been allocated and cleared before calling the function
	int wlength=wsize*2+1;
	double x,y;
	for(int i=-wsize;i<=wsize;i++)
		for(int j=-wsize;j<=wsize;j++)
		{
			y=y0+i;
			x=x0+j;
			if(x<0 || x>width-1 || y<0 || y>height-1)
				continue;
			BilinearInterpolate(pSrcImage,width,height,nChannels,x,y,pPatch+((i+wsize)*wlength+j+wsize)*nChannels);
		}
}

//------------------------------------------------------------------------------------------------------------
// function to warp an image with respect to flow field
// pWarpIm2 has to be allocated before hands
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::warpImage(T1 *pWarpIm2, const T1 *pIm1, const T1 *pIm2, const T2 *pVx, const T2 *pVy, int width, int height, int nChannels)
{
	memset(pWarpIm2,0,sizeof(T1)*width*height*nChannels);
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double x,y;
			y=i+pVy[offset];
			x=j+pVx[offset];
			offset*=nChannels;
			if(x<0 || x>width-1 || y<0 || y>height-1)
			{
				for(int k=0;k<nChannels;k++)
					pWarpIm2[offset+k]=pIm1[offset+k];
				continue;
			}
			BilinearInterpolate(pIm2,width,height,nChannels,x,y,pWarpIm2+offset);
		}
}

template <class T1,class T2>
void ImageProcessing::warpImageFlow(T1 *pWarpIm2, const T1 *pIm1, const T1 *pIm2, const T2 *pFlow, int width, int height, int nChannels)
{
	memset(pWarpIm2,0,sizeof(T1)*width*height*nChannels);
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double x,y;
			y=i+pFlow[offset*2+1];
			x=j+pFlow[offset*2];
			offset*=nChannels;
			if(x<0 || x>width-1 || y<0 || y>height-1)
			{
				for(int k=0;k<nChannels;k++)
					pWarpIm2[offset+k]=pIm1[offset+k];
				continue;
			}
			BilinearInterpolate(pIm2,width,height,nChannels,x,y,pWarpIm2+offset);
		}
}

template <class T1,class T2>
void ImageProcessing::warpImage(T1 *pWarpIm2,const T1 *pIm2, const T2 *pVx, const T2 *pVy, int width, int height, int nChannels)
{
	memset(pWarpIm2,0,sizeof(T1)*width*height*nChannels);
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double x,y;
			y=i+pVy[offset];
			x=j+pVx[offset];
			offset*=nChannels;
			if(x<0 || x>width-1 || y<0 || y>height-1)
				continue;
			BilinearInterpolate(pIm2,width,height,nChannels,x,y,pWarpIm2+offset);
		}
}

template <class T1,class T2>
void ImageProcessing::warpImage_transpose(T1 *pWarpIm2,const T1 *pIm2, const T2 *pVx, const T2 *pVy, int width, int height, int nChannels)
{
	memset(pWarpIm2,0,sizeof(T1)*width*height*nChannels);
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double x,y;
			y=i+pVy[offset];
			x=j+pVx[offset];
			offset*=nChannels;
			if(x<0 || x>width-1 || y<0 || y>height-1)
				continue;
			//BilinearInterpolate(pIm2,width,height,nChannels,x,y,pWarpIm2+offset);
			BilinearInterpolate_transpose(pIm2+offset,width,height,nChannels,x,y,pWarpIm2);
		}
}

//////////////////////////////////////////////////////////////////////////////////////
// different format
//////////////////////////////////////////////////////////////////////////////////////
template <class T1,class T2>
void ImageProcessing::warpImage(T1 *pWarpIm2,const T1 *pIm2, const T2 *flow, int width, int height, int nChannels)
{
	memset(pWarpIm2,0,sizeof(T1)*width*height*nChannels);
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double x,y;
			y=i+flow[offset*2+1];
			x=j+flow[offset*2];
			offset*=nChannels;
			if(x<0 || x>width-1 || y<0 || y>height-1)
				continue;
			BilinearInterpolate(pIm2,width,height,nChannels,x,y,pWarpIm2+offset);
		}
}

template <class T1,class T2>
void ImageProcessing::warpImage_transpose(T1 *pWarpIm2,const T1 *pIm2, const T2 *flow, int width, int height, int nChannels)
{
	memset(pWarpIm2,0,sizeof(T1)*width*height*nChannels);
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double x,y;
			y=i+flow[offset*2+1];
			x=j+flow[offset*2];
			offset*=nChannels;
			if(x<0 || x>width-1 || y<0 || y>height-1)
				continue;
			//BilinearInterpolate(pIm2,width,height,nChannels,x,y,pWarpIm2+offset);
			BilinearInterpolate_transpose(pIm2+offset,width,height,nChannels,x,y,pWarpIm2);
		}
}


template <class T1,class T2,class T3>
void ImageProcessing::warpImage(T1 *pWarpIm2, T3* pMask,const T1 *pIm1, const T1 *pIm2, const T2 *pVx, const T2 *pVy, int width, int height, int nChannels)
{
	memset(pWarpIm2,0,sizeof(T1)*width*height*nChannels);
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double x,y;
			y=i+pVy[offset];
			x=j+pVx[offset];
			offset*=nChannels;
			if(x<0 || x>width-1 || y<0 || y>height-1)
			{
				for(int k=0;k<nChannels;k++)
					pWarpIm2[offset+k]=pIm1[offset+k];
				pMask[i*width+j]=0;
				continue;
			}
			pMask[i*width+j]=1;
			BilinearInterpolate(pIm2,width,height,nChannels,x,y,pWarpIm2+offset);
		}
}

//------------------------------------------------------------------------------------------------------------
// function to crop an image from the source
// assume that pDstImage has been allocated
// also Left and Top must be valid, DstWidth and DstHeight should ensure that the image lies
// inside the image boundary
//------------------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageProcessing::cropImage(const T1 *pSrcImage, int SrcWidth, int SrcHeight, int nChannels, T2 *pDstImage, int Left, int Top, int DstWidth, int DstHeight)
{
	if(typeid(T1)==typeid(T2))
	{
		for(int i=0;i<DstHeight;i++)
			memcpy(pDstImage+i*DstWidth*nChannels,pSrcImage+((i+Top)*SrcWidth+Left)*nChannels,sizeof(T1)*DstWidth*nChannels);
		return;
	}
	int offsetSrc,offsetDst;
	for(int i=0;i<DstHeight;i++)
		for(int j=0;j<DstWidth;j++)
		{
			offsetSrc=((i+Top)*SrcWidth+Left+j)*nChannels;
			offsetDst=(i*DstWidth+j)*nChannels;
			for(int k=0;k<nChannels;k++)
				pDstImage[offsetDst+k]=pSrcImage[offsetSrc+k];
		}
}

//------------------------------------------------------------------------------------------------------------
// function to generate a 2D Gaussian image
// pImage must be allocated before calling the function
//------------------------------------------------------------------------------------------------------------
template <class T>
void ImageProcessing::generate2DGaussian(T*& pImage, int wsize, double sigma)
{
	if(sigma==-1)
		sigma=wsize/2;
	double alpha=1/(2*sigma*sigma);
	int winlength=wsize*2+1;
	if(pImage==NULL)
		pImage=new T[winlength*winlength];
	double total = 0;
	for(int i=-wsize;i<=wsize;i++)
		for(int j=-wsize;j<=wsize;j++)
		{
			pImage[(i+wsize)*winlength+j+wsize]=exp(-(double)(i*i+j*j)*alpha);
			total += pImage[(i+wsize)*winlength+j+wsize];
		}
	for(int i = 0;i<winlength*winlength;i++)
		pImage[i]/=total;
}

//------------------------------------------------------------------------------------------------------------
// function to generate a 1D Gaussian image
// pImage must be allocated before calling the function
//------------------------------------------------------------------------------------------------------------
template <class T>
void ImageProcessing::generate1DGaussian(T*& pImage, int wsize, double sigma)
{
	if(sigma==-1)
		sigma=wsize/2;
	double alpha=1/(2*sigma*sigma);
	int winlength=wsize*2+1;
	if(pImage==NULL)
		pImage=new T[winlength];
	double total = 0;
	for(int i=-wsize;i<=wsize;i++)
	{
		pImage[i+wsize]=exp(-(double)(i*i)*alpha);
		total += pImage[i+wsize];
	}
	for(int i = 0;i<winlength;i++)
		pImage[i]/=total;
}

#endif
