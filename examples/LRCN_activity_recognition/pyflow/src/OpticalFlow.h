// Author: Ce Liu (c) Dec, 2009; celiu@mit.edu
// Modified By: Deepak Pathak (c) 2016; pathak@berkeley.edu

#pragma once

#include "Image.h"
#include "NoiseModel.h"
#include "Vector.h"
#include <vector>

typedef double _FlowPrecision;

class OpticalFlow
{
public:
	static bool IsDisplay;
public:
	enum InterpolationMethod {Bilinear,Bicubic};
	static InterpolationMethod interpolation;
	enum NoiseModel {GMixture,Lap};
	OpticalFlow(void);
	~OpticalFlow(void);
	static GaussianMixture GMPara;
	static Vector<double> LapPara;
	static NoiseModel noiseModel;
public:
	static void getDxs(DImage& imdx,DImage& imdy,DImage& imdt,const DImage& im1,const DImage& im2);
	static void SanityCheck(const DImage& imdx,const DImage& imdy,const DImage& imdt,double du,double dv);
	static void warpFL(DImage& warpIm2,const DImage& Im1,const DImage& Im2,const DImage& vx,const DImage& vy);
	static void warpFL(DImage& warpIm2,const DImage& Im1,const DImage& Im2,const DImage& flow);


	static void genConstFlow(DImage& flow,double value,int width,int height);
	static void genInImageMask(DImage& mask,const DImage& vx,const DImage& vy,int interval = 0);
	static void genInImageMask(DImage& mask,const DImage& flow,int interval =0 );
	static void SmoothFlowPDE(const DImage& Im1,const DImage& Im2, DImage& warpIm2,DImage& vx,DImage& vy,
														 double alpha,int nOuterFPIterations,int nInnerFPIterations,int nCGIterations);

	static void SmoothFlowSOR(const DImage& Im1,const DImage& Im2, DImage& warpIm2, DImage& vx, DImage& vy,
														 double alpha,int nOuterFPIterations,int nInnerFPIterations,int nSORIterations);

	static void estGaussianMixture(const DImage& Im1,const DImage& Im2,GaussianMixture& para,double prior = 0.9);
	static void estLaplacianNoise(const DImage& Im1,const DImage& Im2,Vector<double>& para);
	static void Laplacian(DImage& output,const DImage& input,const DImage& weight);
	static void testLaplacian(int dim=3);

	// function of coarse to fine optical flow
	static void Coarse2FineFlow(DImage& vx,DImage& vy,DImage &warpI2,const DImage& Im1,const DImage& Im2,double alpha,double ratio,int minWidth,
															int nOuterFPIterations,int nInnerFPIterations,int nCGIterations);

	static void Coarse2FineFlowLevel(DImage& vx,DImage& vy,DImage &warpI2,const DImage& Im1,const DImage& Im2,double alpha,double ratio,int nLevels,
															int nOuterFPIterations,int nInnerFPIterations,int nCGIterations);

	// function to convert image to features
	static void im2feature(DImage& imfeature,const DImage& im);

	// function to load optical flow
	static bool LoadOpticalFlow(const char* filename,DImage& flow);

	static bool LoadOpticalFlow(ifstream& myfile,DImage& flow);

	static bool SaveOpticalFlow(const DImage& flow, const char* filename);

	static bool SaveOpticalFlow(const DImage& flow,ofstream& myfile);

	static bool showFlow(const DImage& vx,const char* filename);

	// function to assemble and dissemble flows
	static void AssembleFlow(const DImage& vx,const DImage& vy,DImage& flow)
	{
		if(!flow.matchDimension(vx.width(),vx.height(),2))
			flow.allocate(vx.width(),vx.height(),2);
		for(int i = 0;i<vx.npixels();i++)
		{
			flow.data()[i*2] = vx.data()[i];
			flow.data()[i*2+1] = vy.data()[i];
		}
	}
	static void DissembleFlow(const DImage& flow,DImage& vx,DImage& vy)
	{
		if(!vx.matchDimension(flow.width(),flow.height(),1))
			vx.allocate(flow.width(),flow.height());
		if(!vy.matchDimension(flow.width(),flow.height(),1))
			vy.allocate(flow.width(),flow.height());
		for(int i =0;i<vx.npixels();i++)
		{
			vx.data()[i] = flow.data()[i*2];
			vy.data()[i] = flow.data()[i*2+1];
		}
	}
	static void ComputeOpticalFlow(const DImage& Im1,const DImage& Im2,DImage& flow)
	{
		if(!Im1.matchDimension(Im2))
		{
			cout<<"The input images for optical flow have different dimensions!"<<endl;
			return;
		}
		if(!flow.matchDimension(Im1.width(),Im1.height(),2))
			flow.allocate(Im1.width(),Im1.height(),2);

		double alpha=0.01;
		double ratio=0.75;
		int minWidth=30;
		int nOuterFPIterations=15;
		int nInnerFPIterations=1;
		int nCGIterations=40;

		DImage vx,vy,warpI2;
		OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,Im1,Im2,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nCGIterations);
		AssembleFlow(vx,vy,flow);
	}
};
