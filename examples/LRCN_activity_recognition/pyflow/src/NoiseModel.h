// Author: Ce Liu (c) Dec, 2009; celiu@mit.edu
// Modified By: Deepak Pathak (c) 2016; pathak@berkeley.edu

#pragma once

#include "stdio.h"
#include "Vector.h"
#include <iostream>

#ifndef PI
#define PI 3.1415926535897932384626433832
#endif

using namespace std;

class GaussianMixture
{
public:
	int nChannels;
	double* alpha;
	double* sigma;
	double* beta;
	double* sigma_square;
	double* beta_square;
public:
	GaussianMixture()
	{
		nChannels = 0;
		alpha = sigma = beta = sigma_square = beta_square = NULL;
	}
	GaussianMixture(int _nChannels)
	{
		nChannels = _nChannels;
		allocate();
		for(int i = 0;i<nChannels;i++)
		{
			alpha[i] = 0.95;
			sigma[i] = 0.05;
			beta[i] = 0.5;
		}
		square();
	}
	GaussianMixture(const GaussianMixture& GM)
	{
		clear();
		copy(GM);
	}
	void copy(const GaussianMixture& GM)
	{
		nChannels = GM.nChannels;
		allocate();
		for(int i  = 0;i<nChannels;i++)
		{
			alpha[i]  = GM.alpha[i];
			sigma[i] = GM.sigma[i];
			beta[i]    = GM.beta[i];
		}
		square();
	}
	void operator=(const GaussianMixture& GM)
	{
		clear();
		copy(GM);
	}
	GaussianMixture shrink(int N)
	{
		GaussianMixture GM(N);
		for(int i = 0;i<N;i++)
		{
			GM.alpha[i]  = alpha[i];
			GM.sigma[i] = sigma[i];
			GM.beta[i]    = beta[i];
		}
		GM.square();
		return GM;
	}
	void allocate()
	{
		alpha = new double[nChannels];
		sigma = new double[nChannels];
		beta = new double[nChannels];
		sigma_square = new double[nChannels];
		beta_square = new double[nChannels];
	}
	void clear()
	{
		if(!alpha)
			delete []alpha;
		if(!sigma)
			delete []sigma;
		if(!beta)
			delete []beta;
		if(!sigma_square)
			delete []sigma_square;
		if(!beta_square)
			delete []beta_square;
		alpha = sigma = beta = sigma_square = beta_square = NULL;
	}
	void reset()
	{
		//for(int i = 0;i<nChannels;i++)
		//	alpha[i] = sigma[i] = beta[i] = sigma_square[i] = beta_square[i] = 0;
		for(int i = 0;i<nChannels;i++)
		{
			alpha[i] = 0.95;
			sigma[i] = 0.05;
			beta[i] = 0.5;
		}
		square();
	}
	void reset(int _nChannels)
	{
		clear();
		nChannels = _nChannels;
		allocate();
		reset();
	}
	double Gaussian(double x,int i,int k) const
	{
		if(i==0)
			 return exp(-x/(2*sigma_square[k]))/(2*PI*sigma[k]);
		else
			return exp(-x/(2*beta_square[k]))/(2*PI*beta[k]);
	}
	~GaussianMixture()
	{
		clear();
	}
	void square()
	{
		for(int i =0;i<nChannels;i++)
		{
			sigma_square[i] = sigma[i]*sigma[i];
			beta_square[i] = beta[i]*beta[i];
		}
	}
	void display()
	{
		for(int i = 0;i<nChannels;i++)
			cout<<"alpha: "<<alpha[i] << " sigma: "<<sigma[i]<<" beta: "<<beta[i]<<" sigma^2: "<<sigma_square[i]<<" beta^2: "<<beta_square[i]<<endl;
	}
	bool write(const char* filename)
	{
		ofstream myfile(filename,ios::out | ios::binary);
		if(myfile.is_open())
		{
			bool foo = write(myfile);
			myfile.close();
			return foo;
		}
		return false;
	}
	bool write(ofstream& myfile)
	{
		myfile.write((char *)&nChannels,sizeof(int));
		myfile.write((char *)alpha,sizeof(double)*nChannels);
		myfile.write((char *)sigma,sizeof(double)*nChannels);
		myfile.write((char *)beta,sizeof(double)*nChannels);
		return true;
	}
	bool read(const char* filename)
	{
		ifstream myfile(filename, ios::in | ios::binary);
		if(myfile.is_open())
		{
			bool foo = read(myfile);
			myfile.close();
			square();
			return foo;
		}
		return false;
	}
	bool read(ifstream& myfile)
	{
		myfile.read((char *)&nChannels,sizeof(int));
		allocate();
		myfile.read((char *)alpha,sizeof(double)*nChannels);
		myfile.read((char *)sigma,sizeof(double)*nChannels);
		myfile.read((char *)beta,sizeof(double)*nChannels);
		square();
		return true;
	}
};

//class Laplacian
//{
//public:
//	int nChannels;
//	Vector<double> scale;
//public:
//	Laplacian()
//	{
//	}
//	Laplacian(int _nChannels)
//	{
//		nChannels = _nChannels;
//		scale.allocate(nChannels);
//	}
//	Laplacian(const Laplacian
//
//};
