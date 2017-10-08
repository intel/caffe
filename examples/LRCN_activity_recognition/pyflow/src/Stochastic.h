// Author: Ce Liu (c) Dec, 2009; celiu@mit.edu
// Modified By: Deepak Pathak (c) 2016; pathak@berkeley.edu

#ifndef STOCHASTIC_H
#define STOCHASTIC_H

#include "math.h"
#include "stdlib.h"
#include "project.h"
#include "memory.h"

#define _Release_2DArray(X,i,length) for(i=0;i<length;i++) if(X[i]!=NULL) delete X[i]; delete []X

#ifndef _abs
#define _abs(x) (x>=0)?x:-x
#endif

#ifndef PI
#define PI 3.1415927
#endif

enum SortType{SortAscending,SortDescending};

class CStochastic
{
public:
	CStochastic(void);
	~CStochastic(void);
	static void ConvertInt2String(int x,char* string,int BitNumber=3);
	static double UniformSampling();
	static int UniformSampling(int R);
	static double GaussianSampling();
	template <class T> static void GetMeanVar(T* signal,int length,double* mean,double* var);
	static int Sampling(double* Density,int NumSamples);
	static double GetMean(double *signal,int length);
	static void Generate1DGaussian(double* pGaussian,int size,double sigma=0);
	static void Generate2DGaussian(double* pGaussian,int size,double sigma=0);
	static double entropy(double* pDensity,int n);

	template <class T> static T sum(int NumData,T* pData);
	template <class T> static void Normalize(int NumData,T* pData);
	template <class T> static T mean(int NumData, T* pData);
	template <class T> static void sort(int number, T* pData,int *pIndex,SortType m_SortType=SortDescending);
	template <class T> static T Min(int NumData, T* pData);
	template <class T> static T Min(int NumData, T* pData1,T* pData2);
	template <class T> static T Max(int NumData ,T* pData);
	template <class T> static int FindMax(int NumData,T* pData);
	template <class T1,class T2> static void ComputeVectorMean(int Dim,int NumData,T1* pData,T2* pMean,double* pWeight=NULL);
	template <class T1,class T2> static void ComputeMeanCovariance(int Dim,int NumData,T1* pData,T2* pMean,T2* pCovarance,double* pWeight=NULL);
	template <class T1,class T2> static double VectorSquareDistance(int Dim,T1* pVector1,T2* pVector2);
	template <class T1> static void KMeanClustering(int Dim,int NumData,int NumClusters,T1* pData,int *pPartition,double** pClusterMean=NULL,int MaxIterationNum=10,int MinClusterSampleNumber=2);
	template <class T> static double norm(T* X,int Dim);
	template <class T1,class T2> static int FindClosestPoint(T1* pPointSet,int NumPoints,int nDim,T2* QueryPoint);
	template <class T1,class T2> static void GaussianFiltering(T1* pSrcArray,T2* pDstArray,int NumPoints,int nChannels,int size,double sigma);
};

template <class T>
void CStochastic::GetMeanVar(T* signal,int length,double* mean,double* var)
{
	double m_mean=0,m_var=0;

	int i;
	for (i=0;i<length;i++)
		m_mean+=signal[i];
	m_mean/=length;
	for (i=0;i<length;i++)
		m_var+=(signal[i]-m_mean)*(signal[i]-m_mean);
	m_var/=length-1;
	*mean=m_mean;
	*var=m_var;
}

template <class T>
T CStochastic::sum(int NumData, T* pData)
{
	T sum=0;
	int i;
	for(i=0;i<NumData;i++)
		sum+=pData[i];
	return sum;
}

template <class T>
void CStochastic::Normalize(int NumData,T* pData)
{
	int i;
	T Sum;
	Sum=sum(NumData,pData);
	for(i=0;i<NumData;i++)
		pData[i]/=Sum;
}

template <class T>
T CStochastic::mean(int NumData,T* pData)
{
	return sum(NumData,pData)/NumData;
}

////////////////////////////////////////////////////////////
// sort data in descending order
template <class T>
void CStochastic::sort(int Number,T* pData,int *pIndex,SortType m_SortType)
{
	int i,j,offset_extreme,*flag;
	double extreme;
	flag=new int[Number];
	memset(flag,0,sizeof(int)*Number);
	for(i=0;i<Number;i++)
	{
		if(m_SortType==SortDescending)
			extreme=-1E100;
		else
			extreme=1E100;
		offset_extreme=0;
		for(j=0;j<Number;j++)
		{
			if(flag[j]==1)
				continue;
			if( (m_SortType==SortDescending && extreme<pData[j]) || (m_SortType==SortAscending && extreme>pData[j]))
			{
				extreme=pData[j];
				offset_extreme=j;
			}
		}
		pIndex[i]=offset_extreme;
		flag[offset_extreme]=1;
	}
	delete flag;
}

template <class T>
T CStochastic::Min(int NumData,T* pData)
{
	int i;
	T result=pData[0];
	for(i=1;i<NumData;i++)
		result=__min(result,pData[i]);
	return result;
}

template <class T>
T CStochastic::Min(int NumData,T* pData1,T* pData2)
{
	int i;
	T result=pData1[0]+pData2[0];
	for(i=1;i<NumData;i++)
		result=__min(result,pData1[i]+pData2[i]);
	return result;
}

template <class T>
T CStochastic::Max(int NumData,T* pData)
{
	int i;
	T result=pData[0];
	for(i=1;i<NumData;i++)
		result=__max(result,pData[i]);
	return result;
}

template <class T>
int CStochastic::FindMax(int NumData,T* pData)
{
	int i,index;
	T result=pData[0];
	index=0;
	for(i=1;i<NumData;i++)
		if(pData[i]>result)
		{
			index=i;
			result=pData[i];
		}
	return index;
}


template <class T1,class T2>
void CStochastic::ComputeMeanCovariance(int Dim,int NumData,T1* pData,T2* pMean,T2* pCovariance,double* pWeight)
{
	int i,j,k;
	memset(pMean,0,sizeof(T2)*Dim);
	memset(pCovariance,0,sizeof(T2)*Dim*Dim);

	bool IsWeightLoaded=false;
	double Sum;
	if(pWeight!=NULL)
		IsWeightLoaded=true;

	// compute mean first
	Sum=0;
	if(IsWeightLoaded)
		for(i=0;i<NumData;i++)
		{
			if(pWeight[i]==0)
				continue;
			for(j=0;j<Dim;j++)
				pMean[j]+=pData[i*Dim+j]*pWeight[i];
			Sum+=pWeight[i];
		}
	else
	{
		for(i=0;i<NumData;i++)
			for(j=0;j<Dim;j++)
				pMean[j]+=pData[i*Dim+j];
		Sum=NumData;
	}
	for(j=0;j<Dim;j++)
		pMean[j]/=Sum;

	//compute covariance;
	T2* pTempVector;
	pTempVector=new T2[Dim];

	for(i=0;i<NumData;i++)
	{
		for(j=0;j<Dim;j++)
			pTempVector[j]=pData[i*Dim+j]-pMean[j];
		if(IsWeightLoaded)
		{
			if(pWeight[i]==0)
				continue;
			for(j=0;j<Dim;j++)
				for(k=0;k<=j;k++)
					pCovariance[j*Dim+k]+=pTempVector[j]*pTempVector[k]*pWeight[i];
		}
		else
			for(j=0;j<Dim;j++)
				for(k=0;k<=j;k++)
					pCovariance[j*Dim+k]+=pTempVector[j]*pTempVector[k];
	}
	for(j=0;j<Dim;j++)
		for(k=j+1;k<Dim;k++)
			pCovariance[j*Dim+k]=pCovariance[k*Dim+j];

	for(j=0;j<Dim*Dim;j++)
		pCovariance[j]/=Sum;

	delete []pTempVector;
}

template <class T1,class T2>
void CStochastic::ComputeVectorMean(int Dim,int NumData,T1* pData,T2* pMean,double* pWeight)
{
	int i,j;
	memset(pMean,0,sizeof(T2)*Dim);
	bool IsWeightLoaded;
	double Sum;
	if(pWeight=NULL)
		IsWeightLoaded=false;
	else
		IsWeightLoaded=true;

	Sum=0;
	if(IsWeightLoaded)
		for(i=0;i<NumData;i++)
		{
			if(pWeight[i]==0)
				continue;
			for(j=0;j<Dim;j++)
				pMean[j]+=pData[i*Dim+j]*pWeight[i];
			Sum+=pWeight[i];
		}
	else
	{
		for(i=0;i<NumData;i++)
			for(j=0;j<Dim;j++)
				pMean[j]+=pData[i*Dim+j];
		Sum=NumData;
	}
	for(j=0;j<Dim;j++)
		pMean[j]/=Sum;
}

template <class T1,class T2>
double CStochastic::VectorSquareDistance(int Dim,T1* pVector1,T2* pVector2)
{
	double result=0,temp;
	int i;
	for(i=0;i<Dim;i++)
	{
		temp=pVector1[i]-pVector2[i];
		result+=temp*temp;
	}
	return result;
}

template <class T1>
void CStochastic::KMeanClustering(int Dim,int NumData,int NumClusters,T1* pData,int *pPartition,double** pClusterMean,int MaxIterationNum, int MinClusterSampleNumber)
{
	int i,j,k,l,Index,ClusterSampleNumber;
	double MinDistance,Distance;
	double** pCenters;
	pCenters=new double*[NumClusters];
	for(i=0;i<NumClusters;i++)
		pCenters[i]=new double[Dim];

	// generate randome guess of the partition
_CStochastic_KMeanClustering_InitializePartition:
	for(i=0;i<NumClusters;i++)
	{
		Index=UniformSampling(NumData);
		for(j=0;j<Dim;j++)
			pCenters[i][j]=pData[Index*Dim+j];
	}

	for(k=0;k<MaxIterationNum;k++)
	{
		// step 1. do partition
		for(i=0;i<NumData;i++)
		{
			MinDistance=1E100;
			for(j=0;j<NumClusters;j++)
			{
				Distance=VectorSquareDistance(Dim,pData+i*Dim,pCenters[j]);
				if(Distance<MinDistance)
				{
					MinDistance=Distance;
					Index=j;
				}
			}
			pPartition[i]=Index;
		}
		// step 2. compute mean
		for(i=0;i<NumClusters;i++)
		{
			memset(pCenters[i],0,sizeof(double)*Dim);
			ClusterSampleNumber=0;
			for(j=0;j<NumData;j++)
				if(pPartition[j]==i)
				{
					for(l=0;l<Dim;l++)
						pCenters[i][l]+=pData[j*Dim+l];
					ClusterSampleNumber++;
				}
			// maybe the initial partition is bad
			// if so just do initial partition again
			if(ClusterSampleNumber<MinClusterSampleNumber)
				goto _CStochastic_KMeanClustering_InitializePartition;
			for(l=0;l<Dim;l++)
				pCenters[i][l]/=ClusterSampleNumber;
		}
	}
	// output the final partition if necessary
	if(pClusterMean!=NULL)
		for(i=0;i<NumClusters;i++)
			for(l=0;l<Dim;l++)
				pClusterMean[i][l]=pCenters[i][l];
	// free buffer
	for(i=0;i<NumClusters;i++)
		delete pCenters[i];
	delete []pCenters;
}

template <class T>
double CStochastic::norm(T* X,int Dim)
{
	double result=0;
	int i;
	for(i=0;i<Dim;i++)
		result+=X[i]*X[i];
	result=sqrt(result);
	return result;
}

template <class T1,class T2>
int CStochastic::FindClosestPoint(T1* pPointSet,int NumPoints,int nDim,T2* QueryPoint)
{
	int i,j,Index=0,offset;
	T1 MinDistance,Distance,x;
	MinDistance=0;
	for(j=0;j<nDim;j++)
		MinDistance+=_abs(pPointSet[j]-QueryPoint[j]);
	for(i=1;i<NumPoints;i++)
	{
		Distance=0;
		offset=i*nDim;
		for(j=0;j<nDim;j++)
		{
			x=pPointSet[offset+j]-QueryPoint[j];
			Distance+=_abs(x);
		}
		if(Distance<MinDistance)
		{
			MinDistance=Distance;
			Index=i;
		}
	}
	return Index;
}

template <class T1,class T2>
void CStochastic::GaussianFiltering(T1* pSrcArray,T2* pDstArray,int NumPoints,int nChannels,int size,double sigma)
{
	int i,j,u,l;
	double *pGaussian,temp;
	pGaussian=new double[2*size+1];
	Generate1DGaussian(pGaussian,size,sigma);
	for(i=0;i<NumPoints;i++)
		for(l=0;l<nChannels;l++)
		{
			temp=0;
			for(j=-size;j<=size;j++)
			{
				u=i+j;
				u=__max(__min(u,NumPoints-1),0);
				temp+=pSrcArray[u*nChannels+l]*pGaussian[j+size];
			}
			pDstArray[i*nChannels+l]=temp;
		}
	delete pGaussian;
}

#endif
