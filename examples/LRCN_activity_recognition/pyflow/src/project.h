// Author: Ce Liu (c) Dec, 2009; celiu@mit.edu
// Modified By: Deepak Pathak (c) 2016; pathak@berkeley.edu

#pragma once
#include "stdio.h"
#include <vector>

// if the files are compiled in linux or mac os then uncomment the following line, otherwise comment it if you compile using visual studio in windows
#define _LINUX_MAC
// #define _OPENCV

template <class T>
void _Release1DBuffer(T* pBuffer)
{
	if(pBuffer!=NULL)
		delete []pBuffer;
	pBuffer=NULL;
}

template <class T>
void _Rlease2DBuffer(T** pBuffer,size_t nElements)
{
	for(size_t i=0;i<nElements;i++)
		delete [](pBuffer[i]);
	delete []pBuffer;
	pBuffer=NULL;
}

// disable matlab support
// #define _MATLAB

#ifdef _MATLAB
#include "mex.h"

#endif


#ifdef _LINUX_MAC

template <class T1,class T2>
T1 __min(T1 a, T2 b)
{
  return (a>b)?b:a;
}

template <class T1,class T2>
T1 __max(T1 a, T2 b)
{
  return (a<b)?b:a;
}

#endif
