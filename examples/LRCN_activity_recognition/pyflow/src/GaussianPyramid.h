// Author: Ce Liu (c) Dec, 2009; celiu@mit.edu
// Modified By: Deepak Pathak (c) 2016; pathak@berkeley.edu

#ifndef _GaussianPyramid_h
#define _GaussianPyramid_h

#include "Image.h"

class GaussianPyramid
{
private:
	DImage* ImPyramid;
	int nLevels;
public:
	GaussianPyramid(void);
	~GaussianPyramid(void);
	void ConstructPyramid(const DImage& image,double ratio=0.8,int minWidth=30);
	void ConstructPyramidLevels(const DImage& image,double ratio =0.8,int _nLevels = 2);
	void displayTop(const char* filename);
	inline int nlevels() const {return nLevels;};
	inline DImage& Image(int index) {return ImPyramid[index];};
};

#endif
