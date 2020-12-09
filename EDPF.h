/**************************************************************************************************************
* Edge Drawing (ED) and Edge Drawing Parameter Free (EDPF) source codes.
* Copyright (C) Cuneyt Akinlar & Cihan Topal
* E-mails of the authors: cuneytakinlar@gmail.com, cihantopal@gmail.com
*
* Please cite the following papers if you use EDPF library: 
*
* [1] C. Topal and C. Akinlar, “Edge Drawing: A Combined Real-Time Edge and Segment Detector,”
*     Journal of Visual Communication and Image Representation, 23(6), 862-872, DOI: 10.1016/j.jvcir.2012.05.004 (2012).
*
* [2] C. Akinlar and C. Topal, “EDPF: A Real-time Parameter-free Edge Segment Detector with a False Detection Control,”
*     International Journal of Pattern Recognition and Artificial Intelligence, 26(1), DOI: 10.1142/S0218001412550026 (2012).
**************************************************************************************************************/

#ifndef  _EDPF_
#define _EDPF_

#include "ED.h"

#define MAX_GRAD_VALUE 128*256
#define EPSILON 1.0

class EDPF : public ED {
public:
	EDPF(cv::Mat srcImage);
	EDPF(ED obj);
	EDPF(EDColor obj);
private:
	double divForTestSegment;
	double *H;
	int np;
	short *gradImg;

	void validateEdgeSegments();
	short *ComputePrewitt3x3(); // differs from base class's prewit function (calculates H)
	void TestSegment(int i, int index1, int index2);
	void ExtractNewSegments();
	double NFA(double prob, int len);		  
};

#endif // ! _EDPF_
