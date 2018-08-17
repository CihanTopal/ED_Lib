/**************************************************************************************************************
* Edge Drawing (ED) and Edge Drawing Parameter Free (EDPF) source codes.
* Copyright (C) 2016, Cuneyt Akinlar & Cihan Topal
* E-mails of the authors: cuneytakinlar@gmail.com, cihant@anadolu.edu.tr
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.

* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.

* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.

* By using this library you are implicitly assumed to have accepted all of the above statements,
* and accept to cite the following papers:
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
