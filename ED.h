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

#ifndef _ED_
#define _ED_

#include <opencv2/opencv.hpp>
#include "EDColor.h"

/// Special defines
#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2

#define ANCHOR_PIXEL  254
#define EDGE_PIXEL    255

#define LEFT  1
#define RIGHT 2
#define UP    3
#define DOWN  4

enum GradientOperator { PREWITT_OPERATOR = 101, SOBEL_OPERATOR = 102, SCHARR_OPERATOR = 103, LSD_OPERATOR = 104 };

struct StackNode {
	int r, c;   // starting pixel
	int parent; // parent chain (-1 if no parent)
	int dir;    // direction where you are supposed to go
};

// Used during Edge Linking
struct Chain {

	int dir;                   // Direction of the chain
	int len;                   // # of pixels in the chain
	int parent;                // Parent of this node (-1 if no parent)
	int children[2];           // Children of this node (-1 if no children)
	cv::Point *pixels;         // Pointer to the beginning of the pixels array
};

class ED {
							
public:
	ED(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR, int _gradThresh = 20, int _anchorThresh = 0, int _scanInterval = 1, int _minPathLen = 10, double _sigma = 1.0, bool _sumFlag = true);
	ED(const ED &cpyObj); 
	ED(short* gradImg, uchar *dirImg, int _width, int _height, int _gradThresh, int _anchorThresh, int _scanInterval = 1, int _minPathLen = 10, bool selectStableAnchors = true);
	ED(EDColor &cpyObj);
	ED();

	cv::Mat getEdgeImage();
	cv::Mat getAnchorImage();
	cv::Mat getSmoothImage();
	cv::Mat getGradImage();
	
	int getSegmentNo();
	int getAnchorNo();
	
	std::vector<cv::Point> getAnchorPoints();
	std::vector<std::vector<cv::Point>> getSegments();
	std::vector<std::vector<cv::Point>> getSortedSegments();
	
	cv::Mat drawParticularSegments(std::vector<int> list);

protected:
	int width; // width of source image
	int height; // height of source image
	uchar *srcImg; 
	std::vector<std::vector< cv::Point> > segmentPoints;
	double sigma; // Gaussian sigma
	cv::Mat smoothImage;
	uchar *edgeImg; // pointer to edge image data
	uchar *smoothImg; // pointer to smoothed image data
	int segmentNos;
	int minPathLen;
	cv::Mat srcImage;

private:
	void ComputeGradient();
	void ComputeAnchorPoints();
	void JoinAnchorPointsUsingSortedAnchors(); 
	void sortAnchorsByGradValue();
	int* sortAnchorsByGradValue1();

	static int LongestChain(Chain *chains, int root);
	static int RetrieveChainNos(Chain *chains, int root, int chainNos[]);

	int anchorNos;
	std::vector<cv::Point> anchorPoints;
	std::vector<cv::Point> edgePoints;

	cv::Mat edgeImage;
	cv::Mat gradImage;

	uchar *dirImg; // pointer to direction image data
	short *gradImg; // pointer to gradient image data

	GradientOperator op; // operation used in gradient calculation
	int gradThresh; // gradient threshold
	int anchorThresh; // anchor point threshold
	int scanInterval;
	bool sumFlag;
};


#endif