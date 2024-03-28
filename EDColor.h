/**************************************************************************************************************
* Color Edge Drawing (ED) and Color Edge Drawing Parameter Free (EDPF) source codes.
* Copyright (C) Cihan Topal & Cuneyt Akinlar 
* E-mails of the authors:  cihantopal@gmail.com, cuneytakinlar@gmail.com
*
* Please cite the following papers if you use Edge Drawing library:
*
* [1] C. Topal and C. Akinlar, “Edge Drawing: A Combined Real-Time Edge and Segment Detector,”
*     Journal of Visual Communication and Image Representation, 23(6), 862-872, doi:10.1016/j.jvcir.2012.05.004 (
*
* [2] C. Akinlar and C. Topal, “EDPF: A Real-time Parameter-free Edge Segment Detector with a False Detection Con
*     International Journal of Pattern Recognition and Artificial Intelligence, 26(1), doi:10.1142/S0218001412550
*
* [3] C. Akinlar, C. Topal, "ColorED: Color Edge and Segment Detection by Edge Drawing (ED),"
*     submitted to the Journal of Visual Communication and Image Representation (2017).
**************************************************************************************************************/

#ifndef  _EDColor_
#define _EDColor_

#include <opencv2/opencv.hpp>

// Look up table size for fast color space conversion
#define LUT_SIZE (1024*4096)

// Special defines
#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2
#define EDGE_45         3
#define EDGE_135        4

#define MAX_GRAD_VALUE 128*256
#define EPSILON 1.0
#define MIN_PATH_LEN 10


class EDColor {
public:
	EDColor(cv::Mat srcImage, int gradThresh = 20, int anchor_thresh = 4, double sigma = 1.5, bool validateSegments=false);
	cv::Mat getEdgeImage();
	std::vector<std::vector<cv::Point>> getSegments();
	int getSegmentNo();
	
	int getWidth();
	int getHeight();

	cv::Mat inputImage;
private:
	uchar *L_Img;
	uchar *a_Img;
	uchar *b_Img;

	uchar *smooth_L;
	uchar *smooth_a;
	uchar *smooth_b;

	uchar *dirImg;
	short *gradImg;
	
	cv::Mat edgeImage;
	uchar *edgeImg;

	const uchar *blueImg;
	const uchar *greenImg;
	const uchar *redImg;
	
	int width;
	int height;

	double divForTestSegment;
	double *H;
	int np;
	int segmentNo;

	std::vector<std::vector<cv::Point>> segments;

	static double LUT1[LUT_SIZE + 1];
	static double LUT2[LUT_SIZE + 1];
	static bool LUT_Initialized;

	void MyRGB2LabFast();
	void ComputeGradientMapByDiZenzo();
	void smoothChannel(uchar *src, uchar *smooth, double sigma);
	void validateEdgeSegments();
	void testSegment(int i, int index1, int index2);
	void extractNewSegments();
	double NFA(double prob, int len);

	static void fixEdgeSegments(std::vector<std::vector<cv::Point>> map, int noPixels);

	static void InitColorEDLib();
};

#endif // ! _EDColor_
