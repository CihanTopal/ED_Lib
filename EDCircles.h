/**************************************************************************************************************
* EDCircles source codes.
* Copyright (C) Cuneyt Akinlar, Cihan Topal
* E-mails of the authors: cuneytakinlar@gmail.com, cihantopal@gmail.com
*
* Please cite the following papers if you use EDCircles library:
*
* [1] C. Akinlar and C. Topal, “EDCircles: A Real-time Circle Detector with a False Detection Control,”
*     Pattern Recognition, 46(3), 725-740, 2013.
*
* [2] C. Akinlar and C. Topal, “EDCircles: Realtime Circle Detection by Edge Drawing (ED),”
*     IEEE Int'l Conf. on Acoustics,  Speech, and Signal Processing (ICASSP)}, March 2012.
**************************************************************************************************************/

#ifndef  _EDCIRCLES_
#define _EDCIRCLES_

#include "EDPF.h"
#include "EDLines.h"

#define PI      3.141592653589793238462
#define TWOPI  (2*PI)

// Circular arc, circle thresholds
#define VERY_SHORT_ARC_ERROR     0.40  // Used for very short arcs (>= CANDIDATE_CIRCLE_RATIO1 && < CANDIDATE_CIRCLE_RATIO2)
#define SHORT_ARC_ERROR          1.00  // Used for short arcs (>= CANDIDATE_CIRCLE_RATIO2 && < HALF_CIRCLE_RATIO)
#define HALF_ARC_ERROR           1.25  // Used for arcs with length (>=HALF_CIRCLE_RATIO && < FULL_CIRCLE_RATIO)
#define LONG_ARC_ERROR           1.50  // Used for long arcs (>= FULL_CIRCLE_RATIO)

#define CANDIDATE_CIRCLE_RATIO1  0.25  // 25% -- If only 25% of the circle is detected, it may be a candidate for validation
#define CANDIDATE_CIRCLE_RATIO2  0.33  // 33% -- If only 33% of the circle is detected, it may be a candidate for validation
#define HALF_CIRCLE_RATIO        0.50  // 50% -- If 50% of a circle is detected at any point during joins, we immediately make it a candidate
#define FULL_CIRCLE_RATIO        0.67  // 67% -- If 67% of the circle is detected, we assume that it is fully covered

// Ellipse thresholds
#define CANDIDATE_ELLIPSE_RATIO  0.50  // 50% -- If 50% of the ellipse is detected, it may be candidate for validation
#define ELLIPSE_ERROR            1.50  // Used for ellipses. (used to be 1.65 for what reason?)

#define BOOKSTEIN 0       // method1 for ellipse fit
#define FPF       1       // method2 for ellipse fit

enum ImageStyle{NONE=0, CIRCLES, ELLIPSES, BOTH};

// Circle equation: (x-xc)^2 + (y-yc)^2 = r^2
struct mCircle {
	cv::Point2d center;
	double r;
	mCircle(cv::Point2d _center, double _r) { center = _center; r = _r; }
};

// Ellipse equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
struct mEllipse {
	cv::Point2d center;
	cv::Size axes;
	double theta;
	mEllipse(cv::Point2d _center, cv::Size _axes, double _theta) { center = _center; axes = _axes; theta = _theta; }
};

//----------------------------------------------------------
// Ellipse Equation is of the form:
// Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
//
struct EllipseEquation {
	double coeff[7];  // coeff[1] = A

	EllipseEquation() {
		for (int i = 0; i<7; i++) coeff[i] = 0;
	} //end-EllipseEquation

	double A() { return coeff[1]; }
	double B() { return coeff[2]; }
	double C() { return coeff[3]; }
	double D() { return coeff[4]; }
	double E() { return coeff[5]; }
	double F() { return coeff[6]; }
};

// ================================ CIRCLES ================================
struct Circle {
	double xc, yc, r;        // Center (xc, yc) & radius.
	double circleFitError;   // circle fit error
	double coverRatio;       // Percentage of the circle covered by the arcs making up this circle [0-1]

	double *x, *y;           // Pointers to buffers containing the pixels making up this circle
	int noPixels;            // # of pixels making up this circle

							 // If this circle is better approximated by an ellipse, we set isEllipse to true & eq contains the ellipse's equation
	EllipseEquation eq;
	double ellipseFitError;  // ellipse fit error
	bool isEllipse;
	double majorAxisLength;     // Length of the major axis
	double minorAxisLength;     // Length of the minor axis
};

// ------------------------------------------- ARCS ----------------------------------------------------
struct MyArc {
	double xc, yc, r;           // center x, y and radius
	double circleFitError;      // Error during circle fit

	double sTheta, eTheta;      // Start & end angle in radius
	double coverRatio;          // Ratio of the pixels covered on the covering circle [0-1] (noPixels/circumference)

	int turn;                   // Turn direction: 1 or -1

	int segmentNo;              // SegmentNo where this arc belongs

	int sx, sy;                 // Start (x, y) coordinate
	int ex, ey;                 // End (x, y) coordinate of the arc

	double *x, *y;              // Pointer to buffer containing the pixels making up this arc
	int noPixels;               // # of pixels making up the arc    

	bool isEllipse;             // Did we fit an ellipse to this arc? 
	EllipseEquation eq;         // If an ellipse, then the ellipse's equation
	double ellipseFitError;     // Error during ellipse fit  
};

// =============================== AngleSet ==================================

//-------------------------------------------------------------------------
// add a circular arc to the list of arcs
//
inline double ArcLength(double sTheta, double eTheta) {
	if (eTheta > sTheta) return eTheta - sTheta;
	else                 return TWOPI - sTheta + eTheta;
} // end-ArcLength

// A fast implementation of the AngleSet class. The slow implementation is really bad. About 10 times slower than this!
struct AngleSetArc {
	double sTheta;
	double eTheta;
	int next;         // Next AngleSetArc in the linked list
};

struct AngleSet {
	AngleSetArc angles[360];
	int head;
	int next;   // Next AngleSetArc to be allocated
	double overlapAmount;   // Total overlap of the arcs in angleSet. Computed during set() function

	AngleSet() { clear(); } //end-AngleSet
	void clear() { head = -1; next = 0; overlapAmount = 0; }
	double overlapRatio() { return overlapAmount / (TWOPI); }

	void _set(double sTheta, double eTheta);
	void set(double sTheta, double eTheta);

	double _overlap(double sTheta, double eTheta);
	double overlap(double sTheta, double eTheta);

	void computeStartEndTheta(double &sTheta, double &eTheta);
	double coverRatio();
};


struct EDArcs {
	MyArc *arcs;
	int noArcs;

public:
	EDArcs(int size = 10000) {
		arcs = new MyArc[size];
		noArcs = 0;
	} //end-EDArcs

	~EDArcs() {
		delete[] arcs;
	} //end-~EDArcs
};


//-----------------------------------------------------------------
// Buffer manager
struct BufferManager {
	double *x, *y;
	int index;

	BufferManager(int maxSize) {
		x = new double[maxSize];
		y = new double[maxSize];
		index = 0;
	} //end-BufferManager

	~BufferManager() {
		delete[] x;
		delete[] y;
	} //end-~BufferManager  

	double *getX() { return &x[index]; }
	double *getY() { return &y[index]; }
	void move(int size) { index += size; }
};

struct Info {
	int sign;     // -1 or 1: sign of the cross product
	double angle; // angle with the next line (in radians)
	bool taken;   // Is this line taken during arc detection
};


class EDCircles: public EDPF {
public:
	EDCircles(cv::Mat srcImage);
	EDCircles(ED obj);
	EDCircles(EDColor obj);

	cv::Mat drawResult(bool, ImageStyle);

	std::vector<mCircle> getCircles();
	std::vector<mEllipse> getEllipses();
	int getCirclesNo();
	int getEllipsesNo();

private:
	int noEllipses;
	int noCircles;
	std::vector<mCircle> circles;
	std::vector<mEllipse> ellipses;

	Circle *circles1;
	Circle *circles2;
	Circle *circles3;
	int noCircles1;
	int noCircles2;
	int noCircles3;

	EDArcs *edarcs1;
	EDArcs *edarcs2;
	EDArcs *edarcs3;
	EDArcs *edarcs4;

	int *segmentStartLines;
	BufferManager *bm;
	Info *info;
	NFALUT *nfa;

	void GenerateCandidateCircles();
	void DetectArcs(std::vector<LineSegment> lines);
	void ValidateCircles();
	void JoinCircles();
	void JoinArcs1();
	void JoinArcs2();
	void JoinArcs3();
	
	// circle utility functions
	static Circle *addCircle(Circle *circles, int &noCircles,double xc, double yc, double r, double circleFitError, double *x, double *y, int noPixels);
	static Circle *addCircle(Circle *circles, int &noCircles,double xc, double yc, double r, double circleFitError, EllipseEquation *pEq, double ellipseFitError, double *x, double *y, int noPixels);
	static void sortCircles(Circle *circles, int noCircles);
	static bool CircleFit(double *x, double *y, int N, double *pxc, double *pyc, double *pr, double *pe);
	static void ComputeCirclePoints(double xc, double yc, double r, double *px, double *py, int *noPoints);
	static void sortCircle(Circle *circles, int noCircles);
	
	// ellipse utility functions
	static bool EllipseFit(double *x, double *y, int noPoints, EllipseEquation *pResult, int mode=FPF);
	static double **AllocateMatrix(int noRows, int noColumns);
	static void A_TperB(double **_A, double **_B, double **_res, int _righA, int _colA, int _righB, int _colB);
	static void choldc(double **a, int n, double **l);
	static int inverse(double **TB, double **InvB, int N);
	static void DeallocateMatrix(double **m, int noRows);
	static void AperB_T(double **_A, double **_B, double **_res, int _righA, int _colA, int _righB, int _colB);
	static void AperB(double **_A, double **_B, double **_res, int _righA, int _colA, int _righB, int _colB);
	static void jacobi(double **a, int n, double d[], double **v, int nrot);
	static void ROTATE(double **a, int i, int j, int k, int l, double tau, double s);
	static double computeEllipsePerimeter(EllipseEquation *eq);
	static double ComputeEllipseError(EllipseEquation *eq, double *px, double *py, int noPoints);
	static double ComputeEllipseCenterAndAxisLengths(EllipseEquation *eq, double *pxc, double *pyc, double *pmajorAxisLength, double *pminorAxisLength);
	static void ComputeEllipsePoints(double *pvec, double *px, double *py, int noPoints);

	// arc utility functions
	static void joinLastTwoArcs(MyArc *arcs, int &noArcs);
	static void addArc(MyArc *arcs, int &noArchs,double xc, double yc, double r, double circleFitError, // Circular arc
		double sTheta, double eTheta, int turn, int segmentNo,
		int sx, int sy, int ex, int ey,
		double *x, double *y, int noPixels, double overlapRatio = 0.0);
	static void addArc(MyArc *arcs, int &noArchs, double xc, double yc, double r, double circleFitError, // Elliptic arc
		double sTheta, double eTheta, int turn, int segmentNo,
		EllipseEquation *pEq, double ellipseFitError,
		int sx, int sy, int ex, int ey,
		double *x, double *y, int noPixels, double overlapRatio = 0.0);

	static void ComputeStartAndEndAngles(double xc, double yc, double r,
		double *x, double *y, int len,
		double *psTheta, double *peTheta);

	static void sortArc(MyArc *arcs, int noArcs);
};

#endif // ! _EDCIRCLES_

