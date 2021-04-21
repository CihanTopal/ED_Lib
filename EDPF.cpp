#include "EDPF.h"

using namespace cv;
using namespace std;

EDPF::EDPF(Mat srcImage)
	:ED(srcImage, PREWITT_OPERATOR, 11, 3)
{
	// Validate Edge Segments
	sigma /= 2.5;
	GaussianBlur(srcImage, smoothImage, Size(), sigma); // calculate kernel from sigma
	
	validateEdgeSegments();
}

EDPF::EDPF(ED obj)
	:ED(obj)
{
	// Validate Edge Segments
	sigma /= 2.5;
	GaussianBlur(srcImage, smoothImage, Size(), sigma); // calculate kernel from sigma

	validateEdgeSegments();
}

EDPF::EDPF(EDColor obj)
	:ED(obj)
{
}

void EDPF::validateEdgeSegments()
{
	divForTestSegment = 2.25; // Some magic number :-)
	memset(edgeImg, 0, width*height); // clear edge image
	
	H = new double[MAX_GRAD_VALUE];
	memset(H, 0, sizeof(double)*MAX_GRAD_VALUE);

	gradImg = ComputePrewitt3x3();

	// Compute np: # of segment pieces
#if 1
	// Does this underestimate the number of pieces of edge segments?
	// What's the correct value?
	np = 0;
	for (int i = 0; i < segmentNos; i++) {
		int len = (int)segmentPoints[i].size();
		np += (len*(len - 1)) / 2;
	} //end-for

	  //  np *= 32;
#elif 0
	// This definitely overestimates the number of pieces of edge segments
	int np = 0;
	for (int i = 0; i < segmentNos; i++) {
		np += segmentPoints[i].size();
	} //end-for
	np = (np*(np - 1)) / 2;
#endif

	// Validate segments
	for (int i = 0; i< segmentNos; i++) {
		TestSegment(i, 0, (int)segmentPoints[i].size() - 1);
	} //end-for

	ExtractNewSegments();

	// clean space		  
	delete[] H;
	delete[] gradImg;
}

short * EDPF::ComputePrewitt3x3()
{
	short *gradImg = new short[width*height];
	memset(gradImg, 0, sizeof(short)*width*height);

	int *grads = new int[MAX_GRAD_VALUE];
	memset(grads, 0, sizeof(int)*MAX_GRAD_VALUE);

	for (int i = 1; i<height - 1; i++) {
		for (int j = 1; j<width - 1; j++) {
			// Prewitt Operator in horizontal and vertical direction
			// A B C
			// D x E
			// F G H
			// gx = (C-A) + (E-D) + (H-F)
			// gy = (F-A) + (G-B) + (H-C)
			//
			// To make this faster: 
			// com1 = (H-A)
			// com2 = (C-F)
			// Then: gx = com1 + com2 + (E-D) = (H-A) + (C-F) + (E-D) = (C-A) + (E-D) + (H-F)
			//       gy = com1 - com2 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
			// 
			int com1 = smoothImg[(i + 1)*width + j + 1] - smoothImg[(i - 1)*width + j - 1];
			int com2 = smoothImg[(i - 1)*width + j + 1] - smoothImg[(i + 1)*width + j - 1];

			int gx = abs(com1 + com2 + (smoothImg[i*width + j + 1] - smoothImg[i*width + j - 1]));
			int gy = abs(com1 - com2 + (smoothImg[(i + 1)*width + j] - smoothImg[(i - 1)*width + j]));

			int g = gx + gy;

			gradImg[i*width + j] = g;
			grads[g]++;
		} // end-for
	} //end-for

	 // Compute probability function H
	int size = (width - 2)*(height - 2);
	
	for (int i = MAX_GRAD_VALUE - 1; i>0; i--)
		grads[i - 1] += grads[i];
	
	for (int i = 0; i < MAX_GRAD_VALUE; i++)
		H[i] = (double)grads[i] / ((double)size);

	delete[] grads;
	return gradImg;
}

//----------------------------------------------------------------------------------
// Resursive validation using half of the pixels as suggested by DMM algorithm
// We take pixels at Nyquist distance, i.e., 2 (as suggested by DMM)
//
void EDPF::TestSegment(int i, int index1, int index2)
{

	int chainLen = index2 - index1 + 1;
	if (chainLen < minPathLen) 
		return;

	// Test from index1 to index2. If OK, then we are done. Otherwise, split into two and 
	// recursively test the left & right halves

	// First find the min. gradient along the segment
	int minGrad = 1 << 30;
	int minGradIndex;
	for (int k = index1; k <= index2; k++) {
		int r = segmentPoints[i][k].y;
		int c = segmentPoints[i][k].x;
		if (gradImg[r*width + c] < minGrad) { minGrad = gradImg[r*width + c]; minGradIndex = k; }
	} //end-for

	 // Compute nfa
	double nfa = NFA(H[minGrad], (int)(chainLen / divForTestSegment));

	if (nfa <= EPSILON) {
		for (int k = index1; k <= index2; k++) {
			int r = segmentPoints[i][k].y;
			int c = segmentPoints[i][k].x;

			edgeImg[r*width + c] = 255;
		} //end-for

		return;
	} //end-if  

	// Split into two halves. We divide at the point where the gradient is the minimum
	int end = minGradIndex - 1;
	while (end > index1) {
		int r = segmentPoints[i][end].y;
		int c = segmentPoints[i][end].x;

		if (gradImg[r*width + c] <= minGrad) end--;
		else break;
	} //end-while

	int start = minGradIndex + 1;
	while (start < index2) {
		int r = segmentPoints[i][start].y;
		int c = segmentPoints[i][start].x;

		if (gradImg[r*width + c] <= minGrad) start++;
		else break;
	} //end-while

	TestSegment(i, index1, end);
	TestSegment(i, start, index2);
}

//----------------------------------------------------------------------------------------------
// After the validation of the edge segments, extracts the valid ones
// In other words, updates the valid segments' pixel arrays and their lengths
// 
void EDPF::ExtractNewSegments()
{
	//vector<Point> *segments = &segmentPoints[segmentNos];
	vector< vector<Point> > validSegments;
	int noSegments = 0;

	for (int i = 0; i < segmentNos; i++) {
		int start = 0;
		while (start < segmentPoints[i].size()) {

			while (start < segmentPoints[i].size()) {
				int r = segmentPoints[i][start].y;
				int c = segmentPoints[i][start].x;

				if (edgeImg[r*width + c]) break;
				start++;
			} //end-while

			int end = start + 1;
			while (end < segmentPoints[i].size()) {
				int r = segmentPoints[i][end].y;
				int c = segmentPoints[i][end].x;

				if (edgeImg[r*width + c] == 0) break;
				end++;
			} //end-while

			int len = end - start;
			if (len >= 10) {
				// A new segment. Accepted only only long enough (whatever that means)
				//segments[noSegments].pixels = &map->segments[i].pixels[start];
				//segments[noSegments].noPixels = len;
				validSegments.push_back(vector<Point>());
				vector<Point> subVec(&segmentPoints[i][start], &segmentPoints[i][end - 1]);
				validSegments[noSegments] = subVec;
				noSegments++;
			} //end-else

			start = end + 1;
		} //end-while
	} //end-for

	 // Copy to ed
	segmentPoints = validSegments;

	segmentNos = noSegments;
}

//---------------------------------------------------------------------------
// Number of false alarms code as suggested by Desolneux, Moisan and Morel (DMM)
//
double EDPF::NFA(double prob, int len)
{
	double nfa = np;
	for (int i = 0; i<len && nfa > EPSILON; i++) 
		nfa *= prob;

	return nfa;
}
