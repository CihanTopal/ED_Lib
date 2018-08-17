#include "EDColor.h"
#include "ED.h"

using namespace cv;
using namespace std;

EDColor::EDColor(Mat srcImage, int gradThresh, int anchor_thresh , double sigma, bool validateSegments)		   
{   
	inputImage = srcImage.clone(); 

	// check parameters for sanity
	if (sigma < 1) sigma = 1;
	if (gradThresh < 1) gradThresh = 1;
	if (anchor_thresh < 0) anchor_thresh = 0;

	if (validateSegments) { // setup for validation
		anchor_thresh = 0;
		divForTestSegment = 2.25;
	}

	// split channels (OpenCV uses BGR)
	Mat bgr[3];
	split(srcImage, bgr);
	blueImg = bgr[0].data;
	greenImg = bgr[1].data;
	redImg = bgr[2].data;

	height = srcImage.rows;
	width = srcImage.cols;

	// Allocate space for L*a*b color space
	L_Img = new uchar[width*height];
	a_Img = new uchar[width*height];
	b_Img = new uchar[width*height];

	// Convert RGB2Lab
	MyRGB2LabFast();

	// Allocate space for smooth channels
	smooth_L = new uchar[width*height];
	smooth_a = new uchar[width*height];
	smooth_b = new uchar[width*height]; 

	// Smooth Channels
	smoothChannel(L_Img, smooth_L, sigma);
	smoothChannel(a_Img, smooth_a, sigma);
	smoothChannel(b_Img, smooth_b, sigma);

	// Allocate space for direction and gradient images
	dirImg = new uchar[width*height];
	gradImg = new short[width*height];

	// Compute Gradient & Edge Direction Maps
	ComputeGradientMapByDiZenzo();
	
	
	// Validate edge segments if the flag is set
	if (validateSegments) {
		// Get Edge Image using ED 
		ED edgeObj = ED(gradImg, dirImg, width, height, gradThresh, anchor_thresh, 1, 10, false);
		segments = edgeObj.getSegments();
		edgeImage = edgeObj.getEdgeImage();

		sigma /= 2.5;
		smoothChannel(L_Img, smooth_L, sigma);
		smoothChannel(a_Img, smooth_a, sigma);
		smoothChannel(b_Img, smooth_b, sigma);

		edgeImg = edgeImage.data; // validation steps uses pointer to edgeImage

		validateEdgeSegments();

		// Extract the new edge segments after validation
		extractNewSegments();
	}
	
	else {
		ED edgeObj = ED(gradImg, dirImg, width, height, gradThresh, anchor_thresh);
		segments = edgeObj.getSegments();
		edgeImage = edgeObj.getEdgeImage();
		segmentNo = edgeObj.getSegmentNo();
	}
	
	// Fix 1 pixel errors in the edge map
	fixEdgeSegments(segments, 1);

	// clean space
	delete[] L_Img;
	delete[] a_Img;
	delete[] b_Img;

	delete[] smooth_L;
	delete[] smooth_a;
	delete[] smooth_b;
	
	delete[] gradImg;
	delete[] dirImg;
}

cv::Mat EDColor::getEdgeImage()
{
	return edgeImage;
}


std::vector<std::vector<cv::Point>> EDColor::getSegments()
{
	return segments;
}

int EDColor::getSegmentNo()
{
	return segmentNo;
}

int EDColor::getWidth()
{
	return width;
}

int EDColor::getHeight()
{
	return height;
}

void EDColor::MyRGB2LabFast()
{
	// Inialize LUTs if necessary
	if (!LUT_Initialized) 
		InitColorEDLib();

	// First RGB 2 XYZ
	double red, green, blue;
	double x, y, z;

	// Space for temp. allocation
	double *L = new double[width*height];
	double *a = new double[width*height];
	double *b = new double[width*height];

	for (int i = 0; i<width*height; i++) {
		red = redImg[i] / 255.0;
		green = greenImg[i] / 255.0;
		blue = blueImg[i] / 255.0;

		red = LUT1[(int)(red*LUT_SIZE + 0.5)];
		green = LUT1[(int)(green*LUT_SIZE + 0.5)];
		blue = LUT1[(int)(blue*LUT_SIZE + 0.5)];

		red = red * 100;
		green = green * 100;
		blue = blue * 100;

		//Observer. = 2°, Illuminant = D65
		x = red*0.4124564 + green*0.3575761 + blue*0.1804375;
		y = red*0.2126729 + green*0.7151522 + blue*0.0721750;
		z = red*0.0193339 + green*0.1191920 + blue*0.9503041;

		// Now xyz 2 Lab
		double refX = 95.047;
		double refY = 100.000;
		double refZ = 108.883;

		x = x / refX;          //ref_X =  95.047   Observer= 2°, Illuminant= D65
		y = y / refY;          //ref_Y = 100.000
		z = z / refZ;          //ref_Z = 108.883

		x = LUT2[(int)(x*LUT_SIZE + 0.5)];
		y = LUT2[(int)(y*LUT_SIZE + 0.5)];
		z = LUT2[(int)(z*LUT_SIZE + 0.5)];

		L[i] = (116.0*y) - 16;
		a[i] = 500 * (x / y);
		b[i] = 200 * (y - z);
	} //end-for
	
	// Scale L to [0-255]
	double min = 1e10;
	double max = -1e10;
	for (int i = 0; i<width*height; i++) {
		if (L[i]<min) min = L[i];
		else if (L[i]>max) max = L[i];
	} //end-for

	double scale = 255.0 / (max - min);
	for (int i = 0; i<width*height; i++) { L_Img[i] = (unsigned char)((L[i] - min)*scale); }

	// Scale a to [0-255]
	min = 1e10;
	max = -1e10;
	for (int i = 0; i<width*height; i++) {
		if (a[i]<min) min = a[i];
		else if (a[i]>max) max = a[i];
	} //end-for

	scale = 255.0 / (max - min);
	for (int i = 0; i<width*height; i++) { a_Img[i] = (unsigned char)((a[i] - min)*scale); }

	// Scale b to [0-255]
	min = 1e10;
	max = -1e10;
	for (int i = 0; i<width*height; i++) {
		if (b[i]<min) min = b[i];
		else if (b[i]>max) max = b[i];
	} //end-for

	scale = 255.0 / (max - min);
	for (int i = 0; i<width*height; i++) { b_Img[i] = (unsigned char)((b[i] - min)*scale); }

									   
	// clean space
	delete[] L;
	delete[] a;
	delete[] b;
}

void EDColor::ComputeGradientMapByDiZenzo()
{
	memset(gradImg, 0, sizeof(short)*width*height);

	int max = 0;

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
#if 1
			// Prewitt for channel1
			int com1 = smooth_L[(i + 1)*width + j + 1] - smooth_L[(i - 1)*width + j - 1];
			int com2 = smooth_L[(i - 1)*width + j + 1] - smooth_L[(i + 1)*width + j - 1];

			int gxCh1 = com1 + com2 + (smooth_L[i*width + j + 1] - smooth_L[i*width + j - 1]);
			int gyCh1 = com1 - com2 + (smooth_L[(i + 1)*width + j] - smooth_L[(i - 1)*width + j]);

			// Prewitt for channel2
			com1 = smooth_a[(i + 1)*width + j + 1] - smooth_a[(i - 1)*width + j - 1];
			com2 = smooth_a[(i - 1)*width + j + 1] - smooth_a[(i + 1)*width + j - 1];

			int gxCh2 = com1 + com2 + (smooth_a[i*width + j + 1] - smooth_a[i*width + j - 1]);
			int gyCh2 = com1 - com2 + (smooth_a[(i + 1)*width + j] - smooth_a[(i - 1)*width + j]);

			// Prewitt for channel3
			com1 = smooth_b[(i + 1)*width + j + 1] - smooth_b[(i - 1)*width + j - 1];
			com2 = smooth_b[(i - 1)*width + j + 1] - smooth_b[(i + 1)*width + j - 1];

			int gxCh3 = com1 + com2 + (smooth_b[i*width + j + 1] - smooth_b[i*width + j - 1]);
			int gyCh3 = com1 - com2 + (smooth_b[(i + 1)*width + j] - smooth_b[(i - 1)*width + j]);
#else
			// Sobel for channel1
			int com1 = smooth_L[(i + 1)*width + j + 1] - smooth_L[(i - 1)*width + j - 1];
			int com2 = smooth_L[(i - 1)*width + j + 1] - smooth_L[(i + 1)*width + j - 1];

			int gxCh1 = com1 + com2 + 2 * (smooth_L[i*width + j + 1] - smooth_L[i*width + j - 1]);
			int gyCh1 = com1 - com2 + 2 * (smooth_L[(i + 1)*width + j] - smooth_L[(i - 1)*width + j]);

			// Sobel for channel2
			com1 = smooth_a[(i + 1)*width + j + 1] - smooth_a[(i - 1)*width + j - 1];
			com2 = smooth_a[(i - 1)*width + j + 1] - smooth_a[(i + 1)*width + j - 1];

			int gxCh2 = com1 + com2 + 2 * (smooth_a[i*width + j + 1] - smooth_a[i*width + j - 1]);
			int gyCh2 = com1 - com2 + 2 * (smooth_a[(i + 1)*width + j] - smooth_a[(i - 1)*width + j]);

			// Sobel for channel3
			com1 = smooth_b[(i + 1)*width + j + 1] - smooth_b[(i - 1)*width + j - 1];
			com2 = smooth_b[(i - 1)*width + j + 1] - smooth_b[(i + 1)*width + j - 1];

			int gxCh3 = com1 + com2 + 2 * (smooth_b[i*width + j + 1] - smooth_b[i*width + j - 1]);
			int gyCh3 = com1 - com2 + 2 * (smooth_b[(i + 1)*width + j] - smooth_b[(i - 1)*width + j]);
#endif
			int gxx = gxCh1*gxCh1 + gxCh2*gxCh2 + gxCh3*gxCh3;
			int gyy = gyCh1*gyCh1 + gyCh2*gyCh2 + gyCh3*gyCh3;
			int gxy = gxCh1*gyCh1 + gxCh2*gyCh2 + gxCh3*gyCh3;

#if 1
			// Di Zenzo's formulas from Gonzales & Woods - Page 337
			double theta = atan2(2.0*gxy, (double)(gxx - gyy)) / 2; // Gradient Direction
			int grad = (int)(sqrt(((gxx + gyy) + (gxx - gyy)*cos(2 * theta) + 2 * gxy*sin(2 * theta)) / 2.0) + 0.5); // Gradient Magnitude
#else
			// Koschan & Abidi - 2005 - Signal Processing Magazine
			double theta = atan2(2.0*gxy, (double)(gxx - gyy)) / 2; // Gradient Direction

			double cosTheta = cos(theta);
			double sinTheta = sin(theta);
			int grad = (int)(sqrt(gxx*cosTheta*cosTheta + 2 * gxy*sinTheta*cosTheta + gyy*sinTheta*sinTheta) + 0.5); // Gradient Magnitude
#endif

			// Gradient is perpendicular to the edge passing through the pixel	
			if (theta >= -3.14159 / 4 && theta <= 3.14159 / 4)
				dirImg[i*width + j] = EDGE_VERTICAL;
			else
				dirImg[i*width + j] = EDGE_HORIZONTAL;

			gradImg[i*width + j] = grad;
			if (grad > max) max = grad;

		}
	} // end outer for

	// Scale the gradient values to 0-255
	double scale = 255.0 / max;
	for (int i = 0; i<width*height; i++)
		gradImg[i] = (short)(gradImg[i] * scale);
}

void EDColor::smoothChannel(uchar *src, uchar *smooth, double sigma)
{
	Mat srcImage = Mat(height, width, CV_8UC1, src);
	Mat smoothImage = Mat(height, width, CV_8UC1, smooth);

	if (sigma == 1.0)
		GaussianBlur(srcImage, smoothImage, Size(5, 5), 1);
	else if (sigma == 1.5)
		GaussianBlur(srcImage, smoothImage, Size(7, 7), 1.5);  // seems to be better?
	else
		GaussianBlur(srcImage, smoothImage, Size(), sigma);
}


//--------------------------------------------------------------------------------------------------------------------
// Validate the edge segments using the Helmholtz principle (for color images) channel1, channel2 and channel3 images
//
void EDColor::validateEdgeSegments()
{
	int maxGradValue = MAX_GRAD_VALUE;
	H = new double[maxGradValue];
	memset(H, 0, sizeof(double)*maxGradValue);

	memset(edgeImg, 0, width*height); // clear edge image
	
	// Compute the gradient
	memset(gradImg, 0, sizeof(short)*width*height); // reset gradient Image pixels to zero

	int *grads = new int[maxGradValue];
	memset(grads, 0, sizeof(int)*maxGradValue);

	for (int i = 1; i<height - 1; i++) {
		for (int j = 1; j<width - 1; j++) {
			// Gradient for channel1
			int com1 = smooth_L[(i + 1)*width + j + 1] - smooth_L[(i - 1)*width + j - 1];
			int com2 = smooth_L[(i - 1)*width + j + 1] - smooth_L[(i + 1)*width + j - 1];

			int gxCh1 = abs(com1 + com2 + (smooth_L[i*width + j + 1] - smooth_L[i*width + j - 1]));
			int gyCh1 = abs(com1 - com2 + (smooth_L[(i + 1)*width + j] - smooth_L[(i - 1)*width + j]));
			int ch1Grad = gxCh1 + gyCh1;

			// Gradient for channel2
			com1 = smooth_a[(i + 1)*width + j + 1] - smooth_a[(i - 1)*width + j - 1];
			com2 = smooth_a[(i - 1)*width + j + 1] - smooth_a[(i + 1)*width + j - 1];

			int gxCh2 = abs(com1 + com2 + (smooth_a[i*width + j + 1] - smooth_a[i*width + j - 1]));
			int gyCh2 = abs(com1 - com2 + (smooth_a[(i + 1)*width + j] - smooth_a[(i - 1)*width + j]));
			int ch2Grad = gxCh2 + gyCh2;

			// Gradient for channel3
			com1 = smooth_b[(i + 1)*width + j + 1] - smooth_b[(i - 1)*width + j - 1];
			com2 = smooth_b[(i - 1)*width + j + 1] - smooth_b[(i + 1)*width + j - 1];

			int gxCh3 = abs(com1 + com2 + (smooth_b[i*width + j + 1] - smooth_b[i*width + j - 1]));
			int gyCh3 = abs(com1 - com2 + (smooth_b[(i + 1)*width + j] - smooth_b[(i - 1)*width + j]));
			int ch3Grad = gxCh3 + gyCh3;

			// Take average
			int grad = (ch1Grad + ch2Grad + ch3Grad + 2) / 3;

			gradImg[i*width + j] = grad;
			grads[grad]++;
		} //end-for
	} //end-for

	Mat gradImage = Mat(height, width, CV_16SC1, gradImg);
	imwrite("newGrad.pgm", gradImage);

	// Compute probability function H
	int size = (width - 2)*(height - 2);
	//  size -= grads[0];
	
	for (int i = maxGradValue - 1; i>0; i--) 
		grads[i - 1] += grads[i];
	
	for (int i = 0; i<maxGradValue; i++) 
		H[i] = (double)grads[i] / ((double)size);

	// Compute np: # of segment pieces
	np = 0;
	for (int i = 0; i<segments.size(); i++) {
		int len = segments[i].size();
		np += (len*(len - 1)) / 2;
	} //end-for

	// Validate segments
	for (int i = 0; i< segments.size(); i++) {
		testSegment(i, 0, segments[i].size() - 1);
	} //end-for

	// clear space
	delete[] H;
	delete[] grads;
}


//----------------------------------------------------------------------------------
// Resursive validation using half of the pixels as suggested by DMM algorithm
// We take pixels at Nyquist distance, i.e., 2 (as suggested by DMM)
//
void EDColor::testSegment(int i, int index1, int index2)
{

	int chainLen = index2 - index1 + 1;
	if (chainLen < MIN_PATH_LEN)
		return;

	// Test from index1 to index2. If OK, then we are done. Otherwise, split into two and 
	// recursively test the left & right halves

	// First find the min. gradient along the segment
	int minGrad = 1 << 30;
	int minGradIndex;
	for (int k = index1; k <= index2; k++) {
		int r = segments[i][k].y;
		int c = segments[i][k].x;
		if (gradImg[r*width + c] < minGrad) { minGrad = gradImg[r*width + c]; minGradIndex = k; }
	} //end-for

	  // Compute nfa
	double nfa = NFA(H[minGrad], (int)(chainLen / divForTestSegment));

	if (nfa <= EPSILON) {
		for (int k = index1; k <= index2; k++) {
			int r = segments[i][k].y;
			int c = segments[i][k].x;

			edgeImg[r*width + c] = 255;
		} //end-for

		return;
	} //end-if  

	// Split into two halves. We divide at the point where the gradient is the minimum
	int end = minGradIndex - 1;
	while (end > index1) {
		int r = segments[i][end].y;
		int c = segments[i][end].x;

		if (gradImg[r*width + c] <= minGrad) end--;
		else break;
	} //end-while

	int start = minGradIndex + 1;
	while (start < index2) {
		int r = segments[i][start].y;
		int c = segments[i][start].x;

		if (gradImg[r*width + c] <= minGrad) start++;
		else break;
	} //end-while

	testSegment(i, index1, end);
	testSegment(i, start, index2);
}

//----------------------------------------------------------------------------------------------
// After the validation of the edge segments, extracts the valid ones
// In other words, updates the valid segments' pixel arrays and their lengths
// 
void EDColor::extractNewSegments()
{
	vector< vector<Point> > validSegments;
	int noSegments = 0;

	for (int i = 0; i < segments.size(); i++) {
		int start = 0;
		while (start < segments[i].size()) {

			while (start < segments[i].size()) {
				int r = segments[i][start].y;
				int c = segments[i][start].x;

				if (edgeImg[r*width + c]) break;
				start++;
			} //end-while

			int end = start + 1;
			while (end < segments[i].size()) {
				int r = segments[i][end].y;
				int c = segments[i][end].x;

				if (edgeImg[r*width + c] == 0) break;
				end++;
			} //end-while

			int len = end - start;
			if (len >= 10) {
				// A new segment. Accepted only only long enough (whatever that means)
				//segments[noSegments].pixels = &map->segments[i].pixels[start];
				//segments[noSegments].noPixels = len;
				validSegments.push_back(vector<Point>());
				vector<Point> subVec(&segments[i][start], &segments[i][end - 1]);
				validSegments[noSegments] = subVec;
				noSegments++;
			} //end-else

			start = end + 1;
		} //end-while
	} //end-for

	// Update
	segments = validSegments;
	segmentNo = noSegments; // = validSegments.size()

}


double EDColor::NFA(double prob, int len)
{
	double nfa = np;
	for (int i = 0; i<len && nfa > EPSILON; i++)
		nfa *= prob;

	return nfa;
}


//---------------------------------------------------------
// Fix edge segments having one or two pixel fluctuations
// An example one pixel problem getting fixed:
//  x
// x x --> xxx
//
// An example two pixel problem getting fixed:
//  xx
// x  x --> xxxx
//
void EDColor::fixEdgeSegments(std::vector<std::vector<cv::Point>> map, int noPixels)
{
	/// First fix one pixel problems: There are four cases
	for (int i = 0; i < map.size(); i++) {
		int cp = map[i].size() - 2;  // Current pixel index
		int n2 = 0;  // next next pixel index

		while (n2 < map[i].size()) {
			int n1 = cp + 1; // next pixel

			cp = cp % map[i].size(); // Roll back to the beginning
			n1 = n1 % map[i].size(); // Roll back to the beginning

			int r = map[i][cp].y;
			int c = map[i][cp].x;

			int r1 = map[i][n1].y;
			int c1 = map[i][n1].x;

			int r2 = map[i][n2].y;
			int c2 = map[i][n2].x;

			// 4 cases to fix
			if (r2 == r - 2 && c2 == c) {
				if (c1 != c) {
					map[i][n1].x = c;
				} //end-if

				cp = n2;
				n2 += 2;

			}
			else if (r2 == r + 2 && c2 == c) {
				if (c1 != c) {
					map[i][n1].x = c;
				} //end-if

				cp = n2;
				n2 += 2;

			}
			else if (r2 == r && c2 == c - 2) {
				if (r1 != r) {
					map[i][n1].y = r;
				} //end-if

				cp = n2;
				n2 += 2;

			}
			else if (r2 == r && c2 == c + 2) {
				if (r1 != r) {
					map[i][n1].y = r;
				} //end-if

				cp = n2;
				n2 += 2;

			}
			else {
				cp++;
				n2++;
			} //end-else
		} //end-while
	} // end-for
}

void EDColor::InitColorEDLib()
{
	if (LUT_Initialized) 
		return;

	double inc = 1.0 / LUT_SIZE;
	for (int i = 0; i <= LUT_SIZE; i++) {
		double d = i * inc;

		if (d >= 0.04045) LUT1[i] = pow(((d + 0.055) / 1.055), 2.4);
		else              LUT1[i] = d / 12.92;
	} //end-for

	inc = 1.0 / LUT_SIZE;
	for (int i = 0; i <= LUT_SIZE; i++) {
		double d = i * inc;

		if (d > 0.008856) LUT2[i] = pow(d, 1.0 / 3.0);
		else              LUT2[i] = (7.787*d) + (16.0 / 116.0);
	} //end-for

	LUT_Initialized = true;
}

bool EDColor::LUT_Initialized = false;
double EDColor::LUT1[LUT_SIZE + 1] = {0};
double EDColor::LUT2[LUT_SIZE + 1] = {0};