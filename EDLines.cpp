#include "EDLines.h"
#include "EDColor.h"
#include "NFA.h"

using namespace cv;
using namespace std;

EDLines::EDLines(Mat srcImage ,  double _line_error, int _min_line_len, double _max_distance_between_two_lines , double _max_error)
	:ED(srcImage, SOBEL_OPERATOR, 36, 8) 
{
	min_line_len = _min_line_len;
	line_error = _line_error;
	max_distance_between_two_lines = _max_distance_between_two_lines;
	max_error = _max_error;

	if(min_line_len == -1) // If no initial value given, compute it 
		min_line_len = ComputeMinLineLength();

	if (min_line_len < 9) // avoids small line segments in the result. Might be deleted!
		min_line_len = 9;



	// Temporary buffers used during line fitting
	size_t buffer_size = (width + height) * 8;
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		auto segment_size = segmentPoints[segmentNumber].size();
		buffer_size = std::max(buffer_size, segment_size);
	}
	double* x = new double[buffer_size];
	double* y = new double[buffer_size];

	linesNo = 0;
	
	// Use the whole segment
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		int k = 0;
		std::vector<Point> segment = segmentPoints[segmentNumber];
		for (int k = 0; k < segment.size(); k++) {
			x[k] = segment[k].x;
			y[k] = segment[k].y;
		}
		SplitSegment2Lines(x, y, (int)segment.size(), segmentNumber);
	}

	/*----------- JOIN COLLINEAR LINES ----------------*/
	JoinCollinearLines();

	/*----------- VALIDATE LINES ----------------*/
#define PRECISON_ANGLE 22.5 
	prec = (PRECISON_ANGLE / 180)*M_PI;
	double prob = 0.125;
#undef PRECISON_ANGLE

	double logNT = 2.0*(log10((double)width) + log10((double)height));

	int lutSize = (width + height) / 8;
	nfa = new NFALUT(lutSize, prob, logNT); // create look up table
	
	ValidateLineSegments();

	// Delete redundant space from lines
	// Pop them back
	int size = (int)lines.size();
	for (int i = 1; i <= size - linesNo; i++)
		lines.pop_back();
	
	for (int i = 0; i<linesNo; i++) {
		Point2d start(lines[i].sx, lines[i].sy);
		Point2d end(lines[i].ex, lines[i].ey);
		
		linePoints.push_back(LS(start, end));
	} //end-for

	delete[] x;
	delete[] y;
	delete nfa;
}


EDLines::EDLines(ED obj, double _line_error, int _min_line_len, double _max_distance_between_two_lines, double _max_error)
	:ED(obj) 
{
	min_line_len = _min_line_len;
	line_error = _line_error;
	max_distance_between_two_lines = _max_distance_between_two_lines;
	max_error = _max_error;

	if (min_line_len == -1) // If no initial value given, compute it 
		min_line_len = ComputeMinLineLength();

	if (min_line_len < 9) // avoids small line segments in the result. Might be deleted!
		min_line_len = 9;

	

	// Temporary buffers used during line fitting
	size_t buffer_size = (width + height) * 8;
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		auto segment_size = segmentPoints[segmentNumber].size();
		buffer_size = std::max(buffer_size, segment_size);
	}
	double* x = new double[buffer_size];
	double* y = new double[buffer_size];

	linesNo = 0;

	// Use the whole segment
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		int k = 0;
		std::vector<Point> segment = segmentPoints[segmentNumber];
		for (int k = 0; k < segment.size(); k++) {
			x[k] = segment[k].x;
			y[k] = segment[k].y;
		}
		SplitSegment2Lines(x, y, (int)segment.size(), segmentNumber);
	}

	/*----------- JOIN COLLINEAR LINES ----------------*/
	JoinCollinearLines();

	/*----------- VALIDATE LINES ----------------*/
#define PRECISON_ANGLE 22.5 
	prec = (PRECISON_ANGLE / 180)*M_PI;
	double prob = 0.125;
#undef PRECISON_ANGLE

	double logNT = 2.0*(log10((double)width) + log10((double)height));

	int lutSize = (width + height) / 8;
	nfa = new NFALUT(lutSize, prob, logNT); // create look up table

	ValidateLineSegments();

	// Delete redundant space from lines
	// Pop them back
	int size = (int)lines.size();
	for (int i = 1; i <= size - linesNo; i++)
		lines.pop_back();


	for (int i = 0; i<linesNo; i++) {
		Point2d start(lines[i].sx, lines[i].sy);
		Point2d end(lines[i].ex, lines[i].ey);

		linePoints.push_back(LS(start, end));
	} //end-for

	delete[] x;
	delete[] y;
	delete nfa;
}

EDLines::EDLines(EDColor obj, double _line_error, int _min_line_len, double _max_distance_between_two_lines, double _max_error)
	:ED(obj)
{
	min_line_len = _min_line_len;
	line_error = _line_error;
	max_distance_between_two_lines = _max_distance_between_two_lines;
	max_error = _max_error;

	if (min_line_len == -1) // If no initial value given, compute it 
		min_line_len = ComputeMinLineLength();

	if (min_line_len < 9) // avoids small line segments in the result. Might be deleted!
		min_line_len = 9;


	// Temporary buffers used during line fitting
	size_t buffer_size = (width + height) * 8;
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		auto segment_size = segmentPoints[segmentNumber].size();
		buffer_size = std::max(buffer_size, segment_size);
	}
	double* x = new double[buffer_size];
	double* y = new double[buffer_size];

	linesNo = 0;

	// Use the whole segment
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		int k = 0;
		std::vector<Point> segment = segmentPoints[segmentNumber];
		for (int k = 0; k < segment.size(); k++) {
			x[k] = segment[k].x;
			y[k] = segment[k].y;
		}
		SplitSegment2Lines(x, y, (int)segment.size(), segmentNumber);
	}

	/*----------- JOIN COLLINEAR LINES ----------------*/
	JoinCollinearLines();

	/*----------- VALIDATE LINES ----------------*/
#define PRECISON_ANGLE 22.5 
	prec = (PRECISON_ANGLE / 180)*M_PI;
	double prob = 0.125;
#undef PRECISON_ANGLE

	double logNT = 2.0*(log10((double)width) + log10((double)height));

	int lutSize = (width + height) / 8;
	nfa = new NFALUT(lutSize, prob, logNT); // create look up table

	// Since edge segments are validated in ed color, 
	// Validation is not performed again in line segment detection  
	// TODO :: further validation might be applied.
	// ValidateLineSegments(); 

	// Delete redundant space from lines
	// Pop them back
	int size = (int)lines.size();
	for (int i = 1; i <= size - linesNo; i++)
		lines.pop_back();


	for (int i = 0; i<linesNo; i++) {
		Point2d start(lines[i].sx, lines[i].sy);
		Point2d end(lines[i].ex, lines[i].ey);

		linePoints.push_back(LS(start, end));
	} //end-for

	delete[] x;
	delete[] y;
	delete nfa;
}

EDLines::EDLines()
{
	//
}

vector<LS> EDLines::getLines()
{
	return linePoints;
}

int EDLines::getLinesNo()
{
	return linesNo;
}

Mat EDLines::getLineImage()
{
	Mat lineImage = Mat(height, width, CV_8UC1, Scalar(255));
	for (int i = 0; i < linesNo; i++) {
		line(lineImage, linePoints[i].start, linePoints[i].end, Scalar(0), 1, LINE_AA, 0);
	}

	return lineImage;
}

Mat EDLines::drawOnImage()
{
	Mat colorImage = Mat(height, width, CV_8UC1, srcImg);
	cvtColor(colorImage, colorImage, COLOR_GRAY2BGR);
	for (int i = 0; i < linesNo; i++) {
		line(colorImage, linePoints[i].start, linePoints[i].end, Scalar(0, 255, 0), 1, LINE_AA, 0); // draw lines as green on image
	}

	return colorImage;
}

//-----------------------------------------------------------------------------------------
// Computes the minimum line length using the NFA formula given width & height values
int EDLines::ComputeMinLineLength() {
	// The reason we are dividing the theoretical minimum line length by 2 is because
	// we now test short line segments by a line support region rectangle having width=2.
	// This means that within a line support region rectangle for a line segment of length "l" 
	// there are "2*l" many pixels. Thus, a line segment of length "l" has a chance of getting
	// validated by NFA.

	double logNT = 2.0*(log10((double)width) + log10((double)height));
	return (int) round((-logNT / log10(0.125))*0.5);
} //end-ComputeMinLineLength

//-----------------------------------------------------------------
// Given a full segment of pixels, splits the chain to lines
// This code is used when we use the whole segment of pixels
//
void EDLines::SplitSegment2Lines(double * x, double * y, int noPixels, int segmentNo)
{

	// First pixel of the line segment within the segment of points
	int firstPixelIndex = 0;

	while (noPixels >= min_line_len) {
		// Start by fitting a line to MIN_LINE_LEN pixels
		bool valid = false;
		double lastA, lastB, error;
		int lastInvert;

		while (noPixels >= min_line_len) {
			LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
			if (error <= 0.5) { valid = true; break; }

#if 1
			noPixels -= 1;   // Go slowly
			x += 1; y += 1;
			firstPixelIndex += 1;
#else
			noPixels -= 2;   // Go faster (for speed)
			x += 2; y += 2;
			firstPixelIndex += 2;
#endif
		} //end-while

		if (valid == false) return;

		// Now try to extend this line
		int index = min_line_len;
		int len = min_line_len;

		while (index < noPixels) {
			int startIndex = index;
			int lastGoodIndex = index - 1;
			int goodPixelCount = 0;
			int badPixelCount = 0;
			while (index < noPixels) {
				double d = ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);

				if (d <= line_error) {
					lastGoodIndex = index;
					goodPixelCount++;
					badPixelCount = 0;

				}
				else {
					badPixelCount++;
					if (badPixelCount >= 5) break;
				} //end-if

				index++;
			} //end-while

			if (goodPixelCount >= 2) {
				len += lastGoodIndex - startIndex + 1;
				LineFit(x, y, len, lastA, lastB, lastInvert);  // faster LineFit
				index = lastGoodIndex + 1;
			} // end-if

			if (goodPixelCount < 2 || index >= noPixels) {
				// End of a line segment. Compute the end points
				double sx, sy, ex, ey;

				int index = 0;
				while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index++;
				ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx, sy);
				int noSkippedPixels = index;

				index = lastGoodIndex;
				while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index--;
				ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex, ey);

				if ((sx == ex) & (sy == ey))
					break;

				// Add the line segment to lines
				lines.push_back(LineSegment(lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo, firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1));
				linesNo++;
				len = index + 1;
				break;
			} //end-else
		} //end-while

		noPixels -= len;
		x += len;
		y += len;
		firstPixelIndex += len;
	} //end-while
}

//------------------------------------------------------------------
// Goes over the original line segments and combines collinear lines that belong to the same segment
//
void EDLines::JoinCollinearLines()
{
	int lastLineIndex = -1;   //Index of the last line in the joined lines
	int i = 0;
	while (i < linesNo) {
		int segmentNo = lines[i].segmentNo;

		lastLineIndex++;
		if (lastLineIndex != i) 
			lines[lastLineIndex] = lines[i];
		
		int firstLineIndex = lastLineIndex;  // Index of the first line in this segment

		int count = 1;
		for (int j = i + 1; j< linesNo; j++) {
			if (lines[j].segmentNo != segmentNo) break;

			// Try to combine this line with the previous line in this segment
			if (TryToJoinTwoLineSegments(&lines[lastLineIndex], &lines[j],
				lastLineIndex) == false) {
				lastLineIndex++;
				if (lastLineIndex != j) 
					lines[lastLineIndex] = lines[j];
				
			} //end-if

			count++;
		} //end-for

		  // Try to join the first & last line of this segment
		if (firstLineIndex != lastLineIndex) {
			if (TryToJoinTwoLineSegments(&lines[firstLineIndex], &lines[lastLineIndex],
				firstLineIndex)) {
				lastLineIndex--;
			} //end-if
		} //end-if

		i += count;
	} //end-while

	linesNo = lastLineIndex + 1;
}

void EDLines::ValidateLineSegments()
{

	int *x = new int[(width + height) * 4];
	int *y = new int[(width + height) * 4];

	int noValidLines = 0;
	int eraseOffset = 0;
	for (int i = 0; i< linesNo; i++) {
		LineSegment *ls = &lines[i];

		// Compute Line's angle
		double lineAngle;

		if (ls->invert == 0) {
			// y = a + bx
			lineAngle = atan(ls->b);

		}
		else {
			// x = a + by
			lineAngle = atan(1.0 / ls->b);
		} //end-else

		if (lineAngle < 0) lineAngle += M_PI;

		Point *pixels = &(segmentPoints[ls->segmentNo][0]);
		int noPixels = ls->len;

		bool valid = false;

		// Accept very long lines without testing. They are almost never invalidated.
		if (ls->len >= 80) {
			valid = true;

			// Validate short line segments by a line support region rectangle having width=2
		}
		else if (ls->len <= 25) {
			valid = ValidateLineSegmentRect( x, y, ls);

		}
		else {
			// Longer line segments are first validated by a line support region rectangle having width=1 (for speed)
			// If the line segment is still invalid, then a line support region rectangle having width=2 is tried
			// If the line segment fails both tests, it is discarded
			int aligned = 0;
			int count = 0;
			for (int j = 0; j<noPixels; j++) {
				int r = pixels[j].x;
				int c = pixels[j].y;

				if (r <= 0 || r >= height - 1 || c <= 0 || c >= width - 1) continue;

				count++;

				// compute gx & gy using the simple [-1 -1 -1]
				//                                  [ 1  1  1]  filter in both directions
				// Faster method below
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
				//       gy = com2 - com1 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
				// 
				int com1 = srcImg[(r + 1)*width + c + 1] - srcImg[(r - 1)*width + c - 1];
				int com2 = srcImg[(r - 1)*width + c + 1] - srcImg[(r + 1)*width + c - 1];

				int gx = com1 + com2 + srcImg[r*width + c + 1] - srcImg[r*width + c - 1];
				int gy = com1 - com2 + srcImg[(r + 1)*width + c] - srcImg[(r - 1)*width + c];
				
				double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);
				double diff = fabs(lineAngle - pixelAngle);

				if (diff <= prec || diff >= M_PI - prec) aligned++;
			} //end-for

			// Check validation by NFA computation (fast due to LUT)
			valid = nfa->checkValidationByNFA(count, aligned);
			if (valid == false) valid = ValidateLineSegmentRect(x, y, ls);
		} //end-else

		if (valid) {
			if (i != noValidLines) lines[noValidLines] = lines[i];
			noValidLines++;
		}
		else {
			invalidLines.push_back(lines[i]);
		} //end-else
	} //end-for

	linesNo = noValidLines;

	delete x;
	delete y;
}

bool EDLines::ValidateLineSegmentRect(int * x, int * y, LineSegment * ls)
{

	// Compute Line's angle
	double lineAngle;

	if (ls->invert == 0) {
		// y = a + bx
		lineAngle = atan(ls->b);

	}
	else {
		// x = a + by
		lineAngle = atan(1.0 / ls->b);
	} //end-else

	if (lineAngle < 0) lineAngle += M_PI;

	int noPoints = 0;

	// Enumerate all pixels that fall within the bounding rectangle
	EnumerateRectPoints(ls->sx, ls->sy, ls->ex, ls->ey, x, y, &noPoints);

	int count = 0;
	int aligned = 0;

	for (int i = 0; i<noPoints; i++) {
		int r = y[i];
		int c = x[i];

		if (r <= 0 || r >= height - 1 || c <= 0 || c >= width - 1) continue;

		count++;

		// compute gx & gy using the simple [-1 -1 -1]
		//                                  [ 1  1  1]  filter in both directions
		// Faster method below
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
		//       gy = com2 - com1 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
		// 
		int com1 = srcImg[(r + 1)*width + c + 1] - srcImg[(r - 1)*width + c - 1];
		int com2 = srcImg[(r - 1)*width + c + 1] - srcImg[(r + 1)*width + c - 1];

		int gx = com1 + com2 + srcImg[r*width + c + 1] - srcImg[r*width + c - 1];
		int gy = com1 - com2 + srcImg[(r + 1)*width + c] - srcImg[(r - 1)*width + c];
		double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);
		
		double diff = fabs(lineAngle - pixelAngle);

		if (diff <= prec || diff >= M_PI - prec) aligned++;
	} //end-for

	return nfa->checkValidationByNFA(count, aligned);
}



double EDLines::ComputeMinDistance(double x1, double y1, double a, double b, int invert)
{
	double x2, y2;

	if (invert == 0) {
		if (b == 0) {
			x2 = x1;
			y2 = a;

		}
		else {
			// Let the line passing through (x1, y1) that is perpendicular to a+bx be c+dx
			double d = -1.0 / (b);
			double c = y1 - d*x1;

			x2 = (a - c) / (d - b);
			y2 = a + b*x2;
		} //end-else

	}
	else {
		/// invert = 1
		if (b == 0) {
			x2 = a;
			y2 = y1;

		}
		else {
			// Let the line passing through (x1, y1) that is perpendicular to a+by be c+dy
			double d = -1.0 / (b);
			double c = x1 - d*y1;

			y2 = (a - c) / (d - b);
			x2 = a + b*y2;
		} //end-else
	} //end-else

	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

//---------------------------------------------------------------------------------
// Given a point (x1, y1) and a line equation y=a+bx (invert=0) OR x=a+by (invert=1)
// Computes the (x2, y2) on the line that is closest to (x1, y1)
//
void EDLines::ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut)
{
	double x2, y2;

	if (invert == 0) {
		if (b == 0) {
			x2 = x1;
			y2 = a;

		}
		else {
			// Let the line passing through (x1, y1) that is perpendicular to a+bx be c+dx
			double d = -1.0 / (b);
			double c = y1 - d*x1;

			x2 = (a - c) / (d - b);
			y2 = a + b*x2;
		} //end-else

	}
	else {
		/// invert = 1
		if (b == 0) {
			x2 = a;
			y2 = y1;

		}
		else {
			// Let the line passing through (x1, y1) that is perpendicular to a+by be c+dy
			double d = -1.0 / (b);
			double c = x1 - d*y1;

			y2 = (a - c) / (d - b);
			x2 = a + b*y2;
		} //end-else
	} //end-else

	xOut = x2;
	yOut = y2;
}

//-----------------------------------------------------------------------------------
// Fits a line of the form y=a+bx (invert == 0) OR x=a+by (invert == 1)
// Assumes that the direction of the line is known by a previous computation
//
void EDLines::LineFit(double * x, double * y, int count, double &a, double &b, int invert)
{
	if (count<2) return;

	double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
	for (int i = 0; i<count; i++) {
		Sx += x[i];
		Sy += y[i];
	} //end-for

	if (invert) {
		// Vertical line. Swap x & y, Sx & Sy
		double *t = x;
		x = y;
		y = t;

		double d = Sx;
		Sx = Sy;
		Sy = d;
	} //end-if

	  // Now compute Sxx & Sxy
	for (int i = 0; i<count; i++) {
		Sxx += x[i] * x[i];
		Sxy += x[i] * y[i];
	} //end-for

	double D = S*Sxx - Sx*Sx;
	a = (Sxx*Sy - Sx*Sxy) / D;
	b = (S  *Sxy - Sx* Sy) / D;
}


//-----------------------------------------------------------------------------------
// Fits a line of the form y=a+bx (invert == 0) OR x=a+by (invert == 1)
//
void EDLines::LineFit(double * x, double * y, int count, double &a, double &b, double &e, int &invert)
{
	if (count<2) return;

	double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
	for (int i = 0; i<count; i++) {
		Sx += x[i];
		Sy += y[i];
	} //end-for

	double mx = Sx / count;
	double my = Sy / count;

	double dx = 0.0;
	double dy = 0.0;
	for (int i = 0; i < count; i++) {
		dx += (x[i] - mx)*(x[i] - mx);
		dy += (y[i] - my)*(y[i] - my);
	} //end-for

	if (dx < dy) {
		// Vertical line. Swap x & y, Sx & Sy
		invert = 1;
		double *t = x;
		x = y;
		y = t;

		double d = Sx;
		Sx = Sy;
		Sy = d;

	}
	else {
		invert = 0;
	} //end-else  

	  // Now compute Sxx & Sxy
	for (int i = 0; i<count; i++) {
		Sxx += x[i] * x[i];
		Sxy += x[i] * y[i];
	} //end-for

	double D = S*Sxx - Sx*Sx;
	a = (Sxx*Sy - Sx*Sxy) / D;
	b = (S  *Sxy - Sx* Sy) / D;

	if (b == 0.0) {
		// Vertical or horizontal line
		double error = 0.0;
		for (int i = 0; i<count; i++) {
			error += fabs((a) - y[i]);
		} //end-for
		e = error / count;

	}
	else {
		double error = 0.0;
		for (int i = 0; i<count; i++) {
			// Let the line passing through (x[i], y[i]) that is perpendicular to a+bx be c+dx
			double d = -1.0 / (b);
			double c = y[i] - d*x[i];
			double x2 = ((a) - c) / (d - (b));
			double y2 = (a) + (b)*x2;

			double dist = (x[i] - x2)*(x[i] - x2) + (y[i] - y2)*(y[i] - y2);
			error += dist;
		} //end-for

		e = sqrt(error / count);
	} //end-else
}

//-----------------------------------------------------------------
// Checks if the given line segments are collinear & joins them if they are
// In case of a join, ls1 is updated. ls2 is NOT changed
// Returns true if join is successful, false otherwise
//
bool EDLines::TryToJoinTwoLineSegments(LineSegment * ls1, LineSegment * ls2, int changeIndex)
{	
	int which;
	double dist = ComputeMinDistanceBetweenTwoLines(ls1, ls2, &which);
	if (dist > max_distance_between_two_lines) return false;

	// Compute line lengths. Use the longer one as the ground truth
	double dx = ls1->sx - ls1->ex;
	double dy = ls1->sy - ls1->ey;
	double prevLen = sqrt(dx*dx + dy*dy);

	dx = ls2->sx - ls2->ex;
	dy = ls2->sy - ls2->ey;
	double nextLen = sqrt(dx*dx + dy*dy);

	// Use the longer line as the ground truth
	LineSegment *shorter = ls1;
	LineSegment *longer = ls2;

	if (prevLen > nextLen) { shorter = ls2; longer = ls1; }

#if 0
	// Use 5 points to check for collinearity  
#define POINT_COUNT 5
	double decr = 1.0 / (POINT_COUNT - 1);
	double alpha = 1.0;
	dist = 0.0;

	while (alpha >= 0.0) {
		double px = alpha*shorter->sx + (1.0 - alpha)*shorter->ex;
		double py = alpha*shorter->sy + (1.0 - alpha)*shorter->ey;

		dist += ComputeMinDistance(px, py, longer->a, longer->b, longer->invert);

		alpha -= decr;
	} //end-while

	dist /= POINT_COUNT;

#undef POINT_COUNT

#else
	// Just use 3 points to check for collinearity  
	dist = ComputeMinDistance(shorter->sx, shorter->sy, longer->a, longer->b, longer->invert);
	dist += ComputeMinDistance((shorter->sx + shorter->ex) / 2.0, (shorter->sy + shorter->ey) / 2.0, longer->a, longer->b, longer->invert);
	dist += ComputeMinDistance(shorter->ex, shorter->ey, longer->a, longer->b, longer->invert);

	dist /= 3.0;
#endif

	if (dist > max_error) return false;

#if 0
	// Update the end points of ls1
	if (which == 0) {       // SS
		ls1->sx = ls2->ex;
		ls1->sy = ls2->ey;

	}
	else if (which == 1) { // SE
		ls1->sx = ls2->sx;
		ls1->sy = ls2->sy;

	}
	else if (which == 2) { // ES
		ls1->ex = ls2->ex;
		ls1->ey = ls2->ey;

	}
	else {                // EE
		ls1->ex = ls2->sx;
		ls1->ey = ls2->sy;
	} //end-else

#else
	/// 4 cases: 1:(s1, s2), 2:(s1, e2), 3:(e1, s2), 4:(e1, e2)

	/// case 1: (s1, s2)
	dx = fabs(ls1->sx - ls2->sx);
	dy = fabs(ls1->sy - ls2->sy);
	double d = dx + dy;
	double max = d;
	which = 1;

	/// case 2: (s1, e2)
	dx = fabs(ls1->sx - ls2->ex);
	dy = fabs(ls1->sy - ls2->ey);
	d = dx + dy;
	if (d > max) {
		max = d;
		which = 2;
	} //end-if

	  /// case 3: (e1, s2)
	dx = fabs(ls1->ex - ls2->sx);
	dy = fabs(ls1->ey - ls2->sy);
	d = dx + dy;
	if (d > max) {
		max = d;
		which = 3;
	} //end-if

	  /// case 4: (e1, e2)
	dx = fabs(ls1->ex - ls2->ex);
	dy = fabs(ls1->ey - ls2->ey);
	d = dx + dy;
	if (d > max) {
		max = d;
		which = 4;
	} //end-if

	if (which == 1) {
		// (s1, s2)
		ls1->ex = ls2->sx;
		ls1->ey = ls2->sy;

	}
	else if (which == 2) {
		// (s1, e2)
		ls1->ex = ls2->ex;
		ls1->ey = ls2->ey;

	}
	else if (which == 3) {
		// (e1, s2)
		ls1->sx = ls2->sx;
		ls1->sy = ls2->sy;

	}
	else {
		// (e1, e2)
		ls1->sx = ls1->ex;
		ls1->sy = ls1->ey;

		ls1->ex = ls2->ex;
		ls1->ey = ls2->ey;
	} //end-else

#endif

	  // Update the first line's parameters
	if (ls1->firstPixelIndex + ls1->len + 5 >= ls2->firstPixelIndex) ls1->len += ls2->len;
	else if (ls2->len > ls1->len) {
		ls1->firstPixelIndex = ls2->firstPixelIndex;
		ls1->len = ls2->len;
	} //end-if

	UpdateLineParameters(ls1);
	lines[changeIndex] = *ls1;

	return true;
}

//-------------------------------------------------------------------------------
// Computes the minimum distance between the end points of two lines
//
double EDLines::ComputeMinDistanceBetweenTwoLines(LineSegment * ls1, LineSegment * ls2, int * pwhich)
{
	double dx = ls1->sx - ls2->sx;
	double dy = ls1->sy - ls2->sy;
	double d = sqrt(dx*dx + dy*dy);
	double min = d;
	int which = SS;

	dx = ls1->sx - ls2->ex;
	dy = ls1->sy - ls2->ey;
	d = sqrt(dx*dx + dy*dy);
	if (d < min) { min = d; which = SE; }

	dx = ls1->ex - ls2->sx;
	dy = ls1->ey - ls2->sy;
	d = sqrt(dx*dx + dy*dy);
	if (d < min) { min = d; which = ES; }

	dx = ls1->ex - ls2->ex;
	dy = ls1->ey - ls2->ey;
	d = sqrt(dx*dx + dy*dy);
	if (d < min) { min = d; which = EE; }

	if (pwhich) *pwhich = which;
	return min;
}

//-----------------------------------------------------------------------------------
// Uses the two end points (sx, sy)----(ex, ey) of the line segment
// and computes the line that passes through these points (a, b, invert)
//
void EDLines::UpdateLineParameters(LineSegment * ls)
{
	double dx = ls->ex - ls->sx;
	double dy = ls->ey - ls->sy;

	if (fabs(dx) >= fabs(dy)) {
		/// Line will be of the form y = a + bx
		ls->invert = 0;
		if (fabs(dy) < 1e-3) { ls->b = 0; ls->a = (ls->sy + ls->ey) / 2; }
		else {
			ls->b = dy / dx;
			ls->a = ls->sy - (ls->b)*ls->sx;
		} //end-else

	}
	else {
		/// Line will be of the form x = a + by
		ls->invert = 1;
		if (fabs(dx) < 1e-3) { ls->b = 0; ls->a = (ls->sx + ls->ex) / 2; }
		else {
			ls->b = dx / dy;
			ls->a = ls->sx - (ls->b)*ls->sy;
		} //end-else
	} //end-else
}

void EDLines::EnumerateRectPoints(double sx, double sy, double ex, double ey, int ptsx[], int ptsy[], int * pNoPoints)
{
	double vxTmp[4], vyTmp[4];
	double vx[4], vy[4];
	int n, offset;

	double x1 = sx;
	double y1 = sy;
	double x2 = ex;
	double y2 = ey;
	double width = 2;

	double dx = x2 - x1;
	double dy = y2 - y1;
	double vLen = sqrt(dx*dx + dy*dy);

	// make unit vector
	dx = dx / vLen;
	dy = dy / vLen;

	/* build list of rectangle corners ordered
	in a circular way around the rectangle */
	vxTmp[0] = x1 - dy * width / 2.0;
	vyTmp[0] = y1 + dx * width / 2.0;
	vxTmp[1] = x2 - dy * width / 2.0;
	vyTmp[1] = y2 + dx * width / 2.0;
	vxTmp[2] = x2 + dy * width / 2.0;
	vyTmp[2] = y2 - dx * width / 2.0;
	vxTmp[3] = x1 + dy * width / 2.0;
	vyTmp[3] = y1 - dx * width / 2.0;

	/* compute rotation of index of corners needed so that the first
	point has the smaller x.

	if one side is vertical, thus two corners have the same smaller x
	value, the one with the largest y value is selected as the first.
	*/
	if (x1 < x2 && y1 <= y2) offset = 0;
	else if (x1 >= x2 && y1 < y2) offset = 1;
	else if (x1 > x2 && y1 >= y2) offset = 2;
	else                          offset = 3;

	/* apply rotation of index. */
	for (n = 0; n<4; n++) {
		vx[n] = vxTmp[(offset + n) % 4];
		vy[n] = vyTmp[(offset + n) % 4];
	} //end-for

	  /* Set a initial condition.

	  The values are set to values that will cause 'ri_inc' (that will
	  be called immediately) to initialize correctly the first 'column'
	  and compute the limits 'ys' and 'ye'.

	  'y' is set to the integer value of vy[0], the starting corner.

	  'ys' and 'ye' are set to very small values, so 'ri_inc' will
	  notice that it needs to start a new 'column'.

	  The smaller integer coordinate inside of the rectangle is
	  'ceil(vx[0])'. The current 'x' value is set to that value minus
	  one, so 'ri_inc' (that will increase x by one) will advance to
	  the first 'column'.
	  */
	int x = (int)ceil(vx[0]) - 1;
	int y = (int)ceil(vy[0]);
	double ys = -DBL_MAX, ye = -DBL_MAX;

	int noPoints = 0;
	while (1) {
		/* if not at end of exploration,
		increase y value for next pixel in the 'column' */
		y++;

		/* if the end of the current 'column' is reached,
		and it is not the end of exploration,
		advance to the next 'column' */
		while (y > ye && x <= vx[2]) {
			/* increase x, next 'column' */
			x++;

			/* if end of exploration, return */
			if (x > vx[2]) break;

			/* update lower y limit (start) for the new 'column'.

			We need to interpolate the y value that corresponds to the
			lower side of the rectangle. The first thing is to decide if
			the corresponding side is

			vx[0],vy[0] to vx[3],vy[3] or
			vx[3],vy[3] to vx[2],vy[2]

			Then, the side is interpolated for the x value of the
			'column'. But, if the side is vertical (as it could happen if
			the rectangle is vertical and we are dealing with the first
			or last 'columns') then we pick the lower value of the side
			by using 'inter_low'.
			*/
			if ((double)x < vx[3]) {
				/* interpolation */
				if (fabs(vx[0] - vx[3]) <= 0.01) {
					if (vy[0]<vy[3]) ys = vy[0];
					else if (vy[0]>vy[3]) ys = vy[3];
					else     ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);
				}
				else
					ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);

			}
			else {
				/* interpolation */
				if (fabs(vx[3] - vx[2]) <= 0.01) {
					if (vy[3]<vy[2]) ys = vy[3];
					else if (vy[3]>vy[2]) ys = vy[2];
					else     ys = vy[3] + (x - vx[3]) * (y2 - vy[3]) / (vx[2] - vx[3]);
				}
				else
					ys = vy[3] + (x - vx[3]) * (vy[2] - vy[3]) / (vx[2] - vx[3]);
			} //end-else

			  /* update upper y limit (end) for the new 'column'.

			  We need to interpolate the y value that corresponds to the
			  upper side of the rectangle. The first thing is to decide if
			  the corresponding side is

			  vx[0],vy[0] to vx[1],vy[1] or
			  vx[1],vy[1] to vx[2],vy[2]

			  Then, the side is interpolated for the x value of the
			  'column'. But, if the side is vertical (as it could happen if
			  the rectangle is vertical and we are dealing with the first
			  or last 'columns') then we pick the lower value of the side
			  by using 'inter_low'.
			  */
			if ((double)x < vx[1]) {
				/* interpolation */
				if (fabs(vx[0] - vx[1]) <= 0.01) {
					if (vy[0]<vy[1]) ye = vy[1];
					else if (vy[0]>vy[1]) ye = vy[0];
					else     ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);
				}
				else
					ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);

			}
			else {
				/* interpolation */
				if (fabs(vx[1] - vx[2]) <= 0.01) {
					if (vy[1]<vy[2]) ye = vy[2];
					else if (vy[1]>vy[2]) ye = vy[1];
					else     ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
				}
				else
					ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
			} //end-else

			  /* new y */
			y = (int)ceil(ys);
		} //end-while

		  // Are we done?
		if (x > vx[2]) break;

		ptsx[noPoints] = x;
		ptsy[noPoints] = y;
		noPoints++;
	} //end-while

	*pNoPoints = noPoints;
}

void EDLines::SplitSegment2Lines(double * x, double * y, int noPixels, int segmentNo, vector<LineSegment> &lines, int min_line_len, double line_error)
{
	// First pixel of the line segment within the segment of points
	int firstPixelIndex = 0;

	while (noPixels >= min_line_len) {
		// Start by fitting a line to MIN_LINE_LEN pixels
		bool valid = false;
		double lastA, lastB, error;
		int lastInvert;

		while (noPixels >= min_line_len) {
			LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
			if (error <= 0.5) { valid = true; break; }

#if 1
			noPixels -= 1;   // Go slowly
			x += 1; y += 1;
			firstPixelIndex += 1;
#else
			noPixels -= 2;   // Go faster (for speed)
			x += 2; y += 2;
			firstPixelIndex += 2;
#endif
		} //end-while

		if (valid == false) return;

		// Now try to extend this line
		int index = min_line_len;
		int len = min_line_len;

		while (index < noPixels) {
			int startIndex = index;
			int lastGoodIndex = index - 1;
			int goodPixelCount = 0;
			int badPixelCount = 0;
			while (index < noPixels) {
				double d = ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);

				if (d <= line_error) {
					lastGoodIndex = index;
					goodPixelCount++;
					badPixelCount = 0;

				}
				else {
					badPixelCount++;
					if (badPixelCount >= 5) break;
				} //end-if

				index++;
			} //end-while

			if (goodPixelCount >= 2) {
				len += lastGoodIndex - startIndex + 1;
				LineFit(x, y, len, lastA, lastB, lastInvert);  // faster LineFit
				index = lastGoodIndex + 1;
			} // end-if

			if (goodPixelCount < 2 || index >= noPixels) {
				// End of a line segment. Compute the end points
				double sx, sy, ex, ey;

				int index = 0;
				while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index++;
				ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx, sy);
				int noSkippedPixels = index;

				index = lastGoodIndex;
				while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index--;
				ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex, ey);

				// Add the line segment to lines
				lines.push_back(LineSegment(lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo, firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1));
				//linesNo++;
				len = index + 1;
				break;
			} //end-else
		} //end-while

		noPixels -= len;
		x += len;
		y += len;
		firstPixelIndex += len;
	} //end-while
}
