#include "EDCircles.h"

using namespace cv;
using namespace std;

EDCircles::EDCircles(Mat srcImage)
	: EDPF(srcImage)
{
	// Arcs & circles to be detected
	// If the end-points of the segment is very close to each other, 
	// then directly fit a circle/ellipse instread of line fitting
	noCircles1 = 0;
	circles1 = new Circle[(width + height) * 8];

	// ----------------------------------- DETECT LINES ---------------------------------
	int bufferSize = 0;
	for (int i = 0; i < segmentPoints.size(); i++) 
		bufferSize += (int)segmentPoints[i].size();

	// Compute the starting line number for each segment
	segmentStartLines = new int[segmentNos + 1];

	bm = new BufferManager(bufferSize * 8);
	vector<LineSegment> lines;


#define CIRCLE_MIN_LINE_LEN 6

	for (int i = 0; i < segmentNos; i++) {

		// Make note of the starting line number for this segment
		segmentStartLines[i] = (int)lines.size();

		int noPixels = (int)segmentPoints[i].size();

		if (noPixels < 2 * CIRCLE_MIN_LINE_LEN) 
			continue;

		double *x = bm->getX();
		double *y = bm->getY();

		for (int j = 0; j<noPixels; j++) {
			x[j] = segmentPoints[i][j].x;
			y[j] = segmentPoints[i][j].y;
		} //end-for

		  // If the segment is reasonably long, then see if the segment traverses the boundary of a closed shape
		if (noPixels >= 4 * CIRCLE_MIN_LINE_LEN) {
			// If the end-points of the segment is close to each other, then assume a circular/elliptic structure
			double dx = x[0] - x[noPixels - 1];
			double dy = y[0] - y[noPixels - 1];
			double d = sqrt(dx*dx + dy*dy);
			double r = noPixels / TWOPI;      // Assume a complete circle

			double maxDistanceBetweenEndPoints = MAX(3, r / 4);

			// If almost closed loop, then try to fit a circle/ellipse
			if (d <= maxDistanceBetweenEndPoints) {
				double xc, yc, r, circleFitError = 1e10;

				CircleFit(x, y, noPixels, &xc, &yc, &r, &circleFitError);

				EllipseEquation eq;
				double ellipseFitError = 1e10;

				if (circleFitError > LONG_ARC_ERROR) {
					// Try fitting an ellipse
					if (EllipseFit(x, y, noPixels, &eq)) 
						ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);
				} //end-if

				if (circleFitError <= LONG_ARC_ERROR) {
					addCircle(circles1, noCircles1, xc ,yc, r, circleFitError, x, y, noPixels);
					bm->move(noPixels);
					continue;

				}
				else if (ellipseFitError <= ELLIPSE_ERROR) {
					double major, minor;
					ComputeEllipseCenterAndAxisLengths(&eq, &xc, &yc, &major, &minor);

					// Assume major is longer. Otherwise, swap
					if (minor > major) { double tmp = major; major = minor; minor = tmp; }

					if (major < 8 * minor) {
						addCircle(circles1, noCircles1,xc, yc, r, circleFitError, &eq, ellipseFitError, x, y, noPixels);
						bm->move(noPixels);
					} //end-if

					continue;
				} //end-else
			} //end-if 
		} //end-if
		
		// Otherwise, split to lines
		EDLines::SplitSegment2Lines(x, y, noPixels, i, lines);
	}

	segmentStartLines[segmentNos] = (int)lines.size();

	// ------------------------------- DETECT ARCS ---------------------------------

	info = new Info[lines.size()];

	// Compute the angle information for each line segment
	for (int i = 0; i<segmentNos; i++) {
		for (int j = segmentStartLines[i]; j<segmentStartLines[i + 1]; j++) {
			LineSegment *l1 = &lines[j];
			LineSegment *l2;

			if (j == segmentStartLines[i + 1] - 1) l2 = &lines[segmentStartLines[i]];
			else                               l2 = &lines[j + 1];

#if 1
			// If the end points of the lines are far from each other, then stop at this line
			double dx = l1->ex - l2->sx;
			double dy = l1->ey - l2->sy;
			double d = sqrt(dx*dx + dy*dy);
			if (d >= 15) { info[j].angle = 10; info[j].sign = 2; info[j].taken = false; continue; }
#endif

			// Compute the angle between the lines & their turn direction
			double v1x = l1->ex - l1->sx;
			double v1y = l1->ey - l1->sy;
			double v1Len = sqrt(v1x*v1x + v1y*v1y);

			double v2x = l2->ex - l2->sx;
			double v2y = l2->ey - l2->sy;
			double v2Len = sqrt(v2x*v2x + v2y*v2y);

			double dotProduct = (v1x*v2x + v1y*v2y) / (v1Len*v2Len);
			if (dotProduct > 1.0) dotProduct = 1.0;
			else if (dotProduct < -1.0) dotProduct = -1.0;

			info[j].angle = acos(dotProduct);
			info[j].sign = (v1x*v2y - v2x*v1y) >= 0 ? 1 : -1;  // compute cross product
			info[j].taken = false;
		} //end-for
	} //end-for

	// This is how much space we will allocate for circles buffers
	int maxNoOfCircles = (int)lines.size() / 3 + noCircles1 * 2;

	edarcs1 = new EDArcs(maxNoOfCircles);
	DetectArcs(lines);    // Detect all arcs

	// Try to join arcs that are almost perfectly circular. 
	// Use the distance between the arc end-points as a metric in choosing in choosing arcs to join
	edarcs2 = new EDArcs(maxNoOfCircles);
	JoinArcs1();

	// Try to join arcs that belong to the same segment
	edarcs3 = new EDArcs(maxNoOfCircles);
	JoinArcs2();

	// Try to combine arcs that belong to different segments
	edarcs4 = new EDArcs(maxNoOfCircles);     // The remaining arcs
	JoinArcs3();

	// Finally, go over the arcs & circles, and generate candidate circles
	GenerateCandidateCircles();

	//----------------------------- VALIDATE CIRCLES --------------------------
	noCircles2 = 0;
	circles2 = new Circle[maxNoOfCircles];
	GaussianBlur(srcImage, smoothImage, Size(), 0.50); // calculate kernel from sigma;

	ValidateCircles();

	//----------------------------- JOIN CIRCLES --------------------------
	noCircles3 = 0;
	circles3 = new Circle[maxNoOfCircles];
	JoinCircles();

	noCircles = 0;
	noEllipses = 0;
	for (int i = 0; i < noCircles3; i++) {
		if (circles3[i].isEllipse)
			noEllipses++;
		else
			noCircles++;
	}

	for (int i = 0; i<noCircles3; i++) {
		if (circles3[i].isEllipse) {
			EllipseEquation eq = circles3[i].eq;
			double xc;
			double yc;
			double a;
			double b;
			double theta = ComputeEllipseCenterAndAxisLengths(&eq, &xc, &yc, &a, &b);
			ellipses.push_back(mEllipse(Point2d(xc, yc), Size2d(a, b), theta));

		}
		else {
			double r = circles3[i].r;
			double xc = circles3[i].xc;
			double yc = circles3[i].yc;

			circles.push_back(mCircle(Point2d(xc, yc), r));
			
		} //end-else
	} //end-for


	// clean up
	delete edarcs1;
	delete edarcs2;
	delete edarcs3;
	delete edarcs4;

	delete[] circles1;
	delete[] circles2;
	delete[] circles3;

	delete bm;
	delete[] segmentStartLines;
	delete[] info;
}

EDCircles::EDCircles(ED obj)
	:EDPF(obj)
{
	// Arcs & circles to be detected
	// If the end-points of the segment is very close to each other, 
	// then directly fit a circle/ellipse instread of line fitting
	noCircles1 = 0;
	circles1 = new Circle[(width + height) * 8];

	// ----------------------------------- DETECT LINES ---------------------------------
	int bufferSize = 0;
	for (int i = 0; i < segmentPoints.size(); i++)
		bufferSize += (int)segmentPoints[i].size();

	// Compute the starting line number for each segment
	segmentStartLines = new int[segmentNos + 1];

	bm = new BufferManager(bufferSize * 8);
	vector<LineSegment> lines;


#define CIRCLE_MIN_LINE_LEN 6

	for (int i = 0; i < segmentNos; i++) {

		// Make note of the starting line number for this segment
		segmentStartLines[i] = (int)lines.size();

		int noPixels = (int)segmentPoints[i].size();

		if (noPixels < 2 * CIRCLE_MIN_LINE_LEN)
			continue;

		double *x = bm->getX();
		double *y = bm->getY();

		for (int j = 0; j<noPixels; j++) {
			x[j] = segmentPoints[i][j].x;
			y[j] = segmentPoints[i][j].y;
		} //end-for

		  // If the segment is reasonably long, then see if the segment traverses the boundary of a closed shape
		if (noPixels >= 4 * CIRCLE_MIN_LINE_LEN) {
			// If the end-points of the segment is close to each other, then assume a circular/elliptic structure
			double dx = x[0] - x[noPixels - 1];
			double dy = y[0] - y[noPixels - 1];
			double d = sqrt(dx*dx + dy*dy);
			double r = noPixels / TWOPI;      // Assume a complete circle

			double maxDistanceBetweenEndPoints = MAX(3, r / 4);

			// If almost closed loop, then try to fit a circle/ellipse
			if (d <= maxDistanceBetweenEndPoints) {
				double xc, yc, r, circleFitError = 1e10;

				CircleFit(x, y, noPixels, &xc, &yc, &r, &circleFitError);

				EllipseEquation eq;
				double ellipseFitError = 1e10;

				if (circleFitError > LONG_ARC_ERROR) {
					// Try fitting an ellipse
					if (EllipseFit(x, y, noPixels, &eq))
						ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);
				} //end-if

				if (circleFitError <= LONG_ARC_ERROR) {
					addCircle(circles1, noCircles1, xc, yc, r, circleFitError, x, y, noPixels);
					bm->move(noPixels);
					continue;

				}
				else if (ellipseFitError <= ELLIPSE_ERROR) {
					double major, minor;
					ComputeEllipseCenterAndAxisLengths(&eq, &xc, &yc, &major, &minor);

					// Assume major is longer. Otherwise, swap
					if (minor > major) { double tmp = major; major = minor; minor = tmp; }

					if (major < 8 * minor) {
						addCircle(circles1, noCircles1, xc, yc, r, circleFitError, &eq, ellipseFitError, x, y, noPixels);
						bm->move(noPixels);
					} //end-if

					continue;
				} //end-else
			} //end-if 
		} //end-if

		  // Otherwise, split to lines
		EDLines::SplitSegment2Lines(x, y, noPixels, i, lines);
	}

	segmentStartLines[segmentNos] = (int)lines.size();

	// ------------------------------- DETECT ARCS ---------------------------------

	info = new Info[lines.size()];

	// Compute the angle information for each line segment
	for (int i = 0; i<segmentNos; i++) {
		for (int j = segmentStartLines[i]; j<segmentStartLines[i + 1]; j++) {
			LineSegment *l1 = &lines[j];
			LineSegment *l2;

			if (j == segmentStartLines[i + 1] - 1) l2 = &lines[segmentStartLines[i]];
			else                               l2 = &lines[j + 1];

#if 1
			// If the end points of the lines are far from each other, then stop at this line
			double dx = l1->ex - l2->sx;
			double dy = l1->ey - l2->sy;
			double d = sqrt(dx*dx + dy*dy);
			if (d >= 15) { info[j].angle = 10; info[j].sign = 2; info[j].taken = false; continue; }
#endif

			// Compute the angle between the lines & their turn direction
			double v1x = l1->ex - l1->sx;
			double v1y = l1->ey - l1->sy;
			double v1Len = sqrt(v1x*v1x + v1y*v1y);

			double v2x = l2->ex - l2->sx;
			double v2y = l2->ey - l2->sy;
			double v2Len = sqrt(v2x*v2x + v2y*v2y);

			double dotProduct = (v1x*v2x + v1y*v2y) / (v1Len*v2Len);
			if (dotProduct > 1.0) dotProduct = 1.0;
			else if (dotProduct < -1.0) dotProduct = -1.0;

			info[j].angle = acos(dotProduct);
			info[j].sign = (v1x*v2y - v2x*v1y) >= 0 ? 1 : -1;  // compute cross product
			info[j].taken = false;
		} //end-for
	} //end-for

	  // This is how much space we will allocate for circles buffers
	int maxNoOfCircles = (int)lines.size() / 3 + noCircles1 * 2;

	edarcs1 = new EDArcs(maxNoOfCircles);
	DetectArcs(lines);    // Detect all arcs

						  // Try to join arcs that are almost perfectly circular. 
						  // Use the distance between the arc end-points as a metric in choosing in choosing arcs to join
	edarcs2 = new EDArcs(maxNoOfCircles);
	JoinArcs1();

	// Try to join arcs that belong to the same segment
	edarcs3 = new EDArcs(maxNoOfCircles);
	JoinArcs2();

	// Try to combine arcs that belong to different segments
	edarcs4 = new EDArcs(maxNoOfCircles);     // The remaining arcs
	JoinArcs3();

	// Finally, go over the arcs & circles, and generate candidate circles
	GenerateCandidateCircles();

	//----------------------------- VALIDATE CIRCLES --------------------------
	noCircles2 = 0;
	circles2 = new Circle[maxNoOfCircles];
	GaussianBlur(srcImage, smoothImage, Size(), 0.50); // calculate kernel from sigma;

	ValidateCircles();

	//----------------------------- JOIN CIRCLES --------------------------
	noCircles3 = 0;
	circles3 = new Circle[maxNoOfCircles];
	JoinCircles();

	noCircles = 0;
	noEllipses = 0;
	for (int i = 0; i < noCircles3; i++) {
		if (circles3[i].isEllipse)
			noEllipses++;
		else
			noCircles++;
	}

	for (int i = 0; i<noCircles3; i++) {
		if (circles3[i].isEllipse) {
			EllipseEquation eq = circles3[i].eq;
			double xc;
			double yc;
			double a;
			double b;
			double theta = ComputeEllipseCenterAndAxisLengths(&eq, &xc, &yc, &a, &b);
			ellipses.push_back(mEllipse(Point2d(xc, yc), Size2d(a, b), theta));

		}
		else {
			double r = circles3[i].r;
			double xc = circles3[i].xc;
			double yc = circles3[i].yc;

			circles.push_back(mCircle(Point2d(xc, yc), r));

		} //end-else
	} //end-for


	// clean up
	delete edarcs1;
	delete edarcs2;
	delete edarcs3;
	delete edarcs4;

	delete[] circles1;
	delete[] circles2;
	delete[] circles3;

	delete bm;
	delete[] segmentStartLines;
	delete[] info;
}

EDCircles::EDCircles(EDColor obj)
	:EDPF(obj)
{
	// Arcs & circles to be detected
	// If the end-points of the segment is very close to each other, 
	// then directly fit a circle/ellipse instread of line fitting
	noCircles1 = 0;
	circles1 = new Circle[(width + height) * 8];

	// ----------------------------------- DETECT LINES ---------------------------------
	int bufferSize = 0;
	for (int i = 0; i < segmentPoints.size(); i++)
		bufferSize += (int)segmentPoints[i].size();

	// Compute the starting line number for each segment
	segmentStartLines = new int[segmentNos + 1];

	bm = new BufferManager(bufferSize * 8);
	vector<LineSegment> lines;


#define CIRCLE_MIN_LINE_LEN 6

	for (int i = 0; i < segmentNos; i++) {

		// Make note of the starting line number for this segment
		segmentStartLines[i] = (int)lines.size();

		int noPixels = (int)segmentPoints[i].size();

		if (noPixels < 2 * CIRCLE_MIN_LINE_LEN)
			continue;

		double *x = bm->getX();
		double *y = bm->getY();

		for (int j = 0; j<noPixels; j++) {
			x[j] = segmentPoints[i][j].x;
			y[j] = segmentPoints[i][j].y;
		} //end-for

		  // If the segment is reasonably long, then see if the segment traverses the boundary of a closed shape
		if (noPixels >= 4 * CIRCLE_MIN_LINE_LEN) {
			// If the end-points of the segment is close to each other, then assume a circular/elliptic structure
			double dx = x[0] - x[noPixels - 1];
			double dy = y[0] - y[noPixels - 1];
			double d = sqrt(dx*dx + dy*dy);
			double r = noPixels / TWOPI;      // Assume a complete circle

			double maxDistanceBetweenEndPoints = MAX(3, r / 4);

			// If almost closed loop, then try to fit a circle/ellipse
			if (d <= maxDistanceBetweenEndPoints) {
				double xc, yc, r, circleFitError = 1e10;

				CircleFit(x, y, noPixels, &xc, &yc, &r, &circleFitError);

				EllipseEquation eq;
				double ellipseFitError = 1e10;

				if (circleFitError > LONG_ARC_ERROR) {
					// Try fitting an ellipse
					if (EllipseFit(x, y, noPixels, &eq))
						ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);
				} //end-if

				if (circleFitError <= LONG_ARC_ERROR) {
					addCircle(circles1, noCircles1, xc, yc, r, circleFitError, x, y, noPixels);
					bm->move(noPixels);
					continue;

				}
				else if (ellipseFitError <= ELLIPSE_ERROR) {
					double major, minor;
					ComputeEllipseCenterAndAxisLengths(&eq, &xc, &yc, &major, &minor);

					// Assume major is longer. Otherwise, swap
					if (minor > major) { double tmp = major; major = minor; minor = tmp; }

					if (major < 8 * minor) {
						addCircle(circles1, noCircles1, xc, yc, r, circleFitError, &eq, ellipseFitError, x, y, noPixels);
						bm->move(noPixels);
					} //end-if

					continue;
				} //end-else
			} //end-if 
		} //end-if

		  // Otherwise, split to lines
		EDLines::SplitSegment2Lines(x, y, noPixels, i, lines);
	}

	segmentStartLines[segmentNos] = (int)lines.size();

	// ------------------------------- DETECT ARCS ---------------------------------

	info = new Info[lines.size()];

	// Compute the angle information for each line segment
	for (int i = 0; i<segmentNos; i++) {
		for (int j = segmentStartLines[i]; j<segmentStartLines[i + 1]; j++) {
			LineSegment *l1 = &lines[j];
			LineSegment *l2;

			if (j == segmentStartLines[i + 1] - 1) l2 = &lines[segmentStartLines[i]];
			else                               l2 = &lines[j + 1];

#if 1
			// If the end points of the lines are far from each other, then stop at this line
			double dx = l1->ex - l2->sx;
			double dy = l1->ey - l2->sy;
			double d = sqrt(dx*dx + dy*dy);
			if (d >= 15) { info[j].angle = 10; info[j].sign = 2; info[j].taken = false; continue; }
#endif

			// Compute the angle between the lines & their turn direction
			double v1x = l1->ex - l1->sx;
			double v1y = l1->ey - l1->sy;
			double v1Len = sqrt(v1x*v1x + v1y*v1y);

			double v2x = l2->ex - l2->sx;
			double v2y = l2->ey - l2->sy;
			double v2Len = sqrt(v2x*v2x + v2y*v2y);

			double dotProduct = (v1x*v2x + v1y*v2y) / (v1Len*v2Len);
			if (dotProduct > 1.0) dotProduct = 1.0;
			else if (dotProduct < -1.0) dotProduct = -1.0;

			info[j].angle = acos(dotProduct);
			info[j].sign = (v1x*v2y - v2x*v1y) >= 0 ? 1 : -1;  // compute cross product
			info[j].taken = false;
		} //end-for
	} //end-for

	  // This is how much space we will allocate for circles buffers
	int maxNoOfCircles = (int)lines.size() / 3 + noCircles1 * 2;

	edarcs1 = new EDArcs(maxNoOfCircles);
	DetectArcs(lines);    // Detect all arcs

						  // Try to join arcs that are almost perfectly circular. 
						  // Use the distance between the arc end-points as a metric in choosing in choosing arcs to join
	edarcs2 = new EDArcs(maxNoOfCircles);
	JoinArcs1();

	// Try to join arcs that belong to the same segment
	edarcs3 = new EDArcs(maxNoOfCircles);
	JoinArcs2();

	// Try to combine arcs that belong to different segments
	edarcs4 = new EDArcs(maxNoOfCircles);     // The remaining arcs
	JoinArcs3();

	// Finally, go over the arcs & circles, and generate candidate circles
	GenerateCandidateCircles();

	// EDCircles does not use validation when constructed wia EDColor object.
	// TODO :: apply validation to color image 
	
	//----------------------------- VALIDATE CIRCLES --------------------------
	//noCircles2 = 0;
	//circles2 = new Circle[maxNoOfCircles];
	//GaussianBlur(srcImage, smoothImage, Size(), 0.50); // calculate kernel from sigma;

	//ValidateCircles();

	//----------------------------- JOIN CIRCLES --------------------------
	//noCircles3 = 0;
	//circles3 = new Circle[maxNoOfCircles];
	//JoinCircles();


	noCircles = 0;
	noEllipses = 0;
	for (int i = 0; i < noCircles1; i++) {
		if (circles1[i].isEllipse)
			noEllipses++;
		else
			noCircles++;
	}

	for (int i = 0; i<noCircles1; i++) {
		if (circles1[i].isEllipse) {
			EllipseEquation eq = circles1[i].eq;
			double xc;
			double yc;
			double a;
			double b;
			double theta = ComputeEllipseCenterAndAxisLengths(&eq, &xc, &yc, &a, &b);
			ellipses.push_back(mEllipse(Point2d(xc, yc), Size2d(a, b), theta));

		}
		else {
			double r = circles1[i].r;
			double xc = circles1[i].xc;
			double yc = circles1[i].yc;

			circles.push_back(mCircle(Point2d(xc, yc), r));

		} //end-else
	} //end-for


	  // clean up
	delete edarcs1;
	delete edarcs2;
	delete edarcs3;
	delete edarcs4;

	delete[] circles1;
	//delete[] circles2;
	//delete[] circles3;

	delete bm;
	delete[] segmentStartLines;
	delete[] info;
}

cv::Mat EDCircles::drawResult(bool onImage, ImageStyle style)
{
	Mat colorImage;
	if (onImage)
	{
		colorImage = Mat(height, width, CV_8UC1, srcImg);
		cvtColor(colorImage, colorImage, COLOR_GRAY2BGR);
	}
	else 
		colorImage = Mat(height, width, CV_8UC3, Scalar(0,0,0));
	
	//Circles will be indicated in green
	if (style == ImageStyle::CIRCLES || style == ImageStyle::BOTH)
		for (int i = 0; i < circles.size(); i++)
			circle(colorImage, circles[i].center, (int)circles[i].r, Scalar(0, 255, 0), 1, LINE_AA);
	
	//Ellipses will be indicated in red
	if (style == ImageStyle::ELLIPSES|| style == ImageStyle::BOTH)
		for (int i = 0; i < ellipses.size(); i++) 
		{
			double degree = (ellipses[i].theta * 180) / PI; // convert radian to degree (opencv's ellipse function works with degree)
			ellipse(colorImage, ellipses[i].center, ellipses[i].axes, degree, 0.0, 360.0, Scalar(0, 0, 255), 1, LINE_AA);
		}

	return colorImage;
}


vector<mCircle> EDCircles::getCircles()
{
	return circles;
}

vector<mEllipse> EDCircles::getEllipses()
{
	return ellipses;
}

int EDCircles::getCirclesNo()
{
	return noCircles;
}

int EDCircles::getEllipsesNo()
{
	return noEllipses;
}

void EDCircles::GenerateCandidateCircles()
{
	// Now, go over the circular arcs & add them to circles1
	MyArc *arcs = edarcs4->arcs;
	for (int i = 0; i<edarcs4->noArcs; i++) {
		if (arcs[i].isEllipse) {
			// Ellipse
			if (arcs[i].coverRatio >= CANDIDATE_ELLIPSE_RATIO && arcs[i].ellipseFitError <= ELLIPSE_ERROR) {
				addCircle(circles1, noCircles1, arcs[i].xc, arcs[i].yc, arcs[i].r, arcs[i].circleFitError, &arcs[i].eq, arcs[i].ellipseFitError,
					arcs[i].x, arcs[i].y, arcs[i].noPixels);

			}
			else {
				//        double coverRatio = arcs[i].coverRatio;
				double coverRatio = MAX(ArcLength(arcs[i].sTheta, arcs[i].eTheta) / TWOPI, arcs[i].coverRatio);
				if ((coverRatio >= FULL_CIRCLE_RATIO && arcs[i].circleFitError <= LONG_ARC_ERROR) ||
					(coverRatio >= HALF_CIRCLE_RATIO && arcs[i].circleFitError <= HALF_ARC_ERROR) ||
					(coverRatio >= CANDIDATE_CIRCLE_RATIO2 && arcs[i].circleFitError <= SHORT_ARC_ERROR)) {
					addCircle(circles1, noCircles1, arcs[i].xc, arcs[i].yc, arcs[i].r, arcs[i].circleFitError, arcs[i].x, arcs[i].y, arcs[i].noPixels);
				} //end-if
			} //end-else

		}
		else {
			// If a very short arc, ignore
			if (arcs[i].coverRatio < CANDIDATE_CIRCLE_RATIO1) continue;

			// If the arc is long enough and the circleFitError is small enough, assume a circle
			if ((arcs[i].coverRatio >= FULL_CIRCLE_RATIO && arcs[i].circleFitError <= LONG_ARC_ERROR) ||
				(arcs[i].coverRatio >= HALF_CIRCLE_RATIO && arcs[i].circleFitError <= HALF_ARC_ERROR) ||
				(arcs[i].coverRatio >= CANDIDATE_CIRCLE_RATIO2 && arcs[i].circleFitError <= SHORT_ARC_ERROR)){

				addCircle(circles1, noCircles1, arcs[i].xc, arcs[i].yc, arcs[i].r, arcs[i].circleFitError, arcs[i].x, arcs[i].y, arcs[i].noPixels);

				continue;
			} //end-if

			if (arcs[i].coverRatio < CANDIDATE_CIRCLE_RATIO2) continue;

			// Circle is not possible. Try an ellipse
			EllipseEquation eq;
			double ellipseFitError = 1e10;
			double coverRatio;

			int noPixels = arcs[i].noPixels;
			if (EllipseFit(arcs[i].x, arcs[i].y, noPixels, &eq)) {
				ellipseFitError = ComputeEllipseError(&eq, arcs[i].x, arcs[i].y, noPixels);
				coverRatio = noPixels / computeEllipsePerimeter(&eq);
			} //end-if

			if (arcs[i].coverRatio > coverRatio) coverRatio = arcs[i].coverRatio;

			if (coverRatio >= CANDIDATE_ELLIPSE_RATIO && ellipseFitError <= ELLIPSE_ERROR) {
				addCircle(circles1, noCircles1, arcs[i].xc, arcs[i].yc, arcs[i].r, arcs[i].circleFitError, &eq, ellipseFitError, arcs[i].x, arcs[i].y, arcs[i].noPixels);
			} //end-if
			} //end-else
		} //end-for
}

void EDCircles::DetectArcs(vector<LineSegment> lines)
{

	double maxLineLengthThreshold = MAX(width, height) / 5;

	double MIN_ANGLE = PI / 30;  // 6 degrees
	double MAX_ANGLE = PI / 3;   // 60 degrees

	for (int iter = 1; iter <= 2; iter++) {
		if (iter == 2) MAX_ANGLE = PI / 1.9;  // 95 degrees
											  //    if (iter == 2) MAX_ANGLE = PI/2.25;  // 80 degrees

		for (int curSegmentNo = 0; curSegmentNo<segmentNos; curSegmentNo++) {
			int firstLine = segmentStartLines[curSegmentNo];
			int stopLine = segmentStartLines[curSegmentNo + 1];

			// We need at least 2 line segments
			if (stopLine - firstLine <= 1) continue;

			// Process the info for the lines of this segment
			while (firstLine < stopLine - 1) {
				// If the line is already taken during the previous step, continue
				if (info[firstLine].taken) { firstLine++; continue; }

				// very long lines cannot be part of an arc
				if (lines[firstLine].len >= maxLineLengthThreshold) { firstLine++; continue; }

				// Skip lines that cannot be part of an arc
				if (info[firstLine].angle < MIN_ANGLE || info[firstLine].angle > MAX_ANGLE) { firstLine++; continue; }

				// Find a group of lines (at least 3) with the same sign & angle < MAX_ANGLE degrees
				int lastLine = firstLine + 1;
				while (lastLine < stopLine - 1) {
					if (info[lastLine].taken) break;
					if (info[lastLine].sign != info[firstLine].sign) break;

					if (lines[lastLine].len >= maxLineLengthThreshold) break; // very long lines cannot be part of an arc
					if (info[lastLine].angle < MIN_ANGLE) break;
					if (info[lastLine].angle > MAX_ANGLE) break;

					lastLine++;
				} //end-while

				bool specialCase = false;
				int wrapCase = -1;  // 1: wrap the first two lines with the last line, 2: wrap the last two lines with the first line
#if 0
									// If we do not have 3 lines, then continue
				if (lastLine - firstLine <= 1) { firstLine = lastLine; continue; }
#else
									//        if (lastLine-firstLine == 0){firstLine=lastLine; continue;}
				if (lastLine - firstLine == 1) {
					// Just 2 lines. If long enough, then try to combine. Angle between 15 & 45 degrees. Min. length = 40
					int totalLineLength = lines[firstLine].len + lines[firstLine + 1].len;
					int shorterLen = lines[firstLine].len;
					int longerLen = lines[firstLine + 1].len;

					if (lines[firstLine + 1].len < shorterLen) {
						shorterLen = lines[firstLine + 1].len;
						longerLen = lines[firstLine].len;
					} //end-if

					if (info[firstLine].angle >= PI / 12 && info[firstLine].angle <= PI / 4 && totalLineLength >= 40 && shorterLen * 2 >= longerLen) {
						specialCase = true;
					} //end-if

					  // If the two lines do not make up for arc generation, then try to wrap the lines to the first OR last line.
					  // There are two wrapper cases: 
					if (specialCase == false) {
						// Case 1: Combine the first two lines with the last line of the segment
						if (firstLine == segmentStartLines[curSegmentNo] && info[stopLine - 1].angle >= MIN_ANGLE && info[stopLine - 1].angle <= MAX_ANGLE) {
							wrapCase = 1;
							specialCase = true;
						} //end-if            

						  // Case 2: Combine the last two lines with the first line of the segment
						else if (lastLine == stopLine - 1 && info[lastLine].angle >= MIN_ANGLE && info[lastLine].angle <= MAX_ANGLE) {
							wrapCase = 2;
							specialCase = true;
						} //end-if            
					} // end-if

					  // If still not enough for arc generation, then skip
					if (specialCase == false) {
						firstLine = lastLine;
						continue;
					} //end-else
				} //end-if
#endif

				  // Copy the pixels of this segment to an array
				int noPixels = 0;
				double *x = bm->getX();
				double *y = bm->getY();

				// wrapCase 1: Combine the first two lines with the last line of the segment
				if (wrapCase == 1) {
					int index = lines[stopLine - 1].firstPixelIndex;

					for (int n = 0; n<lines[stopLine - 1].len; n++) {
						x[noPixels] = segmentPoints[curSegmentNo][index + n].x;
						y[noPixels] = segmentPoints[curSegmentNo][index + n].y;
						noPixels++;
					} //end-for
				} //end-if

				for (int m = firstLine; m <= lastLine; m++) {
					int index = lines[m].firstPixelIndex;

					for (int n = 0; n<lines[m].len; n++) {
						x[noPixels] = segmentPoints[curSegmentNo][index + n].x;
						y[noPixels] = segmentPoints[curSegmentNo][index + n].y;
						noPixels++;
					} //end-for
				} //end-for

				  // wrapCase 2: Combine the last two lines with the first line of the segment
				if (wrapCase == 2) {
					int index = lines[segmentStartLines[curSegmentNo]].firstPixelIndex;

					for (int n = 0; n<lines[segmentStartLines[curSegmentNo]].len; n++) {
						x[noPixels] = segmentPoints[curSegmentNo][index + n].x;
						y[noPixels] = segmentPoints[curSegmentNo][index + n].y;
						noPixels++;
					} //end-for
				} //end-if

				  // Move buffer pointers
				bm->move(noPixels);

				// Try to fit a circle to the entire arc of lines
				double xc, yc, radius, circleFitError;
				CircleFit(x, y, noPixels, &xc, &yc, &radius, &circleFitError);

				double coverage = noPixels / (TWOPI*radius);

				// In the case of the special case, the arc must cover at least 22.5 degrees
				if (specialCase && coverage < 1.0 / 16) { info[firstLine].taken = true; firstLine = lastLine; continue; }

				// If only 3 lines, use the SHORT_ARC_ERROR
				double MYERROR = SHORT_ARC_ERROR;
				if (lastLine - firstLine >= 3) MYERROR = LONG_ARC_ERROR;
				if (circleFitError <= MYERROR) {
					// Add this to the list of arcs
					if (wrapCase == 1) {
						x += lines[stopLine - 1].len;
						y += lines[stopLine - 1].len;
						noPixels -= lines[stopLine - 1].len;

					}
					else if (wrapCase == 2) {
						noPixels -= lines[segmentStartLines[curSegmentNo]].len;
					} //end-else

					if ((coverage >= FULL_CIRCLE_RATIO && circleFitError <= LONG_ARC_ERROR)) {
						addCircle(circles1, noCircles1, xc, yc, radius, circleFitError, x, y, noPixels);

					}
					else {
						double sTheta, eTheta;
						ComputeStartAndEndAngles(xc, yc, radius, x, y, noPixels, &sTheta, &eTheta);

						addArc(edarcs1->arcs, edarcs1->noArcs, xc, yc, radius, circleFitError, sTheta, eTheta, info[firstLine].sign, curSegmentNo,
							(int)x[0], (int)y[0], (int)x[noPixels - 1], (int)y[noPixels - 1], x, y, noPixels);
					} //end-else

					for (int m = firstLine; m<lastLine; m++) info[m].taken = true;
					firstLine = lastLine;
					continue;
				} //end-if

				  // Check if this is an almost closed loop (i.e, if 60% of the circle is present). If so, try to fit an ellipse to the entire arc of lines
				double dx = x[0] - x[noPixels - 1];
				double dy = y[0] - y[noPixels - 1];
				double distanceBetweenEndPoints = sqrt(dx*dx + dy*dy);

				bool isAlmostClosedLoop = (distanceBetweenEndPoints <= 1.72*radius && coverage >= FULL_CIRCLE_RATIO);
				if (isAlmostClosedLoop || (iter == 1 && coverage >= 0.25)) {  // an arc covering at least 90 degrees
					EllipseEquation eq;
					double ellipseFitError = 1e10;

					bool valid = EllipseFit(x, y, noPixels, &eq);
					if (valid) ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);

					MYERROR = ELLIPSE_ERROR;
					if (isAlmostClosedLoop == false) MYERROR = 0.75;

					if (ellipseFitError <= MYERROR) {
						// Add this to the list of arcs
						if (wrapCase == 1) {
							x += lines[stopLine - 1].len;
							y += lines[stopLine - 1].len;
							noPixels -= lines[stopLine - 1].len;

						}
						else if (wrapCase == 2) {
							noPixels -= lines[segmentStartLines[curSegmentNo]].len;
						} //end-else

						if (isAlmostClosedLoop) {
							addCircle(circles1, noCircles1, xc, yc, radius, circleFitError, &eq, ellipseFitError, x, y, noPixels);  // Add an ellipse for validation

						}
						else {
							double sTheta, eTheta;
							ComputeStartAndEndAngles(xc, yc, radius, x, y, noPixels, &sTheta, &eTheta);

							addArc(edarcs1->arcs, edarcs1->noArcs,  xc, yc, radius, circleFitError, sTheta, eTheta, info[firstLine].sign, curSegmentNo, &eq, ellipseFitError,
								(int)x[0], (int)y[0], (int)x[noPixels - 1], (int)y[noPixels - 1], x, y, noPixels);
						} //end-else

						for (int m = firstLine; m<lastLine; m++) info[m].taken = true;
						firstLine = lastLine;
						continue;
					} //end-if
				} //end-if

				if (specialCase) { info[firstLine].taken = true; firstLine = lastLine; continue; }

				// Continue until we finish all lines that belong to arc of lines
				while (firstLine <= lastLine - 2) {
					// Fit an initial arc and extend it
					int curLine = firstLine + 2;

					// Fit a circle to the pixels of these lines and see if the error is less than a threshold
					double XC, YC, R, Error = 1e10;
					bool found = false;

					noPixels = 0;
					while (curLine <= lastLine) {
						noPixels = 0;
						for (int m = firstLine; m <= curLine; m++) noPixels += lines[m].len;

						// Fit circle
						CircleFit(x, y, noPixels, &XC, &YC, &R, &Error);
						if (Error <= SHORT_ARC_ERROR) { found = true; break; } // found if the error is smaller than the threshold

																			   // Not found. Move to the next set of lines
						x += lines[firstLine].len;
						y += lines[firstLine].len;

						firstLine++;
						curLine++;
					} //end-while

					  // If no initial arc found, then we are done with this arc of lines
					if (!found) break;

					// If we found an initial arc, then extend it
					for (int m = curLine - 2; m <= curLine; m++) info[m].taken = true;
					curLine++;
					while (curLine <= lastLine) {
						int index = lines[curLine].firstPixelIndex;
						int noPixelsSave = noPixels;

						noPixels += lines[curLine].len;

						double xc, yc, r, error;
						CircleFit(x, y, noPixels, &xc, &yc, &r, &error);
						if (error > LONG_ARC_ERROR) { noPixels = noPixelsSave; break; }   // Adding this line made the error big. So, we do not use this line

																						  // OK. Longer arc
						XC = xc;
						YC = yc;
						R = r;
						Error = error;

						info[curLine].taken = true;
						curLine++;
					} //end-while

					double coverage = noPixels / (TWOPI*radius);
					if ((coverage >= FULL_CIRCLE_RATIO && circleFitError <= LONG_ARC_ERROR)) {
						addCircle(circles1, noCircles1, XC, YC, R, Error, x, y, noPixels);

					}
					else {
						// Add this to the list of arcs
						double sTheta, eTheta;
						ComputeStartAndEndAngles(XC, YC, R, x, y, noPixels, &sTheta, &eTheta);

						addArc(edarcs1->arcs, edarcs1->noArcs, XC, YC, R, Error, sTheta, eTheta, info[firstLine].sign, curSegmentNo,
							(int)x[0], (int)y[0], (int)x[noPixels - 1], (int)y[noPixels - 1], x, y, noPixels);
					} //end-else

					x += noPixels;
					y += noPixels;

					firstLine = curLine;
				} //end-while-current-arc-of-lines

				firstLine = lastLine;
			} //end-while-entire-edge-segment
		} //end-for
	} //end-for-iter
}

//-----------------------------------------------------------------
// Go over all circles & ellipses and validate them
// The idea here is to look at all pixels of a circle/ellipse
// rather than only the pixels of the lines making up the circle/ellipse
//
void EDCircles::ValidateCircles()
{
	double prec = PI / 16;  // Alignment precision
	double prob = 1.0 / 8;  // probability of alignment

	int points_buffer_size = 8 * (width + height);
	double *px = new double[points_buffer_size];
	double *py = new double[points_buffer_size];

	// logNT & LUT for NFA computation
	double logNT = 2 * log10((double)(width*height)) + log10((double)(width + height));

	int lutSize = (width + height) / 8;
	nfa = new NFALUT(lutSize, prob, logNT); // create look up table

	// Validate circles & ellipses
	bool validateAgain;
	int count = 0;
	for (int i = 0; i<noCircles1; ) {
		Circle *circle = &circles1[i];
		double xc = circle->xc;
		double yc = circle->yc;
		double radius = circle->r;

		// Skip potential invalid circles (sometimes these kinds of candidates get generated!)
		if (radius > MAX(width, height)) { i++; continue; }

		validateAgain = false;

		int noPoints = (int)(computeEllipsePerimeter(&circle->eq));

		if (noPoints > points_buffer_size)
		{
			i++;
			continue;
		}

		if (circle->isEllipse) {
			if (noPoints % 2) noPoints--;
			ComputeEllipsePoints(circle->eq.coeff, px, py, noPoints);

		}
		else {
			ComputeCirclePoints(xc, yc, radius, px, py, &noPoints);
		} //end-else

		int pr = -1;  // previous row
		int pc = -1;  // previous column

		int tr = -100;
		int tc = -100;
		int tcount = 0;

		int noPeripheryPixels = 0;
		int noEdgePixels = 0;
		int aligned = 0;
		for (int j = 0; j<noPoints; j++) {
			int r = (int)(py[j] + 0.5);
			int c = (int)(px[j] + 0.5);

			if (r == pr && c == pc) continue;
			noPeripheryPixels++;

			if (r <= 0 || r >= height - 1) continue;
			if (c <= 0 || c >= width - 1) continue;

			pr = r;
			pc = c;

			int dr = abs(r - tr);
			int dc = abs(c - tc);
			if (dr + dc >= 2) {
				tr = r;
				tc = c;
				tcount++;
			} //end-if

#if 1
			  //
			  // See if there is an edge pixel within 1 pixel vicinity
			  //
			if (edgeImg[r*width + c] != 255) {
				//   y-cy=-x-cx    y-cy=x-cx
				//         \       /
				//          \ IV. /
				//           \   / 
				//            \ / 
				//     III.    +   I. quadrant
				//            / \
				//           /   \ 
				//          / II. \
				//         /       \
				//
				// (x, y)-->(x-cx, y-cy)
				//
				
				int x = c;
				int y = r;

				int diff1 = (int)(y - yc - x + xc);
				int diff2 = (int)(y - yc + x - xc);

				if (diff1 < 0) {
					if (diff2 > 0) {
						// I. quadrant
						c = x - 1;
						if (c >= 1 && edgeImg[r*width + c] == 255) goto out;
						c = x + 1;
						if (c<width - 1 && edgeImg[r*width + c] == 255) goto out;

#if 1
						c = x - 2;
						if (c >= 2 && edgeImg[r*width + c] == 255) goto out;
						c = x + 2;
						if (c<width - 2 && edgeImg[r*width + c] == 255) goto out;
#endif
					}
					else {
						// IV. quadrant
						r = y - 1;
						if (r >= 1 && edgeImg[r*width + c] == 255) goto out;
						r = y + 1;
						if (r<height - 1 && edgeImg[r*width + c] == 255) goto out;

#if 1
						r = y - 2;
						if (r >= 2 && edgeImg[r*width + c] == 255) goto out;
						r = y + 2;
						if (r<height - 2 && edgeImg[r*width + c] == 255) goto out;
#endif
					} //end-else

				}
				else {
					if (diff2 > 0) {
						// II. quadrant
						r = y - 1;
						if (r >= 1 && edgeImg[r*width + c] == 255) goto out;
						r = y + 1;
						if (r<height - 1 && edgeImg[r*width + c] == 255) goto out;

#if 1
						r = y - 2;
						if (r >= 2 && edgeImg[r*width + c] == 255) goto out;
						r = y + 2;
						if (r<height - 2 && edgeImg[r*width + c] == 255) goto out;
#endif
					}
					else {
						// III. quadrant
						c = x - 1;
						if (c >= 1 && edgeImg[r*width + c] == 255) goto out;
						c = x + 1;
						if (c<width - 1 && edgeImg[r*width + c] == 255) goto out;

#if 1
						c = x - 2;
						if (c >= 2 && edgeImg[r*width + c] == 255) goto out;
						c = x + 2;
						if (c<width - 2 && edgeImg[r*width + c] == 255) goto out;
#endif
					} //end-else
				} //end-else

				r = pr;
				c = pc;
				continue;  // Ignore non-edge pixels. 
						   // This produces less false positives, but occationally misses on some valid circles
			} //end-if
		out:
			if (edgeImg[r*width + c] == 255) noEdgePixels++;
			//      map->edgeImg[r*width+c] = 254;
#endif

			// compute gx & gy
			int com1 = smoothImg[(r + 1)*width + c + 1] - smoothImg[(r - 1)*width + c - 1];
			int com2 = smoothImg[(r - 1)*width + c + 1] - smoothImg[(r + 1)*width + c - 1];

			int gx = com1 + com2 + smoothImg[r*width + c + 1] - smoothImg[r*width + c - 1];
			int gy = com1 - com2 + smoothImg[(r + 1)*width + c] - smoothImg[(r - 1)*width + c];
			double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);

			double derivX, derivY;
			if (circle->isEllipse) {
				// Ellipse
				derivX = 2 * circle->eq.A()*c + circle->eq.B()*r + circle->eq.D();
				derivY = circle->eq.B()*c + 2 * circle->eq.C()*r + circle->eq.E();

			}
			else {
				// circle
				derivX = c - xc;
				derivY = r - yc;
			} //end-else

			double idealPixelAngle = nfa->myAtan2(derivX, -derivY);
			double diff = fabs(pixelAngle - idealPixelAngle);
			if (diff <= prec || diff >= PI - prec) aligned++;
		} //end-for

		double circumference;
		if (circle->isEllipse) circumference = computeEllipsePerimeter(&circle->eq);
		else                   circumference = TWOPI*radius;

		// Validate by NFA
		bool isValid = nfa->checkValidationByNFA(noPeripheryPixels, aligned);

		if (isValid) {
			circles2[count++] = circles1[i];

		}
		else if (circle->isEllipse == false && circle->coverRatio >= CANDIDATE_ELLIPSE_RATIO) {
			// Fit an ellipse to this circle, and try to revalidate
			double ellipseFitError = 1e10;
			EllipseEquation eq;

			if (EllipseFit(circle->x, circle->y, circle->noPixels, &eq)) {
				ellipseFitError = ComputeEllipseError(&eq, circle->x, circle->y, circle->noPixels);
			} //end-if

			if (ellipseFitError <= ELLIPSE_ERROR) {
				circle->isEllipse = true;
				circle->ellipseFitError = ellipseFitError;
				circle->eq = eq;

				validateAgain = true;
			} //end-if
		} //end-else

		if (validateAgain == false) i++;
	} //end-for

	noCircles2 = count;

	delete[] px;
	delete[] py;
	delete nfa;
}

void EDCircles::JoinCircles()
{
	// Sort the circles wrt their radius
	sortCircle(circles2, noCircles2);

	int noCircles = noCircles2;
	Circle *circles = circles2;

	for (int i = 0; i<noCircles; i++) {
		if (circles[i].isEllipse) {
			ComputeEllipseCenterAndAxisLengths(&circles[i].eq, &circles[i].xc, &circles[i].yc, &circles[i].majorAxisLength, &circles[i].minorAxisLength);
		} //end-if
	} //end-for

	bool *taken = new bool[noCircles];
	for (int i = 0; i<noCircles; i++) taken[i] = false;

	int *candidateCircles = new int[noCircles];
	int noCandidateCircles;

	for (int i = 0; i<noCircles; i++) {
		if (taken[i]) continue;

		// Current arc
		double majorAxisLength, minorAxisLength;

		if (circles[i].isEllipse) {
			majorAxisLength = circles[i].majorAxisLength;
			minorAxisLength = circles[i].minorAxisLength;

		}
		else {
			majorAxisLength = circles[i].r;
			minorAxisLength = circles[i].r;
		} //end-else

		  // Find other circles to join with
		noCandidateCircles = 0;

		for (int j = i + 1; j<noCircles; j++) {
			if (taken[j]) continue;

#define JOINED_SHORT_ARC_ERROR_THRESHOLD  2 // 2.5
#define AXIS_LENGTH_DIFF_THRESHOLD     6 //(JOINED_SHORT_ARC_ERROR_THRESHOLD*2+1)
#define CENTER_DISTANCE_THRESHOLD      12 //(AXIS_LENGTH_DIFF_THRESHOLD*2)

			double dx = circles[i].xc - circles[j].xc;
			double dy = circles[i].yc - circles[j].yc;
			double centerDistance = sqrt(dx*dx + dy*dy);
			if (centerDistance > CENTER_DISTANCE_THRESHOLD) continue;

			double diff1, diff2;
			if (circles[j].isEllipse) {
				diff1 = fabs(majorAxisLength - circles[j].majorAxisLength);
				diff2 = fabs(minorAxisLength - circles[j].minorAxisLength);

			}
			else {
				diff1 = fabs(majorAxisLength - circles[j].r);
				diff2 = fabs(minorAxisLength - circles[j].r);
			} //end-else

			if (diff1 > AXIS_LENGTH_DIFF_THRESHOLD) continue;
			if (diff2 > AXIS_LENGTH_DIFF_THRESHOLD) continue;

			// Add to candidates
			candidateCircles[noCandidateCircles] = j;
			noCandidateCircles++;
		} //end-for

		  // Try to join the current arc with the candidate arc (if there is one)
		double XC = circles[i].xc;
		double YC = circles[i].yc;
		double R = circles[i].r;

		double CircleFitError = circles[i].circleFitError;
		bool CircleFitValid = false;

		EllipseEquation Eq;
		double EllipseFitError;
		bool EllipseFitValid = false;

		if (noCandidateCircles > 0) {
			int noPixels = circles[i].noPixels;
			double *x = bm->getX();
			double *y = bm->getY();
			memcpy(x, circles[i].x, noPixels * sizeof(double));
			memcpy(y, circles[i].y, noPixels * sizeof(double));

			for (int j = 0; j<noCandidateCircles; j++) {
				int CandidateArcNo = candidateCircles[j];

				int noPixelsSave = noPixels;
				memcpy(x + noPixels, circles[CandidateArcNo].x, circles[CandidateArcNo].noPixels * sizeof(double));
				memcpy(y + noPixels, circles[CandidateArcNo].y, circles[CandidateArcNo].noPixels * sizeof(double));
				noPixels += circles[CandidateArcNo].noPixels;

				bool circleFitOK = false;
				if (EllipseFitValid == false && circles[i].isEllipse == false && circles[CandidateArcNo].isEllipse == false) {
					double xc, yc, r, error = 1e10;
					CircleFit(x, y, noPixels, &xc, &yc, &r, &error);

					if (error <= JOINED_SHORT_ARC_ERROR_THRESHOLD) {
						taken[CandidateArcNo] = true;

						XC = xc;
						YC = yc;
						R = r;
						CircleFitError = error;

						circleFitOK = true;
						CircleFitValid = true;
					} //end-if
				} // end-if

				bool ellipseFitOK = false;
				if (circleFitOK == false) {
					// Try to fit an ellipse
					double error = 1e10;
					EllipseEquation eq;
					if (EllipseFit(x, y, noPixels, &eq)) {
						error = ComputeEllipseError(&eq, x, y, noPixels);
					} //end-if

					if (error <= JOINED_SHORT_ARC_ERROR_THRESHOLD) {
						taken[CandidateArcNo] = true;

						Eq = eq;
						EllipseFitError = error;

						ellipseFitOK = true;
						EllipseFitValid = true;
						CircleFitValid = false;
					} // end-if
				} //end-if

				if (circleFitOK == false && ellipseFitOK == false) {
					noPixels = noPixelsSave;
				} //end-if
			} // end-for
		} //end-if

		// Add the new circle/ellipse to circles2
		if (CircleFitValid) {
			addCircle(circles3, noCircles3, XC, YC, R, CircleFitError, NULL, NULL, 0);

		}
		else if (EllipseFitValid) {
			addCircle(circles3, noCircles3, XC, YC, R, CircleFitError, &Eq, EllipseFitError, NULL, NULL, 0);

		}
		else {
			circles3[noCircles3] = circles[i];
			noCircles3++;
		} //end-else
	} //end-for

	delete[] taken;
	delete[] candidateCircles;
}

void EDCircles::JoinArcs1()
{
	AngleSet angles;

	// Sort the arcs with respect to their length so that longer arcs are at the beginning
	sortArc(edarcs1->arcs, edarcs1->noArcs);

	int noArcs = edarcs1->noArcs;
	MyArc *arcs = edarcs1->arcs;

	bool *taken = new bool[noArcs];
	for (int i = 0; i<noArcs; i++) taken[i] = false;

	struct CandidateArc {
		int arcNo;
		int which;    // 1: (SX, SY)-(sx, sy), 2: (SX, SY)-(ex, ey), 3: (EX, EY)-(sx, sy), 4: (EX, EY)-(ex, ey)
		double dist;  // min distance between the end points
	};

	CandidateArc *candidateArcs = new CandidateArc[noArcs];
	int noCandidateArcs;

	for (int i = 0; i<noArcs; i++) {
		if (taken[i]) continue;
		if (arcs[i].isEllipse) { 
			edarcs2->arcs[edarcs2->noArcs++] = arcs[i]; continue; }

		// Current arc
		bool CircleEqValid = false;
		double XC = arcs[i].xc;
		double YC = arcs[i].yc;
		double R = arcs[i].r;
		double CircleFitError = arcs[i].circleFitError;
		int Turn = arcs[i].turn;
		int NoPixels = arcs[i].noPixels;

		int SX = arcs[i].sx;
		int SY = arcs[i].sy;
		int EX = arcs[i].ex;
		int EY = arcs[i].ey;

		// Take the pixels making up this arc
		int noPixels = arcs[i].noPixels;

		double *x = bm->getX();
		double *y = bm->getY();
		memcpy(x, arcs[i].x, noPixels * sizeof(double));
		memcpy(y, arcs[i].y, noPixels * sizeof(double));

		angles.clear();
		angles.set(arcs[i].sTheta, arcs[i].eTheta);

		while (1) {
			bool extendedArc = false;

			// Find other arcs to join with
			noCandidateArcs = 0;

			for (int j = i + 1; j<noArcs; j++) {
				if (taken[j]) continue;
				if (arcs[j].isEllipse) continue;

				double minR = MIN(R, arcs[j].r);
				double maxR = MAX(R, arcs[j].r);
				double radiusDiffThreshold = minR*0.25;

				double diff = fabs(R - arcs[j].r);
				if (diff > radiusDiffThreshold) continue;

				// If 50% of the current arc overlaps with the existing arc, then ignore this arc
				if (angles.overlap(arcs[j].sTheta, arcs[j].eTheta) >= 0.50) continue;

				// Compute the distances
				// 1: (SX, SY)-(sx, sy)
				double dx = SX - arcs[j].sx;
				double dy = SY - arcs[j].sy;
				double d = sqrt(dx*dx + dy*dy);
				int which = 1;

				// 2: (SX, SY)-(ex, ey)
				dx = SX - arcs[j].ex;
				dy = SY - arcs[j].ey;
				double d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 2; }

				// 3: (EX, EY)-(sx, sy)
				dx = EX - arcs[j].sx;
				dy = EY - arcs[j].sy;
				d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 3; }

				// 4: (EX, EY)-(ex, ey)
				dx = EX - arcs[j].ex;
				dy = EY - arcs[j].ey;
				d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 4; }

				// Endpoints must be very close
				double maxDistanceBetweenEndpoints = minR*1.75; //1.5;
				if (d > maxDistanceBetweenEndpoints) continue;

				// This is to give precedence to better matching arc
				d += diff;

				// They have to turn in the same direction
				if (which == 2 || which == 3) {
					if (Turn != arcs[j].turn) continue;
				}
				else {
					if (Turn == arcs[j].turn) continue;
				} //end-else

				  // Add to candidate arcs in sorted order. User insertion sort
				int index = noCandidateArcs - 1;
				while (index >= 0) {
					if (candidateArcs[index].dist < d) break;

					candidateArcs[index + 1] = candidateArcs[index];
					index--;
				} //end-while

				  // Add the new candidate arc to the candidate list
				index++;
				candidateArcs[index].arcNo = j;
				candidateArcs[index].which = which;
				candidateArcs[index].dist = d;
				noCandidateArcs++;
			} //end-for

			  // Try to join the current arc with the candidate arc (if there is one)
			if (noCandidateArcs > 0) {
				for (int j = 0; j<noCandidateArcs; j++) {
					int CandidateArcNo = candidateArcs[j].arcNo;
					int Which = candidateArcs[j].which;

					int noPixelsSave = noPixels;
					memcpy(x + noPixels, arcs[CandidateArcNo].x, arcs[CandidateArcNo].noPixels * sizeof(double));
					memcpy(y + noPixels, arcs[CandidateArcNo].y, arcs[CandidateArcNo].noPixels * sizeof(double));
					noPixels += arcs[CandidateArcNo].noPixels;

					double xc, yc, r, circleFitError;
					CircleFit(x, y, noPixels, &xc, &yc, &r, &circleFitError);

					if (circleFitError > LONG_ARC_ERROR) {
						// No match. Continue with the next candidate
						noPixels = noPixelsSave;

					}
					else {
						// Match. Take it
						extendedArc = true;
						CircleEqValid = true;
						XC = xc;
						YC = yc;
						R = r;
						CircleFitError = circleFitError;
						NoPixels = noPixels;

						taken[CandidateArcNo] = true;
						taken[i] = true;

						angles.set(arcs[CandidateArcNo].sTheta, arcs[CandidateArcNo].eTheta);

						// Update the end points of the new arc
						switch (Which) {
							// (SX, SY)-(sy, sy)
						case 1: SX = EX, SY = EY; EX = arcs[CandidateArcNo].ex; EY = arcs[CandidateArcNo].ey;
							if (Turn == 1) Turn = -1; else Turn = 1; // reverse the turn direction
							break;

							// (SX, SY)-(ex, ey)
						case 2: SX = EX, SY = EY; EX = arcs[CandidateArcNo].sx; EY = arcs[CandidateArcNo].sy;
							if (Turn == 1) Turn = -1; else Turn = 1; // reverse the turn direction
							break;

							// (EX, EY)-(sx, sy)
						case 3: EX = arcs[CandidateArcNo].ex; EY = arcs[CandidateArcNo].ey; break;

							// (EX, EY)-(ex, ey)
						case 4: EX = arcs[CandidateArcNo].sx; EY = arcs[CandidateArcNo].sy; break;
						} //end-switch

						break; // Do not look at the other candidates
					} //end-if
				} //end-for
			} //end-if

			if (extendedArc == false) break;
		} //end-while

		if (CircleEqValid == false) {
			// Add to arcs
			edarcs2->arcs[edarcs2->noArcs++] = arcs[i];

		}
		else {
			// Add the current OR the extended arc to the new arcs
			double sTheta, eTheta;
			angles.computeStartEndTheta(sTheta, eTheta);

			double coverage = ArcLength(sTheta, eTheta) / TWOPI;
			if ((coverage >= FULL_CIRCLE_RATIO && CircleFitError <= LONG_ARC_ERROR))
				addCircle(circles1, noCircles1, XC, YC, R, CircleFitError, x, y, NoPixels);
			else
				addArc(edarcs2->arcs, edarcs2->noArcs,XC, YC, R, CircleFitError, sTheta, eTheta, Turn, arcs[i].segmentNo, SX, SY, EX, EY, x, y, NoPixels, angles.overlapRatio());

			bm->move(NoPixels);
		} //end-if
	} //end-for

	delete[] taken;
	delete[] candidateArcs;
}

void EDCircles::JoinArcs2()
{
	AngleSet angles;

	// Sort the arcs with respect to their length so that longer arcs are at the beginning
	sortArc(edarcs2->arcs, edarcs2->noArcs);

	int noArcs = edarcs2->noArcs;
	MyArc *arcs = edarcs2->arcs;

	bool *taken = new bool[noArcs];
	for (int i = 0; i<noArcs; i++) taken[i] = false;

	struct CandidateArc {
		int arcNo;
		int which;    // 1: (SX, SY)-(sx, sy), 2: (SX, SY)-(ex, ey), 3: (EX, EY)-(sx, sy), 4: (EX, EY)-(ex, ey)
		double dist;  // min distance between the end points
	};

	CandidateArc *candidateArcs = new CandidateArc[noArcs];
	int noCandidateArcs;

	for (int i = 0; i<noArcs; i++) {
		if (taken[i]) continue;

		// Current arc
		bool EllipseEqValid = false;
		EllipseEquation Eq;
		double EllipseFitError;

		double R = arcs[i].r;
		int Turn = arcs[i].turn;
		int NoPixels = arcs[i].noPixels;

		int SX = arcs[i].sx;
		int SY = arcs[i].sy;
		int EX = arcs[i].ex;
		int EY = arcs[i].ey;

		// Take the pixels making up this arc
		int noPixels = arcs[i].noPixels;

		double *x = bm->getX();
		double *y = bm->getY();
		memcpy(x, arcs[i].x, noPixels * sizeof(double));
		memcpy(y, arcs[i].y, noPixels * sizeof(double));

		angles.clear();
		angles.set(arcs[i].sTheta, arcs[i].eTheta);

		while (1) {
			bool extendedArc = false;

			// Find other arcs to join with
			noCandidateArcs = 0;

			for (int j = i + 1; j<noArcs; j++) {
				if (taken[j]) continue;
				if (arcs[j].segmentNo != arcs[i].segmentNo) continue;
				if (arcs[j].turn != Turn) continue;

				double minR = MIN(R, arcs[j].r);
				double radiusDiffThreshold = minR*2.5;

				double diff = fabs(R - arcs[j].r);
				if (diff > radiusDiffThreshold) continue;

				// If 75% of the current arc overlaps with the existing arc, then ignore this arc
				if (angles.overlap(arcs[j].sTheta, arcs[j].eTheta) >= 0.75) continue;

				// Compute the distances
				// 1: (SX, SY)-(sx, sy)
				double dx = SX - arcs[j].sx;
				double dy = SY - arcs[j].sy;
				double d = sqrt(dx*dx + dy*dy);
				int which = 1;

				// 2: (SX, SY)-(ex, ey)
				dx = SX - arcs[j].ex;
				dy = SY - arcs[j].ey;
				double d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 2; }

				// 3: (EX, EY)-(sx, sy)
				dx = EX - arcs[j].sx;
				dy = EY - arcs[j].sy;
				d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 3; }

				// 4: (EX, EY)-(ex, ey)
				dx = EX - arcs[j].ex;
				dy = EY - arcs[j].ey;
				d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 4; }

				// Endpoints must be very close
				double maxDistanceBetweenEndpoints = 5; //10; 
				if (d > maxDistanceBetweenEndpoints) continue;

				// Add to candidate arcs in sorted order. User insertion sort
				int index = noCandidateArcs - 1;
				while (index >= 0) {
					if (candidateArcs[index].dist < d) break;

					candidateArcs[index + 1] = candidateArcs[index];
					index--;
				} //end-while

				  // Add the new candidate arc to the candidate list
				index++;
				candidateArcs[index].arcNo = j;
				candidateArcs[index].which = which;
				candidateArcs[index].dist = d;
				noCandidateArcs++;
			} //end-for

			  // Try to join the current arc with the candidate arc (if there is one)
			if (noCandidateArcs > 0) {
				for (int j = 0; j<noCandidateArcs; j++) {
					int CandidateArcNo = candidateArcs[j].arcNo;
					int Which = candidateArcs[j].which;

					int noPixelsSave = noPixels;
					memcpy(x + noPixels, arcs[CandidateArcNo].x, arcs[CandidateArcNo].noPixels * sizeof(double));
					memcpy(y + noPixels, arcs[CandidateArcNo].y, arcs[CandidateArcNo].noPixels * sizeof(double));
					noPixels += arcs[CandidateArcNo].noPixels;

					// Directly fit an ellipse
					EllipseEquation eq;
					double ellipseFitError = 1e10;
					if (EllipseFit(x, y, noPixels, &eq)) ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);

					if (ellipseFitError > ELLIPSE_ERROR) {
						// No match. Continue with the next candidate
						noPixels = noPixelsSave;

					}
					else {
						// Match. Take it
						extendedArc = true;
						EllipseEqValid = true;
						Eq = eq;
						EllipseFitError = ellipseFitError;
						NoPixels = noPixels;

						taken[CandidateArcNo] = true;
						taken[i] = true;

						R = (R + arcs[CandidateArcNo].r) / 2.0;

						angles.set(arcs[CandidateArcNo].sTheta, arcs[CandidateArcNo].eTheta);

						// Update the end points of the new arc
						switch (Which) {
							// (SX, SY)-(sy, sy)
						case 1: SX = EX, SY = EY; EX = arcs[CandidateArcNo].ex; EY = arcs[CandidateArcNo].ey;
							if (Turn == 1) Turn = -1; else Turn = 1; // reverse the turn direction
							break;

							// (SX, SY)-(ex, ey)
						case 2: SX = EX, SY = EY; EX = arcs[CandidateArcNo].sx; EY = arcs[CandidateArcNo].sy;
							if (Turn == 1) Turn = -1; else Turn = 1; // reverse the turn direction
							break;

							// (EX, EY)-(sx, sy)
						case 3: EX = arcs[CandidateArcNo].ex; EY = arcs[CandidateArcNo].ey; break;

							// (EX, EY)-(ex, ey)
						case 4: EX = arcs[CandidateArcNo].sx; EY = arcs[CandidateArcNo].sy; break;
						} //end-switch

						break; // Do not look at the other candidates
					} //end-if
				} //end-for
			} //end-if

			if (extendedArc == false) break;
		} //end-while

		if (EllipseEqValid == false) {
			// Add to arcs
			edarcs3->arcs[edarcs3->noArcs++] = arcs[i];

		}
		else {
			// Add the current OR the extended arc to the new arcs
			double sTheta, eTheta;
			angles.computeStartEndTheta(sTheta, eTheta);

			double XC, YC, R, CircleFitError;
			CircleFit(x, y, NoPixels, &XC, &YC, &R, &CircleFitError);

			double coverage = ArcLength(sTheta, eTheta) / TWOPI;
			if ((coverage >= FULL_CIRCLE_RATIO && CircleFitError <= LONG_ARC_ERROR))
				addCircle(circles1, noCircles1, XC, YC, R, CircleFitError, x, y, NoPixels);
			else
				addArc(edarcs3->arcs, edarcs3->noArcs, XC, YC, R, CircleFitError, sTheta, eTheta, Turn, arcs[i].segmentNo, &Eq, EllipseFitError, SX, SY, EX, EY, x, y, NoPixels, angles.overlapRatio());

			// Move buffer pointers
			bm->move(NoPixels);
		} //end-if
	} //end-for

	delete[] taken;
	delete[] candidateArcs;
}

void EDCircles::JoinArcs3()
{
	AngleSet angles;

	// Sort the arcs with respect to their length so that longer arcs are at the beginning
	sortArc(edarcs3->arcs, edarcs3->noArcs);

	int noArcs = edarcs3->noArcs;
	MyArc *arcs = edarcs3->arcs;

	bool *taken = new bool[noArcs];
	for (int i = 0; i<noArcs; i++) taken[i] = false;

	struct CandidateArc {
		int arcNo;
		int which;    // 1: (SX, SY)-(sx, sy), 2: (SX, SY)-(ex, ey), 3: (EX, EY)-(sx, sy), 4: (EX, EY)-(ex, ey)
		double dist;  // min distance between the end points
	};

	CandidateArc *candidateArcs = new CandidateArc[noArcs];
	int noCandidateArcs;

	for (int i = 0; i<noArcs; i++) {
		if (taken[i]) continue;

		// Current arc
		bool EllipseEqValid = false;
		EllipseEquation Eq;
		double EllipseFitError;

		double R = arcs[i].r;
		int Turn = arcs[i].turn;
		int NoPixels = arcs[i].noPixels;

		int SX = arcs[i].sx;
		int SY = arcs[i].sy;
		int EX = arcs[i].ex;
		int EY = arcs[i].ey;

		// Take the pixels making up this arc
		int noPixels = arcs[i].noPixels;

		double *x = bm->getX();
		double *y = bm->getY();
		memcpy(x, arcs[i].x, noPixels * sizeof(double));
		memcpy(y, arcs[i].y, noPixels * sizeof(double));

		angles.clear();
		angles.set(arcs[i].sTheta, arcs[i].eTheta);

		while (1) {
			bool extendedArc = false;

			// Find other arcs to join with
			noCandidateArcs = 0;

			for (int j = i + 1; j<noArcs; j++) {
				if (taken[j]) continue;


				/******************************************************************
				* It seems that for minimum false detections,
				* radiusDiffThreshold =  minR*0.5 & maxDistanceBetweenEndpoints = minR*0.75.
				* But these parameters results in many valid misses too!
				******************************************************************/

				double minR = MIN(R, arcs[j].r);
				double diff = fabs(R - arcs[j].r);
				if (diff > minR) continue;

				// If 50% of the current arc overlaps with the existing arc, then ignore this arc
				if (angles.overlap(arcs[j].sTheta, arcs[j].eTheta) >= 0.50) continue;

				// Compute the distances
				// 1: (SX, SY)-(sx, sy)
				double dx = SX - arcs[j].sx;
				double dy = SY - arcs[j].sy;
				double d = sqrt(dx*dx + dy*dy);
				int which = 1;

				// 2: (SX, SY)-(ex, ey)
				dx = SX - arcs[j].ex;
				dy = SY - arcs[j].ey;
				double d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 2; }

				// 3: (EX, EY)-(sx, sy)
				dx = EX - arcs[j].sx;
				dy = EY - arcs[j].sy;
				d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 3; }

				// 4: (EX, EY)-(ex, ey)
				dx = EX - arcs[j].ex;
				dy = EY - arcs[j].ey;
				d2 = sqrt(dx*dx + dy*dy);
				if (d2 < d) { d = d2; which = 4; }

				// Endpoints must be very close
				if (diff <= 0.50*minR) { if (d > minR*0.75) continue; }
				else if (diff <= 0.75*minR) { if (d > minR*0.50) continue; }
				else if (diff <= 1.00*minR) { if (d > minR*0.25) continue; }
				else continue;

				// This is to allow more circular arcs a precedence
				d += diff;

				// They have to turn in the same direction
				if (which == 2 || which == 3) {
					if (Turn != arcs[j].turn) continue;
				}
				else {
					if (Turn == arcs[j].turn) continue;
				} //end-else

				  // Add to candidate arcs in sorted order. User insertion sort
				int index = noCandidateArcs - 1;
				while (index >= 0) {
					if (candidateArcs[index].dist < d) break;

					candidateArcs[index + 1] = candidateArcs[index];
					index--;
				} //end-while

				  // Add the new candidate arc to the candidate list
				index++;
				candidateArcs[index].arcNo = j;
				candidateArcs[index].which = which;
				candidateArcs[index].dist = d;
				noCandidateArcs++;
			} //end-for

			  // Try to join the current arc with the candidate arc (if there is one)
			if (noCandidateArcs > 0) {
				for (int j = 0; j<noCandidateArcs; j++) {
					int CandidateArcNo = candidateArcs[j].arcNo;
					int Which = candidateArcs[j].which;

					int noPixelsSave = noPixels;
					memcpy(x + noPixels, arcs[CandidateArcNo].x, arcs[CandidateArcNo].noPixels * sizeof(double));
					memcpy(y + noPixels, arcs[CandidateArcNo].y, arcs[CandidateArcNo].noPixels * sizeof(double));
					noPixels += arcs[CandidateArcNo].noPixels;

					// Directly fit an ellipse
					EllipseEquation eq;
					double ellipseFitError = 1e10;
					if (EllipseFit(x, y, noPixels, &eq)) ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);

					if (ellipseFitError > ELLIPSE_ERROR) {
						// No match. Continue with the next candidate
						noPixels = noPixelsSave;

					}
					else {
						// Match. Take it
						extendedArc = true;
						EllipseEqValid = true;
						Eq = eq;
						EllipseFitError = ellipseFitError;
						NoPixels = noPixels;

						taken[CandidateArcNo] = true;
						taken[i] = true;

						R = (R + arcs[CandidateArcNo].r) / 2.0;

						angles.set(arcs[CandidateArcNo].sTheta, arcs[CandidateArcNo].eTheta);

						// Update the end points of the new arc
						switch (Which) {
							// (SX, SY)-(sy, sy)
						case 1: SX = EX, SY = EY; EX = arcs[CandidateArcNo].ex; EY = arcs[CandidateArcNo].ey;
							if (Turn == 1) Turn = -1; else Turn = 1; // reverse the turn direction
							break;

							// (SX, SY)-(ex, ey)
						case 2: SX = EX, SY = EY; EX = arcs[CandidateArcNo].sx; EY = arcs[CandidateArcNo].sy;
							if (Turn == 1) Turn = -1; else Turn = 1; // reverse the turn direction
							break;

							// (EX, EY)-(sx, sy)
						case 3: EX = arcs[CandidateArcNo].ex; EY = arcs[CandidateArcNo].ey; break;

							// (EX, EY)-(ex, ey)
						case 4: EX = arcs[CandidateArcNo].sx; EY = arcs[CandidateArcNo].sy; break;
						} //end-switch

						break; // Do not look at the other candidates
					} //end-if
				} //end-for
			} //end-if

			if (extendedArc == false) break;
		} //end-while

		if (EllipseEqValid == false) {
			// Add to arcs
			edarcs4->arcs[edarcs4->noArcs++] = arcs[i];

		}
		else {
			// Add the current OR the extended arc to the new arcs
			double sTheta, eTheta;
			angles.computeStartEndTheta(sTheta, eTheta);

			double XC, YC, R, CircleFitError;
			CircleFit(x, y, NoPixels, &XC, &YC, &R, &CircleFitError);

			double coverage = ArcLength(sTheta, eTheta) / TWOPI;
			if ((coverage >= FULL_CIRCLE_RATIO && CircleFitError <= LONG_ARC_ERROR))
				addCircle(circles1, noCircles1, XC, YC, R, CircleFitError, x, y, NoPixels);
			else
				addArc(edarcs4->arcs, edarcs4->noArcs, XC, YC, R, CircleFitError, sTheta, eTheta, Turn, arcs[i].segmentNo, &Eq, EllipseFitError, SX, SY, EX, EY, x, y, NoPixels, angles.overlapRatio());

			bm->move(NoPixels);
		} //end-if
	} //end-for

	delete[] taken;
	delete[] candidateArcs;
}

Circle * EDCircles::addCircle(Circle *circles, int &noCircles,double xc, double yc, double r, double circleFitError, double * x, double * y, int noPixels)
{
	circles[noCircles].xc = xc;
	circles[noCircles].yc = yc;
	circles[noCircles].r = r;
	circles[noCircles].circleFitError = circleFitError;
	circles[noCircles].coverRatio = noPixels / (TWOPI*r);

	circles[noCircles].x = x;
	circles[noCircles].y = y;
	circles[noCircles].noPixels = noPixels;

	circles[noCircles].isEllipse = false;

	noCircles++;

	return &circles[noCircles - 1];
}

Circle * EDCircles::addCircle(Circle *circles, int &noCircles,double xc, double yc, double r, double circleFitError, EllipseEquation * pEq, double ellipseFitError, double * x, double * y, int noPixels)
{
	circles[noCircles].xc = xc;
	circles[noCircles].yc = yc;
	circles[noCircles].r = r;
	circles[noCircles].circleFitError = circleFitError;
	circles[noCircles].coverRatio = noPixels / computeEllipsePerimeter(pEq);

	circles[noCircles].x = x;
	circles[noCircles].y = y;
	circles[noCircles].noPixels = noPixels;

	circles[noCircles].eq = *pEq;
	circles[noCircles].ellipseFitError = ellipseFitError;
	circles[noCircles].isEllipse = true;

	noCircles++;

	return &circles[noCircles - 1];
}

void EDCircles::sortCircles(Circle *circles, int noCircles)
{
	for (int i = 0; i<noCircles - 1; i++) {
		int max = i;
		for (int j = i + 1; j<noCircles; j++) {
			if (circles[j].r > circles[max].r) max = j;
		} //end-for

		if (max != i) {
			Circle t = circles[i];
			circles[i] = circles[max];
			circles[max] = t;
		} // end-if
	} //end-for
} //end-sort



// ---------------------------------------------------------------------------
// Given an ellipse equation, computes the length of the perimeter of the ellipse
// Calculates the ellipse perimeter wrt the Ramajunan II formula
//
double EDCircles::computeEllipsePerimeter(EllipseEquation * eq)
{
	double mult = 1;

	double A = eq->A()*mult;
	double B = eq->B()*mult;
	double C = eq->C()*mult;
	double D = eq->D()*mult;
	double E = eq->E()*mult;
	double F = eq->F()*mult;

	double A2, B2, C2, D2, E2, F2, theta;  //rotated coefficients
	double			   D3, E3, F3;             //ellipse form coefficients
	double cX, cY, a, b;		               //(cX,cY) center, a & b: semimajor & semiminor axes
	double h;							                 //h = (a-b)^2 / (a+b)^2
	bool rotation = false;

#define pi 3.14159265

	//Normalize coefficients
	B /= A; C /= A; D /= A; E /= A; F /= A; A /= A;

	if (B == 0) //Then not need to rotate the axes
	{
		A2 = A;
		B2 = B;
		C2 = C;
		D2 = D;
		E2 = E;
		F2 = F;

	}

	else if (B != 0) //Rotate the axes
	{
		rotation = true;

		//Determine the rotation angle (in radians)
		theta = atan(B / (A - C)) / 2;

		//Compute the coefficients wrt the new coordinate system 
		A2 = 0.5 * (A * (1 + cos(2 * theta) + B * sin(2 * theta) + C * (1 - cos(2 * theta))));

		B2 = (C - A) * sin(2 * theta) + B * cos(2 * theta); //B2 should turn to be zero?

		C2 = 0.5 * (A * (1 - cos(2 * theta) - B * sin(2 * theta) + C * (1 + cos(2 * theta))));

		D2 = D * cos(theta) + E * sin(theta);

		E2 = -D * sin(theta) + E * cos(theta);

		F2 = F;
	}

	//Transform the conic equation into the ellipse form
	D3 = D2 / A2; //normalize x term's coef 
				  //A3 = 1;     //A2 / A2

	E3 = E2 / C2; //normalize y term's coef
				  //C3 = 1;     //C2 / C2

	cX = -(D3 / 2);	//center X
	cY = -(E3 / 2);	//center Y

	F3 = A2 * pow(cX, 2.0) + C2 * pow(cY, 2.0) - F2;

	//semimajor axis
	a = sqrt(F3 / A2);
	//semiminor axis
	b = sqrt(F3 / C2);

	//Center coordinates have to be re-transformed if rotation is applied!
	if (rotation)
	{
		double tmpX = cX, tmpY = cY;
		cX = tmpX * cos(theta) - tmpY * sin(theta);
		cY = tmpX * sin(theta) + tmpY * cos(theta);
	}

	//Perimeter Computation(s)
	h = pow((a - b), 2.0) / pow((a + b), 2.0);

	//Ramajunan I
	//double P1 = pi * (a + b) * (3 - sqrt(4 - h));
	///printf("Perimeter of the ellipse is %.5f (Ramajunan I)\n", P1);

	//Ramajunan II
	double P2 = pi * (a + b) * (1 + 3 * h / (10 + sqrt(4 - 3 * h)));
	//	printf("Perimeter of the ellipse is %.5f (Ramajunan II)\n", P2);

	//  High-school formula
	//	double P3 = 2 * pi * sqrt(0.5 * (a*a + b*b));
	//	printf("Perimeter of the ellipse is %.5f (simple formula)\n", P3);

	return P2;
#undef pi
}

double EDCircles::ComputeEllipseError(EllipseEquation * eq, double * px, double * py, int noPoints)
{
	double error = 0;

	double A = eq->A();
	double B = eq->B();
	double C = eq->C();
	double D = eq->D();
	double E = eq->E();
	double F = eq->F();

	double xc, yc, major, minor;
	ComputeEllipseCenterAndAxisLengths(eq, &xc, &yc, &major, &minor);

	for (int i = 0; i<noPoints; i++) {
		double dx = px[i] - xc;
		double dy = py[i] - yc;

		double min;
		double xs, ys;

		if (fabs(dx) > fabs(dy)) {
			// The line equation is of the form: y = mx+n
			double m = dy / dx;
			double n = yc - m*xc;

			// a*x^2 + b*x + c
			double a = A + B*m + C*m*m;
			double b = B*n + 2 * C*m*n + D + E*m;
			double c = C*n*n + E*n + F;
			double det = b*b - 4 * a*c;
			if (det<0) det = 0;
			double x1 = -(b + sqrt(det)) / (2 * a);
			double x2 = -(b - sqrt(det)) / (2 * a);

			double y1 = m*x1 + n;
			double y2 = m*x2 + n;

			dx = px[i] - x1;
			dy = py[i] - y1;
			double d1 = dx*dx + dy*dy;

			dx = px[i] - x2;
			dy = py[i] - y2;
			double d2 = dx*dx + dy*dy;

			if (d1<d2) { min = d1; xs = x1; ys = y1; }
			else { min = d2; xs = x2; ys = y2; }

		}
		else {
			// The line equation is of the form: x = my+n
			double m = dx / dy;
			double n = xc - m*yc;

			// a*y^2 + b*y + c
			double a = A*m*m + B*m + C;
			double b = 2 * A*m*n + B*n + D*m + E;
			double c = A*n*n + D*n + F;
			double det = b*b - 4 * a*c;
			if (det<0) det = 0;
			double y1 = -(b + sqrt(det)) / (2 * a);
			double y2 = -(b - sqrt(det)) / (2 * a);

			double x1 = m*y1 + n;
			double x2 = m*y2 + n;

			dx = px[i] - x1;
			dy = py[i] - y1;
			double d1 = dx*dx + dy*dy;

			dx = px[i] - x2;
			dy = py[i] - y2;
			double d2 = dx*dx + dy*dy;

			if (d1<d2) { min = d1; xs = x1; ys = y1; }
			else { min = d2; xs = x2; ys = y2; }
		} //end-else

		  // Refine the search in the vicinity of (xs, ys)
		double delta = 0.5;
		double x = xs;
		while (1) {
			x += delta;

			double a = C;
			double b = B*x + E;
			double c = A*x*x + D*x + F;
			double det = b*b - 4 * a*c;
			if (det < 0) det = 0;

			double y1 = -(b + sqrt(det)) / (2 * a);
			double y2 = -(b - sqrt(det)) / (2 * a);

			dx = px[i] - x;
			dy = py[i] - y1;
			double d1 = dx*dx + dy*dy;

			dy = py[i] - y2;
			double d2 = dx*dx + dy*dy;

			if (d1 <= min) { min = d1; }
			else if (d2 <= min) { min = d2; }
			else break;
		} //end-while

		x = xs;
		while (1) {
			x -= delta;

			double a = C;
			double b = B*x + E;
			double c = A*x*x + D*x + F;
			double det = b*b - 4 * a*c;
			if (det < 0) det = 0;

			double y1 = -(b + sqrt(det)) / (2 * a);
			double y2 = -(b - sqrt(det)) / (2 * a);

			dx = px[i] - x;
			dy = py[i] - y1;
			double d1 = dx*dx + dy*dy;

			dy = py[i] - y2;
			double d2 = dx*dx + dy*dy;

			if (d1 <= min) { min = d1; }
			else if (d2 <= min) { min = d2; }
			else break;
		} //end-while

		error += min;
	} //end-for

	error = sqrt(error / noPoints);

	return error;
}

// also returns rotate angle theta
double EDCircles::ComputeEllipseCenterAndAxisLengths(EllipseEquation * eq, double * pxc, double * pyc, double * pmajorAxisLength, double * pminorAxisLength)
{
	double mult = 1;

	double A = eq->A()*mult;
	double B = eq->B()*mult;
	double C = eq->C()*mult;
	double D = eq->D()*mult;
	double E = eq->E()*mult;
	double F = eq->F()*mult;

	double A2, B2, C2, D2, E2, F2, theta;  //rotated coefficients
	double			   D3, E3, F3;             //ellipse form coefficients
	double cX, cY, a, b;		               //(cX,cY) center, a & b: semimajor & semiminor axes
	bool rotation = false;

#define pi 3.14159265

	//Normalize coefficients
	B /= A; C /= A; D /= A; E /= A; F /= A; A /= A;

	if (B == 0) //Then not need to rotate the axes
	{
		A2 = A;
		B2 = B;
		C2 = C;
		D2 = D;
		E2 = E;
		F2 = F;

		//		printf("No Rotation is applied\n\n");
	}
	else if (B != 0) //Rotate the axes
	{
		rotation = true;

		//Determine the rotation angle (in radians)
		theta = atan(B / (A - C)) / 2;

		//Compute the coefficients wrt the new coordinate system 
		A2 = 0.5 * (A * (1 + cos(2 * theta) + B * sin(2 * theta) + C * (1 - cos(2 * theta))));

		B2 = (C - A) * sin(2 * theta) + B * cos(2 * theta); //B2 should turn to be zero?

		C2 = 0.5 * (A * (1 - cos(2 * theta) - B * sin(2 * theta) + C * (1 + cos(2 * theta))));

		D2 = D * cos(theta) + E * sin(theta);

		E2 = -D * sin(theta) + E * cos(theta);

		F2 = F;

		//		printf("Rotation in degrees: %.2f\n\n", theta * 180 / pi);
	}

	//Transform the conic equation into the ellipse form
	D3 = D2 / A2; //normalize x term's coef 
				  //A3 = 1;     //A2 / A2

	E3 = E2 / C2; //normalize y term's coef
				  //C3 = 1;     //C2 / C2

	cX = -(D3 / 2);	//center X
	cY = -(E3 / 2);	//center Y

	F3 = A2 * pow(cX, 2.0) + C2 * pow(cY, 2.0) - F2;

	//semimajor axis
	a = sqrt(F3 / A2);
	//semiminor axis
	b = sqrt(F3 / C2);

	//Center coordinates have to be re-transformed if rotation is applied!
	if (rotation)
	{
		double tmpX = cX, tmpY = cY;
		cX = tmpX * cos(theta) - tmpY * sin(theta);
		cY = tmpX * sin(theta) + tmpY * cos(theta);
	}

	*pxc = cX;
	*pyc = cY;

	*pmajorAxisLength = a;
	*pminorAxisLength = b;

	//if (a > b) {
	//	*pmajorAxisLength = a;
	//	*pminorAxisLength = b;
	//}q
	//else {
	//	*pmajorAxisLength = b;
	//	*pminorAxisLength = a;
	//} //end-else

	return theta;
#undef pi  
}

// ---------------------------------------------------------------------------
// Given an ellipse equation, computes "noPoints" many consecutive points
// on the ellipse periferi. These points can be used to draw the ellipse
// noPoints must be an even number.
//
void EDCircles::ComputeEllipsePoints(double * pvec, double * px, double * py, int noPoints)
{
	if (noPoints % 2) noPoints--;
	int npts = noPoints / 2;

	double **u = AllocateMatrix(3, npts + 1);
	double **Aiu = AllocateMatrix(3, npts + 1);
	double **L = AllocateMatrix(3, npts + 1);
	double **B = AllocateMatrix(3, npts + 1);
	double **Xpos = AllocateMatrix(3, npts + 1);
	double **Xneg = AllocateMatrix(3, npts + 1);
	double **ss1 = AllocateMatrix(3, npts + 1);
	double **ss2 = AllocateMatrix(3, npts + 1);
	double *lambda = new double[npts + 1];
	double **uAiu = AllocateMatrix(3, npts + 1);
	double **A = AllocateMatrix(3, 3);
	double **Ai = AllocateMatrix(3, 3);
	double **Aib = AllocateMatrix(3, 2);
	double **b = AllocateMatrix(3, 2);
	double **r1 = AllocateMatrix(2, 2);
	double Ao, Ax, Ay, Axx, Ayy, Axy;

	double pi = 3.14781;
	double theta;
	int i;
	int j;
	double kk;

	memset(lambda, 0, sizeof(double)*(npts + 1));

	Ao = pvec[6];
	Ax = pvec[4];
	Ay = pvec[5];
	Axx = pvec[1];
	Ayy = pvec[3];
	Axy = pvec[2];

	A[1][1] = Axx; A[1][2] = Axy / 2;
	A[2][1] = Axy / 2; A[2][2] = Ayy;
	b[1][1] = Ax; b[2][1] = Ay;

	// Generate normals linspace
	for (i = 1, theta = 0.0; i <= npts; i++, theta += (pi / npts))
	{
		u[1][i] = cos(theta);
		u[2][i] = sin(theta);
	}

	inverse(A, Ai, 2);

	AperB(Ai, b, Aib, 2, 2, 2, 1);
	A_TperB(b, Aib, r1, 2, 1, 2, 1);
	r1[1][1] = r1[1][1] - 4 * Ao;

	AperB(Ai, u, Aiu, 2, 2, 2, npts);
	for (i = 1; i <= 2; i++)
		for (j = 1; j <= npts; j++)
			uAiu[i][j] = u[i][j] * Aiu[i][j];

	for (j = 1; j <= npts; j++)
	{
		if ((kk = (r1[1][1] / (uAiu[1][j] + uAiu[2][j]))) >= 0.0)
			lambda[j] = sqrt(kk);
		else
			lambda[j] = -1.0;
	}

	// Builds up B and L
	for (j = 1; j <= npts; j++)
		L[1][j] = L[2][j] = lambda[j];
	for (j = 1; j <= npts; j++)
	{
		B[1][j] = b[1][1];
		B[2][j] = b[2][1];
	}

	for (j = 1; j <= npts; j++)
	{
		ss1[1][j] = 0.5 * (L[1][j] * u[1][j] - B[1][j]);
		ss1[2][j] = 0.5 * (L[2][j] * u[2][j] - B[2][j]);
		ss2[1][j] = 0.5 * (-L[1][j] * u[1][j] - B[1][j]);
		ss2[2][j] = 0.5 * (-L[2][j] * u[2][j] - B[2][j]);
	}

	AperB(Ai, ss1, Xpos, 2, 2, 2, npts);
	AperB(Ai, ss2, Xneg, 2, 2, 2, npts);

	for (j = 1; j <= npts; j++)
	{
		if (lambda[j] == -1.0)
		{
			px[j - 1] = -1;
			py[j - 1] = -1;
			px[j - 1 + npts] = -1;
			py[j - 1 + npts] = -1;
		}
		else
		{
			px[j - 1] = Xpos[1][j];
			py[j - 1] = Xpos[2][j];
			px[j - 1 + npts] = Xneg[1][j];
			py[j - 1 + npts] = Xneg[2][j];
		}
	}

	DeallocateMatrix(u, 3);
	DeallocateMatrix(Aiu, 3);
	DeallocateMatrix(L, 3);
	DeallocateMatrix(B, 3);
	DeallocateMatrix(Xpos, 3);
	DeallocateMatrix(Xneg, 3);
	DeallocateMatrix(ss1, 3);
	DeallocateMatrix(ss2, 3);
	delete[] lambda;
	DeallocateMatrix(uAiu, 3);
	DeallocateMatrix(A, 3);
	DeallocateMatrix(Ai, 3);
	DeallocateMatrix(Aib, 3);
	DeallocateMatrix(b, 3);
	DeallocateMatrix(r1, 2);
}


// Tries to join the last two arcs if their end-points are very close to each other
// and if they are part of the same segment. This is useful in cases where an arc on a segment
// is broken due to a noisy patch along the arc, and the long arc is broken into two or more arcs.
// This function will join such broken arcs
//
void EDCircles::joinLastTwoArcs(MyArc *arcs, int &noArcs)
{
	if (noArcs < 2) return;

	int prev = noArcs - 2;
	int last = noArcs - 1;

	if (arcs[prev].segmentNo != arcs[last].segmentNo) return;
	if (arcs[prev].turn != arcs[last].turn) return;
	if (arcs[prev].isEllipse || arcs[last].isEllipse) return;

	// The radius difference between the arcs must be very small
	double minR = MIN(arcs[prev].r, arcs[last].r);
	double radiusDiffThreshold = minR*0.25;

	double diff = fabs(arcs[prev].r - arcs[last].r);
	if (diff > radiusDiffThreshold) return;

	// End-point distance
	double dx = arcs[prev].ex - arcs[last].sx;
	double dy = arcs[prev].ey - arcs[last].sy;
	double d = sqrt(dx*dx + dy*dy);

	double endPointDiffThreshold = 10;
	if (d > endPointDiffThreshold) return;

	// Try join
	int noPixels = arcs[prev].noPixels + arcs[last].noPixels;

	double xc, yc, r, circleFitError;
	CircleFit(arcs[prev].x, arcs[prev].y, noPixels, &xc, &yc, &r, &circleFitError);

	if (circleFitError <= LONG_ARC_ERROR) {
		arcs[prev].noPixels = noPixels;
		arcs[prev].circleFitError = circleFitError;

		arcs[prev].xc = xc;
		arcs[prev].yc = yc;
		arcs[prev].r = r;
		arcs[prev].ex = arcs[last].ex;
		arcs[prev].ey = arcs[last].ey;
		//    arcs[prev].eTheta = arcs[last].eTheta;   -- Fails in a very nasty way in a very special case (recall circles9 with cvsmooth(7x7)!) So, do not use.

		AngleSet angles;
		angles.set(arcs[prev].sTheta, arcs[prev].eTheta);
		angles.set(arcs[last].sTheta, arcs[last].eTheta);
		angles.computeStartEndTheta(arcs[prev].sTheta, arcs[prev].eTheta);

		//    arcs[prev].coverRatio = noPixels/(TWOPI*r);
		arcs[prev].coverRatio = ArcLength(arcs[prev].sTheta, arcs[prev].eTheta) / (TWOPI);

		noArcs--;
	} //end-if
}


//-----------------------------------------------------------------------
// Add a new arc to arcs
//
void EDCircles::addArc(MyArc *arcs, int &noArcs, double xc, double yc, double r, double circleFitError, double sTheta, double eTheta, int turn, int segmentNo, int sx, int sy, int ex, int ey, double * x, double * y, int noPixels, double overlapRatio)
{
	arcs[noArcs].xc = xc;
	arcs[noArcs].yc = yc;
	arcs[noArcs].r = r;
	arcs[noArcs].circleFitError = circleFitError;

	arcs[noArcs].sTheta = sTheta;
	arcs[noArcs].eTheta = eTheta;
	arcs[noArcs].coverRatio = ArcLength(sTheta, eTheta) / (TWOPI);

	arcs[noArcs].turn = turn;

	arcs[noArcs].segmentNo = segmentNo;

	arcs[noArcs].isEllipse = false;

	arcs[noArcs].sx = sx;
	arcs[noArcs].sy = sy;
	arcs[noArcs].ex = ex;
	arcs[noArcs].ey = ey;

	arcs[noArcs].x = x;
	arcs[noArcs].y = y;
	arcs[noArcs].noPixels = noPixels;

	noArcs++;

	// See if you can join the last two arcs
	joinLastTwoArcs(arcs, noArcs);
}

//-------------------------------------------------------------------------
// Add an elliptic arc to the list of arcs
//
void EDCircles::addArc(MyArc * arcs, int & noArcs, double xc, double yc, double r, double circleFitError, double sTheta, double eTheta, int turn, int segmentNo, EllipseEquation * pEq, double ellipseFitError, int sx, int sy, int ex, int ey, double * x, double * y, int noPixels, double overlapRatio)
{
	arcs[noArcs].xc = xc;
	arcs[noArcs].yc = yc;
	arcs[noArcs].r = r;
	arcs[noArcs].circleFitError = circleFitError;

	arcs[noArcs].sTheta = sTheta;
	arcs[noArcs].eTheta = eTheta;
	arcs[noArcs].coverRatio = (double)((1.0 - overlapRatio)*noPixels) / computeEllipsePerimeter(pEq);
	//  arcs[noArcs].coverRatio = noPixels/ComputeEllipsePerimeter(pEq);
	//  arcs[noArcs].coverRatio = ArcLength(sTheta, eTheta)/(TWOPI);
	
	arcs[noArcs].turn = turn;

	arcs[noArcs].segmentNo = segmentNo;

	arcs[noArcs].isEllipse = true;
	arcs[noArcs].eq = *pEq;
	arcs[noArcs].ellipseFitError = ellipseFitError;

	arcs[noArcs].sx = sx;
	arcs[noArcs].sy = sy;
	arcs[noArcs].ex = ex;
	arcs[noArcs].ey = ey;

	arcs[noArcs].x = x;
	arcs[noArcs].y = y;
	arcs[noArcs].noPixels = noPixels;

	noArcs++;
}

//--------------------------------------------------------------
// Given a circular arc, computes the start & end angles of the arc in radians
//
void EDCircles::ComputeStartAndEndAngles(double xc, double yc, double r, double * x, double * y, int len, double * psTheta, double * peTheta)
{
	double sx = x[0];
	double sy = y[0];
	double ex = x[len - 1];
	double ey = y[len - 1];
	double mx = x[len / 2];
	double my = y[len / 2];

	double d = (sx - xc) / r;
	if (d > 1.0) d = 1.0; else if (d < -1.0) d = -1.0;
	double theta1 = acos(d);

	double sTheta;
	if (sx >= xc) {
		if (sy >= yc) {
			// I. quadrant
			sTheta = theta1;
		}
		else {
			// IV. quadrant
			sTheta = TWOPI - theta1;
		} //end-else
	}
	else {
		if (sy >= yc) {
			// II. quadrant
			sTheta = theta1;
		}
		else {
			// III. quadrant
			sTheta = TWOPI - theta1;
		} //end-else
	} //end-else

	d = (ex - xc) / r;
	if (d > 1.0) d = 1.0; else if (d < -1.0) d = -1.0;
	theta1 = acos(d);

	double eTheta;
	if (ex >= xc) {
		if (ey >= yc) {
			// I. quadrant
			eTheta = theta1;
		}
		else {
			// IV. quadrant
			eTheta = TWOPI - theta1;
		} //end-else
	}
	else {
		if (ey >= yc) {
			// II. quadrant
			eTheta = theta1;
		}
		else {
			// III. quadrant
			eTheta = TWOPI - theta1;
		} //end-else
	} //end-else

	  // Determine whether the arc is clockwise (CW) or counter-clockwise (CCW)
	double circumference = TWOPI*r;
	double ratio = len / circumference;

	if (ratio <= 0.25 || ratio >= 0.75) {
		double angle1, angle2;

		if (eTheta > sTheta) {
			angle1 = eTheta - sTheta;
			angle2 = TWOPI - eTheta + sTheta;

		}
		else {
			angle1 = sTheta - eTheta;
			angle2 = TWOPI - sTheta + eTheta;
		} //end-else

		angle1 = angle1 / TWOPI;
		angle2 = angle2 / TWOPI;

		double diff1 = fabs(ratio - angle1);
		double diff2 = fabs(ratio - angle2);

		if (diff1 < diff2) {
			// angle1 is correct
			if (eTheta > sTheta) {
				;
			}
			else {
				double tmp = sTheta;
				sTheta = eTheta;
				eTheta = tmp;
			} // end-else

		}
		else {
			// angle2 is correct
			if (eTheta > sTheta) {
				double tmp = sTheta;
				sTheta = eTheta;
				eTheta = tmp;

			}
			else {
				;
			} // end-else
		} //end-else

	}
	else {
		double v1x = mx - sx;
		double v1y = my - sy;
		double v2x = ex - mx;
		double v2y = ey - my;

		// cross product
		double cross = v1x*v2y - v1y*v2x;
		if (cross < 0) {
			// swap sTheta & eTheta
			double tmp = sTheta;
			sTheta = eTheta;
			eTheta = tmp;
		} //end-if
	} //end-else

	double diff = fabs(sTheta - eTheta);
	if (diff < (TWOPI / 120)) {
		sTheta = 0;
		eTheta = 6.26;   // 359 degrees
	} //end-if

	  // Round the start & etheta to 0 if very close to 6.28 or 0
	if (sTheta >= 6.26) sTheta = 0;
	if (eTheta < 1.0 / TWOPI) eTheta = 6.28;  // if less than 1 degrees, then round to 6.28

	*psTheta = sTheta;
	*peTheta = eTheta;
}

void EDCircles::sortArc(MyArc * arcs, int noArcs)
{
	for (int i = 0; i<noArcs - 1; i++) {
		int max = i;
		for (int j = i + 1; j<noArcs; j++) {
			if (arcs[j].coverRatio > arcs[max].coverRatio) max = j;
		} //end-for

		if (max != i) {
			MyArc t = arcs[i];
			arcs[i] = arcs[max];
			arcs[max] = t;
		} // end-if
	} //end-for
}


//---------------------------------------------------------------------
// Fits a circle to a given set of points. There must be at least 2 points
// The circle equation is of the form: (x-xc)^2 + (y-yc)^2 = r^2
// Returns true if there is a fit, false in case no circles can be fit
//
bool EDCircles::CircleFit(double * x, double * y, int N, double * pxc, double * pyc, double * pr, double * pe)
{
	*pe = 1e20;
	if (N < 3) return false;

	double xAvg = 0;
	double yAvg = 0;

	for (int i = 0; i<N; i++) {
		xAvg += x[i];
		yAvg += y[i];
	} //end-for

	xAvg /= N;
	yAvg /= N;

	double Suu = 0;
	double Suv = 0;
	double Svv = 0;
	double Suuu = 0;
	double Suvv = 0;
	double Svvv = 0;
	double Svuu = 0;
	for (int i = 0; i<N; i++) {
		double u = x[i] - xAvg;
		double v = y[i] - yAvg;

		Suu += u*u;
		Suv += u*v;
		Svv += v*v;
		Suuu += u*u*u;
		Suvv += u*v*v;
		Svvv += v*v*v;
		Svuu += v*u*u;
	} //end-for

	  // Now, we solve for the following linear system of equations
	  // Av = b, where v = (uc, vc) is the center of the circle
	  //
	  // |N    Suv| |uc| = |b1|
	  // |Suv  Svv| |vc| = |b2|
	  //
	  // where b1 = 0.5*(Suuu+Suvv) and b2 = 0.5*(Svvv+Svuu)
	  //
	double detA = Suu*Svv - Suv*Suv;
	if (detA == 0) return false;

	double b1 = 0.5*(Suuu + Suvv);
	double b2 = 0.5*(Svvv + Svuu);

	double uc = (Svv*b1 - Suv*b2) / detA;
	double vc = (Suu*b2 - Suv*b1) / detA;

	double R = sqrt(uc*uc + vc*vc + (Suu + Svv) / N);

	*pxc = uc + xAvg;
	*pyc = vc + yAvg;

	// Compute mean square error
	double error = 0;
	for (int i = 0; i<N; i++) {
		double dx = x[i] - *pxc;
		double dy = y[i] - *pyc;
		double d = sqrt(dx*dx + dy*dy) - R;
		error += d*d;
	} //end-for

	*pr = R;
	*pe = sqrt(error / N);

	return true;
}


//------------------------------------------------------------------------------------
// Computes the points making up a circle
//
void EDCircles::ComputeCirclePoints(double xc, double yc, double r, double * px, double * py, int * noPoints)
{
	int len = (int)(TWOPI*r + 0.5);
	double angleInc = TWOPI / len;
	double angle = 0;

	int prevX = -1;
	int prevY = -1;
	int count = 0;

	while (angle < TWOPI) {
		int x = (int)(cos(angle)*r + xc + 0.5);
		int y = (int)(sin(angle)*r + yc + 0.5);

		angle += angleInc;

		px[count] = x;
		py[count] = y;
		count++;
	} //end-while

	*noPoints = count;
}

void EDCircles::sortCircle(Circle * circles, int noCircles)
{
	for (int i = 0; i<noCircles - 1; i++) {
		int max = i;
		for (int j = i + 1; j<noCircles; j++) {
			if (circles[j].r > circles[max].r) max = j;
		} //end-for

		if (max != i) {
			Circle t = circles[i];
			circles[i] = circles[max];
			circles[max] = t;
		} // end-if
	} //end-for
}

bool EDCircles::EllipseFit(double * x, double * y, int noPoints, EllipseEquation * pResult, int mode)
{
	double **D = AllocateMatrix(noPoints + 1, 7);
	double **S = AllocateMatrix(7, 7);
	double **Const = AllocateMatrix(7, 7);
	double **temp = AllocateMatrix(7, 7);
	double **L = AllocateMatrix(7, 7);
	double **C = AllocateMatrix(7, 7);

	double **invL = AllocateMatrix(7, 7);
	double *d = new double[7];
	double **V = AllocateMatrix(7, 7);
	double **sol = AllocateMatrix(7, 7);
	double tx, ty;
	int nrot = 0;

	memset(d, 0, sizeof(double) * 7);


	switch (mode)
	{
	case (FPF):
		//fprintf(stderr, "EllipseFit: FPF mode");
		Const[1][3] = -2;
		Const[2][2] = 1;
		Const[3][1] = -2;
		break;
	case (BOOKSTEIN):
		//fprintf(stderr, "EllipseFit: BOOKSTEIN mode");
		Const[1][1] = 2;
		Const[2][2] = 1;
		Const[3][3] = 2;
	} //end-switch

	if (noPoints < 6)
		return false;

	// Now first fill design matrix
	for (int i = 1; i <= noPoints; i++) {
		tx = x[i - 1];
		ty = y[i - 1];

		D[i][1] = tx * tx;
		D[i][2] = tx * ty;
		D[i][3] = ty * ty;
		D[i][4] = tx;
		D[i][5] = ty;
		D[i][6] = 1.0;
	} //end-for

	  //pm(Const,"Constraint");
	  // Now compute scatter matrix  S
	A_TperB(D, D, S, noPoints, 6, noPoints, 6);
	//pm(S,"Scatter");

	choldc(S, 6, L);
	//pm(L,"Cholesky");

	inverse(L, invL, 6);
	//pm(invL,"inverse");

	AperB_T(Const, invL, temp, 6, 6, 6, 6);
	AperB(invL, temp, C, 6, 6, 6, 6);
	//pm(C,"The C matrix");

	jacobi(C, 6, d, V, nrot);
	//pm(V,"The Eigenvectors");  /* OK */
	//pv(d,"The eigevalues");

	A_TperB(invL, V, sol, 6, 6, 6, 6);
	//pm(sol,"The GEV solution unnormalized");  /* SOl */

	// Now normalize them 
	for (int j = 1; j <= 6; j++)  /* Scan columns */
	{
		double mod = 0.0;
		for (int i = 1; i <= 6; i++)
			mod += sol[i][j] * sol[i][j];
		for (int i = 1; i <= 6; i++)
			sol[i][j] /= sqrt(mod);
	}

	//pm(sol,"The GEV solution");  /* SOl */

	double zero = 10e-20;
	double minev = 10e+20;
	int solind = 0;
	int i;
	switch (mode)
	{
	case (BOOKSTEIN):  // smallest eigenvalue	 	   
		for (i = 1; i <= 6; i++)
			if (d[i] < minev && fabs(d[i]) > zero)
				solind = i;
		break;
	case (FPF):
		for (i = 1; i <= 6; i++)
			if (d[i] < 0 && fabs(d[i]) > zero)
				solind = i;
	}

#if 0
	fprintf(stderr, "SOLUTIONS: Selected: %d\n", solind);
	for (i = 1; i <= 6; i++) {
		fprintf(stderr, "d[i]: %e, a: %.7lf, b: %.7lf, c: %.7lf, d: %.7lf, e: %.7lf, f: %.7lf\n",
			d[i], sol[1][i], sol[2][i], sol[3][i], sol[4][i], sol[5][i], sol[6][i]);
	} //end-for

#endif

	bool valid = true;
	if (solind == 0) valid = false;

	if (valid) {
		// Now fetch the right solution
		for (int j = 1; j <= 6; j++) {
			pResult->coeff[j] = sol[j][solind];
		} //end-for
	} //end-if

	DeallocateMatrix(D, noPoints + 1);
	DeallocateMatrix(S, 7);
	DeallocateMatrix(Const, 7);
	DeallocateMatrix(temp, 7);
	DeallocateMatrix(L, 7);
	DeallocateMatrix(C, 7);
	DeallocateMatrix(invL, 7);
	delete[] d;
	DeallocateMatrix(V, 7);
	DeallocateMatrix(sol, 7);

	if (valid) {
		int len = (int)computeEllipsePerimeter(pResult);
		if (len <= 0 || len > 50000) valid = false;
	} //end-else

	return valid;
}

double ** EDCircles::AllocateMatrix(int noRows, int noColumns)
{
	double **m = new double *[noRows];
	
	for (int i = 0; i<noRows; i++) {
		m[i] = new double[noColumns];
		memset(m[i], 0, sizeof(double)*noColumns);
	} // end-for

	return m;
}

void EDCircles::A_TperB(double ** _A, double ** _B, double ** _res, int _righA, int _colA, int _righB, int _colB)
{
	int p, q, l;
	for (p = 1; p <= _colA; p++)
		for (q = 1; q <= _colB; q++)
		{
			_res[p][q] = 0.0;
			for (l = 1; l <= _righA; l++)
				_res[p][q] = _res[p][q] + _A[l][p] * _B[l][q];
		}
}

//-----------------------------------------------------------
// Perform the Cholesky decomposition    
// Return the lower triangular L  such that L*L'=A  
//
void EDCircles::choldc(double ** a, int n, double ** l)
{
	int i, j, k;
	double sum;
	double *p = new double[n + 1];
	memset(p, 0, sizeof(double)*(n + 1));


	for (i = 1; i <= n; i++)
	{
		for (j = i; j <= n; j++)
		{
			for (sum = a[i][j], k = i - 1; k >= 1; k--) sum -= a[i][k] * a[j][k];
			if (i == j)
			{
				if (sum <= 0.0)
					// printf("\nA is not poitive definite!");
				{
				}
				else
					p[i] = sqrt(sum);
			}
			else
			{
				a[j][i] = sum / p[i];
			}
		}
	}
	for (i = 1; i <= n; i++) {
		for (j = i; j <= n; j++) {
			if (i == j)
				l[i][i] = p[i];
			else
			{
				l[j][i] = a[j][i];
				l[i][j] = 0.0;
			}
		} //end-for-inner
	} // end-for-outer

	delete[] p;
}

int EDCircles::inverse(double ** TB, double ** InvB, int N)
{
	int k, i, j, p, q;
	double mult;
	double D, temp;
	double maxpivot;
	int npivot;
	double **B = AllocateMatrix(N + 1, N + 2);
	double **A = AllocateMatrix(N + 1, 2 * N + 2);
	double **C = AllocateMatrix(N + 1, N + 1);
	double eps = 10e-20;

	for (k = 1; k <= N; k++)
		for (j = 1; j <= N; j++)
			B[k][j] = TB[k][j];

	for (k = 1; k <= N; k++)
	{
		for (j = 1; j <= N + 1; j++)
			A[k][j] = B[k][j];
		for (j = N + 2; j <= 2 * N + 1; j++)
			A[k][j] = (double)0;
		A[k][k - 1 + N + 2] = (double)1;
	}
	for (k = 1; k <= N; k++)
	{
		maxpivot = fabs((double)A[k][k]);
		npivot = k;
		for (i = k; i <= N; i++)
			if (maxpivot < fabs((double)A[i][k]))
			{
				maxpivot = fabs((double)A[i][k]);
				npivot = i;
			}
		if (maxpivot >= eps)
		{
			if (npivot != k)
				for (j = k; j <= 2 * N + 1; j++)
				{
					temp = A[npivot][j];
					A[npivot][j] = A[k][j];
					A[k][j] = temp;
				};
			D = A[k][k];
			for (j = 2 * N + 1; j >= k; j--)
				A[k][j] = A[k][j] / D;
			for (i = 1; i <= N; i++)
			{
				if (i != k)
				{
					mult = A[i][k];
					for (j = 2 * N + 1; j >= k; j--)
						A[i][j] = A[i][j] - mult * A[k][j];
				}
			}
		}
		else
		{  // printf("\n The matrix may be singular !!") ;

			DeallocateMatrix(B, N + 1);
			DeallocateMatrix(A, N + 1);
			DeallocateMatrix(C, N + 1);

			return (-1);
		} //end-else
	}
	/**   Copia il risultato nella matrice InvB  ***/
	for (k = 1, p = 1; k <= N; k++, p++)
		for (j = N + 2, q = 1; j <= 2 * N + 1; j++, q++)
			InvB[p][q] = A[k][j];

	DeallocateMatrix(B, N + 1);
	DeallocateMatrix(A, N + 1);
	DeallocateMatrix(C, N + 1);

	return (0);
}

void EDCircles::DeallocateMatrix(double ** m, int noRows)
{
	for (int i = 0; i<noRows; i++) delete[] m[i];
	delete[] m;
}

void EDCircles::AperB_T(double ** _A, double ** _B, double ** _res, int _righA, int _colA, int _righB, int _colB)
{
	int p, q, l;
	for (p = 1; p <= _colA; p++)
		for (q = 1; q <= _colB; q++)
		{
			_res[p][q] = 0.0;
			for (l = 1; l <= _righA; l++)
				_res[p][q] = _res[p][q] + _A[p][l] * _B[q][l];
		}
}

void EDCircles::AperB(double ** _A, double ** _B, double ** _res, int _righA, int _colA, int _righB, int _colB)
{
	int p, q, l;
	for (p = 1; p <= _righA; p++)
		for (q = 1; q <= _colB; q++)
		{
			_res[p][q] = 0.0;
			for (l = 1; l <= _colA; l++)
				_res[p][q] = _res[p][q] + _A[p][l] * _B[l][q];
		}
}

void EDCircles::jacobi(double ** a, int n, double d[], double ** v, int nrot)
{
	int j, iq, ip, i;
	double tresh, theta, tau, t, sm, s, h, g, c;

	double *b = new double[n + 1];
	double *z = new double[n + 1];
	memset(b, 0, sizeof(double)*(n + 1));
	memset(z, 0, sizeof(double)*(n + 1));

	for (ip = 1; ip <= n; ip++)
	{
		for (iq = 1; iq <= n; iq++) v[ip][iq] = 0.0;
		v[ip][ip] = 1.0;
	}
	for (ip = 1; ip <= n; ip++)
	{
		b[ip] = d[ip] = a[ip][ip];
		z[ip] = 0.0;
	}
	nrot = 0;
	for (i = 1; i <= 50; i++)
	{
		sm = 0.0;
		for (ip = 1; ip <= n - 1; ip++)
		{
			for (iq = ip + 1; iq <= n; iq++)
				sm += fabs(a[ip][iq]);
		}
		if (sm == 0.0)
		{
			delete[] b;
			delete[] z;
			return;
		}
		if (i < 4)
			tresh = 0.2 * sm / (n * n);
		else
			tresh = 0.0;
		for (ip = 1; ip <= n - 1; ip++)
		{
			for (iq = ip + 1; iq <= n; iq++)
			{
				g = 100.0 * fabs(a[ip][iq]);
				//				if (i > 4 && fabs(d[ip]) + g == fabs(d[ip])
				//				&& fabs(d[iq]) + g == fabs(d[iq]))

				if (i > 4 && g == 0.0)
					a[ip][iq] = 0.0;
				else if (fabs(a[ip][iq]) > tresh)
				{
					h = d[iq] - d[ip];
					if (g == 0.0)
						t = (a[ip][iq]) / h;
					else
					{
						theta = 0.5 * h / (a[ip][iq]);
						t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
						if (theta < 0.0) t = -t;
					}
					c = 1.0 / sqrt(1 + t * t);
					s = t * c;
					tau = s / (1.0 + c);
					h = t * a[ip][iq];
					z[ip] -= h;
					z[iq] += h;
					d[ip] -= h;
					d[iq] += h;
					a[ip][iq] = 0.0;
					for (j = 1; j <= ip - 1; j++)
					{
						ROTATE(a, j, ip, j, iq, tau, s);
					}
					for (j = ip + 1; j <= iq - 1; j++)
					{
						ROTATE(a, ip, j, j, iq, tau, s);
					}
					for (j = iq + 1; j <= n; j++)
					{
						ROTATE(a, ip, j, iq, j, tau, s);
					}
					for (j = 1; j <= n; j++)
					{
						ROTATE(v, j, ip, j, iq, tau, s);
					}
					++nrot;
				}
			}
		}
		for (ip = 1; ip <= n; ip++)
		{
			b[ip] += z[ip];
			d[ip] = b[ip];
			z[ip] = 0.0;
		}
	}
	//printf("Too many iterations in routine JACOBI");
	delete[] b;
	delete[] z;
}

void EDCircles::ROTATE(double ** a, int i, int j, int k, int l, double tau, double s)
{
	double g, h;
	g = a[i][j]; h = a[k][l]; a[i][j] = g - s * (h + g * tau);
	a[k][l] = h + s * (g - h * tau);
}

void AngleSet::_set(double sTheta, double eTheta) {
	int arc = next++;

	angles[arc].sTheta = sTheta;
	angles[arc].eTheta = eTheta;
	angles[arc].next = -1;

	// Add the current arc to the linked list
	int prev = -1;
	int current = head;
	while (1) {
		// Empty list?
		if (head < 0) { head = arc; break; }

		// End of the list. Add to the end
		if (current < 0) {
			angles[prev].next = arc;
			break;
		} //end-if

		if (angles[arc].eTheta <= angles[current].sTheta) {
			// Add before current
			if (prev < 0) {
				angles[arc].next = current;
				head = arc;

			}
			else {
				angles[arc].next = current;
				angles[prev].next = arc;
			} //end-else

			break;

		}
		else if (angles[arc].sTheta >= angles[current].eTheta) {
			// continue
			prev = current;
			current = angles[current].next;

			// End of the list?
			if (current < 0) {
				angles[prev].next = arc;
				break;
			} //end-if

		}
		else {
			// overlaps with current. Join
			// First delete current from the list
			if (prev < 0) head = angles[head].next;
			else          angles[prev].next = angles[current].next;

			// Update overlap amount.
			if (angles[arc].eTheta < angles[current].eTheta) {
				overlapAmount += angles[arc].eTheta - angles[current].sTheta;
			}
			else {
				overlapAmount += angles[current].eTheta - angles[arc].sTheta;
			} //end-else

			  // Now join current with arc
			if (angles[current].sTheta < angles[arc].sTheta) angles[arc].sTheta = angles[current].sTheta;
			if (angles[current].eTheta > angles[arc].eTheta) angles[arc].eTheta = angles[current].eTheta;
			current = angles[current].next;
		} //end-else
	} //end-while
}

void AngleSet::set(double sTheta, double eTheta)
{
	if (eTheta > sTheta) {
		_set(sTheta, eTheta);

	}
	else {
		_set(sTheta, TWOPI);
		_set(0, eTheta);
	} //end-else
}

double AngleSet::_overlap(double sTheta, double eTheta)
{
	double o = 0;

	int current = head;
	while (current >= 0) {
		if (sTheta > angles[current].eTheta) { current = angles[current].next; continue; }
		else if (eTheta < angles[current].sTheta) break;

		// 3 cases.
		if (sTheta < angles[current].sTheta && eTheta > angles[current].eTheta) {
			o += angles[current].eTheta - angles[current].sTheta;

		}
		else if (sTheta < angles[current].sTheta) {
			o += eTheta - angles[current].sTheta;

		}
		else {
			o += angles[current].eTheta - sTheta;
		} //end-else

		current = angles[current].next;
	} //end-while

	return o;
}

double AngleSet::overlap(double sTheta, double eTheta)
{
	double o;

	if (eTheta > sTheta) {
		o = _overlap(sTheta, eTheta);

	}
	else {
		o = _overlap(sTheta, TWOPI);
		o += _overlap(0, eTheta);
	} //end-else

	return o / ArcLength(sTheta, eTheta);
}

void AngleSet::computeStartEndTheta(double & sTheta, double & eTheta)
{
	// Special case: Just one arc
	if (angles[head].next < 0) {
		sTheta = angles[head].sTheta;
		eTheta = angles[head].eTheta;

		return;
	} //end-if

	  // OK. More than one arc. Find the biggest gap
	int current = head;
	int nextArc = angles[current].next;

	double biggestGapSTheta = angles[current].eTheta;
	double biggestGapEtheta = angles[nextArc].sTheta;
	double biggestGapLength = biggestGapEtheta - biggestGapSTheta;

	double start, end, len;
	while (1) {
		current = nextArc;
		nextArc = angles[nextArc].next;
		if (nextArc < 0) break;

		start = angles[current].eTheta;
		end = angles[nextArc].sTheta;
		len = end - start;

		if (len > biggestGapLength) {
			biggestGapSTheta = start;
			biggestGapEtheta = end;
			biggestGapLength = len;
		} //end-if

	} //end-while

	  // Compute the gap between the last arc & the first arc
	start = angles[current].eTheta;
	end = angles[head].sTheta;
	len = TWOPI - start + end;
	if (len > biggestGapLength) {
		biggestGapSTheta = start;
		biggestGapEtheta = end;
	} //end-if

	sTheta = biggestGapEtheta;
	eTheta = biggestGapSTheta;
}

double AngleSet::coverRatio()
{
	int current = head;

	double total = 0;
	while (current >= 0) {
		total += angles[current].eTheta - angles[current].sTheta;
		current = angles[current].next;
	} //end-while

	return total / (TWOPI);
}


