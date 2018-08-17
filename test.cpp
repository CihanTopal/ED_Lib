#include "EDLib.h"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{	
	//***************************** ED Edge Segment Detection *****************************
	//Detection of edge segments from an input image	
	Mat testImg = imread("billiard.jpg", 0);	
	imshow("Source Image", testImg);

	//Call ED constructor
	ED testED = ED(testImg, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true); // apply ED algorithm
	
	//Show resulting edge image
	Mat edgeImg = testED.getEdgeImage();
	imshow("Edge Image - PRESS ANY KEY TO CONTINUE", edgeImg);
	waitKey();
		
	//Output number of segments
	int noSegments = testED.getSegmentNo();
	std::cout << "Number of edge segments: " << noSegments << std::endl;
		
	//Get edges in segment form (getSortedSegments() gives segments sorted w.r.t. legnths) 
	std::vector< std::vector<Point> > segments = testED.getSegments();
	
	
	//***************************** EDLINES Line Segment Detection *****************************
	//Detection of line segments from the same image
	EDLines testEDLines = EDLines(testImg);
	Mat lineImg = testEDLines.getLineImage();	//draws on an empty image
	imshow("Line Image 1 - PRESS ANY KEY TO CONTINUE", lineImg);

	//Detection of lines segments from edge segments instead of input image
	//Therefore, redundant detection of edge segmens can be avoided
	testEDLines = EDLines(testED);
	lineImg = testEDLines.drawOnImage();	//draws on the input image
	imshow("Line Image 2  - PRESS ANY KEY TO CONTINUE", lineImg);

	//Acquiring line information, i.e. start & end points
	vector<LS> lines = testEDLines.getLines();
	int noLines = testEDLines.getLinesNo();
	std::cout << "Number of line segments: " << noLines << std::endl;

	waitKey();

	//************************** EDPF Parameter-free Edge Segment Detection **************************
	// Detection of edge segments with parameter free ED (EDPF)

	EDPF testEDPF = EDPF(testImg);
	Mat edgePFImage = testEDPF.getEdgeImage();
	imshow("Edge Image Parameter Free", edgePFImage);
	cout << "Number of edge segments found by EDPF: " << testEDPF.getSegmentNo() << endl;
	waitKey();

	//***************************** EDCIRCLES Circle Segment Detection *****************************
	//Detection of circles directly from the input image

	EDCircles testEDCircles = EDCircles(testImg);
	Mat circleImg = testEDCircles.drawResult(false, ImageStyle::CIRCLES);
	imshow("Circle Image 1", circleImg);

	//Detection of circles from already available EDPF or ED image
	testEDCircles = EDCircles(testEDPF);
	
	//Get circle information as [cx, cy, r]
	vector<mCircle> circles = testEDCircles.getCircles();

	//Get ellipse information as [cx, cy, a, b, theta]
	vector<mEllipse> ellipses = testEDCircles.getEllipses();

	//Circles and ellipses will be indicated in green and red, resp.
	circleImg = testEDCircles.drawResult(true, ImageStyle::BOTH);
	imshow("CIRCLES and ELLIPSES RESULT IMAGE", circleImg);

	int noCircles = testEDCircles.getCirclesNo();
	std::cout << "Number of circles: " << noCircles << std::endl;
	waitKey();
	
	//*********************** EDCOLOR Edge Segment Detection from Color Images **********************
		
	Mat colorImg = imread("billiard.jpg");	
	//Mat colorImg = imread("billiardNoise.jpg");
	EDColor testEDColor = EDColor(colorImg, 36, 4, 1.5, true); //last parameter for validation
	imshow("Color Edge Image - PRESS ANY KEY TO QUIT", testEDColor.getEdgeImage());
	cout << "Number of edge segments detected by EDColor: " << testEDColor.getSegmentNo() << endl;	
	waitKey();	
	
	// get lines from color image
	EDLines colorLine = EDLines(testEDColor);
	imshow("Color Line", colorLine.getLineImage());
	std::cout << "Number of line segments: " << colorLine.getLinesNo() << std::endl;
	waitKey();

	// get circles from color image
	EDCircles colorCircle = EDCircles(testEDColor);
	// TO DO :: drawResult doesnt overlay (onImage = true) when input is from EDColor
	circleImg = colorCircle.drawResult(false, ImageStyle::BOTH);
	imshow("Color Circle", circleImg);
	std::cout << "Number of line segments: " << colorCircle.getCirclesNo() << std::endl;
	waitKey();

	return 0;
}



