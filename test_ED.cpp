#include "EDLib.h"
#include <iostream>
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

int main(int argc, char** argv)
{
    char* filename;
    if (argc > 1)
        filename = argv[1];
    else
        filename = "billiard.jpg";

    Mat testImg = imread(filename, 0);
    TickMeter tm;

    for (int i = 1; i < 5; i++)
    {
        cout << "\n#################################################";
        cout << "\n####### ( " << i << " ) ORIGINAL & OPENCV COMPARISON ######";
        cout << "\n#################################################\n";
        Ptr<EdgeDrawing> ed = createEdgeDrawing();
        ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
        ed->params.GradientThresholdValue = 36;
        ed->params.AnchorThresholdValue = 8;
        vector<Vec6d> ellipses;
        vector<Vec4f> lines;

        //Detection of edge segments from an input image    
        tm.start();
        //Call ED constructor
        ED testED = ED(testImg, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true);
        tm.stop();
        std::cout << "\ntestED.getEdgeImage()  (Original)  : " << tm.getTimeMilli() << endl;

        tm.reset();
        tm.start();
        ed->detectEdges(testImg);
        tm.stop();
        std::cout << "detectEdges()            (OpenCV)  : " << tm.getTimeMilli() << endl;

        Mat edgeImg0 = testED.getEdgeImage();
        Mat edgeImg1, diff;
        ed->getEdgeImage(edgeImg1);
        absdiff(edgeImg0, edgeImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;
        imwrite("edgeImg0.png", edgeImg0);
        imwrite("edgeImg1.png", edgeImg1);
        imwrite("diff0.png", diff);

        //***************************** EDLINES Line Segment Detection *****************************
        //Detection of lines segments from edge segments instead of input image
        //Therefore, redundant detection of edge segmens can be avoided
        tm.reset();
        tm.start();
        EDLines testEDLines = EDLines(testED);
        tm.stop();
        cout << "-------------------------------------------------\n";
        cout << "testEDLines.getLineImage()         : " << tm.getTimeMilli() << endl;
        Mat lineImg0 = testEDLines.getLineImage();    //draws on an empty image
        imwrite("lineImg0.png", lineImg0);

        tm.reset();
        tm.start();
        ed->detectLines(lines);
        tm.stop();
        cout << "detectLines()            (OpenCV)  : " << tm.getTimeMilli() << endl;

        Mat lineImg1 = Mat(lineImg0.rows, lineImg0.cols, CV_8UC1, Scalar(255));

        for (int i = 0; i < lines.size(); i++)
            line(lineImg1, Point2d(lines[i][0], lines[i][1]), Point2d(lines[i][2], lines[i][3]), Scalar(0), 1, LINE_AA);

        absdiff(lineImg0, lineImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;
        imwrite("lineImg1.png", lineImg1);
        imwrite("diff1.png", diff);

        //***************************** EDCIRCLES Circle Segment Detection *****************************
        //Detection of circles from already available EDPF or ED image
        tm.reset();
        tm.start();
        EDCircles testEDCircles = EDCircles(testEDLines);
        tm.stop();
        cout << "-------------------------------------------------\n";
        cout << "EDCircles(testEDLines)             : " << tm.getTimeMilli() << endl;

        tm.reset();
        tm.start();
        ed->detectEllipses(ellipses);
        tm.stop();
        cout << "detectEllipses()         (OpenCV)  : " << tm.getTimeMilli() << endl;
        cout << "-------------------------------------------------\n";

        //************************** EDPF Parameter-free Edge Segment Detection **************************
        // Detection of edge segments with parameter free ED (EDPF)
        tm.reset();
        tm.start();
        EDPF testEDPF = EDPF(testImg);
        tm.stop();
        cout << "testEDPF.getEdgeImage()            : " << tm.getTimeMilli() << endl;

        ed->params.EdgeDetectionOperator = EdgeDrawing::PREWITT;
        ed->params.GradientThresholdValue = 11;
        ed->params.AnchorThresholdValue = 3;
        ed->params.PFmode = true;

        tm.reset();
        tm.start();
        ed->detectEdges(testImg);
        tm.stop();
        std::cout << "detectEdges()  PF        (OpenCV)  : " << tm.getTimeMilli() << endl;

        edgeImg0 = testEDPF.getEdgeImage();
        ed->getEdgeImage(edgeImg1);
        absdiff(edgeImg0, edgeImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;
        imwrite("edgePFImage0.png", edgeImg0);
        imwrite("edgePFImage1.png", edgeImg1);
        imwrite("diff2.png", diff);
        //*********************** EDCOLOR Edge Segment Detection from Color Images **********************

        Mat colorImg = imread(filename);
        tm.reset();
        tm.start();
        EDColor testEDColor = EDColor(colorImg, 36);
        tm.stop();
        cout << "-------------------------------------------------\n";
        cout << "testEDColor                        : " << tm.getTimeMilli() << endl;

        tm.reset();
        tm.start();
        // get lines from color image
        EDLines colorLine = EDLines(testEDColor);
        tm.stop();
        cout << "get lines from color image         : " << tm.getTimeMilli() << endl;

        tm.reset();
        tm.start();
        // get circles from color image
        EDCircles colorCircle = EDCircles(testEDColor);
        tm.stop();
        cout << "get circles from color image       : " << tm.getTimeMilli() << endl;
    }
    return 0;
}
