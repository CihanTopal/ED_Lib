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
        Mat anchImg0 = testED.getAnchorImage();
        Mat gradImg0 = testED.getGradImage();
        Mat edgeImg1, diff;
        ed->getEdgeImage(edgeImg1);
        absdiff(edgeImg0, edgeImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;
        imwrite("EqualGradImage.png", gradImg0);
        imwrite("EqualAnchorImage.png", anchImg0);
        imwrite("EqualEdgeImage.png", edgeImg0);

        //***************************** EDLINES Line Segment Detection *****************************
        //Detection of lines segments from edge segments instead of input image
        //Therefore, redundant detection of edge segmens can be avoided
        tm.reset();
        tm.start();
        EDLines testEDLines = EDLines(testED);
        tm.stop();
        cout << "-------------------------------------------------\n";
        cout << "testEDLines.getLineImage()         : " << tm.getTimeMilli() << endl;
        Mat lineImg0 = testEDLines.getLineImage();
        imwrite("EqualLineImg.png", lineImg0);

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

        vector<mCircle> found_circles = testEDCircles.getCircles();
        vector<mEllipse> found_ellipses = testEDCircles.getEllipses();
        Mat ellipsImg0, ellipsImg1;
        cvtColor(testImg, ellipsImg0, COLOR_GRAY2BGR);
        cvtColor(testImg, ellipsImg1, COLOR_GRAY2BGR);

        for (int i = 0; i < found_circles.size(); i++)
        {
            Point center((int)found_circles[i].center.x, (int)found_circles[i].center.y);
            Size axes((int)found_circles[i].r, (int)found_circles[i].r);
            double angle(0.0);
            Scalar color = Scalar(0, 255, 0);

            ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 1, LINE_AA);
        }

        for (int i = 0; i < found_ellipses.size(); i++)
        {
            Point center((int)found_ellipses[i].center.x, (int)found_ellipses[i].center.y);
            Size axes((int)found_ellipses[i].axes.width, (int)found_ellipses[i].axes.height);
            double angle = found_ellipses[i].theta * 180 / CV_PI;
            Scalar color = Scalar(255, 255, 0);

            ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 1, LINE_AA);
        }

        for (size_t i = 0; i < ellipses.size(); i++)
        {
            Point center((int)ellipses[i][0], (int)ellipses[i][1]);
            Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
            double angle(ellipses[i][5]);
            Scalar color = ellipses[i][2] == 0 ? Scalar(255, 255, 0) : Scalar(0, 255, 0);

            ellipse(ellipsImg1, center, axes, angle, 0, 360, color, 1, LINE_AA);
        }

        imwrite("EllipsImg0.png", ellipsImg0);
        imwrite("EllipsImg1.png", ellipsImg1);

        //************************** EDPF Parameter-free Edge Segment Detection **************************
        // Detection of edge segments with parameter free ED (EDPF)
        tm.reset();
        tm.start();
        EDPF testEDPF = EDPF(testImg);
        tm.stop();
        cout << "testEDPF.getEdgeImage()            : " << tm.getTimeMilli() << endl;

        Ptr<EdgeDrawing> ed1 = createEdgeDrawing();
        ed1->params.EdgeDetectionOperator = EdgeDrawing::PREWITT;
        ed1->params.GradientThresholdValue = 11;
        ed1->params.AnchorThresholdValue = 3;
        ed1->params.PFmode = true;

        tm.reset();
        tm.start();
        ed1->detectEdges(testImg);
        tm.stop();
        std::cout << "detectEdges()  PF        (OpenCV)  : " << tm.getTimeMilli() << endl;

        edgeImg0 = testEDPF.getEdgeImage();
        ed->getEdgeImage(edgeImg1);
        absdiff(edgeImg0, edgeImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;
        imwrite("EdgePFImage0.png", edgeImg0);
        imwrite("EdgePFImage1.png", edgeImg1);
        imwrite("DiffEdgePFImage.png", diff);
        //*********************** EDCOLOR Edge Segment Detection from Color Images **********************

        Mat colorImg = imread(filename);
        tm.reset();
        tm.start();
        EDColor testEDColor = EDColor(colorImg, 36);
        edgeImg0 = testEDColor.getEdgeImage();
        tm.stop();
        cout << "-------------------------------------------------\n";
        cout << "testEDColor.getEdgeImage()         : " << tm.getTimeMilli() << endl;

        Ptr<EdgeDrawing> ed2 = createEdgeDrawing();
        ed2->params.EdgeDetectionOperator = EdgeDrawing::PREWITT;
        ed2->params.GradientThresholdValue = 36;
        ed2->params.AnchorThresholdValue = 4;
        ed2->params.Sigma = 1.5;

        tm.reset();
        tm.start();
        ed2->detectEdges(colorImg);
        tm.stop();
        cout << "detectEdges()            (OpenCV)  : " << tm.getTimeMilli() << endl;

        ed2->getEdgeImage(edgeImg1);
        absdiff(edgeImg0, edgeImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;
        imwrite("EqualColorEdgeImg.png", edgeImg0);

        tm.reset();
        tm.start();
        // get lines from color image
        EDLines colorLine = EDLines(testEDColor);
        tm.stop();
        cout << "-------------------------------------------------\n";
        cout << "get lines from color image         : " << tm.getTimeMilli() << endl;
        lineImg0 = colorLine.getLineImage();
        imwrite("EqualColorLineImg.png", lineImg0);
        tm.reset();
        tm.start();
        ed2->detectLines(lines);
        tm.stop();
        cout << "detectLines()            (OpenCV)  : " << tm.getTimeMilli() << endl;

        lineImg1 = Mat(lineImg0.rows, lineImg0.cols, CV_8UC1, Scalar(255));

        for (int i = 0; i < lines.size(); i++)
            line(lineImg1, Point2d(lines[i][0], lines[i][1]), Point2d(lines[i][2], lines[i][3]), Scalar(0), 1, LINE_AA);

        absdiff(lineImg0, lineImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;
        cout << "-------------------------------------------------\n";
        tm.reset();
        tm.start();
        // get circles from color image
        EDCircles colorCircle = EDCircles(testEDColor);
        tm.stop();
        cout << "get circles from color image       : " << tm.getTimeMilli() << endl;
        tm.reset();
        tm.start();
        ed2->detectEllipses(ellipses);
        tm.stop();
        cout << "detectEllipses()         (OpenCV)  : " << tm.getTimeMilli() << endl;

        found_circles = colorCircle.getCircles();
        found_ellipses = colorCircle.getEllipses();

        ellipsImg0 = colorImg.clone();
        ellipsImg1 = colorImg.clone();
        for (int i = 0; i < found_circles.size(); i++)
        {
            Point center((int)found_circles[i].center.x, (int)found_circles[i].center.y);
            Size axes((int)found_circles[i].r, (int)found_circles[i].r);
            double angle(0.0);
            Scalar color = Scalar(0, 255, 0);

            ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 2, LINE_AA);
        }

        for (int i = 0; i < found_ellipses.size(); i++)
        {
            Point center((int)found_ellipses[i].center.x, (int)found_ellipses[i].center.y);
            Size axes((int)found_ellipses[i].axes.width, (int)found_ellipses[i].axes.height);
            double angle = found_ellipses[i].theta * 180 / CV_PI;
            Scalar color = Scalar(255, 255, 0);

            ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 2, LINE_AA);
        }



        for (size_t i = 0; i < ellipses.size(); i++)
        {
            Point center((int)ellipses[i][0], (int)ellipses[i][1]);
            Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
            double angle(ellipses[i][5]);
            Scalar color = ellipses[i][2] == 0 ? Scalar(255, 255, 0) : Scalar(0, 255, 0);

            ellipse(ellipsImg1, center, axes, angle, 0, 360, color, 2, LINE_AA);
        }

        imwrite("ColorEllipsImg0.png", ellipsImg0);
        imwrite("ColorEllipsImg1.png", ellipsImg1);

    }
    return 0;
}
