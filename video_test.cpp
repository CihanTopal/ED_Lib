#include "EDLib.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>


using namespace cv;
using namespace std;
using namespace cv::ximgproc;

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{scale|1|}"
        "{counter|99999|}"
        "{@filename|vtest.avi|}"
    );

    String filename = parser.get<string>("@filename");
    double scale = parser.get<double>("scale");
    int test_counter = parser.get<int>("counter");
    Mat src, gray;
    TickMeter tm0, tm1;
    int counter = 0;

    VideoCapture capture(samples::findFileOrKeep(filename));
    if (capture.isOpened())
    {
        cout << "Capture is opened" << endl;
        cout << "Frame [width,height] : ["  << capture.get(CAP_PROP_FRAME_WIDTH) << "," << capture.get(CAP_PROP_FRAME_HEIGHT) << "]" << endl;
        cout << "  scaled Frame [w,h] : [" << capture.get(CAP_PROP_FRAME_WIDTH) * scale << "," << capture.get(CAP_PROP_FRAME_HEIGHT) * scale << "]" << endl;;

        Ptr<EdgeDrawing> ed = createEdgeDrawing();
        ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
        ed->params.GradientThresholdValue = 36;
        ed->params.AnchorThresholdValue = 8;
        vector<Vec6d> ellipses;
        vector<Vec4f> lines;

        for (;;)
        {
            capture >> src;
            test_counter--;

            if (src.empty() || test_counter < 0)
                break;

            resize(src, src, Size(), scale, scale);
            cvtColor(src, gray, COLOR_BGR2GRAY);

            tm0.start();
            ed->detectEdges(gray);
            ed->detectLines(lines);
            ed->detectEllipses(ellipses);
            tm0.stop();
            counter++;

            tm1.start();
            ED testED = ED(gray, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true);
            EDLines testEDLines = EDLines(testED);
            EDCircles testEDCircles = EDCircles(testEDLines);
            tm1.stop();

            /*std::vector<LS> linesegments = testEDLines.getLines();

            Mat lineImg0 = testEDLines.getLineImage();    //draws on an empty image
            Mat lineImg1 = Mat(lineImg0.rows, lineImg0.cols, CV_8UC1, Scalar(255));

            for (int i = 0; i < lines.size(); i++)
                line(lineImg1, Point2d(lines[i][0], lines[i][1]), Point2d(lines[i][2], lines[i][3]), Scalar(0), 1, LINE_AA);

            Mat diff;
            absdiff(lineImg0, lineImg1, diff);
            imshow("", diff);
            waitKey();*/
        }

        cout << "OpenCV    processed " << counter << " frames in    " << tm0.getTimeMilli() << " ms.";
        cout << "\t\tfps : " << counter * 1000 / tm0.getTimeMilli() << endl;

        cout << "EDCircles processed " << counter << " frames in    " << tm1.getTimeMilli() << " ms.";
        cout << "\t\tfps : " << counter * 1000 / tm1.getTimeMilli() << endl;
    }
    return 0;
}
