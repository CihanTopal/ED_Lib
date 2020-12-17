#include "EDLib.h"
#include "timer.h"

#include <iostream>
#include <chrono>
#include <experimental/filesystem>


using namespace cv;
using namespace std;

int main(int argc, char** argv)
{	
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " [img_dir]" << std::endl;
        return 1;
    }

    // Is directory or an image?
    std::string img_dir = argv[1];
    std::vector<cv::String> img_lists;
    // Image path or image file name.
    if (std::experimental::filesystem::is_directory(img_dir))
    {
        std::cout << "Reading img lists ..." << std::endl;
//        std::string file_pattern = img_dir + "/*" + img_format;
        std::string file_pattern = img_dir + "/*png";
        cv::glob(file_pattern, img_lists, false);
        if (img_lists.empty())
        {
            file_pattern = img_dir + "/*jpg";
            cv::glob(file_pattern, img_lists, false);
        }
        std::sort(img_lists.begin(), img_lists.end());
        assert(img_lists.size()!=0 && "No images in lists.");
    }
    else    // An image.
    {
        img_lists.push_back(cv::String(img_dir));
    }

    cv::Mat img;

    RNG rng(12345);
    // While loop
    while(!img_lists.empty())
    {
        auto img_name = img_lists.front();  // Image name.
        img = cv::imread(img_name, 0);     // Read img, must be gray.
        img_lists.erase(img_lists.begin());     // Then erase.

        // Show
        cv::imshow("src_img", img);

        // Timer
        Timer timer;
        //Call ED constructor
        ED testED = ED(img, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true); // apply ED algorithm
        std::cout << "ED Edge took: " << timer.getTimeUs() << " us." << std::endl;

        //Show resulting edge image
//        Mat edgeImg = testED.getEdgeImage();
//        imshow("Edge Image - PRESS ANY KEY TO CONTINUE", edgeImg);

        //Output number of segments
        int noSegments = testED.getSegmentNo();
        std::cout << "Number of edge segments: " << noSegments << std::endl;

        //Get edges in segment form (getSortedSegments() gives segments sorted w.r.t. legnths)
        std::vector< std::vector<Point> > segments = testED.getSegments();

        // Create black image.
        cv::Mat seg_img = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
        for (auto pts : segments)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(50,255), rng.uniform(50,255), rng.uniform(50,255));
//            cv::drawContours(seg_img, std::vector<std::vector<Point> >{pts}, -1, color);
            cv::polylines(seg_img, pts, false, color);
        }
        cv::imshow("seg_edges", seg_img);

        //Detection of lines segments from edge segments instead of input image
        //Therefore, redundant detection of edge segmens can be avoided
        timer.reset();
        EDLines testEDLines = EDLines(testED);
        std::cout << "ED Lines on edge img took: " << timer.getTimeUs() << " us." << std::endl;

        cv::Mat lineImg = testEDLines.drawOnImage();	//draws on the input image
        imshow("Line Image 2  - PRESS ANY KEY TO CONTINUE", lineImg);

        //Acquiring line information, i.e. start & end points
        vector<LS> lines = testEDLines.getLines();
        int noLines = testEDLines.getLinesNo();
        std::cout << "Number of line segments: " << noLines << std::endl;

        //************************** EDPF Parameter-free Edge Segment Detection **************************
        // Detection of edge segments with parameter free ED (EDPF)
        timer.reset();
        EDPF testEDPF = EDPF(img);
        std::cout << "EDPF took: " << timer.getTimeUs() << " us." << std::endl;

//        Mat edgePFImage = testEDPF.getEdgeImage();
//        imshow("Edge Image Parameter Free", edgePFImage);
        cout << "Number of edge segments found by EDPF: " << testEDPF.getSegmentNo() << endl;

        //Get edges in segment form (getSortedSegments() gives segments sorted w.r.t. legnths)
        segments = testEDPF.getSegments();

        // Create black image.
        seg_img = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
        for (auto pts : segments)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(50,255), rng.uniform(50,255), rng.uniform(50,255));
//            cv::drawContours(seg_img, std::vector<std::vector<Point> >{pts}, -1, color);
            cv::polylines(seg_img, pts, false, color);
        }
        cv::imshow("seg_edges_1", seg_img);

        // EDLines on EDPF
        timer.reset();
        testEDLines = EDLines(testEDPF);
        std::cout << "EDLines on EDPF took: " << timer.getTimeUs() << " us." << std::endl;

        lineImg = testEDLines.drawOnImage();	//draws on the input image
        imshow("Line Image EDPF  - PRESS ANY KEY TO CONTINUE", lineImg);

        int key = cv::waitKey(1);
        if ((key&0xff) == 27)   // esc
        {
            std::cout << "Quit.\n";
            break;
        }
        else if ((key&0xff) == 32)        // space
            cv::waitKey(0);
    }

    cv::waitKey(0);

    return 0;
}



