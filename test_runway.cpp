#include "src/EDLib.h"
#include "timer.h"

#include <iostream>
#include <chrono>
#include <experimental/filesystem>


using namespace cv;
using namespace std;

const bool SAVE_TIME = true;

// Build Pyramid for roi_img
void buildPyramid(const cv::Mat& src_img, cv::Mat& dst_img, float& pyr_factor, int max_width=256)
{
    int src_size = std::max(src_img.rows, src_img.cols);
    // No need scale.
    if (src_size < max_width)
    {
        src_img.copyTo(dst_img);
        pyr_factor = 1.0;
        return;
    }

    // Calc pyramid factor.
    pyr_factor = static_cast<float>(src_size)/max_width;
    // Resize width & height.
    cv::Size resize(src_img.cols/pyr_factor, src_img.rows/pyr_factor);
    // dsize = Size(round(fx*src.cols), round(fy*src.rows)
    // (double)dsize.width/src.cols
    cv::resize(src_img, dst_img, resize);

}



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

    cv::Mat src_img, img;

    RNG rng(12345);

    std::FILE* time_took_ed = nullptr;
    if (SAVE_TIME)
    {
        std::string time_fname = "time_ed.txt";
        time_took_ed = std::fopen(time_fname.c_str(), "w");
        // Write header
        std::fprintf(time_took_ed, "IMG_CNT\tED\tEDPF\tED_LINES\n");
    }

    double ed_time = 0, edpf_time = 0,edline_time = 0;
    uint64_t img_cnt = 0;


    // While loop
    while(!img_lists.empty())
    {
        auto img_name = img_lists.front();  // Image name.
        src_img = cv::imread(img_name, 0);     // Read img, must be gray.
        img_lists.erase(img_lists.begin());     // Then erase.
        // Get image name.
        std::size_t start = img_dir.size();
//        std::size_t end = aimg.find(img_fmt);
        std::string img_nm = img_name.substr(start, img_name.size()-start);
        std::size_t mid = img_nm.find("_");
        img_cnt = std::stoul(img_nm.substr(0, mid));
        std::printf( "img_cnt: %ld\n", img_cnt);

        // Show
        cv::imshow("src_img", src_img);


        float pyr_factor = 1.0;
        buildPyramid(src_img, img, pyr_factor, 256);
        std::cout << "\033[33m" << "pyr_factor: " << pyr_factor
                  << "\033[0m" << std::endl;

        // Timer
        Timer timer;
        //Call ED constructor
        ED testED = ED(img, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true); // apply ED algorithm
        ed_time = timer.getTimeUs();
        std::cout << "ED Edge took: " << ed_time << " us." << std::endl;

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
        edline_time = timer.getTimeUs();
        std::cout << "ED Lines on edge img took: " << edline_time << " us." << std::endl;

//        cv::Mat lineImg = testEDLines.drawOnImage();	//draws on the input image
//        cv::Mat lineImg = img.clone();	//draws on the input image
//        cv::cvtColor(lineImg, lineImg, CV_GRAY2BGR);
        cv::Mat lineImg = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
        auto lines = testEDLines.getLines();
        for (auto aline : lines)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(50,255), rng.uniform(50,255), rng.uniform(50,255));
            cv::line( lineImg, aline.start, aline.end, color, 1, cv::LINE_AA);
        }
        imshow("Line Image 2  - PRESS ANY KEY TO CONTINUE", lineImg);

        //Acquiring line information, i.e. start & end points
//        vector<LS> lines = testEDLines.getLines();
        int noLines = testEDLines.getLinesNo();
        std::cout << "Number of line segments: " << noLines << std::endl;

        //************************** EDPF Parameter-free Edge Segment Detection **************************
        // Detection of edge segments with parameter free ED (EDPF)
        timer.reset();
        EDPF testEDPF = EDPF(img);
        edpf_time = timer.getTimeUs();
        std::cout << "EDPF took: " << edpf_time << " us." << std::endl;

        cv::Mat grad_img = testEDPF.getGradImage();
        cv::imshow("grad_img", grad_img);
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
        // double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3
        testEDLines = EDLines(testEDPF, 1.0, img.rows/4, img.rows/4);
        std::cout << "EDLines on EDPF took: " << timer.getTimeUs() << " us." << std::endl;

//        lineImg = testEDLines.drawOnImage();	//draws on the input image
        lineImg = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
        lines = testEDLines.getLines();
        for (auto aline : lines)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(50,255), rng.uniform(50,255), rng.uniform(50,255));
            cv::line( lineImg, aline.start, aline.end, color, 1, cv::LINE_AA);
        }
        imshow("Line Image EDPF  - PRESS ANY KEY TO CONTINUE", lineImg);

        // Write file.
        if (SAVE_TIME)
        {
            // Write time to file.
            std::fprintf(time_took_ed, "%ld\t%f\t%f\t%f\n", img_cnt, ed_time, edpf_time, edline_time);
        }

        int key = cv::waitKey(5);
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



