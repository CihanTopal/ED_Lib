# ED_Lib
EDGE DRAWING LIBRARY FOR GEOMETRIC FEATURE EXTRACTION AND VALIDATION

Keywords: edge detection, edge segment detection, color edge detection, line detection, line segment detection, circle detection, ellipse detection.

Edge Drawing (ED) algorithm is an proactive approach on edge detection problem.
In contrast to many other existing edge detection algorithms which follow a subtractive approach (i.e. after applying gradient filters onto an image eliminating pixels w.r.t. several rules, e.g. non-maximal suppression and hysteresis in Canny), ED algorithm works via an additive strategy, i.e. it picks edge pixels one by one, hence the name Edge Drawing.
Then we process those random shaped edge segments to extract higher level edge features, i.e. lines, circles, ellipses, etc.
The popular method of extraction edge pixels from the thresholded gradient magnitudes is non-maximal supressiun that tests every pixel whether it has the maximum gradient response along its gradient direction and eliminates if it does not.
However, this method does not check status of the neighboring pixels, and therefore might result low quality (in terms of edge continuity, smoothness, thinness, localization) edge segments.
Instead of non-maximal supression, ED points a set of edge pixels and join them by maximizing the total gradient response of edge segments.
Therefore it can extract high quality edge segments without need for an additional hysteresis step. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The list of algorithms in this library:

ED: Edge Drawing detection algorithm. Detects edge segments in an input image and provides the result in segment form (a vector of edge segment pixels).

Paper: C. Topal, C. Akinlar, Edge Drawing: A Combined Real-Time Edge and Segment Detector, Journal of Visual Communication and Image Representation, vol.23, no.6, pp.862-872, August 2012.

[https://www.sciencedirect.com/science/article/pii/S1047320312000831]

Demo video: https://www.youtube.com/watch?v=-Bpb_OLfOts

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

EDPF: Edge Drawing - Parameter Free: Detects edge segments without need for parameter tuning. It runs naive ED with all parameters at their extreme to detect all possible edge segments, then invalidates false detected segments due to Helmholtz Principle.

Paper: C. Akinlar, C. Topal, EDPF: A Real-time Parameter-free Edge Segment Detector with a False Detection Control, Int’l Journal of Pattern Recognition and Artificial Intelligence, vol.26, no.1, 2012.

[https://doi.org/10.1142/S0218001412550026]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

EDLines: Detects line segments in the image and provides the result as a vector list consisting of starting and ending points.
EDLines is alos a parameter-free algorithm which validates all detected lines via Helmholtz Principle.

Paper: C. Akinlar, C. Topal, EDLines: A real-time line segment detector with a false detection control, Pattern Recognition Letters, vol.32, iss.13, pg. 1633-1642, October 2011. 

[https://www.sciencedirect.com/science/article/pii/S0167865511001772]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

EDCircles: Detects circles and ellipses (up to a level of eccentricity) in an input image and returns the result in a list of circle and ellipse parameters. Just like EDPF and EDLines, EDCircles is alos a parameter-free algorithm that applies validation via Helmholtz Principle.

Paper: C. Akinlar, C. Topal, EDCircles: A Real-time Circle Detector with a False Detection Control, Pattern Recognition, 46(3), 725-740, March 2013.

[https://www.sciencedirect.com/science/article/pii/S0031320312004268]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

EDColor: Edge Drawing - Color: Detects edges in segment form on color image using a color gradient computation, hence provides better results than converting the image to grayscale.

Paper: C. Akinlar, C. Topal, ColorED: Color edge and segment detection by Edge Drawing (ED), Journal of Visual Communication and Image Representation, vol.44, pp.82-94, April 2017.

[https://www.sciencedirect.com/science/article/pii/S1047320317300305]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Instructions:
You can detect edge segments with ED, EDPF and EDColor; and lines and circles with EDLines and EDCircles, easily.
It is also possible to feed ED instances to EDLines or EDCircles to extract line segments, circles and ellipses without need to run edge segment detection again.
In this way, redundant computation can be avoided.
Results might slightly differ between using directly EDLines and ED + EDX (EDLines, EDCircles, etc.) because of the differences in default gradient operators and parameters in the algorithms.
There are several usage scenarios available in test.cpp. 

Dependencies: OpenCV 3.4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Disclaimer for Edge Drawing Library

This software library is provided "as is" and "with all faults." Authors of this library make no warranties of any kind concerning the safety, suitability, lack of viruses, inaccuracies, typographical errors, or other harmful components of this software product. 
There are inherent dangers in the use of any software, and you are solely responsible for determining whether this software product is compatible with your equipment and other software installed on your equipment. 
You are also solely responsible for the protection of your equipment and backup of your data, and the authors of this software product will not be liable for any damages you may suffer in connection with using, modifying, or distributing this software product. 
By using this library in any scientific work, you are implicitly presumed to have accepted all of the above statements, and accept to cite the aforementioned papers.
Please contact for commercial use.

