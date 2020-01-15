/*

CV project 2
Histogram Matching


By: Dan Stoianovici
2/5/2019

*/


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <Windows.h>

using namespace cv;
using namespace std;

//initialize Mat locations
Mat A;
Mat B;
Mat C;
Mat target, target_hist;
Mat source_, source_hist;



string targetFile = "grumpyman_target.jpg";
//string targetFile = "target_hist/leaf_target.jpg"

string sourceFile = "under_exposed.jpg";
//string imageFile = "sample_images/over_exposed.jpg";

void show_histogram(std::string const& name, cv::Mat1b const& image)
{
	// Set histogram bins count
	int bins = 256;
	int histSize[] = { bins };
	// Set ranges for histogram bins
	float lranges[] = { 0, 256 };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };

	// create matrix for histogram visualization
	int const hist_height = 256;
	cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

	cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

	double max_val = 0;
	minMaxLoc(hist, 0, &max_val);

	// visualize each bin
	for (int b = 0; b < bins; b++) {
		float const binVal = hist.at<float>(b);
		int   const height = cvRound(binVal*hist_height / max_val);
		cv::line
		(hist_image
			, cv::Point(b, hist_height - height), cv::Point(b, hist_height)
			, cv::Scalar::all(255)
		);
	}
	cv::imshow(name, hist_image);
}


//Function to equalize histogram

int main() {
	Mat target = imread(targetFile, CV_LOAD_IMAGE_GRAYSCALE); // Read target image
	Mat source = imread(sourceFile, CV_LOAD_IMAGE_GRAYSCALE);

	//Check for images
	if (target.empty()) // Check for invalid input
	{
		cout << "Could not open target image" << std::endl;
		while(1);
	}
	else if (source.empty()) // Check for invalid input
	{
		cout << "Could not open the source image" << std::endl;
		while (1);
	}

	////Display images
	//namedWindow("target image", WINDOW_AUTOSIZE);
	//imshow("target image", target);
	//waitKey(0);
	//namedWindow("source image", WINDOW_AUTOSIZE);
	//imshow("source image", source);
	//waitKey(0);
	
	cout << "# of Channels Target: " << target.channels() << endl;
	cout << "# of Channels Source: " << source.channels() << endl;
	

	//waitKey(1000);

	int bins = 256;
	int histSize[] = { bins };
	float histRanges[] = {0, 256};
	const float* ranges[] = { histRanges };
	//int 

	int const hist_height = 256;
	Mat3b target_hist = Mat3b::zeros(hist_height, bins);

	cv::calcHist(&target, 1, 0 , Mat(), &target_hist, 32 ,histSize, histRanges,true, false);


	return 0;
}