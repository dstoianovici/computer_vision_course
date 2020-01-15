/*

Bayer Pattern Demosaicing Implementation for BGGR pattern
usinng standard Bilinear and advanced Malvar-He-Cutler (MHC) Interpolation
Computer Vision Project 1


By: Dan Stoianovici
1/29/2019

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

//Variabeles for padding
int top, bottom;
int left1, right1;
int borderType = BORDER_REPLICATE;
//int borderType = BORDER_CONSTANT;

//Initialize processing matricies
Mat A;
Mat B; //Padding
Mat C; //Bilinear
Mat D; //Melvar-He-Cutler
Mat E; //Comparisson Image

//Set File Here from included Samples by uncommenting
//string FileName = "peppersBayer";
string FileName = "pearsBayer";
//string FileName = "onionBayer";
//string FileName = "officeBayer";




/*
Bileaner Interpolation functions
*/

	//Green Pixel Demozaic BGR Format
	Mat on_green(Mat in, Mat out, int i, int j, int pad) {

		Vec3b intensity_in = in.at<Vec3b>(i, j ); //Reads the intensity per channel
		//cout << intensity_in << endl;
		uchar green_in = intensity_in.val[0]; //All values in Mat_IN are the same, taking just one value. 
	
		Vec3b intensity_out = out.at<Vec3b>(i-pad, j - pad); //Acess intensity from blank image
	
		//get neighboring pixel values from gray image
		uchar r_up = in.at<Vec3b>(i, j-1).val[0]; 
		uchar r_down = in.at<Vec3b>(i, j+1).val[0];
		uchar b_left = in.at<Vec3b>(i-1, j).val[0];
		uchar b_right= in.at<Vec3b>(i+1, j).val[0];

		uchar blue_out = (r_up + r_down) / 2; //interpolate blue
		uchar green_out = green_in;
		uchar red_out = (b_left + b_right) / 2; //interpolate green

		intensity_out.val[0] = blue_out; //write new value to out pixel
		intensity_out.val[1] = green_in;
		intensity_out.val[2] = red_out;
		out.at<Vec3b>(i - pad, j - pad) = intensity_out;
		//cout << intensity_out;
		return out;

		//cout << "Green: " << "[" << (int)blue_out << "," << (int)green_out << "," << (int)red_out << "]" << endl;

	}

	//Blue Pixel Demoziac 
	Mat on_blue(Mat in, Mat  out, int i, int j, int pad) {

		Vec3b intensity_in = in.at<Vec3b>(i, j); //Reads the intensity per channel
	
		uchar blue_in = intensity_in.val[0]; //All values in Mat IN are the same, taking just one value. 

		Vec3b intensity_out = out.at<Vec3b>(i - pad, j - pad); //access intensity from blank image

		//get neighboring pixel values from gray image
		uchar r_up_right = in.at<Vec3b>(i-1, j+1).val[0];
		uchar r_up_left = in.at<Vec3b>(i-1, j-1).val[0];
		uchar r_down_right = in.at<Vec3b>(i+1, j+1).val[0];
		uchar r_down_left = in.at<Vec3b>(i+1, j-1).val[0];

		uchar g_up = in.at<Vec3b>(i-1, j).val[0];
		uchar g_down = in.at<Vec3b>(i+1, j).val[0];
		uchar g_left = in.at<Vec3b>(i, j-1).val[0];
		uchar g_right = in.at<Vec3b>(i, j+1).val[0];


		//interpolating
		uchar blue_out = blue_in;
		uchar green_out = (g_up + g_down + g_left +g_right) / 4;
		uchar red_out = (r_up_right + r_up_left + r_down_right + r_down_left) / 4;


		//write new value to out pixel
		intensity_out.val[0] = blue_out; 
		intensity_out.val[1] = green_out;
		intensity_out.val[2] = red_out;
		out.at<Vec3b>(i - pad, j - pad) = intensity_out;
		//cout << intensity_out;

		return out;

		//cout << "Blue: " << "[" << (int)blue_out << "," << (int)green_out << "," << (int)red_out << "]" << endl;
	}

	//Red Pixel Demoziac 
	Mat on_red(Mat in, Mat out, int i, int j, int pad) {

		Vec3b intensity_in = in.at<Vec3b>(i, j); //Reads the intensity per channel
		//cout << intensity_in << endl;
		uchar red_in = intensity_in.val[0]; //All values in Mat IN are the same, taking just one value. 

		Vec3b intensity_out = out.at<Vec3b>(i - pad, j - pad); //access intensity from blank image

		//get neighboring pixel values from gray image
		uchar b_up_right = in.at<Vec3b>(i - 1, j + 1).val[0];
		uchar b_up_left = in.at<Vec3b>(i - 1, j - 1).val[0];
		uchar b_down_right = in.at<Vec3b>(i + 1, j + 1).val[0];
		uchar b_down_left = in.at<Vec3b>(i + 1, j - 1).val[0];

		uchar g_up = in.at<Vec3b>(i - 1, j).val[0];
		uchar g_down = in.at<Vec3b>(i + 1, j).val[0];
		uchar g_left = in.at<Vec3b>(i, j - 1).val[0];
		uchar g_right = in.at<Vec3b>(i, j + 1).val[0];


		//interpolating
		uchar blue_out = (b_up_right + b_up_left + b_down_right + b_down_left) / 4;
		uchar green_out = (g_up + g_down + g_left + g_right) / 4;
		uchar red_out = red_in;

		//write new value to out pixel
		intensity_out.val[0] = blue_out; 
		intensity_out.val[1] = green_out;
		intensity_out.val[2] = red_out;

		out.at<Vec3b>(i - pad, j - pad) = intensity_out;
		//cout << intensity_out;
		return out;

		//cout << "Red: " << "[" << (int)blue_out << "," << (int)green_out << "," << (int)red_out << "]" << endl;
	}


/*
Advanced Melvar-He-Culter interpolation functions
*/

	Mat MHC_blue(Mat in, Mat out, int i, int j, int pad) {

		Vec3b intensity_IN = in.at<Vec3b>(i, j); //Reads the intensity per channel
	
		uchar blue_IN = intensity_IN.val[0]; //All values in Mat IN are the same, taking just one value. 

		Vec3b intensity_OUT = out.at<Vec3b>(i - pad, j - pad); //access intensity from blank image

		//get neighboring pixel values from gray image
		uchar r_ur = in.at<Vec3b>(i - 1, j + 1).val[0];
		uchar r_ul = in.at<Vec3b>(i - 1, j - 1).val[0];
		uchar r_dr = in.at<Vec3b>(i + 1, j + 1).val[0];
		uchar r_dl = in.at<Vec3b>(i + 1, j - 1).val[0];

		uchar g_u = in.at<Vec3b>(i - 1, j).val[0];
		uchar g_d = in.at<Vec3b>(i + 1, j).val[0];
		uchar g_l = in.at<Vec3b>(i, j - 1).val[0];
		uchar g_r = in.at<Vec3b>(i, j + 1).val[0];

		uchar b_uu = in.at<Vec3b>(i - 2, j).val[0];
		uchar b_dd = in.at<Vec3b>(i + 2, j).val[0];
		uchar b_ll = in.at<Vec3b>(i, j - 2).val[0];
		uchar b_rr = in.at<Vec3b>(i, j + 2).val[0];

		//interpolating
		uchar blue_OUT = blue_IN;
		uchar green_OUT = (0.125) * (4*(blue_IN) + 2*(g_u + g_d + g_l + g_r) + (-1)*(b_uu + b_dd + b_ll + b_rr));	
		uchar red_OUT = (0.125) * (6*(blue_IN) + (-1.5)*(b_uu + b_dd + b_ll + b_rr) + 2*(r_ur + r_ul + r_dr + r_dl));

		//write new value to out pixel
		intensity_OUT.val[0] = blue_OUT;
		intensity_OUT.val[1] = green_OUT;
		intensity_OUT.val[2] = red_OUT;

		out.at<Vec3b>(i - pad, j - pad) = intensity_OUT;
		//cout << "blue" <<intensity_out <<endl;
		//cout << "blue: " << "[" << (int)blue_OUT << "," << (int)green_OUT << "," << (int)red_OUT << "]" << endl;
		return out;

	
	}

	Mat MHC_red(Mat in, Mat out, int i, int j, int pad) {

		Vec3b intensity_in = in.at<Vec3b>(i, j); //Reads the intensity per channel
		//cout << intensity_in << endl;
		uchar red_in = intensity_in.val[0]; //All values in Mat IN are the same, taking just one value. 

		Vec3b intensity_out = out.at<Vec3b>(i - pad, j - pad); //access intensity from blank image

		//get neighboring pixel values from gray image
		uchar b_ur = in.at<Vec3b>(i - 1, j + 1).val[0];
		uchar b_ul = in.at<Vec3b>(i - 1, j - 1).val[0];
		uchar b_dr = in.at<Vec3b>(i + 1, j + 1).val[0];
		uchar b_dl = in.at<Vec3b>(i + 1, j - 1).val[0];

		uchar g_up = in.at<Vec3b>(i - 1, j).val[0];
		uchar g_down = in.at<Vec3b>(i + 1, j).val[0];
		uchar g_left = in.at<Vec3b>(i, j - 1).val[0];
		uchar g_right = in.at<Vec3b>(i, j + 1).val[0];

		uchar r_uu = in.at<Vec3b>(i - 2, j).val[0];
		uchar r_dd = in.at<Vec3b>(i + 2, j).val[0];
		uchar r_ll = in.at<Vec3b>(i, j - 2).val[0];
		uchar r_rr = in.at<Vec3b>(i, j + 2).val[0];

		//interpolating
		uchar blue_out = (0.125) * (6*(red_in) + (-1.5)*(r_uu + r_dd + r_ll + r_rr) + 2*(b_ur + b_ul + b_dr + b_dl));;
		uchar green_out = (0.125) * (4*(red_in) + 2*(g_up + g_down + g_left + g_right) + (-1)*(r_uu + r_dd + r_ll + r_rr));
		uchar red_out = red_in;

		//write new value to out pixel
		intensity_out.val[0] = blue_out;
		intensity_out.val[1] = green_out;
		intensity_out.val[2] = red_out;

		out.at<Vec3b>(i - pad, j - pad) = intensity_out;
		//cout << "red" << intensity_out << endl;
		//cout << "Red: " << "[" << (int)blue_out << "," << (int)green_out << "," << (int)red_out << "]" << endl;
		return out;

	
	}

	Mat MHC_green_rb(Mat in, Mat out, int i, int j, int pad) {

		Vec3b intensity_in = in.at<Vec3b>(i, j); //Reads the intensity per channel

		uchar green_in = intensity_in.val[0]; //All values in Mat IN are the same, taking just one value. 

		Vec3b intensity_out = out.at<Vec3b>(i - pad, j - pad); //access intensity from blank image

		//get neighboring pixel values from gray image
		uchar r_l = in.at<Vec3b>(i, j - 1).val[0];
		uchar r_r = in.at<Vec3b>(i, j + 1).val[0];

		uchar g_ul = in.at<Vec3b>(i - 1, j - 1).val[0];
		uchar g_ur = in.at<Vec3b>(i - 1, j + 1).val[0];
		uchar g_dr = in.at<Vec3b>(i + 1, j + 1).val[0];
		uchar g_dl = in.at<Vec3b>(i + 1, j - 1).val[0];

		uchar b_u = in.at<Vec3b>(i - 1, j).val[0];
		uchar b_d = in.at<Vec3b>(i + 1, j).val[0];


		uchar g_uu = in.at<Vec3b>(i - 2, j).val[0];
		uchar g_dd = in.at<Vec3b>(i + 2, j).val[0];
		uchar g_ll = in.at<Vec3b>(i, j - 2).val[0];
		uchar g_rr = in.at<Vec3b>(i, j + 2).val[0];

		//interpolating
		uchar blue_out = (.125) * (5*(green_in) + 4*(b_u + b_d) + (-1)*(g_ul + g_ur + g_dl + g_dr + g_uu + g_dd) + (0.5)*(g_ll + g_rr));
		uchar green_out = green_in;
		uchar red_out = (.125) *  (5*(green_in) + 4*(r_l + r_r) + (-1)*(g_ul + g_ur + g_dl + g_dr + g_ll + g_rr) + (0.5)*(g_uu + g_dd));

		//write new value to out pixel
		intensity_out.val[0] = blue_out;
		intensity_out.val[1] = green_out;
		intensity_out.val[2] = red_out;

		out.at<Vec3b>(i - pad, j - pad) = intensity_out;
		//cout << "green_rb" << intensity_out << endl;
		//cout << "green rb: " << "[" << (int)blue_out << "," << (int)green_out << "," << (int)red_out << "]" << endl;
		return out;

	
	}

	Mat MHC_green_br(Mat in, Mat out, int i, int j, int pad) {

		Vec3b intensity_in = in.at<Vec3b>(i, j); //Reads the intensity per channel
		uchar green_in = intensity_in.val[0]; //All values in Mat IN are the same, taking just one value. 

		Vec3b intensity_out = out.at<Vec3b>(i - pad, j - pad); //access intensity from blank image

		//get neighboring pixel values from gray image
		uchar b_l = in.at<Vec3b>(i, j - 1).val[0];
		uchar b_r = in.at<Vec3b>(i, j + 1).val[0];

		uchar g_ul = in.at<Vec3b>(i - 1, j - 1).val[0];
		uchar g_ur = in.at<Vec3b>(i - 1, j + 1).val[0];
		uchar g_dr = in.at<Vec3b>(i + 1, j + 1).val[0];
		uchar g_dl = in.at<Vec3b>(i + 1, j - 1).val[0];

		uchar r_u = in.at<Vec3b>(i - 1, j).val[0];
		uchar r_d = in.at<Vec3b>(i + 1, j).val[0];


		uchar g_uu = in.at<Vec3b>(i - 2, j).val[0];
		uchar g_dd = in.at<Vec3b>(i + 2, j).val[0];
		uchar g_ll = in.at<Vec3b>(i, j - 2).val[0];
		uchar g_rr = in.at<Vec3b>(i, j + 2).val[0];

		//interpolating
		uchar blue_out = (0.125) * (5*(green_in) + 4*(b_l + b_r) + (-1)*(g_ul + g_ur + g_dl + g_dr + g_ll + g_rr) + (0.5)*(g_uu + g_dd));
		uchar green_out = green_in;
		uchar red_out = (0.125) *  (5*(green_in)+ 4*(r_u + r_d) + (-1)*(g_ul + g_ur + g_dl + g_dr + g_dd + g_uu) + (0.5)*(g_ll + g_rr));

		//write new value to out pixel
		intensity_out.val[0] = blue_out;
		intensity_out.val[1] = green_out;
		intensity_out.val[2] = red_out;

		out.at<Vec3b>(i - pad, j - pad) = intensity_out;
		//cout <<"green_br" << intensity_out <<endl;
		//cout << "green_br: " << "[" << (int)blue_out << "," << (int)green_out << "," << (int)red_out << "]" << endl;
		return out;

	
	}


/*
PSNR Algorithm from OpenCV tutorials
https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
*/
	double getPSNR(const Mat& I1, const Mat& I2)
	{
		Mat s1;
		absdiff(I1, I2, s1);       // |I1 - I2|
		s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
		s1 = s1.mul(s1);           // |I1 - I2|^2

		Scalar s = sum(s1);         // sum elements per channel

		double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		if (sse <= 1e-10) // for small values return zero
			return 0;
		else
		{
			double  mse = sse / (double)(I1.channels() * I1.total());
			double psnr = 10.0*log10((255 * 255) / mse);
			return psnr;
		}
	}






int main()
{
	//Read Bayer Image into matrix with header A. 
	Mat A = imread(FileName+".png"); // Read the file
	cout << "Image: " + FileName << endl;
	//Check for image
	if (A.empty()) // Check for invalid input
	{
		cout << "Could not open or find the original image" << std::endl;
		while (1);
	}

	//Print Matrix size
	int rows = A.rows;
	int cols = A.cols;

	Size s = A.size();
	rows = s.height;
	cols = s.width;
	cout << "Size A: " << s << endl << "Rows: " << rows << endl << "Columns: " << cols << endl;

	//Create matrix for Storing Demosaiced Image BiLinear
	Mat C(rows, cols, CV_8UC3, Scalar(0, 0, 0)); 
	cout << "Size C: " << C.size() << endl;

	//Create matrix for Storing Demosaiced Image Malvar_He_Cutler (MHC) Interpolation
	Mat D(rows, cols, CV_8UC3, Scalar(0, 0, 0));
	cout << "Size D: " << C.size() << endl;
	
	
	//////Display Bayer Image
	////namedWindow("Bayer Image", WINDOW_AUTOSIZE); // Create a window for display.
	////imshow("Bayer Image", A);                   // Show our image inside it.
	////waitKey(0); // Wait for a keystroke in the window

	//Pad Image
	int padSize = 11; //must be odd number
	RNG rng(12345);
	Scalar value(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	top = padSize;
	bottom = top;
	cout << "Width of Horizontal Padding: " << top << endl;
	left1 = padSize;
	cout << "Width of Vertical Padding: " << left1 << endl;
	right1 = left1;
	cv::copyMakeBorder(A, B, top, bottom, left1, right1, borderType,value);

	////Show size of B
	//cout << "Size B: " << B.size() << endl;
	//cout << "Size A: " << A.size() << endl;

	//Show Padded Image
	namedWindow("Padded Bayer Image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Padded Bayer Image", B);                   // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window 
	


	//BiLinear Interpolation
	for (int i = top; i < (A.rows + padSize -1); i++)
	{
		for (int j = left1; j < (A.cols+ padSize -1); j++)
		{
			

			if (i%2 != 0 && j%2 != 0) {
				//cout << i << "," << j << endl;
				C = on_blue(B, C, i, j, padSize);
				//Vec3b intensity_in = C.at<Vec3b>(i - padSize, j - padSize); //Reads the intensity per channel
				//cout << intensity_in << endl;
				
				//cout << "blue" << endl;
			}

			else if (i%2 == 0 && j%2 == 0) {
				//cout << i << "," << j << endl;
				C = on_red(B, C, i, j, padSize);
				//Vec3b intensity_in = C.at<Vec3b>(i - padSize, j - padSize); //Reads the intensity per channel
				//cout << intensity_in << endl;
				//cout << "red" << endl;
			}

			else {
				//cout << i << "," << j << endl;
				C = on_green(B, C, i, j, padSize);
				//Vec3b intensity_in = C.at<Vec3b>(i-padSize, j - padSize); //Reads the intensity per channel
				//cout << intensity_in << endl;
				//cout << "green" << endl;
			}

			//cout << (int)B.at<uchar>(i, j) << " "; break;
			//cout << intensity << " ";
		}
		//cout << endl;
		//waitKey(1000);
		//break;
	}

	
	//Show Demozaiced Image BiLinear Interpolation
	namedWindow("Demozaiced Image BiLinear", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Demozaiced Image BiLinear", C);                 
	waitKey(0);


	//Write C to processed folder
	imwrite("processed/" + FileName + "_demosiac_bilinear.png", C);
	waitKey(500);
	cout << "Image C Written" << endl;
	waitKey(100);


	//Advanced Melvar-He-Cutler Interpolation
	for (int i = top; i < (A.rows + padSize - 1); i++) //moves in y
	{
		for (int j = left1; j < (A.cols + padSize - 1); j++) //moves in x
		{


			if (i % 2 != 0 && j % 2 != 0) { //blue
				//cout << i << "," << j << endl;
				//cout << intensity_in << endl;
				D = MHC_blue(B, D, i, j, padSize);
				//cout << "blue" << endl;
			}

			else if (i % 2 == 0 && j % 2 == 0) { //red
				//cout << i << "," << j << endl;
				
				D = MHC_red(B, D, i, j, padSize);
				//cout << intensity_in << endl;
				//cout << "red" << endl;
			}

			else if (i % 2 == 0 && j % 2 != 0) { //Green on blue row & red col
				//cout << i << "," << j << endl;
				//cout << intensity_in << endl;
				D = MHC_green_rb(B, D, i, j, padSize);
				//cout << "green_rb" << endl;
			}
			else if (i % 2 != 0 && j % 2 == 0) { //Green on red row & blue col
				//cout << i << "," << j << endl;
				//cout << intensity_in << endl;
				D = MHC_green_br(B, D, i, j, padSize);
				//cout << "green_br" << endl;

			}


			

			//cout << (int)B.at<uchar>(i, j) << " "; break;
			//cout << intensity << " ";
		}
	}

	//Show Demozaiced Image MHC INterpolation
	imshow("Demozaiced Image MHC", D);                   
	waitKey(0);

	//Write D to processed folder
	imwrite("processed/"+FileName+"_demosiac_MHC.png", D);
	waitKey(500);
	cout << "Image D Written" << endl;
	waitKey(100);

	//Open Comparisson Image for PSNR comparisson
	Mat E = imread(FileName + "_color.png"); // Read the file

	//Check for image
	if (E.empty()) // Check for invalid input
	{
		cout << "Could not open or find the comparisson image" << std::endl;
		while (1);
	}
	//check image size
	cout << "Size E:" << E.size() << endl;

	cout << "Bilinear PSNR: " << getPSNR(E, C) << endl;
	cout << "Malvar-He-Cutler PSNR: " << getPSNR(E, D) << endl;



	//Poor quit button implementation
	char k;
	cout << "enter q to quit:" << endl;
	while (1) {
		cin >> k;

		if (k == 'q') {
			return 0;
		}
	}
}