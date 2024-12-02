
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CUDA_Histogram.h"

using namespace std;
using namespace cv;

int main() {

	Mat Input_Image = imread("Test_Image.png", 0); // Read Gray Scale Image

	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;

	int Histogram_GrayScale[256] = { 0 };

	Histogram_Calculation_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Histogram_GrayScale);

	imwrite("Histogram_Image.png", Input_Image);

	for (int i = 0; i < 256; i++) {
		cout << "Histogram_GrayScale[" << i << "]: " << Histogram_GrayScale[i] << endl;
	}
	system("pause");//to stop when every thing executes
	return 0;
}
