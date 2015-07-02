// opencv3test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <chrono>
//#include <time>


using namespace std;
using namespace cv;

//This is very slow and should not be used, apparently opencv3 is not designed for color segmentation
int thresh3UMat(UMat image, Scalar lowerBounds, Scalar upperBounds)
{
	Mat mimage;
	Mat chans[3];
	UMat uchans[3];
	image.copyTo(mimage);
	inRange(mimage, lowerBounds, upperBounds, mimage);
	split(mimage, chans);
	
	for (int i = 0; i < 3; i++)
	{
		chans[i].copyTo(uchans[i]);
		threshold(uchans[i], uchans[i], lowerBounds[i], 255, THRESH_TOZERO);
		threshold(uchans[i], uchans[i], upperBounds[i], 255, THRESH_TOZERO_INV);
		uchans[i].copyTo(chans[i]);
	}
	
	/*
	for (int i = 0; i < 3; i++)
	{
		threshold(chans[i], chans[i], lowerBounds[i], 255, THRESH_TOZERO);
		threshold(chans[i], chans[i], upperBounds[i], 255, THRESH_TOZERO_INV);
	}
	*/
	merge(chans, 3, mimage);
	mimage.copyTo(image);
	return 0;
}


int _tmain(int argc, _TCHAR* argv[])
{	
	
	VideoCapture cap(0); // open the default camera
	//VideoCapture cap("E:/test.avi"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	//Creating erosion/dilation element
	int erosion_size = 1;
	Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));
	UMat uelement;
	element.copyTo(uelement);

	UMat developed, frame;
	UMat lowB, upB;
	lowB = UMat(1, 1, CV_8UC3, Scalar(50));
	upB = UMat(1, 1, CV_8UC3, Scalar(200));
	char capstr[50];
	char devstr[50];
	int frames = 0;
	bool isFrame;
	float capms = 0, procms = 0;
	namedWindow("frame", 1);
	namedWindow("developed", 1);
	std::chrono::time_point<std::chrono::system_clock> start, end, capstart, capend;
	std::chrono::duration<double> cap_elapsed, elapsed_seconds;
	for (;;)
	{
		frames++;

		capstart = std::chrono::system_clock::now();

		isFrame = cap.read(frame); // get a new frame from camera
		if (!isFrame) break;

		//Upscale resolution to better emulate computational complexity with 2 ximea cameras.
		resize(frame, frame, cv::Size(2560, 1024));
		developed = frame.clone();

		capend = std::chrono::system_clock::now();
		cap_elapsed = capend - capstart;
		capms += (cap_elapsed.count() * 1000.0);
		//std::cout << "Capturing took " << cap_elapsed.count() * 1000.0 << "ms to run.\n";

		start = std::chrono::system_clock::now();

		cvtColor(frame, developed, CV_BGR2HSV);
		cvtColor(developed, developed, CV_HSV2RGB);

		//cvtColor(developed, developed, CV_BGR2GRAY);

		GaussianBlur(developed, developed, Size(11, 11), 9, 9);

		//thresh3UMat(developed, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 0));
		//thresh3UMat(developed, cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 255));

		//threshold(developed, developed, 50, 255, THRESH_TOZERO);
		//threshold(developed, developed, 200, 255, THRESH_TOZERO_INV);
		//inRange(developed, lowB, upB, developed);

		erode(developed, developed, uelement);
		dilate(developed, developed, uelement);

		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		procms += (elapsed_seconds.count() * 1000);
		//std::cout << "Processing took " << elapsed_seconds.count() * 1000 << "ms to run.\n";


		//Assembling display windows
		//Raw capture window
		sprintf_s(capstr, "CAPDUR: %.2f ms", cap_elapsed.count() * 1000.0);
		putText(frame, capstr, Point2f(5, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
		sprintf_s(capstr, "AVG: %.2f ms", capms / frames);
		putText(frame, capstr, Point2f(360, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);

		//Processed window
		sprintf_s(devstr, "PROCDUR: %.2f ms", elapsed_seconds.count() * 1000);
		putText(developed, devstr, Point2f(5, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 2);
		sprintf_s(devstr, "RATE: %.2f fps", 1/elapsed_seconds.count());
		putText(developed, devstr, Point2f(5, 60), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 2);
		sprintf_s(devstr, "AVG: %.2f ms", procms / frames);
		putText(developed, devstr, Point2f(360, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 2);
		sprintf_s(devstr, "AVG: %.2f fps", frames / (procms / 1000));
		putText(developed, devstr, Point2f(360, 60), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 2);
		//Canny(edges, edges, 0, 30, 3);
		resize(frame, frame, cv::Size(960, 384));
		resize(developed, developed, cv::Size(960, 384));
		imshow("frame", frame);
		imshow("developed", developed);
		if (waitKey(30) >= 0) break;
	}
	cap.release();
	waitKey(0);
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

