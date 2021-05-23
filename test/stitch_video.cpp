#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	const string file1("/home/vance/Pictures/good_result_0412/2021-04-12-ORMSLAM3-badResult-做对比用-hall2.mp4");
	const string file2("/home/vance/Pictures/good_result_0412/2021-04-12-PL-SLAM-Very-Good-Result.mp4");
	const string file3("/home/vance/Videos/ORBSLAM3_VS_OURS.avi");

	VideoCapture video1(file1);
	VideoCapture video2(file2);
	VideoWriter  video3;
	video3.open(file3, CV_FOURCC('M', 'J', 'P', 'G'), 30.0, Size(960, 540*2));

	if (!video1.isOpened() || !video2.isOpened()) {
		cerr << "Open video file error!" << endl;
		return -1;
	}

	const int step = 1;
	int frameCnt1 = 0, frameCnt2 = 0;

	Mat frame1, frame2;
	Mat frameStitch, frameOut;

	// skip 30 frames for video2
	while (frameCnt2++ < 90) {
		video2.read(frame2);
	}
	cout << "frame size: " << frame2.size() << endl;

	// stitch each frames
	while (1) {
		video1.read(frame1);
		video2.read(frame2);

		if (frame1.empty() || frame2.empty())
			break;

		if (frameCnt1++ % step != 0)
			continue;

		// putText(frame1, to_string(frameCnt1), Point(100, 100), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(0,0,255));
		// putText(frame2, to_string(frameCnt2), Point(100, 100), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(0,0,255));
		vconcat(frame1, frame2, frameStitch);
		resize(frameStitch, frameOut, Size(960, 540*2));
		// imshow("ORB_SLAM3(Left) & OURS(right)", frameOut);
		video3.write(frameOut);
		// waitKey(20);
	}

	video1.release();
	video2.release();
	video3.release();
	
	return 0;
}


