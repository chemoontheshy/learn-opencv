#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <time.h>
#include <vector>
#include <cctype>
#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>

using namespace cv;
using namespace std;

enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };


class calib
{
public:
	const char* liveCaptureHelp =
		"When the live video from camera is used as input, the following hot-keys may be used:\n"
		"  <ESC>, 'q' - quit the program\n"
		"  'g' - start capturing images\n"
		"  'u' - switch undistortion on/off\n";
	//data
	Size boardSize, imageSize;
	float squareSize, aspectRatio;
	Mat cameraMatrix, distCoeffs;
	
	Pattern pattern = CHESSBOARD;
	int i, nframes;
	bool writeExtrinsics, writePoints;
	bool undistortImage = false;
	int flags = 0;
	VideoCapture capture;
	bool flipVertical;
	bool showUndistorted;
	bool videofile;
	int delay;
	clock_t prevTimestamp = 0;
	int mode = DETECTION;
	int cameraId = 0;
	string videoFile = "";
	vector<vector<Point2f> > imagePoints;
	string outputFilename;
	string inputFilename = "";

	//method
	calib(cv::CommandLineParser parser);
	void help();

    // ext matrix and homography matrix
	bool calc_homgraphy(cv::Mat &img, cv::Size board_size, float square_size, vector<cv::Point2f> &corners, cv::Mat &H_mat, double &error);
	
	//intri matrix
	void cam_calibration();
	bool readStringList(const string& filename, vector<string>& l);
	bool runAndSave(const string& outputFilename,
		const vector<vector<Point2f> >& imagePoints,
		Size imageSize, Size boardSize, Pattern patternType, float squareSize,
		float aspectRatio, int flags, Mat& cameraMatrix, Mat& distCoeffs);
	bool runCalibration(vector<vector<Point2f> > imagePoints,
		Size imageSize, Size boardSize, Pattern patternType,
		float squareSize, float aspectRatio,
		int flags, Mat& cameraMatrix, Mat& distCoeffs,
		vector<Mat>& rvecs, vector<Mat>& tvecs,
		vector<float>& reprojErrs,
		double& totalAvgErr);
	void saveCameraParams(const string& filename,
		Size imageSize, Size boardSize,
		float squareSize, float aspectRatio, int flags,
		const Mat& cameraMatrix, const Mat& distCoeffs, double totalAvgErr);
	double computeReprojectionErrors(
		const vector<vector<Point3f> >& objectPoints,
		const vector<vector<Point2f> >& imagePoints,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const Mat& cameraMatrix, const Mat& distCoeffs,
		vector<float>& perViewErrors);
	void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType);
};
