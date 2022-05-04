#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>

using namespace cv;
using namespace std;

extern vector<Point2f> kps;

class homo
{
public:
	//data
	const char* liveCaptureHelp =
		"When the live video from camera is used as input, the following hot-keys may be used:\n"
		"  <ESC>, 'q' - quit the program\n";
	Size boardSize, imageSize;
	float squareSize, aspectRatio;
	string outputFilename;
	string inputFilename = "";
	string videoFile = "";
	int cameraId = 0;
	Mat camera_matrix;
	Mat dist_coeffs;

	//for output
	Mat H_mat;
	Mat ext_param;
	bool flag_hand;

	//mothod
	homo(cv::CommandLineParser parser);
	void help();
	void cam_homography();
	bool calc_homgraphy(cv::Mat &img, cv::Size board_size, float square_size, vector<cv::Point2f> &corners, cv::Mat &H_mat, double &error);
	bool calc_homgraphy_byhand(cv::Mat &img, cv::Size board_size, float square_size, vector<cv::Point2f> &corners, cv::Mat &H_mat, double &error);
	void calcChessboardCorners(cv::Size board_size, float square_size, vector<cv::Point3f>& corners);
	double text_pts_coords(cv::Mat &img, vector<cv::Point2f> corners, Eigen::Matrix3f H, vector<cv::Point2f> pt_chessboard);
	Eigen::Vector3f get_depth(int pt2d_x, int pt2d_y, Eigen::Matrix3f mat3f_H);
	cv::Mat get_ext_param(cv::Mat mat3f_inter_param, cv::Mat mat3f_H);
	//void on_mouse(int event, int x, int y, int flags, void *ustc);
};
