#include"homo.h"

vector<Point2f> kps;
homo::homo(cv::CommandLineParser parser)
{
	if (parser.has("help"))
	{
		help();
		cerr << "pls input correct args" << endl;
	}
	boardSize.width = parser.get<int>("w");
	boardSize.height = parser.get<int>("h");
	squareSize = parser.get<float>("s");
	inputFilename = parser.get<string>("@input_data");
	if (isdigit(parser.get<string>("V")[0]))
		cameraId = parser.get<int>("V");
	else
		videoFile = parser.get<string>("V");
	if (parser.has("o"))
		outputFilename = parser.get<string>("o");
	if (squareSize <= 0)
		cerr << "Invalid board square width\n";
	if (boardSize.width <= 0)
		cerr << "Invalid board width\n";
	if (boardSize.height <= 0)
		cerr << "Invalid board height\n";

	cv::FileStorage input_fs;
	input_fs.open(inputFilename, cv::FileStorage::READ);
	cout << "read intrinsic param from:" + inputFilename << endl;
	input_fs["camera_matrix"] >> camera_matrix;
	input_fs["distortion_coefficients"] >> dist_coeffs;
	flag_hand = parser.has("hand") ? 1 : 0;
}

void homo::help()
{
	printf("This is a homography matrix.\n"
		"Usage: calibration\n"
		"     -w=<board_width>         # the number of inner corners per one of board dimension\n"
		"     -h=<board_height>        # the number of inner corners per another board dimension\n"
		"     [-s=<squareSize>]        # square size in some user-defined units (1 by default)\n"
		"     [-o=<out_homo_params>]   # the output filename for homography parameters\n"
		"     [-zt]                    # assume zero tangential distortion\n"
		"     [-V]                     # use a video file, and not an image list, uses\n"
		"                              # [input_data] string for the video file name\n"
		"     [-su]                    # show undistorted images after calibration\n"
		"     [input_data]             # input camera matrix data\n"
		"                              # if input_data not specified, a live view from the camera is used\n"
		"     [-e]                     # whether compute homography matrix"
		"\n");
}

bool homo::calc_homgraphy(cv::Mat &img, cv::Size board_size, float square_size, vector<cv::Point2f> &corners, cv::Mat &H_mat, double &error)
{
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	// fing chess board corners
	bool patternfound = findChessboardCorners(img_gray, board_size, corners,
		cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
	if (patternfound)
		cornerSubPix(img_gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
	else
	{
		cout << "Error, cannot find the chessboard corners in both images." << endl;
		return patternfound;
	}
	drawChessboardCorners(img, board_size, cv::Mat(corners), patternfound);

	// calculate homography matrix
	if (patternfound)
	{
		vector<cv::Point3f> objectPoints;
		calcChessboardCorners(board_size, square_size, objectPoints);
		vector<cv::Point2f> objectPointsPlanar;
		for (size_t i = 0; i < objectPoints.size(); i++)
		{
			objectPointsPlanar.push_back(cv::Point2f(objectPoints[i].x, objectPoints[i].y));
		}
		//undistortPoints(corners, imagePoints, cameraMatrix, distCoeffs); //undistor
		H_mat = findHomography(objectPointsPlanar, corners);

		Eigen::Matrix3f Eig_H;
		Eig_H << H_mat.at<double>(0, 0), H_mat.at<double>(0, 1), H_mat.at<double>(0, 2),
			H_mat.at<double>(1, 0), H_mat.at<double>(1, 1), H_mat.at<double>(1, 2),
			H_mat.at<double>(2, 0), H_mat.at<double>(2, 1), H_mat.at<double>(2, 2);
		error = text_pts_coords(img, corners, Eig_H, objectPointsPlanar);
	}
	return patternfound;
}
void on_mouse(int event, int x, int y, int flags, void *ustc)
//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号    
{
	Mat& img = *(cv::Mat*) ustc;
	if (event == EVENT_LBUTTONDOWN)
	{
		kps.push_back(Point2f(y, x));
		cout << "Point:" << x << "," << y << endl;
		circle(img, Point2f(x,y), 2, Scalar(0, 255, 0));
		imshow("Image", img);
		waitKey(10);
	}
}


bool homo::calc_homgraphy_byhand(cv::Mat &img, cv::Size board_size, float square_size, vector<cv::Point2f> &corners, cv::Mat &H_mat, double &error)
{
	int num_kps = board_size.height*board_size.width;
	Mat img_tmp=img.clone();
	kps.clear();
	while (kps.size() < num_kps){
		imshow("Image", img);
		waitKey(5);
		setMouseCallback("Image", on_mouse, (void *)&img_tmp);
		//kps.push_back(kp);
	}

	vector<cv::Point3f> objectPoints;
	calcChessboardCorners(board_size, square_size, objectPoints);
	vector<cv::Point2f> objectPointsPlanar;
	for (size_t i = 0; i < objectPoints.size(); i++)
	{
		objectPointsPlanar.push_back(cv::Point2f(objectPoints[i].x, objectPoints[i].y));
	}
	//undistortPoints(corners, imagePoints, cameraMatrix, distCoeffs); //undistor
	H_mat = findHomography(objectPointsPlanar, kps);

	Eigen::Matrix3f Eig_H;
	Eig_H << H_mat.at<double>(0, 0), H_mat.at<double>(0, 1), H_mat.at<double>(0, 2),
		H_mat.at<double>(1, 0), H_mat.at<double>(1, 1), H_mat.at<double>(1, 2),
		H_mat.at<double>(2, 0), H_mat.at<double>(2, 1), H_mat.at<double>(2, 2);
	error = text_pts_coords(img, kps, Eig_H, objectPointsPlanar);
	return 1;
}

void homo::calcChessboardCorners(cv::Size board_size, float square_size, vector<cv::Point3f>& corners)
// function for generating chess board pts
{
	corners.resize(0);
	for (int i = board_size.height - 1; i >= 0; i--)
	for (int j = 0; j < board_size.width; j++)
		corners.push_back(cv::Point3f(float(j*square_size),
		float(i*square_size), 0));
}

Eigen::Vector3f homo::get_depth(int pt2d_x, int pt2d_y, Eigen::Matrix3f mat3f_H)
// function 
// get_depth: input pixel in image and homograph matrix, 
// calcate the position of the point in ground plane
// input: 
//       int pt2d_x: pixel x coordination
//       int pt2d_y: pixel y coordination
//       Eigen:Matrix3f mat3f_H: homography matrix
// ouput:
//       float depth
{
	float depth;
	Eigen::Vector3f v_homo_pt2d;
	v_homo_pt2d << pt2d_x, pt2d_y, 1;
	Eigen::Vector3f v_pt3d = mat3f_H.inverse()* v_homo_pt2d;
	//depth = v_pt3d(1) / v_pt3d(2);
	return v_pt3d;
}

// put every pts coords on img
double homo::text_pts_coords(cv::Mat &img, vector<cv::Point2f> corners, Eigen::Matrix3f H, vector<cv::Point2f> pt_chessboard)
{
	double error_match = 0;
	vector<cv::Point2f>::iterator iter_chessboard = pt_chessboard.begin();
	for (vector<cv::Point2f>::iterator iter = corners.begin(); iter != corners.end(); iter++)
	{
		cv::Point2f pt = *iter;
		Eigen::Vector3f pts_ground = get_depth(pt.x, pt.y, H);
		char text_output[200];
		sprintf(text_output, "(%0.1f,%0.1f)", pts_ground[0] / pts_ground[2], pts_ground[1] / pts_ground[2]);
		cv::putText(img, text_output, cv::Point2f(int(pt.x) - 20, int(pt.y) + 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255));
		error_match += (pts_ground[0] / pts_ground[2] - (*iter_chessboard).x)*(pts_ground[0] / pts_ground[2] - (*iter_chessboard).x) +
			(pts_ground[1] / pts_ground[2] - (*iter_chessboard).y)*(pts_ground[1] / pts_ground[2] - (*iter_chessboard).y);
		iter_chessboard++;
	}


	return error_match;
}

//function: get_exter_depth
// calc extrenal parameter
// input: interal parameter matrix mat3f_inter_param, and homography matrix mat3f_H
// output : exter parameter depth in chessboard croodnation
cv::Mat homo::get_ext_param(cv::Mat mat3f_inter_param, cv::Mat mat3f_H)
{
	cv::Mat mat3f_trans = mat3f_inter_param.inv()* mat3f_H;
	// Normalization to ensure that ||c1|| = 1
	double norm = sqrt(mat3f_trans.at<double>(0, 0)*mat3f_trans.at<double>(0, 0) +
		mat3f_trans.at<double>(1, 0)*mat3f_trans.at<double>(1, 0) +
		mat3f_trans.at<double>(2, 0)*mat3f_trans.at<double>(2, 0));
	mat3f_trans /= norm;
	cv::Mat c1 = mat3f_trans.col(0);
	cv::Mat c2 = mat3f_trans.col(1);
	cv::Mat c3 = c1.cross(c2);
	cv::Mat tvec = mat3f_trans.col(2);
	cv::Mat R(3, 3, CV_64F);
	for (int i = 0; i < 3; i++)
	{
		R.at<double>(i, 0) = c1.at<double>(i, 0);
		R.at<double>(i, 1) = c2.at<double>(i, 0);
		R.at<double>(i, 2) = c3.at<double>(i, 0);
	}
	cv::Mat W, U, Vt;
	SVDecomp(R, W, U, Vt);
	R = U*Vt;

	cv::Mat tmp;
	cv::hconcat(R, tvec, tmp);

	cv::Mat T_c_g(4, 4, CV_64F);
	T_c_g = cv::Mat::eye(cv::Size(4, 4), CV_64F);
	tmp.copyTo(T_c_g(cv::Range(0, 3), cv::Range(0, 4)));
	cv::Mat T_g_c = T_c_g.inv();
	return T_g_c;
}

void homo::cam_homography()
{
	cv::VideoCapture cap;
	if (videoFile.empty())
	{
		cap.open(cameraId);
		cout << "read video from devices:" << cameraId << endl;
	}
	else{
		cap.open(videoFile);
		cout << "read video from devices:" << videoFile << endl;
	}

	Mat frame, frame_resized;
	bool b_find_chesspts;
	int frame_num = -1;
	vector<cv::Point2f> corners;
	double error_match = 0, error_match_last = 10000;
	cv::FileStorage output_fs;
	if (cap.isOpened())
		printf("%s", liveCaptureHelp);
	namedWindow("Image", 1);
	//cv::Size board_size(board_size_x, board_size_y);
	while (1)
	{
		cap >> frame;
		if (!frame.empty())
		{
			Mat temp = frame.clone();
			undistort(temp, frame, camera_matrix, dist_coeffs);
			frame_num++;
			if (frame_num % 10 != 0)
				continue;
			
			if (flag_hand)
			{
				b_find_chesspts = calc_homgraphy_byhand(frame, boardSize, squareSize, corners, H_mat, error_match);
			}
			else{
				b_find_chesspts = calc_homgraphy(frame, boardSize, squareSize, corners, H_mat, error_match);
			}
			
			b_find_chesspts = 1;
			if (b_find_chesspts)
			{
				/*cv::Mat H_tmp = (Mat_ <double>(3, 3) << 0.988293964070367, 0.631586063280910, 497, 0.0263573071424133, 0.0997606763224965, 890, -9.63510909429974e-06, 0.000687200845403121, 1.0);
				cv::Mat camera_matrix_tmp = (Mat_ <double>(3, 3) << 1.2885500058350538e+03, 0., 9.4133259560279487e+02, 0.,1.2911371695370810e+03, 6.1575421804645441e+02, 0., 0., 1.);
				ext_param = get_ext_param(camera_matrix_tmp, H_tmp);*/
				ext_param = get_ext_param(camera_matrix, H_mat);
				cout << ext_param << endl;
				corners.clear();

				if (error_match < error_match_last)
				{
					output_fs.open(outputFilename, cv::FileStorage::WRITE);
					output_fs << "homography" << H_mat;
					output_fs << "ext_param" << ext_param;
					error_match_last = error_match;
					output_fs.release();
					cout << "reproject error is" << error_match << endl;
					cout << " write homography matrix" << endl;
				}
			}
			cv::resize(frame, frame_resized, cv::Size(960, 540));
			cv::imshow("Image", frame);
			char key = (char)waitKey(cap.isOpened() ? 50 : 500);

			if (key == 27)
				break;
		}
	}
	cv::destroyWindow("Image");
	cap.release();
}