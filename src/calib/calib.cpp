#include"calib.h"
#include "utils/vsnc_utils.h"

calib::calib(cv::CommandLineParser parser)
{
	if (parser.has("help"))
	{
		help();
		cerr << "pls input correct args"<< endl;
	}
	boardSize.width = parser.get<int>("w");
	boardSize.height = parser.get<int>("h");
	if (parser.has("pt"))
	{
		string val = parser.get<string>("pt");
		if (val == "circles")
			pattern = CIRCLES_GRID;
		else if (val == "acircles")
			pattern = ASYMMETRIC_CIRCLES_GRID;
		else if (val == "chessboard")
			pattern = CHESSBOARD;
		else{
			cerr << "Invalid pattern type: must be chessboard or circles\n" << endl;
		}
	}
	squareSize = parser.get<float>("s");
	nframes = parser.get<int>("n");
	aspectRatio = parser.get<float>("a");
	delay = parser.get<int>("d");
	if (parser.has("zt"))
		flags |= CALIB_ZERO_TANGENT_DIST;

	if (parser.has("o"))
		outputFilename = parser.get<string>("o");
	showUndistorted = parser.has("su");
	if (isdigit(parser.get<string>("V")[0]))
		cameraId = parser.get<int>("V");
	else
		videoFile = parser.get<string>("V");
	if (!parser.check())
	{
		help();
		parser.printErrors();
		cerr << "error arg!!!" << endl;
	}
	if (squareSize <= 0)
		cerr<<"Invalid board square width\n";
	if (nframes <= 3)
		cerr<<"Invalid number of images\n";
	if (aspectRatio <= 0)
		cerr<<"Invalid aspect ratio\n";
	if (delay <= 0)
		cerr<<"Invalid delay\n";
	if (boardSize.width <= 0)
		cerr<<"Invalid board width\n";
	if (boardSize.height <= 0)
		cerr<<"Invalid board height\n";
}

void calib::help()
{
	printf("This is a camera calibration sample.\n"
		"Usage: calibration\n"
		"     -w=<board_width>         # the number of inner corners per one of board dimension\n"
		"     -h=<board_height>        # the number of inner corners per another board dimension\n"
		"     [-pt=<pattern>]          # the type of pattern: chessboard or circles' grid\n"
		"     [-n=<number_of_frames>]  # the number of frames to use for calibration\n"
		"                              # (if not specified, it will be set to the number\n"
		"                              #  of board views actually available)\n"
		"     [-d=<delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
		"                              # (used only for video capturing)\n"
		"     [-s=<squareSize>]        # square size in some user-defined units (1 by default)\n"
		"     [-o=<out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
		"     [-zt]                    # assume zero tangential distortion\n"
		"     -V=<videoname>                     # use a video file, and not an image list, uses\n"
		"                              # [input_data] string for the video file name\n"
		"     [-su]                    # show undistorted images after calibration\n"
		"\n");
	cout << liveCaptureHelp<<endl;
}

bool calib::readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

void calib::calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
	corners.resize(0);

	switch (patternType)
	{
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(float(j*squareSize),
			float(i*squareSize), 0));
		break;

	case ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(float((2 * j + i % 2)*squareSize),
			float(i*squareSize), 0));
		break;

	default:
		CV_Error(Error::StsBadArg, "Unknown pattern type\n");
	}
}

double calib::computeReprojectionErrors(
	const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); i++)
	{
		// 根据三维坐标和相机与世界坐标系时间的关系估算内角点像素坐标
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
			cameraMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

bool calib::runCalibration(vector<vector<Point2f> > imagePoints,
	Size imageSize, Size boardSize, Pattern patternType,
	float squareSize, float aspectRatio,
	int flags, Mat& cameraMatrix, Mat& distCoeffs,
	vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs,
	double& totalAvgErr)
{
	//单位矩阵?
	cameraMatrix = Mat::eye(3, 3, CV_64F);

	//
	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	/*
	objectPoints ：世界坐标系中的点。
	imagePoint: 其对应的图像点。
	imagesize:array: 图像的大小，仅用于初始化相机的内参矩阵。
	cameraMatrix: 输入输出3x3的浮点相机内参矩阵
	distCoeffs:array: 输入或输出的系数数组。
	revecs:array:旋转向量。
	tvecs:aarary:位移向量。
	返回：
	ret 
	mtx:内参矩阵
	dist:畸变系数
	revecs:array
	tvecs:
	*/
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs, flags | CALIB_FIX_K4 | CALIB_FIX_K5);
	///*|CALIB_FIX_K3*/|CALIB_FIX_K4|CALIB_FIX_K5);
	printf("RMS error reported by calibrateCamera: %g\n", rms);

	// 检查输入的矩阵的每一个元素，并确定该元素是否给定范围内。
	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

void calib::saveCameraParams(const string& filename,
	Size imageSize, Size boardSize,
	float squareSize, float aspectRatio, int flags,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	double totalAvgErr)
{
	FileStorage fs(filename, FileStorage::WRITE);

	time_t tt;
	time(&tt);
	struct tm *t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf)-1, "%c", t2);
	
	fs << "calibration_time" << buf;

	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << boardSize.width;
	fs << "board_height" << boardSize.height;
	fs << "square_size" << squareSize;

	if (flags != 0)
	{
		sprintf(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
		//cvWriteComment( *fs, buf, 0 );
	}

	//fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;

	if (!imagePoints.empty())
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (int i = 0; i < (int)imagePoints.size(); i++)
		{
			Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		//fs << "image_points" << imagePtMat;
	}
}

bool calib::runAndSave(const string& outputFilename,
	const vector<vector<Point2f> >& imagePoints,
	Size imageSize, Size boardSize, Pattern patternType, float squareSize,
	float aspectRatio, int flags, Mat& cameraMatrix, Mat& distCoeffs)
{
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
		aspectRatio, flags, cameraMatrix, distCoeffs,
		rvecs, tvecs, reprojErrs, totalAvgErr);
	//平均重投影误差
	printf("%s. avg reprojection error = %.2f\n",
		ok ? "Calibration succeeded" : "Calibration failed",
		totalAvgErr);

	//保存相机参数
	if (ok)
		saveCameraParams(outputFilename, imageSize,
		boardSize, squareSize, aspectRatio,
		flags, cameraMatrix, distCoeffs, totalAvgErr);
	return ok;
}

void calib::cam_calibration()
{
	if (videoFile.empty())
	{
		capture.open(cameraId);
		cout << "read video from devices:" << cameraId << endl;
	}
	else{
		capture.open(videoFile);
		cout << "read video from devices:" << videoFile << endl;
	}

	if (!capture.isOpened())
		cerr << "Could not initialize video"+ to_string(cameraId)+ " capture\n";

	if (capture.isOpened())
		printf("%s", liveCaptureHelp);

	namedWindow("Image View", 1);

	int64 now = 0;
	int64 temp = 0;
	for (i = 0;; i++)
	{
		std::cout << vsnc::utils::__utc() - now << std::endl;
		now = vsnc::utils::__utc();
		Mat view, viewGray;
		bool blink = false;

		if (capture.isOpened())
		{
			Mat view0;
			capture >> view0;
			view0.copyTo(view);
		}
	
		temp = vsnc::utils::__utc();
		//结束
		if (view.empty())
		{
			if (imagePoints.size() > 0)
				runAndSave(outputFilename, imagePoints, imageSize,
				boardSize, pattern, squareSize, aspectRatio,
				flags, cameraMatrix, distCoeffs);
			break;
		}
		std::cout << "runAndSave: " << vsnc::utils::__utc()-temp << std::endl;
		//图片的宽高
		imageSize = view.size();

		temp = vsnc::utils::__utc();
		vector<Point2f> pointbuf;
		//转换成灰色
		cvtColor(view, viewGray, COLOR_BGR2GRAY);
		std::cout << "cvtColor: " << vsnc::utils::__utc() - temp << std::endl;

		temp = vsnc::utils::__utc();
		/*
		cv::CALIB_CB_ADAPTIVE_THRESH : 该函数的默认方式是根据图像的平均亮度值进行图像的二值化。设立此标志的含义是采用变化的阈值进行自适应二值化。
		cv::CALIB_CB_FAST_CHECK ：快速检测选项，对于检测角点极可能不成功检查的情况，这个标志可以使函数效率提示。
		cv::CALIB_CB_NORMALIZE_IMAGE ： 在二值化完成后，调用EqualizeHist()函数进行图像归一化处理。
		*/
		bool found;
		switch (pattern)
		{
		case CHESSBOARD:
			found = findChessboardCorners(view, boardSize, pointbuf,
				CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
			break;
		case CIRCLES_GRID:
			found = findCirclesGrid(view, boardSize, pointbuf);
			break;
		case ASYMMETRIC_CIRCLES_GRID:
			found = findCirclesGrid(view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID);
			break;
		default:
			cerr<< "Unknown pattern type\n";
		}
		std::cout << "pattern: " << vsnc::utils::__utc() - temp << std::endl;
		temp = vsnc::utils::__utc();
		// improve the found corners' coordinate accuracy
		//对检查到的角点进一步的优化计算，可使角点的精度达到亚像素级别。
		// cv::TermCriteria 停止优化标志
		// 30,要计算的最大迭代或元素数。
		// 0.1 迭代算法停止时所需的精度或者参数变化。
		if (pattern == CHESSBOARD && found) cornerSubPix(viewGray, pointbuf, Size(11, 11),
			Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

		//存入角点
		if (mode == CAPTURING && found &&
			(!capture.isOpened() || clock() - prevTimestamp > delay*1e-3*CLOCKS_PER_SEC))
		{
			imagePoints.push_back(pointbuf);
			prevTimestamp = clock();
			blink = capture.isOpened();
		}

		//把角点画回原始图像里
		if (found)
			drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

		//提示输入“g”开始
		string msg = mode == CAPTURING ? "100/100" :
			mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		if (mode == CAPTURING)
		{
			if (undistortImage)
				msg = format("%d/%d Undist", (int)imagePoints.size(), nframes);
			else
				msg = format("%d/%d", (int)imagePoints.size(), nframes);
		}

		putText(view, msg, textOrigin, 1, 1,
			mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

		//图片取反色
		if (blink)
			bitwise_not(view, view);


		//是否查看畸变后的数据
		if (mode == CALIBRATED && undistortImage)
		{
			Mat temp = view.clone();
			// 计算去畸变和校正变换映射
			undistort(temp, view, cameraMatrix, distCoeffs);
		}
		std::cout << "test: " << vsnc::utils::__utc() - temp << std::endl;
		temp = vsnc::utils::__utc();
		// 显示图片
		imshow("Image View", view);
		// 等待时间
		char key = (char)waitKey(capture.isOpened() ? 50 : 500);
		
		std::cout << "show and wait: " << vsnc::utils::__utc() - temp << std::endl;
		temp = vsnc::utils::__utc();
		if (key == 27)
			break;

		//查看数据
		if (key == 'u' && mode == CALIBRATED)
			undistortImage = !undistortImage;

		//开始标定
		if (capture.isOpened() && key == 'g')
		{
			mode = CAPTURING;
			imagePoints.clear();
		}

		//当保存的数据的数量大于目标设定的数量时，保存输出参数
		if (mode == CAPTURING && imagePoints.size() >= (unsigned)nframes)
		{
			if (runAndSave(outputFilename, imagePoints, imageSize,
				boardSize, pattern, squareSize, aspectRatio,
				flags, cameraMatrix, distCoeffs))
				mode = CALIBRATED;
			else
				mode = DETECTION;
			if (!capture.isOpened())
				break;
		}
		std::cout << "last: " << vsnc::utils::__utc() - temp << std::endl;
	}

	if (!capture.isOpened() && showUndistorted)
	{
		Mat view, rview, map1, map2;
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
			getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
			imageSize, CV_16SC2, map1, map2);
	}

}