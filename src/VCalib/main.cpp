#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <iostream>
#include <Eigen/Dense>

void runAndSave(const std::string& outputFileName, const std::vector<std::vector<cv::Point2f>>& imagePoints,
	const cv::Size imageSize,const cv::Size boardSize, const float squareSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs)
{
	auto __compute_reprojection_errors = [](
		const std::vector<std::vector<cv::Point3f>>& objectPoints,
		const std::vector<std::vector<cv::Point2f>>& imagePoints,
		const std::vector<cv::Mat>& rvecs,
		const std::vector<cv::Mat>& tvecs,
		const cv::Mat& cameraMatrix,
		const cv::Mat& distCoeffs,
		std::vector<float>& perViewErrors
		)->double
	{
		std::vector<cv::Point2f> tempImagePoint;
		int i = 0;
		int totalPoints = 0;
		double totalErr = 0;
		double err;
		perViewErrors.resize(objectPoints.size());
		for (i; i < objectPoints.size(); i++)
		{
			cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i],
				cameraMatrix, distCoeffs, tempImagePoint);
			err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(tempImagePoint), cv::NORM_L2);
			int n = (int)objectPoints[i].size();
			perViewErrors[i] = static_cast<float>(std::sqrt(err * err / n));
			totalErr += err * err;
			totalPoints += n;
		}
		return std::sqrt(totalErr / totalPoints);
	};

	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<float> reprojErrs;
	double totalAvgErr = 0;
	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

	std::vector<std::vector<cv::Point3f>> objectPoints(1);
	{
		objectPoints[0].resize(0);
		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				objectPoints[0].push_back(cv::Point3f(float(j * squareSize), float(i * squareSize), 0));
			}
		}
	}
	objectPoints.resize(imagePoints.size(),objectPoints[0]);

	auto rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);
	std::cout << "RMS error reported by calibrateCamera: " << rms << std::endl;
	bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);

	totalAvgErr = __compute_reprojection_errors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);
	std::string msg = (ok ? "Calibration succeeded" : "Calibration failed");
	std::cout << msg + "avg reprojection error = "+ std::to_string(totalAvgErr) << std::endl;
}

int main()
{
	cv::VideoCapture capture;
	capture.open("..\\..\\3rdparty\\video\\video.mp4");
	if (!capture.isOpened())
	{
		std::cout << "error capture open fail!" << std::endl;
		return -1;
	}
	//ԭʼͼƬ
	cv::Mat view;
	//�Ҷ�ͼ
	cv::Mat viewGray;
	//�洢�ڽǵ������
	std::vector<cv::Point2f> pointBuf;
	// һ�����ݣ���ͬ�Ƕȵ��ڽǵ����ݣ�
	std::vector<std::vector<cv::Point2f>> imagePoints;
	size_t num = 0;
	auto bStart = true;
	auto bBlink = false;
	while (true)
	{
		capture >> view;
		auto a = view.empty();
		if (view.empty()) break;
		//�ڽǵ�
		cv::Size boardSize{ 8,6 };
		cv::cvtColor(view, viewGray, cv::COLOR_BGR2GRAY);
		auto found = false;
		/*
		cv::CALIB_CB_ADAPTIVE_THRESH : �ú�����Ĭ�Ϸ�ʽ�Ǹ���ͼ���ƽ������ֵ����ͼ��Ķ�ֵ���������˱�־�ĺ����ǲ��ñ仯����ֵ��������Ӧ��ֵ����
		cv::CALIB_CB_FAST_CHECK �����ټ��ѡ����ڼ��ǵ㼫���ܲ��ɹ���������������־����ʹ����Ч����ʾ��
		 cv::CALIB_CB_NORMALIZE_IMAGE �� �ڶ�ֵ����ɺ󣬵���EqualizeHist()��������ͼ���һ������
		*/
		found = cv::findChessboardCorners(view, boardSize, pointBuf, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			//�Լ�鵽�Ľǵ��һ�����Ż����㣬��ʹ�ǵ�ľ��ȴﵽ�����ؼ���
			// cv::TermCriteria ֹͣ�Ż���־
			// 30,Ҫ�������������Ԫ������
			// 0.1 �����㷨ֹͣʱ����ľ��Ȼ��߲����仯��
			cv::cornerSubPix(viewGray, pointBuf, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS,30, 0.1));
		}
		if (found)
		{
			//�ѽǵ㻭��ԭʼͼ����
			cv::drawChessboardCorners(view, boardSize, pointBuf, found);
		}
		num++;
		if (imagePoints.size() >= 20)
		{
			bStart = false;
			std::string path = "er.re";
			cv::Mat cameraMatrix;
			cv::Mat distCoeffs;
			auto imageSize = view.size();
			runAndSave(path, imagePoints, imageSize, boardSize, 0.0245, cameraMatrix, distCoeffs);
			cv::Mat temp = view.clone();
			cv::undistort(temp, view, cameraMatrix, distCoeffs);
			cv::imshow("video", view);
			char key = cv::waitKey(0);
			if (key == 'q')
			{
				return 0;
			}
		}
		if (num % 5 == 0 && bStart)
		{
			imagePoints.push_back(pointBuf);
			cv::bitwise_not(view, view);
		}

		cv::imshow("video", view);
		cv::waitKey((1000 / 30));
	}
	capture.release();
	return 0;
}