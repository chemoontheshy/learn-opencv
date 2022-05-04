#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>

int main()
{
	// 1.��ȡ��Ƶ
	cv::VideoCapture capture;
	//capture.open("..\\..\\3rdparty\\video\\1280x720_30fps_60s.mp4
    capture.open("test.mp4");
	// 2.ѭ����ʾÿһ֡
	if (!capture.isOpened())
	{
		std::cout << "error capture.isOpened fail!" << std::endl;
		return -1;
	}
	cv::Mat frame;
	size_t num = 0;
	while (true)
	{
		capture >> frame;
		if (frame.empty()) break;
		cv::imshow("video", frame);
		/*auto saveFile = "../../3rdparty/video/save/" + std::to_string(vsnc::utils::__utc()) + ".png";
		if (num % 60 == 0)
		{
			cv::imwrite(saveFile, frame);
		}
		num++;*/
		cv::waitKey((1000 / 30));
	
	}
	capture.release();
	return 0;
}