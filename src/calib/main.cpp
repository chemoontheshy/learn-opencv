#include "calib.h"
#include "homo.h"

using namespace std;

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv,
		"{help ||}{w||}{h||}{pt|chessboard|}{n|10|}{d|3000|}{s|1|}{o|out_camera_data.yml|}"
		"{zt||}{a|1|}{V|0|}{su||}"
		"{@input_data|0|}{e||}{hand||}");
	if (parser.has("e"))
	{
		homo c_homo(parser);
		c_homo.cam_homography();
	}
	else{
		calib c_calib(parser);
		c_calib.cam_calibration();
	}

	return 0;
}