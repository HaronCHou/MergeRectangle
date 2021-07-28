#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Scalar icvprGetRandomColor();
void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg);
void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg);
void icvprCcaBySeedFill(const cv::Mat& _binImg, cv::Mat& _lableImg);