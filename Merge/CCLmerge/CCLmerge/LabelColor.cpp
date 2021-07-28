//  Connected Component Analysis/Labeling -- Color Labeling 
//  Author:  www.icvpr.com  
//  Blog  :  http://blog.csdn.net/icvpr 
#include "LabelColor.h"
cv::Scalar icvprGetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}


void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg)
{
	if (_labelImg.empty() ||
		_labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = _labelImg.rows;
	int cols = _labelImg.cols;

	//_colorLabelImg.release();
	//_colorLabelImg.create(rows, cols, CV_8UC3);
	_colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)_labelImg.ptr<int>(i);
		uchar* data_dst = _colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = icvprGetRandomColor();
				}
				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
}

void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
	// ����ͨ�������two-pass�㷨����һ�α�ǣ��ڶ��Σ�������ϲ�
	// connected component analysis (4-component)
	// use two-pass algorithm
	// 1. first pass: label each foreground pixel with a label
	// 2. second pass: visit each labeled pixel and merge neighbor labels
	// 
	// foreground pixel: _binImg(x,y) = 1
	// background pixel: _binImg(x,y) = 0


	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		return;
	}

	// 1. first pass

	_lableImg.release();
	_binImg.convertTo(_lableImg, CV_32SC1);
	//_lableImg /= 255;

	int label = 1;  // start by 2
	std::vector<int> labelSet;
	labelSet.push_back(0);   // background: 0
	labelSet.push_back(1);   // foreground: 1

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	for (int i = 1; i < rows; i++)			// ��ɨ�裬��1��row-1
	{
		int* data_preRow = _lableImg.ptr<int>(i - 1);
		int* data_curRow = _lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)			// ��ǰ����Ϊǰ���� 4-����
			{
				std::vector<int> neighborLabels;
				neighborLabels.reserve(3);
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];	// ����ֵ
				int leftconer = data_preRow[j - 1];	// ���Ͻ�
				int rightconer = data_preRow[j + 1];// ���Ͻ�
				if (leftconer >= 1)			// �������Ϊǰ��
				{
					neighborLabels.push_back(leftconer);
				}
				if (rightconer >= 1)			// �������Ϊǰ��
				{
					neighborLabels.push_back(rightconer);
				}
				if (leftPixel >= 1)			// �������Ϊǰ��
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel >= 1)				// �ϲ�����Ϊǰ��
				{
					neighborLabels.push_back(upPixel);
				}

				if (neighborLabels.empty())		// �ϣ��󣬶���0����ǰ����ָ���±��
				{
					labelSet.push_back(++label);  // assign to a new label
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else							// �Ϻ�����ǰ�����
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());	// ��ͬ�ȱ�����򣬼�ֻ��4��ͨ��8��ͨ����
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;		// ���Ϊ��С��label		{1,3,5}��1

					// save equivalence
					for (size_t k = 1; k < neighborLabels.size(); k++)			// ȷ���ҵ���С�ı�ǩֵ
					{
						int tempLabel = neighborLabels[k];
						int& oldSmallestLabel = labelSet[tempLabel];			
						if (oldSmallestLabel > smallestLabel)					
						{
							labelSet[oldSmallestLabel] = smallestLabel;			
							oldSmallestLabel = smallestLabel;
						}
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel;
						}
					}
				}
			}
		}
	}

	// update equivalent labels
	// assigned with the smallest label in each equivalent label set
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];					// ֻ��Ϊ���Ҹ��ڵ�
		int preLabel = labelSet[curLabel];			
		while (curLabel != preLabel)				
		{
			curLabel = preLabel;					
			preLabel = labelSet[preLabel];			
		}
		labelSet[i] = curLabel;						// �滻Ϊ���ڵ�
	}


	// 2. second pass
	for (int i = 0; i < rows; i++)
	{
		int* data = _lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];		// labelImg��ֵΪ��ǩֵ
		}
	}
	// ����labelSetȥ�����ֵ
}
void icvprCcaBySeedFill(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
	// connected component analysis (4-component)
	// use seed filling algorithm
	// 1. begin with a foreground pixel and push its foreground neighbors into a stack;
	// 2. pop the top pixel on the stack and label it with the same label until the stack is empty
	// 
	// foreground pixel: _binImg(x,y) = 1
	// background pixel: _binImg(x,y) = 0


	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		return;
	}

	_lableImg.release();
	_binImg.convertTo(_lableImg, CV_32SC1);

	int label = 1;  // start by 2

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	for (int i = 100; i < rows - 100; i++)
	{
		int* data = _lableImg.ptr<int>(i);
		for (int j = 100; j < cols - 100; j++)
		{
			if (data[j] == 1)
			{
				std::stack<std::pair<int, int>> neighborPixels;
				neighborPixels.push(std::pair<int, int>(i, j));     // pixel position: <i,j>
				++label;  // begin with a new label
				while (!neighborPixels.empty())
				{
					// get the top pixel on the stack and label it with the same label
					std::pair<int, int> curPixel = neighborPixels.top();
					int curX = curPixel.first;
					int curY = curPixel.second;
					_lableImg.at<int>(curX, curY) = label;

					// pop the top pixel
					neighborPixels.pop();

					// push the 4-neighbors (foreground pixels)
					if (_lableImg.at<int>(curX, curY - 1) == 1)
					{// left pixel
						neighborPixels.push(std::pair<int, int>(curX, curY - 1));
					}
					if (_lableImg.at<int>(curX, curY + 1) == 1)
					{// right pixel
						neighborPixels.push(std::pair<int, int>(curX, curY + 1));
					}
					if (_lableImg.at<int>(curX - 1, curY) == 1)
					{// up pixel
						neighborPixels.push(std::pair<int, int>(curX - 1, curY));
					}
					if (_lableImg.at<int>(curX + 1, curY) == 1)
					{// down pixel
						neighborPixels.push(std::pair<int, int>(curX + 1, curY));
					}
				}
			}
		}
	}
}