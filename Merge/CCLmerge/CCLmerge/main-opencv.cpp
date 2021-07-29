/**
 * @file main-opencv.cpp
 * @date July 2014 
 * @brief An exemplative main file for the use of ViBe and OpenCV
 */
#include <iostream>
#include <forward_list>

//#include <opencv2/cv.h>
//#include <opencv/highgui.h>
#include <opencv.hpp>
#include <opencv2\imgproc\types_c.h>		// CV_BGRGRAYҪ�õ���

#include "vibe-background-sequential.h"
#include "LabelColor.h"
#include "CCL_SAUF.h"
#include <stdlib.h>
#include <math.h>
#include <windows.h>	// DWORD

#define SHOW 1
#define USE_VECTOR_MERGE 0
#define USE_LIST_MERGE 0
#define USE_ORGIN_MERGE 1

using namespace cv;
using namespace std;
void Merge(std::vector<cv::Rect>& rects);
void Merge(std::vector<cv::Rect>& rects, const int&s, const int&t);
bool intetsect(std::vector<cv::Rect>& rects, int&s, int&t);
double intetsect(const cv::Rect& a, const cv::Rect& b);
bool isRectangleOverlap(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2);
bool isCenterClose(double Cenx, double Ceny, double Cenx_new, double Ceny_new);
void Merge(cv::Mat &stats, cv::Mat &centroids, int i, int j);

//----------------�ϲ��ص���������ڽ���---------------------
#define  CENTER_DIST_THRESH 900.0
typedef struct  DetectBox
{
	cv::Rect rect;		// bbox
	int validNum = 0;		// ���/����
	cv::Point2f center_point;
}DetectBox;

std::vector<DetectBox> getMoveRects(int nccomps, cv::Mat stats, cv::Mat centroids);
bool isOverlap(cv::Point2i tl1, cv::Point2i br1, cv::Point2i curTl, cv::Point2i curBR);
std::vector<int> getAllOverlaps(std::vector<DetectBox>move_rects, DetectBox curRect, int index);
bool isCenterClose(cv::Point2f center, cv::Point2f curCenter);
void Merge(std::vector<DetectBox> &move_rects);

//------------------------����ʵ��--------------------------
std::forward_list<DetectBox> getMoveRects_list(int nccomps, cv::Mat stats, cv::Mat centroids);
std::vector<std::forward_list<DetectBox>::iterator> getAllOverlaps_list(std::forward_list<DetectBox>move_rects
	, DetectBox curRect, std::forward_list<DetectBox>::iterator it);
void Merge_list(std::forward_list<DetectBox> &move_rects, unsigned &list_size);
void Statistic(std::vector<cv::Point2i> &points, float &cenx, float &ceny, int &validNumSum
	, std::forward_list<DetectBox>::iterator curr);

// ����������
int cmp1(int a, int b)
{
	return b < a;
}
/** Function Headers */
void processVideo(char* videoFilename);

/**
 * Displays instructions on how to use this program.
 */

void help()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to use ViBe with OpenCV                            " << endl
    << "Usage:"                                                                     << endl
    << "./main-opencv <video filename>"                                             << endl
    << "for example: ./main-opencv video.avi"                                       << endl
    << "--------------------------------------------------------------------------" << endl
    << endl;
}
Mat Array2Mat(uint8_t *a, int row, int col);
/**
 * Main program. It shows how to use the grayscale version (C1R) and the RGB version (C3R). 
 */
#if 1	// Ŀ�ģ�Ϊ����������������ж��main demo
int main(int argc, char* argv[])
{
  /* Print help information. */
  help();

  ///* Check for the input parameter correctness. */
  //if (argc != 2) {
  //  cerr <<"Incorrect input" << endl;
  //  cerr <<"exiting..." << endl;
  //  return EXIT_FAILURE;
  //}

  /* Create GUI windows. */
  //namedWindow("Frame");
  //namedWindow("Segmentation by ViBe");

  processVideo("G:\\pic\\boat.avi");	// G:\\pic\\boat.avi 
							// G:\\202007_work\\data_video\\0304am.avi
  /* Destroy GUI windows. */
  destroyAllWindows();
  return EXIT_SUCCESS;
}
#endif
/**
 * Processes the video. The code of ViBe is included here. 
 *
 * @param videoFilename  The name of the input video file. 
 */
void processVideo(char* videoFilename)
{
	int kernel_size = 5;
	uchar point[21] = { 0 }, point_last[21] = { 0 };
  /* Create the capture object. */
  VideoCapture capture(videoFilename);

  if (!capture.isOpened()) {
    /* Error in opening the video input. */
    cerr << "Unable to open video file: " << videoFilename << endl;
    exit(EXIT_FAILURE);
  }

  /* Variables. */
  static int frameNumber = 1; /* The current frame number */
  Mat frame;                  /* Current frame. */
  Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
  int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */

  double totalNumber = capture.get(CAP_PROP_FRAME_COUNT); /*CV_CAP_PROP_FRAME_COUNT opencv2*/

  /* Model for ViBe. */
  vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */
  /* Read input data. ESC or 'q' for quitting. */

  uint8_t *historyImage = (uint8_t*)malloc(2 * frame.cols * frame.rows * sizeof(uint8_t));
  uint8_t *historyBuffer = (uint8_t*)malloc(18 * frame.cols * frame.rows * sizeof(uint8_t));

  while ((char)keyboard != 'q' && (char)keyboard != 27 /*|| frameNumber <= 20*/) 
  {
    /* Read the current frame. */
    if (!capture.read(frame)) {
      cerr << "Unable to read next frame." << endl;
      cerr << "Exiting..." << endl;
      exit(EXIT_FAILURE);
    }

    if ((frameNumber % 100) == 0) { std::cout << "Frame number = " << frameNumber << endl; }
    /* Applying ViBe.
     * If you want to use the grayscale version of ViBe (which is much faster!):
     * (1) remplace C3R by C1R in this file.
     * (2) uncomment the next line (cvtColor).
     */
    /* cvtColor(frame, frame, CV_BGR2GRAY); */
	cv::Mat frame_bgr;
	frame.copyTo(frame_bgr);
	cvtColor(frame, frame, CV_BGR2GRAY);

	cv::Mat histBuff[20];
	for (int i = 0; i < 20; i++){
		histBuff[i] = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	}
    if (frameNumber == 1) {
      segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
      model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
      libvibeModel_Sequential_AllocInit_8u_C1R(model, frame.data, frame.cols, frame.rows);
    }


    /* ViBe: Segmentation and updating. */
    libvibeModel_Sequential_Segmentation_8u_C1R(model, frame.data, segmentationMap.data);
	libvibeModel_Sequential_Update_8u_C1R(model, frame.data, segmentationMap.data, historyImage, historyBuffer);

    /* Post-processes the segmentation map. This step is not compulsory. 
       Note that we strongly recommend to use post-processing filters, as they 
       always smooth the segmentation map. For example, the post-processing filter 
       used for the Change Detection dataset (see http://www.changedetection.net/ ) 
       is a 5x5 median filter. */
	if (kernel_size > 0)
    medianBlur(segmentationMap, segmentationMap, kernel_size); /* 3x3 median filtering */

    /* Shows the current frame and the segmentation map. */
    //imshow("Frame", frame);
    //imshow("Segmentation by ViBe", segmentationMap);

#if 1 //���ڽ�ǰ��������ԭʼͼ��֡�ϱ��ڹ۲�
	cv::Mat colorMask(frame_bgr.size(), frame_bgr.type(), Scalar(1, 1, 1));
	//colorMask = colorMask.mul(MergeMat);

	for (int i = 0; i < frame.rows; i++)
	for (int j = 0; j < frame.cols; j++){
		if (segmentationMap.at<uchar>(i, j) == 0)
			colorMask.at<Vec3b>(i, j) = Vec3b(1, 1, 1);
		else
			colorMask.at<Vec3b>(i, j) = Vec3b(0, 0, 100);
	}
	cv::Mat res;
	res = colorMask.mul(frame_bgr);
#if SHOW
	cv::imshow("ǰ�������ڱ�����", colorMask.mul(frame_bgr));
#endif

#endif
	segmentationMap /= 255;
	cv::Mat labels, stats, centroids;
	int nccomps = cv::connectedComponentsWithStats(segmentationMap, labels, stats, centroids, 8, 4, CCL_WU);

#if 0
	// ����һ�Σ������ܲ��ܴﵽ��Ч����
	for (int i = 1; i < stats.rows; i++)
	{
		int &xmin = stats.at<int>(i, 0);		// max_ID * 5 �ľ���
		if (xmin == -1) continue;				// �ñ�ǩ�Ѿ��ϲ�����
		int &ymin = stats.at<int>(i, 1);
		int &width = stats.at<int>(i, 2);
		int &height = stats.at<int>(i, 3);
		int &validNum = stats.at<int>(i, 4);

		double Cenx = centroids.at<double>(i,0);
		double Ceny = centroids.at<double>(i, 1);

		for (int j = i + 1; j < stats.rows; j++)
		{
			int &xminJ = stats.at<int>(j, 0);	// max_ID * 5 �ľ���
			if (xminJ == -1) continue;			// �ñ�ǩ�Ѿ��ϲ�����

			int &yminJ = stats.at<int>(j, 1);
			int &widthJ = stats.at<int>(j, 2);
			int &heightJ = stats.at<int>(j, 3);
			int &validNumJ = stats.at<int>(j, 4);

			double Cenx_new = centroids.at<double>(j, 0);
			double Ceny_new = centroids.at<double>(j, 1);

			if (isCenterClose(Cenx, Ceny, Cenx_new, Ceny_new))
				Merge(stats,centroids, i, j);
		}
	}
	// ����һ�Σ������ܲ��ܴﵽ��Ч����
	for (int i = 1; i < stats.rows; i++)
	{
		int &xmin = stats.at<int>(i, 0);		// max_ID * 5 �ľ���
		if (xmin == -1) continue;				// �ñ�ǩ�Ѿ��ϲ�����
		int &ymin = stats.at<int>(i, 1);
		int &width = stats.at<int>(i, 2);
		int &height = stats.at<int>(i, 3);
		int &validNum = stats.at<int>(i, 4);

		double Cenx = centroids.at<double>(i, 0);
		double Ceny = centroids.at<double>(i, 1);

		for (int j = i + 1; j < stats.rows; j++)
		{
			int &xminJ = stats.at<int>(j, 0);	// max_ID * 5 �ľ���
			if (xminJ == -1) continue;			// �ñ�ǩ�Ѿ��ϲ�����

			int &yminJ = stats.at<int>(j, 1);
			int &widthJ = stats.at<int>(j, 2);
			int &heightJ = stats.at<int>(j, 3);
			int &validNumJ = stats.at<int>(j, 4);

			double Cenx_new = centroids.at<double>(j, 0);
			double Ceny_new = centroids.at<double>(j, 1);

			if (isRectangleOverlap(xmin, ymin, width, height, xminJ, yminJ, widthJ, heightJ))
				Merge(stats, centroids, i, j);
		}
	}
#elif USE_VECTOR_MERGE		// ��vector���ݽṹʵ�ֺϲ�
	DWORD start_time = GetTickCount();
	std::vector<DetectBox> move_rects = getMoveRects(nccomps, stats, centroids);
	//	���ٰ��Merge
	Merge(move_rects);
	// if (move_rects.size()>1) Merge(move_rects);

	DWORD end_time = GetTickCount();
	cout << "�ϲ��ص���̬����vector : " << (end_time - start_time) * 1.00 << "ms" << endl;
#elif USE_LIST_MERGE // ʹ������ʵ�ֺϲ�
	DWORD start_time = GetTickCount();
	std::forward_list<DetectBox> move_rects_list = getMoveRects_list(nccomps, stats, centroids);
	//	���ٰ��Merge
	unsigned list_size = nccomps;
	Merge_list(move_rects_list, list_size);
	// ����ӿ�Ϊvector��ͳһһ��
	std::vector<DetectBox> move_rects;
	//std::forward_list<DetectBox>::iterator it;
	//for (it = move_rects_list.begin(); it != move_rects_list.end(); it++)
	//{
	//	move_rects.push_back(*it);
	//}
	for(auto &it : move_rects_list)
		move_rects.push_back(it);
	DWORD end_time = GetTickCount();
	std::cout << "�ϲ��ص���-����forward_listʵ�� : " << (end_time - start_time) * 1.00 << "ms" << endl;
#endif
	// �����������յĳ�ȥ�Ľӿ���vector�����Խӿڻ�Ҫ��ͳһһ��
	//for (int i = 0; i < move_rects.size(); i++)
	//{
	//	rectangle(frame_bgr, move_rects[i].rect, Scalar(0, 0, 255), 1);
	//}
#if 0
	for (int i = 1; i < stats.rows; i++)
	{
		int &xmin = stats.at<int>(i, 0);	// max_ID * 5 �ľ���
		if (xmin == -1) continue;			// �ϲ������
		int &ymin = stats.at<int>(i, 1);
		int &width = stats.at<int>(i, 2);
		int &height = stats.at<int>(i, 3);
		int &validNum = stats.at<int>(i, 4);
		//if (validNum < 10) continue;
		//if (width < 10 || height < 10) continue;
		rectangle(frame_bgr, cv::Rect(xmin, ymin, width, height), Scalar(0, 0, 255), 1);
	}
	// ��ID��������ID����
	// ����rect���飬�������ص������ڣ��ͺϲ�ID��rect��id����ID��ID֮��ϲ���
#endif

#if USE_ORGIN_MERGE
	//-----------------------------------------------------------------
	/* ��ͨ������� */
	cv::Mat1i labelImg;
	//icvprCcaByTwoPass(segmentationMap, labelImg);
	//icvprCcaBySeedFill(segmentationMap, labelImg);
	// ʹ��SAUF�㷨��������ͨ����
	cv::Mat1b binImg = segmentationMap;
	DWORD start_time1 = GetTickCount();
	SAUF sauf_ufpc(binImg);
	sauf_ufpc.PerformLabeling();
	labelImg = sauf_ufpc.img_labels_;
	labelImg += 1;
	unsigned n_labels = sauf_ufpc.n_labels_;

	DWORD end_time1 = GetTickCount();
	//std::cout << "SAUFPC:\t" << (end_time1 - start_time1) * 1.00 / 1000 << "s" << endl;

	// ���չʾ
	cv::Mat colorLabelImg;
	colorLabelImg.create(labelImg.rows, labelImg.cols, CV_8UC3);
	icvprLabelColor(labelImg, colorLabelImg);
	//cv::imshow("��ɫ���ͼ", colorLabelImg);
	//-------------�����߼�------------//
	std::vector<cv::Rect> move_rects;

	//-----------------------------------------------------------------
	double maxp = 0.0, minp = 0.0;
	cv::minMaxIdx(labelImg, &minp, &maxp);
	int maxID = (int)maxp;
	// ����labelImg����¼ID��xmin,ymin,xmax,ymax
	if (maxID >= 2)
	{
		// ʹ��Mat������Calloc����ΪID���ܻ�ܴ󣬺ܿ���û��һ��Ƭ�������ڴ档
		Mat rect(maxID + 1, 6, CV_32S, Scalar(-1)); // ��ͨ��
		//// malloc����ʼ��Ϊ0��ʹ��calloc
		//int *rect = (int *)calloc(5 * maxID, sizeof(int));
		// ���ʷ�ʽ��[xmin,ymin,xmax,ymax,validNum]
		for (int i = 0; i < labelImg.rows; i++)
		{
			for (int j = 0; j < labelImg.cols; j++)
			{
				int id = labelImg.at<int>(i, j);
				if (id >= 2)	// 0 ���� 1 ǰ������2����label
				{
					int &isValid = rect.at<int>(id, 5);	// �Ƿ�����ЧID
					int &xmin = rect.at<int>(id, 0);	// max_ID * 5 �ľ���
					int &ymin = rect.at<int>(id, 1);
					int &xmax = rect.at<int>(id, 2);
					int &ymax = rect.at<int>(id, 3);
					int &validNum = rect.at<int>(id, 4);
					// �õ�ǰID����[xmin,ymin,xmax,ymax,validNum]
					if (isValid == -1)	// ��Ҫ��ʼ��
					{
						xmin = j;	ymin = i; xmax = j; ymax = i; validNum = 0;
						isValid = 0;	// �������ѳ�ʼ��
					}
					else
					{
						xmin = xmin < j ? xmin : j;
						ymin = ymin < i ? ymin : i;

						xmax = xmax > j ? xmax : j;
						ymax = ymax > i ? ymax : i;
						validNum++;
					}

				}
			}
		}
		// ���е����ص����һ������ˣ���ʼɸѡ����
		for (int i = 2; i <= maxID; i++)			// [2,3,...,maxID]�ڻ��������Ϊ������[0,1,...maxID-2]
		{	// �̶��еģ��в��̶�
			int &isValid = rect.at<int>(i, 5);	// �Ƿ�����ЧID
			int &xmin = rect.at<int>(i, 0);	// max_ID * 5 �ľ���
			int &ymin = rect.at<int>(i, 1);
			int &xmax = rect.at<int>(i, 2);
			int &ymax = rect.at<int>(i, 3);
			int &validNum = rect.at<int>(i, 4);

			//if (validNum <= 10) continue;		// ��������
			
			int width = std::abs(xmax - xmin);
			int height = std::abs(ymax - ymin);	// �����
			//int  bar_ratio = 20;
			//if (width >= bar_ratio * height || height >= bar_ratio * width)
			//	continue;
			//if (height < 10) continue;		// ���˳���10�����أ�����
			//�Աȶ�//�Ա߽���չ��������
			int dr = 2;

			int sx = xmin > dr ? xmin - dr : 0;
			int sy = ymin > dr ? ymin - dr : 0;

			int ex = xmax + dr < labelImg.cols ? xmax + dr : labelImg.cols;
			int ey = ymax + dr < labelImg.rows ? ymax + dr : labelImg.rows;

			int rw = ex - sx;
			int rh = ey - sy;
			// sx���Ͻǣ� x��col, y��row
			cv::Rect roi(sx, sy, rw, rh);
			move_rects.push_back(roi);
		}
		//free(rect);
	}
	std::cout << "�ϲ�ǰ: " << move_rects.size() << " rects" << endl;
	DWORD start_time2 = GetTickCount();
	// �ϲ���Χ��
	if (move_rects.size()>1) Merge(move_rects);
	DWORD end_time2 = GetTickCount();
	std::cout << "�ϲ���: " << move_rects.size() << " rects" << endl;
	std::cout << "�ϲ��ص���-ԭʼʵ��: " << (end_time2 - start_time2) * 1.00 << "ms" << endl;;
	// ����
	for (int i = 0; i < move_rects.size(); i++)
	{
		rectangle(frame_bgr, move_rects[i], Scalar(0, 0, 255), 1);
	}
#endif
#if SHOW
	cv::imshow("Frame", frame_bgr);
	cv::waitKey(1000);
#endif
	//------------------------------------------------------------------
#if 0		//��ͼ��������ԭʼͼ����ǰ��ͼ�������ͼ���汳��ģ��ͼ
	string pathfiles = "./img_shake0824";
	string path;
	string pathnum;
	stringstream ss;

	ss << frameNumber;
	ss >> pathnum;
	path = pathfiles + "/" + pathnum;
	string img_mat = ".jpg";
	//����ԭͼ
	imwrite(path + "_current_gray" + img_mat, frame);
	//����ǰ��ͼ
	imwrite(path + "_fg" + img_mat, segmentationMap);
	//�������ͼ
	imwrite(path + "_draw" + img_mat, colorMask.mul(frame_bgr));
#endif
    ++frameNumber;
	std::cout <<frameNumber<<endl;
#if 0 /* ����ǵ����õ� */
	cv::Mat fgCount(frame.rows, frame.cols, CV_8UC1);//�鿴ǰ������
	//�������ڲ���ȡ������������Ҫ����һ������ȥȡ������
	fgCount = Array2Mat(GetFgCount(model), frame.rows, frame.cols);
	histBuff[0] = Array2Mat(historyImage, frame.rows, frame.cols);
	histBuff[1] = Array2Mat(historyImage + frame.rows * frame.cols, frame.rows, frame.cols);
	for (int i = 0; i < frame.rows; i++){
		for (int j = 0; j < frame.cols; j++){
			for (int k = 2; k < 20; k++){
				int t = k - 2;
				histBuff[k].at<uchar>(i, j) = historyBuffer[(i*frame.cols + j) * 18 + t];
			}
		}
	}
	#if 0
	//������Ҫ��ע�ĵ�
	int point_rows = 0, point_cols = 0;//�ζ��ϴ��һ���� 364,122,for wavingtrees_inray
	point[0] = frame.at<uchar>(point_rows, point_cols);
	for (int i = 1; i < 21;i++)
	{
		point[i] = histBuff[i-1].at<uchar>(point_rows,point_cols);
	}
	/* Gets the input from the keyboard. */
	keyboard = waitKey(1);
	for (int i = 0; i < 21; i++)
	{
		point_last[i] = point[i];//�洢��һ�εĽ��
	}
	#endif
#endif
  } /* end of while : main outer loop */

  /* Delete capture object. */
  capture.release();
  /* Frees the model. */
  libvibeModel_Sequential_Free(model);
}

Mat Array2Mat(uint8_t *a, int rows, int cols){
	Mat M(rows, cols, CV_8UC1);
	for (int i = 0; i < M.rows; ++i){
		uchar *p = M.ptr<uchar>(i);
		for (int j = 0; j < M.cols; ++j)
			p[j] = a[i*cols + j];
	}
	return M;
}
void Merge(std::vector<cv::Rect>& rects)
{
	bool is_continue = true;
	int s = 0, t = 0;
	do
	{
		is_continue = intetsect(rects, s, t);
		if (is_continue == true)
		{
			Merge(rects, s, t);
		}
	} while (is_continue);
}
bool intetsect(std::vector<cv::Rect>& rects, int&s, int&t)
{
	unsigned num = rects.size();
	for (unsigned i = 0; i < num; ++i)
	{
		for (unsigned j = i + 1; j < num; ++j)
		{
			const cv::Rect& a = rects[i];
			const cv::Rect& b = rects[j];

			const float& overlap_ratio = intetsect(a, b);

			if (overlap_ratio > 0.1)
			{
				s = i;
				t = j;
				return true;
			}
		}
	}
	return false;
}
void Merge(std::vector<cv::Rect>& rects, const int&s, const int&t)
{
	//t>s
	cv::Rect a = rects[s];
	cv::Rect b = rects[t];

	//�޸�����һ��
	auto pos = std::find(rects.begin(), rects.end(), b);
	if (pos != rects.end())
	{
		int x = a.x < b.x ? a.x : b.x;
		int y = a.y < b.y ? a.y : b.y;

		int w = a.x + a.width  > b.x + b.width ? a.x + a.width : b.x + b.width;
		int h = a.y + a.height > b.y + b.height ? a.y + a.height : b.y + b.height;

		w -= x;	h -= y;

		pos->x = x;
		pos->y = y;
		pos->width = w;
		pos->height = h;
	}
	//ɾ��������һ��
	pos = std::find(rects.begin(), pos, a);
	if (pos != rects.end())	rects.erase(pos);
}
double intetsect(const cv::Rect& a, const cv::Rect& b)
{
	if (a.x + a.width < b.x || b.x + b.width < a.x)  return 0.0f;
	if (a.y > b.y + b.height || b.y > a.y + a.height) return 0.0f;

	double w = min(a.x + a.width, b.x + b.width) - max(a.x, b.x);
	double h = min(a.y + a.height, b.y + b.height) - max(a.y, b.y);

	double area = w*h;
	double area_0 = a.width*a.height;
	double area_1 = b.width*b.height;

	double ratio = area_0 < area_1 ? area / area_0 : area / area_1;

	return ratio;
}

/* �ж��������ο��Ƿ��ص� */
bool isRectangleOverlap(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2)
{
	// rect1=[x1, y1, t1, t2] rect2=[x2, y2, s1, s2]
	int t1 = x1 + w1;
	int t2 = y1 + h1;

	int s1 = x2 + w2;
	int s2 = y2 + h2;	//!(rect1[2] <= rect2[0] || rect1[3]<= rect2[1] || rect1[0] >= rec2[2] || rect[1] >=rec2[3])
	return !(t1 < x2 || t2 <  y2 || x1 > s1 || y1 > s2);
}
bool isCenterClose(double Cenx, double Ceny, double Cenx_new, double Ceny_new)
{
	double dist = (Cenx - Cenx_new) * (Cenx - Cenx_new) + (Ceny - Ceny_new) * (Ceny - Ceny_new);
	return dist <= CENTER_DIST_THRESH;
}

/*
	stats�Ǿ��ο����centroids�����ģ�����IDһֱ��i��j��Ҫmerge��
	min(i,j) = merge(i,j); max(i,j) = -1
*/
void Merge(cv::Mat &stats, cv::Mat &centroids, int i, int j)
{
	int minID = min(i, j);
	int maxID = max(i, j);
	// ���ο�x��y
	int &x = stats.at<int>(minID, 0);
	int &y = stats.at<int>(minID, 1);
	int &w = stats.at<int>(minID, 2);
	int &h = stats.at<int>(minID, 3);
	int &area = stats.at<int>(minID, 4);

	int &x2 = stats.at<int>(maxID, 0);
	int &y2 = stats.at<int>(maxID, 1);
	int &w2 = stats.at<int>(maxID, 2);
	int &h2 = stats.at<int>(maxID, 3);
	int &area2 = stats.at<int>(maxID, 4);

	// �ȼ������½Ǿ�������
	w = max(x + w, x2 + w2);
	h = max(y + h, y2 + h2);
	// ��������ϽǾ�������
	x = min(x, x2);
	y = min(y, y2);
	w -= x;
	h -= y;

	// �ϲ�����
	double &Cenx = centroids.at<double>(minID, 0);
	double &Ceny = centroids.at<double>(minID, 1);
	double &Cenx_new = centroids.at<double>(maxID, 0);
	double &Ceny_new = centroids.at<double>(maxID, 1);
	Cenx = (Cenx * area + Cenx_new * area2) / (area + area2);
	Ceny = (Ceny * area + Ceny_new * area2) / (area + area2);
	Cenx_new = -1;
	Ceny_new = -1;

	// �ϲ����
	area += area2;
	// ��һ������
	x2 = y2 = w2 = h2 = area2 = -1;
	return;
}

/* int nccomps, cv::Mat stats, cv::Mat centroids */
std::vector<DetectBox> getMoveRects(int nccomps, cv::Mat stats, cv::Mat centroids)
{
	std::vector<DetectBox> move_rects;
	std::vector<cv::Point2f> center_points;
	for (int id = 1; id < nccomps; id++)
	{
		int xmin = stats.at<int>(id, 0);	// max_ID * 5 �ľ���
		int ymin = stats.at<int>(id, 1);
		int width = stats.at<int>(id, 2);
		int height = stats.at<int>(id, 3);
		int validNum = stats.at<int>(id, 4);
		// ɸѡ���֣��߶ȡ���ȡ����������������ȵ�
		DetectBox tmp;
		tmp.rect = cv::Rect(xmin, ymin, width, height);
		tmp.validNum = validNum;
		// ��Ӧ������λ��
		float cenx = centroids.at<double>(id, 0);
		float ceny = centroids.at<double>(id, 1);
		tmp.center_point = cv::Point2f(cenx, ceny);
		move_rects.push_back(tmp);
	}
	return move_rects;
}
/* ʹ������ʵ�� */
std::forward_list<DetectBox> getMoveRects_list(int nccomps, cv::Mat stats, cv::Mat centroids)
{
	// ����β��û������
	std::forward_list<DetectBox> move_rects;
	// �Ѿ���ʼ����û����
	auto it = move_rects.before_begin();
	std::vector<cv::Point2f> center_points;
	for (int id = 1; id < nccomps; id++, it++)
	{
		int xmin = stats.at<int>(id, 0);	// max_ID * 5 �ľ���
		int ymin = stats.at<int>(id, 1);
		int width = stats.at<int>(id, 2);
		int height = stats.at<int>(id, 3);
		int validNum = stats.at<int>(id, 4);
		// ɸѡ���֣��߶ȡ���ȡ����������������ȵ�
		DetectBox tmp;
		tmp.rect = cv::Rect(xmin, ymin, width, height);
		tmp.validNum = validNum;
		// ��Ӧ������λ��
		float cenx = centroids.at<double>(id, 0);
		float ceny = centroids.at<double>(id, 1);
		tmp.center_point = cv::Point2f(cenx, ceny);
		// �����Ǻ�壬���������Ҫ����λ��Ϊ��һ��Ԫ�أ�
		// ���޷�������begin()����ɣ�����Ҫ����before_begin()����ʵ�֡�
		move_rects.emplace_after(it, tmp);
	}
	return move_rects;
}
bool isOverlap(cv::Point2i tl1, cv::Point2i br1, cv::Point2i curTl, cv::Point2i curBR)
{
	//[tl1.x, tl1.y, br1.x, br1.y] [curTl.x, curTl.y, curBR.x, curBR.y]
	return !(br1.x < curTl.x || br1.y < curTl.y || tl1.x > curBR.x || tl1.y > curBR.y);
}

bool isCenterClose(cv::Point2f center, cv::Point2f curCenter)
{
	float diff = (center.x - curCenter.x) * (center.x - curCenter.x)
		+ (center.y - curCenter.y) * (center.y - curCenter.y);
	return diff < CENTER_DIST_THRESH;
}

std::vector<int> getAllOverlaps(std::vector<DetectBox>move_rects, DetectBox curRect, int index)
{
	std::vector<int> overlaps_ID;
	overlaps_ID.clear();
	// ��move_rects���ҵ�������tl br��rect�ص��Ŀ򣬲���������
	for (int i = 0; i < move_rects.size(); i++)
	{
		if (i != index)
		{
			if (isOverlap(move_rects[i].rect.tl(), move_rects[i].rect.br(), curRect.rect.tl(), curRect.rect.br()))
				overlaps_ID.push_back(i);
			else if (isCenterClose(move_rects[i].center_point, curRect.center_point))
				overlaps_ID.push_back(i);
		}
	}
	return overlaps_ID;
}
std::vector<std::forward_list<DetectBox>::iterator> getAllOverlaps_list(std::forward_list<DetectBox>move_rects
	, DetectBox curRect, std::forward_list<DetectBox>::iterator it)
{
	std::vector<std::forward_list<DetectBox>::iterator> overlaps_ID;
	overlaps_ID.clear();
	// ˳���������
	for (std::forward_list<DetectBox>::iterator iter = move_rects.before_begin()
		; iter != move_rects.end(); iter++)
	{
		if (iter != it)		// ��ȥ��ǰ��
		{
			if (isOverlap(iter->rect.tl(), iter->rect.br(), curRect.rect.tl(), curRect.rect.br()))
				overlaps_ID.push_back(iter);
			else if (isCenterClose(iter->center_point, curRect.center_point))
				overlaps_ID.push_back(iter);
		}
	}
	return overlaps_ID;
}
/* �ϲ��ص���ͽ��ڿ� */
void Merge(std::vector<DetectBox> &move_rects)
{
	bool finished = false;
	std::cout << "�ϲ�ǰ: " << move_rects.size() << " rects" << endl;
	while (!finished)			// �Ƿ����ص���
	{
		finished = true;
		int index = move_rects.size() - 1;
		while (index >= 0)
		{
			// ����߽�
			DetectBox curRect = move_rects[index];
			// ��ȡ�����ص����ID
			std::vector<int> overlaps = getAllOverlaps(move_rects, curRect, index);
			if (overlaps.size() > 0)				// ���ص��ͺϲ�
			{
				overlaps.push_back(index);
				std::vector<cv::Point2i> points;
				points.clear();
				float cenx = 0.0;
				float ceny = 0.0;
				int validNumSum = 0;
				for (int i = 0; i < overlaps.size(); i++)
				{
					points.push_back(move_rects[overlaps[i]].rect.tl());
					points.push_back(move_rects[overlaps[i]].rect.br());
					// ����ϲ��������
					int num = move_rects[overlaps[i]].validNum;
					cenx += move_rects[overlaps[i]].center_point.x * num;
					ceny += move_rects[overlaps[i]].center_point.y * num;
					validNumSum += num;
				}
				// �ϲ���Ĵ��
				cv::Rect mergeRect = cv::boundingRect(points);
				cenx /= validNumSum;
				ceny /= validNumSum;
				DetectBox mergeNewRect;
				mergeNewRect.rect = mergeRect;
				mergeNewRect.center_point = cv::Point2f(cenx, ceny);
				mergeNewRect.validNum = validNumSum;
				// ɾ���ϲ�ǰ������С������������ɾ����Ч�ʽϸ�
				std::sort(overlaps.begin(), overlaps.end(), cmp1);
				// ��ȡ�������ĵ�һ��ֵ

				for (int i = 0; i < overlaps.size(); i++)
				{
					vector<DetectBox>::iterator   iter = move_rects.begin() + overlaps[i];
					move_rects.erase(iter);
				}
				move_rects.push_back(mergeNewRect);

				finished = false;
				break;
			}
			index -= 1;
		}	// end of inner while
	}	// end of outer while
	std::cout << "�ϲ���: " << move_rects.size() << " rects" << endl;
}

/* �ϲ��ص���ͽ��ڿ� */
void Merge_list(std::forward_list<DetectBox> &move_rects, unsigned &list_size)
{
	bool finished = false;
	// ��ʼ���Ϳ����ˣ����Ǵ˴�Ҫ���������ָ�룬û�а취������ʹ��auto
	std::cout << "�ϲ�ǰ: " << list_size << " rects" << endl;
	while (!finished)							// �Ƿ����ص���
	{
		finished = true;

		auto prev_outer = move_rects.before_begin();

		// ��һ�ζ���ͷ��ʼ
		bool has_overlap = false;
		for (auto it = move_rects.begin(); it != move_rects.end(); prev_outer=it, it++)			// ͷ��Ϳ�����
		{
			// ����߽�-------------------�����ͷ��Ȼ��ÿ�δ�ͷ�����ߣ�����ϣ���ģ�����
			DetectBox curRect = *it;
			std::vector<cv::Point2i> points;
			points.clear();
			float cenx = 0.0;
			float ceny = 0.0;
			int validNumSum = 0;
			Statistic(points, cenx, ceny, validNumSum, it);
			// ��ȡ�����ص����ID
			//std::vector<std::forward_list<DetectBox>::iterator> overlaps;
			//auto overlaps = getAllOverlaps_list(move_rects, curRect, it);
			// ��һ�α�������ɾ���ص���ɾ��Ҫ�ú�ѧϰһ��
				// �����ص����ҵ���ɾ��ͬʱ��¼ͳ����Ϣ��Ϊ����add��׼��
			auto prev = move_rects.before_begin();
			auto curr = move_rects.begin();

			while(curr != move_rects.end())
			{
				if (curr != it)		// ��ȥ��ǰ�� 
				{
					if (isOverlap(curr->rect.tl(), curr->rect.br(), curRect.rect.tl(), curRect.rect.br())
						|| isCenterClose(curr->center_point, curRect.center_point) )
					{
						has_overlap = true;
						Statistic(points, cenx, ceny, validNumSum, curr);
						// ɾ�������
						curr = move_rects.erase_after(prev);
						list_size--;
					}
					else// ����������������+1
					{
						prev = curr;
						curr++;
					}
				}
				else// ����������������+1
				{
					prev = curr;
					curr++;
				}		
			}	// end of inner while ��ǰ����ص���Ѱ�����
			if (has_overlap)
			{
				// �ϲ���Ĵ��
				cv::Rect mergeRect = cv::boundingRect(points);
				cenx /= validNumSum;
				ceny /= validNumSum;
				DetectBox mergeNewRect;
				mergeNewRect.rect = mergeRect;
				mergeNewRect.center_point = cv::Point2f(cenx, ceny);
				mergeNewRect.validNum = validNumSum;
				// ͷ�壺emplace_front()ͷ�������½ڵ㡣
				move_rects.erase_after(prev_outer);			// ���д���©�����µ���ѭ��������
				move_rects.emplace_front(mergeNewRect);
				finished = false;
				break;		// �ϲ��˾����´�ͷ��ʼ
			}
		}	// end of out for ������һ���ο���
	}	// end of outer while
	std::cout << "�ϲ���: " << list_size << " rects" << endl;
}  
// ������Ҫ�ֽ�merge()����ռ�˶���ʱ�䣬ռ��ʱ��Ķ��٣��߼��������      
void Statistic(std::vector<cv::Point2i> &points, float &cenx, float &ceny, int &validNumSum
	,  std::forward_list<DetectBox>::iterator curr)
{
	// ͳ�ƺϲ���Ϣ
	points.push_back(curr->rect.tl());
	points.push_back(curr->rect.br());
	// ����ϲ��������
	int num = curr->validNum;
	cenx += curr->center_point.x * num;
	ceny += curr->center_point.y * num;
	validNumSum += num;
}