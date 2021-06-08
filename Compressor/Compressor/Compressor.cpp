#include <fstream>
#include "opencv2/core/core_c.h"
#include "opencv2/core/types_c.h"
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/flann/flann.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <string>
#include "source.h"

using namespace cv;
using namespace std;

struct Img
{
	IplImage* implImg;
	Mat matImg;
};

void Show(string pic_name, Mat& pic) {
	cvNamedWindow(pic_name.c_str(), WINDOW_NORMAL);
	imshow(pic_name.c_str(), pic);
	cvWaitKey(0);
}

void Show(string pic_name, IplImage* pic) {
	cvNamedWindow(pic_name.c_str(), CV_WINDOW_NORMAL);
	cvShowImage(pic_name.c_str(), pic);
	cvWaitKey(0);
}

IplImage* TransformMatToImg(Mat& mat) {
	auto src_img = cvCreateImage(cvSize(mat.cols, mat.rows), 8, 3);
	IplImage buff = mat;
	cvCopy(&buff, src_img);
	return src_img;
}

Img GetImage(int argc, char* argv[]) {
	Img img;
	String imageName = FILENAME;
	if (argc > 1) {
		imageName = String(argv[1]);
	}
	img.matImg = imread(PATH + imageName);// получаем картинку
	if (img.matImg.empty()) {// Check for invalid input
		cout << "Could not open or find the image" << std::endl;
		exit(-1);
	}
	//transform
	img.implImg = TransformMatToImg(img.matImg);//new_image = cvCreateImage(cvSize(new_image_mat.cols, new_image_mat.rows), 8, 3);//IplImage ipltemp = new_image_mat;//cvCopy(&ipltemp, new_image);
	// покажем изображение
	Show("new", img.implImg);
	return img;
}

vector<Rect> ReadCoord(string file_name) {
	ifstream f(file_name, ios_base::in);
	vector<Rect> pieces;
	CvPoint tl;
	CvPoint br;
	if (f.is_open()) {
		char c[8], b[3], a;
		int x, y;
		int i = 0;
		string num = "";
		while (!f.eof()) {
			f >> a;
			if (a == '-' && i == 2) {
				i = 0;
				Rect r(tl, br);
				pieces.push_back(r);
			}
			while (isdigit((int)a)) {
				num += a;
				f >> a;
			}
			if (a == ';') {
				x = atoi(num.c_str());
				num = "";
				f >> a;
				while (isdigit((int)a)) {
					if (f.eof()) {
						break;
					}
					num += a;
					f >> a;
				}
				f >> a;
				y = atoi(num.c_str());
				num = "";
				if (i == 0) {
					tl = CvPoint(x, y);
				}
				if (i == 1) {
					br = CvPoint(x, y);
				}
				if (f.eof()) {
					Rect r(tl, br);
					pieces.push_back(r);
				}
				i++;
			}
		}
	}
	f.close();

	return pieces;
}

void WriteCoordinates(vector<Rect>& boundRect) {
	ofstream f(NEW_COORD_FILE, ios::trunc);
	for (int i = 0; i < boundRect.size(); i++) {
		f << "BR - " << boundRect[i].br().x << ";" << boundRect[i].br().y << endl;
		f << "TL - " << boundRect[i].tl().x << ";" << boundRect[i].tl().y << endl;
		f << endl;
	}
	f.close();
}

void EnlargeROI(Mat& frm, Rect& boundingBox) {
	int padding = EPSBOUND;
	Rect returnRect = Rect(boundingBox.x - padding, boundingBox.y - padding, boundingBox.width + (padding * 2), boundingBox.height + (padding * 2));
	if (returnRect.x < 0)returnRect.x = 0;
	if (returnRect.y < 0)returnRect.y = 0;
	if (returnRect.x + returnRect.width >= frm.cols)returnRect.width = frm.cols - returnRect.x;
	if (returnRect.y + returnRect.height >= frm.rows)returnRect.height = frm.rows - returnRect.y;
	boundingBox = returnRect;
}

vector<Rect> FindObjects(Img& img) {
	Mat threshold_output = img.matImg, gray_mat, blur_image;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int x = img.implImg->width, y = img.implImg->height;
	// создаём одноканальные картинки
	cvtColor(img.matImg, gray_mat, CV_BGR2GRAY);
	// размываем картинку чтобы убрать шумы
	blur(gray_mat, blur_image, BLURSIZE);
	// применяем пороговое преобразование для выделения объектов
	threshold(blur_image, threshold_output, LOWTHRESHOLD, HIGHTHRESHOLD, THRESH_BINARY);
	Show("threshold", threshold_output);
	/// Поиск контуров
	findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// Аппроксимация контуров к прямоугольникам
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	printf("%d objects on pic\n", contours.size());
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], EPSBOUND, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		EnlargeROI(img.matImg, boundRect[i]);
		printf("tl - (%d %d) - br - (%d %d) \n", boundRect[i].tl().x, boundRect[i].tl().y,
			boundRect[i].br().x, boundRect[i].br().y);
		//отрисовка для наглядности
		drawContours(drawing, contours_poly, i, CV_RGB(0, 255, 0), LINETHICK, LINETYPE, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), CV_RGB(0, 0, 255), LINETHICK, LINETYPE, 0);
		//установка ИОР
		cvSetImageROI(img.implImg, boundRect[i]);
		//вырезание объекта
		Mat subImg = img.matImg(boundRect[i]);
		//никогда не забываем удалять ИОР
		cvResetImageROI(img.implImg);
		string filename = to_string(i);
		imwrite(PATH + filename + EXTENSION, subImg);
		//imwrite(filename + "compressed.jpg", subImg, { IMWRITE_JPEG_QUALITY, 50 });
	}
	Show("objects", drawing);
	return boundRect;
}

bool IsCoordinatesEmpty() {
	fstream file(COORD_FILE);
	if (!file.is_open()) {
		cout << "First launch\n"; // если не открылся
		return true;
	}
	else if (file.peek() == EOF)
		return true;// если первый символ конец файла
	file.close();
	return false;
}

void CopyOldPlaces(Img& img, vector<Rect>& old_coord) {
	Mat drawing = Mat::zeros(img.matImg.size(), CV_8UC3);;
	for (int i = 0; i < old_coord.size(); i++) {
		Mat subImg = img.matImg(old_coord[i]);
		rectangle(drawing, old_coord[i].tl(), old_coord[i].br(), CV_RGB(0, 0, 255), LINETHICK, LINETYPE, 0);
		imwrite(PATH + to_string(i) + MARKEDEXTENSION, subImg);
	}
	Show("old places", drawing);
}

int main(int argc, char* argv[])
{
	//проверка файла с координатами
	bool first_launch = IsCoordinatesEmpty();
	//получение нового изображения
	Img img = GetImage(argc, argv);
	if (first_launch) {
		//если алгоритм запущен впервые, то изображение остается неизменным
		imwrite(ORIG_FILE, img.matImg);
	}
	//поиск объектов на изображении
	auto objects = FindObjects(img);
	if (!first_launch) {
		//сохранение частей изображения с предыдущих положений объектов
		auto old_coord = ReadCoord(COORD_FILE);
		CopyOldPlaces(img, old_coord);
	}
	//запись координат новых положений объектов
	WriteCoordinates(objects);
	// освобождение ресурсов
	cvReleaseImage(&img.implImg);
	// удаление окон
	cvDestroyAllWindows();
	return 0;
}