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
#include <filesystem>

using namespace cv;
using namespace std;

string path = "C:/Users/User1/Documents/Course4/Diploma/";

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

bool IsCoordinatesEmpty() {
	fstream file(path + "coordinates.txt");
	if (!file.is_open()) {
		cout << "Not open\n"; // если не открылся
		return true;
	}
	else if (file.peek() == EOF)
		return true;// если первый символ конец файла
	file.close();
	return false;
}

IplImage* TransformMatToImg(Mat& mat) {
	auto src_img = cvCreateImage(cvSize(mat.cols, mat.rows), 8, 3);
	IplImage buff = mat;
	cvCopy(&buff, src_img);
	return src_img;
}

void InsertOldPlaces(IplImage* orig_img, vector<Rect>& old_coord) {
	for (int i = 0; i < old_coord.size(); i++) {
		Mat subImg = imread(path+to_string(i) + "_old.jpg");
		//wana see this?
		//transform
		auto src_img = TransformMatToImg(subImg);
		//start insert
		cvSetImageROI(orig_img, old_coord[i]);
		// обнулим изображение
		cvZero(orig_img);
		// копируем изображение
		cvCopy(src_img, orig_img);
		cvResetImageROI(orig_img);
		//Show("ROI", orig_img);
	}
}

void InsertNewPlaces(IplImage* orig_img, vector<Rect>& new_coord) {
	for (int i = 0; i < new_coord.size(); i++) {
		Mat src_mat = imread(path+to_string(i) + ".jpg");
		auto src_img = TransformMatToImg(src_mat);
		cvSetImageROI(orig_img, new_coord[i]);
		// обнулим изображение
		cvZero(orig_img);
		//// копируем изображение
		cvCopy(src_img, orig_img);
		cvResetImageROI(orig_img);
		//show("ROI", orig_img);
	}
	Mat out_mat = cvarrToMat(orig_img);
	Show("out", out_mat);
	imwrite(path + "original.jpg", out_mat);
}

void AcceptChanges() {
	string oldpath = path + "coordinates.txt";
	string newpath = path + "new_coordinates.txt";
	remove(oldpath.c_str());
	if (rename(newpath.c_str(), oldpath.c_str())) {
		cout << "Troubled renaming" << endl;
	}
}

void ReadCoords(vector<Rect>& old_coord, vector<Rect>& new_coord) {
	string oldpath = path + "coordinates.txt";
	string newpath = path + "new_coordinates.txt";
	old_coord = ReadCoord(oldpath);
	new_coord = ReadCoord(newpath);
	AcceptChanges();
}

int main(int argc, char* argv[])
{
	vector<Rect> old_coord, new_coord;
	bool first_launch = IsCoordinatesEmpty();
	if (!first_launch) {
		//read original
		ReadCoords(old_coord, new_coord);
		// получаем картинку фон
		Mat orig_mat = imread(path + "original.jpg");
		auto orig_img = TransformMatToImg(orig_mat);
		InsertOldPlaces(orig_img, old_coord);
		InsertNewPlaces(orig_img, new_coord);
	}
	else {
		AcceptChanges();
	}
	// удаляем окна
	cvDestroyAllWindows();
	return 0;
}