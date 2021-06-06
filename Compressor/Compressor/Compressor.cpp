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

using namespace cv;
using namespace std;

struct Img
{
	IplImage* implImg;
	Mat matImg;
};

string path = "C:/Users/User1/Documents/Course4/Diploma/";

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
	String imageName = "ff1.jpg";
	if (argc > 1) {
		imageName = String(argv[1]);
	}
	img.matImg = imread(path + imageName);// получаем картинку
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
	ofstream f(path+ "new_coordinates.txt", ios::trunc);
	for (int i = 0; i < boundRect.size(); i++) {
		f << "BR - " << boundRect[i].br().x << ";" << boundRect[i].br().y << endl;
		f << "TL - " << boundRect[i].tl().x << ";" << boundRect[i].tl().y << endl;
		f << endl;
	}
	f.close();
}

vector<Rect> FindObjects(Img& img) {
	Mat threshold_output = img.matImg, gray_mat, blur_image;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int x = img.implImg->width, y = img.implImg->height;
	////// создаём одноканальные картинки
	cvtColor(img.matImg, gray_mat, CV_BGR2GRAY);
	// Detect edges using Threshold
	blur(gray_mat, blur_image, Size(10, 10)); // apply blur to grayscaled image
	//GaussianBlur(gray_mat, blur_image, Size(10, 10), 0, 0);
	Show("blur", blur_image);
	threshold(blur_image, threshold_output, 70, 255, THRESH_BINARY);
	//Canny(gray_mat, threshold_output, 90, 255, 3);
	Show("threshold", threshold_output);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	
	/*
	
	vector< vector<Point> > hull(contours.size());
	for (int i = 0; i < contours.size(); i++) {

		convexHull(Mat(contours[i]), hull[i], false);
	}
	Mat drawing1 = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++){
		Scalar color_contours = Scalar(0, 255, 0); // green - color for contours
		Scalar color = Scalar(255, 0, 0); // blue - color for convex hull
		// draw ith contour
	//	drawContours(drawing1, contours, i, color_contours, 5, 8, vector<Vec4i>(), 0, Point());
		// draw ith convex hull
		drawContours(drawing1, hull, i, color, 5, 8, vector<Vec4i>(), 0, Point());
	}
	Show("hull", drawing1);
	// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > poly(hull.size());
	vector<Rect> bRect(hull.size());
	printf("%d objects on pic\n", hull.size());
	for (int i = 0; i < hull.size(); i++) {
		approxPolyDP(Mat(hull[i]), poly[i], 1, true);
		bRect[i] = boundingRect(Mat(poly[i]));
		//	minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i < hull.size(); i++)
	{
		printf("tl - (%d %d) - br - (%d %d) \n", bRect[i].tl().x, bRect[i].tl().y,
			bRect[i].br().x, bRect[i].br().y);

		Scalar color = Scalar(RNG(12345).uniform(0, 0), RNG(12345).uniform(0, 255), RNG(12345).uniform(0, 255));

		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, bRect[i].tl(), bRect[i].br(), color, 5, 8, 0);

	}*/

	vector<Rect> boundRect(contours.size());
// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	printf("%d objects on pic\n", contours.size());
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	//	minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}

	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		printf("tl - (%d %d) - br - (%d %d) \n", boundRect[i].tl().x, boundRect[i].tl().y,
			boundRect[i].br().x, boundRect[i].br().y);
		cvRectangle(img.implImg, boundRect[i].tl(), boundRect[i].br(), CV_RGB(255, 0, 0));
		cvSetImageROI(img.implImg, boundRect[i]);
		Mat subImg = img.matImg(boundRect[i]);

		Scalar color = Scalar(RNG(12345).uniform(0, 0), RNG(12345).uniform(0, 255), RNG(12345).uniform(0, 255));

		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 5, 8, 0);

		// Никогда не забывайте удалять ИОР
		cvResetImageROI(img.implImg);
		string filename = to_string(i);
		//imwrite(path+filename + ".jpg", subImg);
		//imwrite(filename + "compressed.jpg", subImg, { IMWRITE_JPEG_QUALITY, 50 });
	}

	Show("objects", drawing);

	Mat drawng = drawing, gry_mat;
	////// создаём одноканальные картинки
	cvtColor(drawing, gry_mat, CV_BGR2GRAY);
	threshold(gry_mat, drawng, 120, 255, THRESH_BINARY);
		vector<vector<Point> > cntours;
		vector<Vec4i> herarchy;

		findContours(drawng, cntours, herarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<vector<Point> > cntours_poly(cntours.size());
		vector<Rect> bundRect(cntours.size());
		vector<Point2f>cnter(cntours.size());
		vector<float>rdius(cntours.size());
		
		for (int i = 0; i < cntours.size(); i++) {
			approxPolyDP(Mat(cntours[i]), cntours_poly[i], 10, true);
			bundRect[i] = boundingRect(Mat(cntours_poly[i]));
			//	minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
		}
		printf("sec time! %d objects on pic\n", cntours.size());
		Mat drwing = Mat::zeros(drawing.size(), CV_8UC3);
		for (int i = 0; i < cntours.size(); i++)
		{
			printf("tl - (%d %d) - br - (%d %d) \n", bundRect[i].tl().x, bundRect[i].tl().y,
				bundRect[i].br().x, bundRect[i].br().y);
	
			Scalar color = Scalar(RNG(12345).uniform(0, 0), RNG(12345).uniform(0, 0), RNG(12345).uniform(0, 255));

			//drawContours(drwing, cntours_poly, i, color, 5, 8, vector<Vec4i>(), 0, Point());
			rectangle(drwing, bundRect[i].tl(), bundRect[i].br(), color, 5, 8, 0);


		}

	
	Show("objects2", drwing);
	return boundRect;
}

bool IsCoordinatesEmpty() {
	fstream file(path+"coordinates.txt");
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
		Scalar color = Scalar(RNG(12345).uniform(0, 0), RNG(12345).uniform(0, 0), RNG(12345).uniform(0, 255));
		rectangle(drawing, old_coord[i].tl(), old_coord[i].br(), color, 5, 8, 0);
		imwrite(path + to_string(i) + "_old.jpg", subImg);
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
		imwrite(path+"original.jpg", img.matImg);
	}
	//поиск объектов на изображении
	auto objects = FindObjects(img);
	if (!first_launch) {
		//сохранение частей изображения с предыдущих положений объектов
		auto old_coord = ReadCoord(path + "coordinates.txt");
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