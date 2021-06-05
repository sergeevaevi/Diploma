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

vector<Rect> old_coord, new_coord;
IplImage* new_image;
Mat new_image_mat;
string path = "C:/Users/User1/Documents/Course4/";

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

void show(string pic_name, Mat pic) {
	cvNamedWindow(pic_name.c_str(), WINDOW_NORMAL);
	imshow(pic_name.c_str(), pic);
	cvWaitKey(0);
}

void show(string pic_name, IplImage* pic) {
	cvNamedWindow(pic_name.c_str(), CV_WINDOW_NORMAL);
	cvShowImage(pic_name.c_str(), pic);
	cvWaitKey(0);
}

vector<Rect> FindObjects(Mat img) {
	Mat threshold_output = img, gray_mat;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int x = new_image->width, y = new_image->height;
	////// создаём одноканальные картинки
	cvtColor(img, gray_mat, CV_BGR2GRAY);
	// Detect edges using Threshold
	threshold(gray_mat, threshold_output, 100, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));
	// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	printf("%d objects on pic\n", contours.size());
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}

	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		printf("tl - (%d %d) - br - (%d %d) \n", boundRect[i].tl().x, boundRect[i].tl().y,
			boundRect[i].br().x, boundRect[i].br().y);
		cvRectangle(new_image, boundRect[i].tl(), boundRect[i].br(), CV_RGB(255, 0, 0));
		cvSetImageROI(new_image, boundRect[i]);
		Mat subImg = img(boundRect[i]);

		Scalar color = Scalar(RNG(12345).uniform(0, 255), RNG(12345).uniform(0, 255), RNG(12345).uniform(0, 255));
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 5, 8, 0);
		// Никогда не забывайте удалять ИОР
		cvResetImageROI(new_image);
		string filename = to_string(i);
		imwrite(filename + ".jpg", subImg);
		//imwrite(filename + "compressed.jpg", subImg, { IMWRITE_JPEG_QUALITY, 50 });
	}
	show("objects", drawing);
	return boundRect;
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

IplImage* TransformMatToImg(Mat mat) {
	auto src_img = cvCreateImage(cvSize(mat.cols, mat.rows), 8, 3);
	IplImage buff = mat;
	cvCopy(&buff, src_img);
	return src_img;
}

void GetImage(int argc, char* argv[]) {
	String imageName = "mew.jpg";
	if (argc > 1) {
		imageName = String(argv[1]);
	}
	new_image_mat = imread(imageName);// получаем картинку
	if (new_image_mat.empty()) {// Check for invalid input
		cout << "Could not open or find the image" << std::endl;
		exit(-1);
	}
	//transform
	new_image = TransformMatToImg(new_image_mat);//new_image = cvCreateImage(cvSize(new_image_mat.cols, new_image_mat.rows), 8, 3);//IplImage ipltemp = new_image_mat;//cvCopy(&ipltemp, new_image);
	// покажем изображение
	show("new", new_image);
}

void CopyOldPlaces() {
	Mat drawing = Mat::zeros(new_image_mat.size(), CV_8UC3);;
	for (int i = 0; i < old_coord.size(); i++) {
		Mat subImg = new_image_mat(old_coord[i]);
		Scalar color = Scalar(RNG(12345).uniform(0, 0), RNG(12345).uniform(0, 0), RNG(12345).uniform(0, 255));
		rectangle(drawing, old_coord[i].tl(), old_coord[i].br(), color, 5, 8, 0);
		imwrite(to_string(i) + "_old.jpg", subImg);
		//imwrite(filename + "_oldcompressed.jpg", subImg, { IMWRITE_JPEG_QUALITY, 50 });
	}
	show("old places", drawing);
}

void InsertOldPlaces(IplImage* orig_img) {
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
		//show("ROI", orig_img);
	}
}

void InsertNewPlaces(IplImage* orig_img) {
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
	show("out", out_mat);
	imwrite("original.jpg", out_mat);
}

void ReadCoords() {
	string oldpath = path + "coordinates.txt";
	string newpath = path + "new_coordinates.txt";
	old_coord = ReadCoord(oldpath);
	new_coord = ReadCoord(newpath);
	remove(oldpath.c_str());
	rename(newpath.c_str(),oldpath.c_str());
}

int main(int argc, char* argv[])
{
	bool first_launch = IsCoordinatesEmpty();

	
	//if (first_launch) {
		//GetImage(argc, argv);
		//imwrite(path+"original.jpg", new_image_mat);
	//}
	if (!first_launch) {
		//read original
		ReadCoords();
		// получаем картинку фон
		Mat orig_mat = imread(path + "original.jpg");
		auto orig_img = TransformMatToImg(orig_mat);
		InsertOldPlaces(orig_img);
		InsertNewPlaces(orig_img);
	}
	// освобождаем ресурсы
	cvReleaseImage(&new_image);
	// удаляем окна
	cvDestroyAllWindows();
	return 0;
}