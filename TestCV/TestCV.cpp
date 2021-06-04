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

vector<Rect> old;
IplImage* new_image;
//IplImage* gray;
//IplImage* bin;
//IplImage* dst;
Mat new_image_mat;
String imageName("mew.jpg");

vector<Rect> ReadOldPieces() {
	ifstream f("data.txt", ios_base::in);
	vector<Rect> old_pieces;
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
				old_pieces.push_back(r);
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
					old_pieces.push_back(r);
				}
				i++;
			}
		}
	}

	return old_pieces;
}

void WriteCoordinates(vector<Rect> boundRect, ofstream& f) {
 	for (int i = 0; i < boundRect.size(); i++) {
		f << "BR - " << boundRect[i].br().x << ";" << boundRect[i].br().y << endl;
		f << "TL - " << boundRect[i].tl().x << ";" << boundRect[i].tl().y << endl;
		f << endl;
	}
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
	//Mat dst2 = Mat(cvSize(x, y), IPL_DEPTH_8U, 1);
	//adaptiveThreshold(gray_mat, dst2, 250, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 7, 1);
	//threshold(gray_mat, dst2, 250, 255, THRESH_BINARY);
	//cvNamedWindow("Threshold", WINDOW_NORMAL);
	//imshow("Threshold", dst2);
	

	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
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

		RNG rng(12345);
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
		cvNamedWindow("rect", WINDOW_NORMAL);
		imshow("rect", drawing);
		cvWaitKey(0);
		
		// Никогда не забывайте удалять ИОР
		cvResetImageROI(new_image);
		string filename = to_string(i);
		imwrite(filename + ".jpg", subImg);
		imwrite(filename + "compressed.jpg", subImg, { IMWRITE_JPEG_QUALITY, 50 });
	}
	
	return boundRect;
}

bool IsCoordinatesEmpty() {
	fstream file("data.txt");
	if (!file.is_open())
		cout << "Not open\n"; // если не открылся
	else if (file.peek() == EOF)
		return true;// если первый символ конец файла
	file.close();
	return false;
}

void GetImage() {
	new_image_mat = imread(imageName);// получаем картинку
	if (new_image_mat.empty()) {// Check for invalid input
		cout << "Could not open or find the image" << std::endl;
		exit(-1);
	}
	//transform
	new_image = cvCreateImage(cvSize(new_image_mat.cols, new_image_mat.rows), 8, 3);
	IplImage ipltemp = new_image_mat;
	cvCopy(&ipltemp, new_image);
	// покажем изображение
	cvNamedWindow("new", WINDOW_NORMAL);
	cvShowImage("new", new_image);
	cvWaitKey(0);
}

IplImage* TransformMatToImg(Mat mat) {
	auto src_img = cvCreateImage(cvSize(mat.cols, mat.rows), 8, 3);
	IplImage buff = mat;
	cvCopy(&buff, src_img);
	return src_img;
}

void CopyOldPlaces() {
	int j = 0;
	Mat drawing = Mat::zeros(new_image_mat.size(), CV_8UC3);;
	for (auto i : old) {
		Mat subImg = new_image_mat(i);
		RNG rng(12345);
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(drawing, i.tl(), i.br(), color, 2, 8, 0);
		cvNamedWindow("rect", WINDOW_NORMAL);
		imshow("rect", drawing);
		cvWaitKey(0);
		string filename = to_string(j);
		imwrite(filename + "_old.jpg", subImg);
		imwrite(filename + "_oldcompressed.jpg", subImg, { IMWRITE_JPEG_QUALITY, 50 });
		j++;
	}
}

void InsertOldPlaces(IplImage* orig_img)   {
	//insert_old
	int j = 0;
	for (auto i : old) {
		Mat subImg = imread(to_string(j) + "_old.jpg");
		//new_image_mat(i);
		//wana see this?
		//imshow("sub", subImg);
		//cvWaitKey(0);
		//transform
		auto src_img = TransformMatToImg(subImg);// cvCreateImage(cvSize(subImg.cols, subImg.rows), 8, 3);
		//IplImage buff = subImg;
		//cvCopy(&buff, src_img);

		//start insert
		cvSetImageROI(orig_img, i);
		// обнулим изображение
		cvZero(orig_img);
		// копируем изображение
		cvCopy(src_img, orig_img);
		cvResetImageROI(orig_img);

		cvNamedWindow("ROI", CV_WINDOW_NORMAL);
		cvShowImage("ROI", orig_img);
	    cvWaitKey(0);

		/*string filename = to_string(j);
		imwrite(filename + "_.jpg", subImg);*/
		j++;
	}
}

void InsertNewPlaces(IplImage* orig_img, vector<Rect> objects) {
	int i_ = 0;
	for (auto i : objects) {
		Mat src_mat = imread(to_string(i_) + ".jpg");
		auto src_img = TransformMatToImg(src_mat);//cvCreateImage(cvSize(src_mat.cols, src_mat.rows), 8, 3);
		//IplImage ipltemp_2 = src_mat;
		//cvCopy(&ipltemp_2, src_img);

		//cvNamedWindow("t", CV_WINDOW_AUTOSIZE);
		//cvShowImage("t", src_img);
		//cvWaitKey(0);
		cvSetImageROI(orig_img, i);
		// обнулим изображение
		cvZero(orig_img);
		//// копируем изображение
		cvCopy(src_img, orig_img);

		cvResetImageROI(orig_img);
		cvNamedWindow("Res", WINDOW_NORMAL);
		cvShowImage("Res", orig_img);
		
		cvWaitKey(0);
		i_++;
	}
	//auto out_mat = Mat(&orig_img, true);
	Mat out_mat = cvarrToMat(orig_img);

	namedWindow("out", WINDOW_NORMAL);
	imshow("out", out_mat);
	cvWaitKey(0);
	imwrite("original.jpg", out_mat);
}


int main(int argc, char* argv[])
{
	bool first_launch = IsCoordinatesEmpty();

	GetImage();

	if (first_launch) {
		imwrite("original.jpg", new_image_mat);
	}

	//find Objects on new image
	auto objects = FindObjects(new_image_mat);

	if (!first_launch) {
		//namedWindow("sub", WINDOW_NORMAL);
		//cvNamedWindow("ROI", CV_WINDOW_NORMAL);
		//read original
		old = ReadOldPieces();
		CopyOldPlaces();
	}
	if (!first_launch) {
		String original_name("original.jpg");
		
		// получаем картинку
		Mat orig_mat = imread(original_name);

		auto orig_img = TransformMatToImg(orig_mat);//cvCreateImage(cvSize(orig_mat.cols, orig_mat.rows), 8, 3);
		//IplImage buff = orig_mat;
		//cvCopy(&buff, orig_img);

		InsertOldPlaces(orig_img);
		InsertNewPlaces(orig_img, objects);
	}

	ofstream f("data.txt", ios::trunc);
	WriteCoordinates(objects, f);
	// ждём нажатия клавиши
	cvWaitKey(0);
	// освобождаем ресурсы
	cvReleaseImage(&new_image);
	//cvReleaseImage(&gray);
	//cvReleaseImage(&bin);
	//cvReleaseImage(&dst);
	// удаляем окна
	cvDestroyAllWindows();
	f.close();

	return 0;
}
