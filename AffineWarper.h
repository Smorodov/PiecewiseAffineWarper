#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <windows.h>
//#include "utils/utils.h"
using namespace std;
using namespace cv;
void DrawLabelsMask(Mat& imgLabel,vector<Point2f>& points,vector<vector<size_t>>& triangles);
void WarpAffine(Mat& img,vector<Point2f>& s_0,vector<Point2f>& s_1, vector<vector<size_t>>& triangles, Mat& dstLabelsMask,Mat& dst);
void CalcCoeffs(vector<Point2f>& s_0,vector<Point2f>& s_1, vector<vector<size_t>>& triangles, Mat& Coeffs);
cv::Rect_<float> boundingRect(vector<Point2f>& pts);