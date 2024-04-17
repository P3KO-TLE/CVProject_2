#include<iostream>
#include<vector>
#include<cmath>
#include<queue>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>

using namespace std;
using namespace cv;

#define GAUSKSIZE 3
#define IMGX 400
#define IMGY 300
#define MAXG 80
#define MING 40

string filepath = "D:/CV/Resources/canny2.jpg";
Size imgsize(IMGX, IMGY);
Mat src,gaus,gradx,grady,grad,dire,cmp,nms,dst;

void read() {
	src = imread(filepath,IMREAD_GRAYSCALE);
}

//2 计算梯度幅值图像,方向图像和边缘图像
void getGradDire(Mat input)
{
	cmp.create(input.size(), CV_8UC1);
	Sobel(input, gradx, CV_32F, 1, 0, 3);
	Sobel(input, grady, CV_32F, 0, 1, 3);
	magnitude(gradx, grady, grad);
	phase(gradx, grady, dire, true);
	int height = input.rows, width = input.cols;
	for (int i = 0; i < height; i++)for (int j = 0; j < width; j++) {
		float theta = dire.at<float>(i, j);
		if (theta >= 0 && theta < 22.5 || theta >= 180 - 22.5 && theta < 180 + 22.5 || theta >= 360 - 22.5 && theta < 360) {
			cmp.at<uchar>(i, j) = 1;
		}
		else if (theta >= 22.5 && theta < 67.5 || theta >= 180 + 22.5 && theta < 180 + 67.5) {
			cmp.at<uchar>(i, j) = 2;
		}
		else if (theta >= 67.5 && theta < 90 + 22.5 || theta >= 180 + 67.5 && theta < 270 + 22.5) {
			cmp.at<uchar>(i, j) = 3;
		}
		else {
			cmp.at<uchar>(i, j) = 4;
		}
	}
}

//3 非极大值抑制图像
void NonmaxSuppress(Mat input){
	nms.create(input.size(), CV_32F);
	int height = input.rows;
	int width = input.cols;
	for (int i = 1; i < height - 1; i++)
		for (int j = 1; j < width - 1; j++)
		{
			switch (cmp.at<uchar>(i, j))
			{
			case 1:
				if (input.at<float>(i, j) >= input.at<float>(i, j - 1) && input.at<float>(i, j) >= input.at<float>(i, j + 1))
					nms.at<float>(i, j) = input.at<float>(i, j);
				else
					nms.at<float>(i, j) = 0;
				break;
			case 2:
				if (input.at<float>(i, j) >= input.at<float>(i + 1, j - 1) && input.at<float>(i, j) >= input.at<float>(i - 1, j + 1))
					nms.at<float>(i, j) = input.at<float>(i, j);
				else
					nms.at<float>(i, j) = 0;
				break;
			case 3:
				if (input.at<float>(i, j) >= input.at<float>(i - 1, j) && input.at<float>(i, j) >= input.at<float>(i + 1, j))
					nms.at<float>(i, j) = input.at<float>(i, j);
				else
					nms.at<float>(i, j) = 0;
				break;
			case 4:
				if (input.at<float>(i, j) >= input.at<float>(i - 1, j - 1) && input.at<float>(i, j) >= input.at<float>(i + 1, j + 1))
					nms.at<float>(i, j) = input.at<float>(i, j);
				else
					nms.at<float>(i, j) = 0;
				break;
			default:
				break;
			}
		}
}

//双阈值(广搜实现)
queue<pair<int, int>>q;

bool J(int x, int y) {
	return (x >= 0 && x < IMGX && y >= 0 && y < IMGY);
}

void thresh(Mat input,float th_high,float th_low) {
	dst.create(input.size(), CV_32F);
	int height = input.rows;
	int width = input.cols;
	vector<vector<int>> vis(height,vector<int>(width));

	for (int i = 0; i < height -1; i++)for (int j = 0; j < width-1; j++) {
		if (input.at<float>(i, j) >= th_high)
			q.push({ i,j }), vis[i][j] = 1;
		else if (input.at<float>(i, j) >= th_low && input.at<float>(i, j) < th_high)
			vis[i][j] = 0;
		else
			vis[i][j] = -1;
	}

	while (!q.empty()) {
		int x = q.front().first, y = q.front().second;
		q.pop();
		for (int dx = -1; dx <= 1; dx++)for (int dy = -1; dy <= 1; dy++) {
			int nx = x + dx, ny = y + dy;
			if (J(nx, ny)&&!(nx==x&&ny==y)&&vis[x][y]==0) {
				vis[nx][ny] = 1;
				q.push({ nx,ny });
			}
		}
	}

	for (int i = 0; i < height; i++)for (int j = 0; j < width; j++) 
		if (vis[i][j]==1)
			dst.at<float>(i, j) = 255;
		else
			dst.at<float>(i, j) = 0;
}

void solve() {
	src.convertTo(src, CV_32F);
	//高斯滤波
	GaussianBlur(src, gaus, Size(GAUSKSIZE, GAUSKSIZE), 0, 0);
	//sobel算子计算梯度值及其方向
	getGradDire(gaus);
	//非极大值抑制
	NonmaxSuppress(grad);
	//双阈值处理边缘
	thresh(nms,MAXG,MING);
}

void output() {
	src.convertTo(src, CV_8UC1);
	imshow("原图", src);
	gaus.convertTo(gaus, CV_8UC1);
	imshow("高斯模糊化图像", gaus);
	grad.convertTo(grad, CV_8UC1);
	imshow("幅度", grad);
	nms.convertTo(nms, CV_8UC1);
	imshow("非极大值抑制", nms);
	dst.convertTo(dst, CV_8UC1);
	imshow("结果", dst);
	Mat stand;
	Canny(src, stand, 50, 150);
	imshow("标配库结果", stand);

	waitKey(0);
	destroyAllWindows();
}

int main() {
	read();
	solve();
	output();
}