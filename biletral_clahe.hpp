#ifndef BILETRAL_CLAHE_H
#define BILETRAL_CLAHE_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include<string>

using namespace std;
using namespace cv;

void biletral(Mat& img,Mat& out, int n=5,double sf=0.1,double sg=0.1){
    Mat im = img.clone();
    bilateralFilter(im,out,n,sf,sg);
}
void clahe(Mat& img, Mat& out, double clip=4.0, int n=8){
    Mat im,tmp,clahed;
    cvtColor(img,im,COLOR_BGR2HSV);
    vector<Mat> hsv;
    split(im,hsv);
    Ptr<CLAHE> cl = createCLAHE(clip,Size(n,n));
    cl->apply(hsv[2],tmp);
    tmp.copyTo(hsv[2]);
    merge(hsv,clahed);
    cvtColor(clahed,out,COLOR_HSV2BGR);
}

#endif