#ifndef HELPER_H
#define HELPER_H

#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void clip_double(Mat& img,double min,double max){
    // assuming img is 32float C1.
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(img.at<float>(i,j)<min)
            img.at<float>(i,j) = min;
            else if(img.at<float>(i,j)>max)
            img.at<float>(i,j) = max;
        }
    }
}

void clip_int(Mat& img,int min,int max){
    // assuming img is 8Uchar C1 type.
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(img.at<uchar>(i,j)<min)
            img.at<uchar>(i,j) = min;
            else if(img.at<uchar>(i,j)>max)
            img.at<uchar>(i,j) = max;
        }
    }
}

void percentile(Mat& img,double per,int* min,int* max){
    // assuming img id 8uchar c1 type.
    int hist[256]={0};
    int i;
    for(i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            hist[img.at<uchar>(i,j)]++;
        }
    }
    int sum=0;
    double index;
    index = per*(img.rows)*(img.cols)/100;
    for(i=0;i<256;i++){
        sum+=hist[i];
        if(sum>=index)
        break;
    }
    *min = i;
    for(sum=0,i=255;i>=0;i--){
        sum+=hist[i];
        if(sum>=index)
        break;
    }
    *max = i;
}

Mat extract_lchannel(Mat img){
    // assuming img is 8uchar BGR image.
    Mat im;
    cvtColor(img,im,COLOR_BGR2Lab);
    vector<Mat> lab;
    split(im,lab);
    Mat l;
    (lab[0]).convertTo(l,CV_32FC1);
    return l;
}

Mat binomial_kernel(){
    float data[] = {1.0,4.0,6.0,4.0,1.0};
    Mat mask = Mat(1,5,CV_32F,data);
    mask = mask/16;
    mask = (mask.t())*mask;
    return mask;
}
#endif