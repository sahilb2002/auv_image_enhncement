#ifndef WEIGHTS_H
#define WEIGHTS_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include"helper_funcs.hpp"
#include<cmath>

using namespace std;
using namespace cv;

Mat local_contrast_weights(Mat img){
    // assuming img is 8uchar BGR image.
    Mat l = extract_lchannel(img);
    Mat mask = binomial_kernel();
    Mat tmp,iwhc;
    filter2D(l,tmp,-1,mask,Point(-1,-1),0.0,BORDER_DEFAULT);
    tmp.convertTo(iwhc,CV_32FC1);
    clip_double(iwhc,0,M_PI/2.75);
    Mat w = cv::abs(l-iwhc);
    return w;
}

Mat exposedness_weights(Mat img,double sigma=0.25){
    // assuming img is 8uchar BGR image.
    Mat l = extract_lchannel(img);
    Mat w;
    Mat _half = 0.5*(Mat::ones(l.rows,l.cols,CV_32FC1));
    Mat tmp;
    pow(l-_half,2.0,tmp);
    cv::exp(-1*(tmp/(2*sigma*sigma)),w);
    return w;
}

Mat saliency_weights(Mat img){
    // assuming img is 8uchar BGR image.
    Mat lab;
    cvtColor(img,lab,COLOR_BGR2Lab);
    Mat tmp1;
    lab.convertTo(tmp1,CV_32FC3);
    Scalar mean;
    mean = cv::mean(tmp1);
    vector<Mat> mean_im;
    for(int i=0;i<3;i++){
        mean_im.push_back((Mat::ones(tmp1.rows,tmp1.cols,CV_32FC1))*mean[i]);
    }
    Mat mean_img;
    merge(mean_im,mean_img);
    Mat tmp2;
    cv::pow(tmp1-mean_img,2,tmp2);
    Mat w;
    transform(tmp2,w,Matx13f(1,1,1));
    Mat w2;
    sqrt(w,w2);
    w2 = w2/3;
    return w2;
}

Mat laplacian_contrast(Mat img){
    // assuming img is 8uchar BGR image.
    Mat l;
    l = extract_lchannel(img);
    Mat tmp;
    Laplacian(l,tmp,-1);
    Mat tmp2,w;
    tmp2 = cv::abs(tmp);
    cv::normalize(tmp2,w);
    return w;
}
Mat weights(Mat inp,double sigma = 0.25){
    // assuming inp is 8uchar BGR image.
    Mat lc1 = local_contrast_weights(inp);
    Mat ew1 = exposedness_weights(inp,sigma);
    Mat sw = saliency_weights(inp);
    Mat lapc = laplacian_contrast(inp);
    Mat w;
    w = lc1+ew1+sw+lapc;
    return w;
}
vector<Mat> normalized_weights(Mat inp1,Mat inp2,double sigma=0.25){
    // assuming inp1, inp2 are 8uchar BGR image.
    Mat w1 = weights(inp1,sigma);
    Mat w2 = weights(inp2,sigma);
    Mat sum = w1+w2;
    Mat w1n,w2n;
    divide(w1,sum,w1n,1,-1);
    divide(w2,sum,w2n,1,-1);
    vector<Mat> out;
    out.push_back(w1n);
    out.push_back(w2n);
    return out;
}
#endif