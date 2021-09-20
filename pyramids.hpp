#ifndef PYRAMIDS_H
#define PYRAMIDS_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include"helper_funcs.hpp"

using namespace std;
using namespace cv;

vector<Mat> gauss_pyramid(Mat img,int level=6,bool apply_mask=false){
    // assuiming img is single channel.
    Mat im;
    img.convertTo(im,CV_32FC1);
    Mat mask = binomial_kernel();
    if(apply_mask){
        Mat tmp;
        filter2D(im,tmp,-1,mask,Point(-1,-1),0.0,BORDER_DEFAULT);
        tmp.copyTo(im);
    }
    vector<Mat> pyr;
    pyr.push_back(im);
    for(int i=1;i<level;i++){
        Mat tmp,tmp1;
        pyrDown(pyr[i-1],tmp,Size(),BORDER_DEFAULT);
        tmp.convertTo(tmp1,CV_32FC1);
        pyr.push_back(tmp1);
    }
    return pyr;
}

vector<Mat> laplace_pyramid_c1(Mat c,int level=6){
    // assuiming c is single channel.
    vector<Mat> gauss = gauss_pyramid(c,level);
    vector<Mat> pyr;
    int h,w;
    for(int i=0;i<level-1;i++){
        Mat tmp1,tmp2;
        pyrUp(gauss[i+1],tmp1,Size(),BORDER_DEFAULT);
        resize(tmp1,tmp2,(gauss[i]).size());
        tmp2 = gauss[i]-tmp2;
        pyr.push_back(tmp2);
    }
    pyr.push_back(gauss[level-1]);
    return pyr;
}

vector<vector<Mat>> laplace_pyramid_c3(Mat img, int level=6){
    vector<Mat> channels;
    split(img,channels);
    vector<vector<Mat>> pyr3;
    vector<Mat> pyr1;
    for(int i=0;i<3;i++){
        pyr1 = laplace_pyramid_c1(channels[i],level);
        pyr3.push_back(pyr1);
    }
    return pyr3;
}

vector<Mat> multiply(vector<Mat> pyr1, vector<Mat> pyr2){
    int level = pyr1.size();
    vector<Mat> pyr;
    Mat tmp,tmp1,tmp2;
    for(int i=0;i<level;i++){
        (pyr1[i]).convertTo(tmp1,CV_32FC1);
        (pyr2[i]).convertTo(tmp2,CV_32FC1);
        // cout<<i<<"/"<<level<<endl;
        tmp = tmp1.mul(tmp2);
        pyr.push_back(tmp);
    }
    return pyr;
}

vector<Mat> add(vector<Mat> pyr1, vector<Mat> pyr2){
    int level = pyr1.size();
    vector<Mat> pyr;
    Mat tmp;
    for(int i=0;i<level;i++){
        tmp = pyr1[i] + pyr2[i];
        pyr.push_back(tmp);
    }
    return pyr;
}

Mat reconstruct_image_c1(vector<Mat> pyr){
    int level = pyr.size();
    for(int i=level-1;i>0;i--){
        Mat tmp1,tmp2;
        pyrUp(pyr[i],tmp1,Size(),BORDER_DEFAULT);
        resize(tmp1,tmp2,(pyr[i-1]).size());
        pyr[i-1] = pyr[i-1] + tmp2;
    }
    return pyr[0];
}

#endif