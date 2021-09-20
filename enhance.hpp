#ifndef ENHANCE_H
#define ENHANCE_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include"white_balance.hpp"
#include"biletral_clahe.hpp"
#include"pyramids.hpp"
#include"weights.hpp"

using namespace std;
using namespace cv;

Mat laplace_blending(Mat inp1,Mat inp2){
    vector<Mat> wgh;
    int level=6;
    wgh = normalized_weights(inp1,inp2);
    vector<Mat> gauss1 = gauss_pyramid(wgh[0],level,true);
    vector<Mat> gauss2 = gauss_pyramid(wgh[1],level,true);
    vector<vector<Mat>> lap1 = laplace_pyramid_c3(inp1,level);
    vector<vector<Mat>> lap2 = laplace_pyramid_c3(inp2,level);
    vector<vector<Mat>> tmp1,tmp2;
    vector<Mat> tmp;
    for(int i=0;i<3;i++){
        tmp = multiply(gauss1,lap1[i]);
        tmp1.push_back(tmp);
        tmp = multiply(gauss2,lap2[i]);
        tmp2.push_back(tmp);
    }
    vector<Mat> channels;
    Mat channel;
    for(int i=0;i<3;i++){
        tmp = add(tmp1[i],tmp2[i]);
        channel = reconstruct_image_c1(tmp);
        channels.push_back(channel);
    }
    Mat out;
    merge(channels,out);
    Mat result;
    out.convertTo(result,CV_8UC3);
    // white_balance(result,0,out);
    return result;
    // return out;
}

Mat enhance(Mat img){
    Mat white_balanced = img.clone();
    // white_balance_algo1(img,5,white_balanced);
    algo_2(img,white_balanced);
    Mat filtered;
    biletral(white_balanced,filtered);
    Mat input_2;
    clahe(filtered,input_2);
    Mat result;
    result = laplace_blending(white_balanced,input_2);
    return result;
}
#endif