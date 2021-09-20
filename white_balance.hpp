#ifndef WHITE_BALANCE_H
#define WHITE_BALANCE_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include"helper_funcs.hpp"
using namespace std;
using namespace cv;
// to compile files involving opencv use the command: "opencv file_name.cpp"


void white_balance_algo1(Mat& image,double per,Mat& im){
    // cout<< typeid(img.at<uint>(0,0)).name();
    Mat ims = image.clone();
    Mat img;
    int chs = 3;
    Mat c = Mat::zeros(image.rows,image.cols,CV_8UC1);
    Mat tmp = Mat::zeros(image.rows,image.cols,CV_8UC1);
    // Mat channels[3] = {Mat::zeros(image.rows,image.cols,CV_8UC1),Mat::zeros(image.rows,image.cols,CV_8UC1),Mat::zeros(image.rows,image.cols,CV_8UC1)};
    vector<Mat> channels;
    vector<Mat> outputs;
    split(ims,channels);
    for(int i=0;i<chs;i++){
        int mi,ma;
        percentile(channels[i],per,&mi,&ma);
        clip_int(channels[i],(double)(mi),(double)(ma));
        channels[i].convertTo(c,CV_32FC1);
        c = (c-mi*(Mat::ones(c.rows,c.cols,CV_32FC1)))/(ma-mi);
        c = c*255.0;
        c.convertTo(tmp,CV_8UC1);
        outputs.push_back(tmp.clone());
    }
    merge(outputs,im);
}

void gray_world_algo(Mat& img,Mat& out,double lambda=0.2){
    // img is 8uc3 image
    Mat ims;
    img.convertTo(ims,CV_32FC3);
    ims = ims/255.0;
    Scalar mn;
    double ilum;
    Mat tmp,tmp2;
    vector<Mat> chs,out_chs;
    split(ims,chs);
    for(int i=0;i<3;i++){
        mn = mean(chs[i]);
        ilum = mn[0];
        ilum = 0.5+lambda/ilum;
        tmp2 = ilum*chs[i];
        tmp = tmp2.clone();
        clip_double(tmp,0.0,1.0);
        out_chs.push_back(tmp);
    }
    Mat tmp3;
    merge(out_chs,tmp3);
    tmp3 = tmp3*255;
    tmp3.convertTo(out,CV_8UC3);
}

void algo_2(Mat& img,Mat& im,double per=5,double lambda=0.2){
    // img is 8uc3 BGR image
    vector<Mat> chs;
    Mat ims;
    img.convertTo(ims,CV_32FC3);
    ims = ims/255.0;
    split(ims,chs);
    Mat one;
    one = Mat::ones(chs[1].rows,chs[1].cols,CV_32FC1);
    Mat rrc,rbc;
    rrc = (chs[1]).mul(one-chs[2]);
    rbc = (chs[1]).mul(one-chs[0]);
    vector<Mat> tmp_chs;
    Mat tmp,tm;
    Scalar mn;
    double men;

    mn = mean(chs[1]-chs[0]);
    men = mn[0];
    tmp = chs[0] + rbc*(men);
    tmp_chs.push_back(tmp);

    tmp_chs.push_back(chs[1]);

    mn = mean(chs[1]-chs[2]);
    men = mn[0];
    tm = chs[2] + rrc*(men);
    tmp_chs.push_back(tm);

    Mat tmp2;
    merge(tmp_chs,tmp2);
    Mat tmp3,tmp4,out;
    out = tmp2*255;
    out.convertTo(tmp3,CV_8UC3);
    white_balance_algo1(tmp3,per,tmp4);
    gray_world_algo(tmp4,im,lambda);
}

#endif