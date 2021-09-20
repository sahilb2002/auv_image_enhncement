#include<iostream>
#include<opencv2/opencv.hpp>
#include"enhance.hpp"
// #include"white_balance.hpp"
#include<string>
#include<chrono>

using namespace std;
using namespace cv;

int main(){
    string image_name;
    cout<<"Enter image name from dataset:"<<endl;
    cin>>image_name;
    image_name = image_name + ".png";
    Mat org1 = imread("../dataset/raw-890/"+image_name);
    Mat ref1 = imread("../dataset/reference-890/"+image_name);
    Mat org,ref;
    if(org1.rows>500){
        resize(org1,org,Size(500,500),0,0);
        resize(ref1,ref,Size(500,500),0,0);
    }
    else{
        org = org1;
        ref = ref1;
    }
    auto start = chrono::high_resolution_clock::now();
    Mat enhanced = enhance(org);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop-start);
    cout<<"time taken = "<<duration.count()<<"ms"<<endl;
    imshow("reference",ref);
    imshow("original",org);
    imshow("enhanced",enhanced);
    waitKey();
}