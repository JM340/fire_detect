#include<opencv2/opencv.hpp>
#include<iostream>
#include<detection.hpp>

int main(int argc,char **argv)
{
    cv::Mat img_src;
    img_src = cv::imread("../test/60.jpg");
    torch::DeviceType device_type = torch::kCPU;
    torch::jit::script::Module module_;
    module_ = torch::jit::load("../model/best.torchscript");
    auto detector = Detector("../model/best.torchscript",device_type);

    float conf = 0.2;
    float iou = 0.2;

    std::vector<std::string> class_name;
    auto result = detector.Run(img_src,conf,iou);
    class_name.push_back("fire");
    detector.DrawPred(img_src,result,class_name,true);
    cv::imshow("dst",img_src);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}