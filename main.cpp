#include<opencv2/opencv.hpp>
#include<iostream>
#include<detection.hpp>
#include<camera.hpp>
#include<SFML/Audio.hpp>

void test_detect()
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
}

void playSound(const std::string& soundFile) {
    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile(soundFile)) {
        std::cerr << "无法加载音频文件" << std::endl;
        return;
    }

    sf::Sound sound;
    sound.setBuffer(buffer);

    sound.play();
    while (sound.getStatus() == sf::Sound::Playing);
    sound.stop();
}

int main(int argc,char **argv)
{
    Camera camera;
    torch::DeviceType device_type = torch::kCPU;
    torch::jit::script::Module module_;
    module_ = torch::jit::load("../model/best.torchscript");
    auto detector = Detector("../model/best.torchscript", device_type);
    cv::VideoWriter writer_1("../video.avi", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30, cv::Size(1280, 720));
    float conf = 0.2;
    float iou = 0.2;

    std::vector<std::string> class_name;
    class_name.push_back("fire");

    std::string alarmSoundFile = "/home/robot/freesound.wav";

    while(camera.get_app())
    {
        cv::Mat img_src = camera.camera_stream_callback();
        writer_1 << img_src;
        /*
        auto result = detector.Run(img_src, conf, iou);

        if (!result.empty()) {
            std::cout << "Fire detected! Activating alarm..." << std::endl;
            playSound(alarmSoundFile);
            cv::imshow("Camer
            a Feed", img_src);
            cv::waitKey(1); 
        } 
        else 
        {
         */
    }
    return 0;
}