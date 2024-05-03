#include "window.hpp" // 包含窗口头文件
#include <iostream> // 标准输入输出流
#include <memory> // 智能指针
#include "libobsensor/hpp/Pipeline.hpp" // obsensor 管道头文件
#include "libobsensor/hpp/Error.hpp" // obsensor 错误头文件

class Camera
{
    public:
        Camera()
        {
            app = std::make_shared<Window>("ColorViewer", 1280, 720);
            config = std::make_shared<ob::Config>();
            auto profiles = pipe.getStreamProfileList(OB_SENSOR_COLOR);
            auto colorProfile = profiles->getVideoStreamProfile(1280, 720, OB_FORMAT_RGB, 30);
            config->enableStream(colorProfile);
            start();
        }
        void start()
        {
            pipe.start(config);
        }

        std::shared_ptr<Window> get_app()
        {
            return this->app;
        }
        
        cv::Mat camera_stream_callback()
        {
            cv::Mat rstMat;
            // Wait for up to 100ms for a frameset in blocking mode.
            auto frameSet = pipe.waitForFrames(500)->colorFrame();
            if(frameSet == nullptr) {
                return rstMat;
            }
            auto videoFrame = frameSet->as<ob::VideoFrame>();
            cv::Mat rawMat(videoFrame->height(), videoFrame->width(), CV_8UC3, videoFrame->data());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_RGB2BGR);
            app->addToRender(frameSet);
            return rstMat;
        }

    private:
        std::shared_ptr<Window> app;
        ob::Pipeline pipe;
        std::shared_ptr<ob::Config> config;
};  
