// This file is part of HSO: Hybrid Sparse Monocular Visual Odometry
// With Online Photometric Calibration
//
// Copyright(c) 2021, Dongting Luo, Dalian University of Technology, Dalian
// Copyright(c) 2021, Robotics Group, Dalian University of Technology
//
// This program is highly based on the previous implementation
// of SVO: https://github.com/uzh-rpg/rpg_svo
// and PL-SVO: https://github.com/rubengooj/pl-svo
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <sophus/se3.h>

#include <boost/thread.hpp>

#include <hso/config.h>
#include <hso/frame_handler_mono.h>
#include <hso/frame_handler_base.h>
#include <hso/map.h>
#include <hso/frame.h>
#include <hso/feature.h>
#include <hso/point.h>
#include <hso/viewer.h>
#include <hso/depth_filter.h>
// #include "hso/PhotomatricCalibration.h"

#include "hso/camera.h"
#include "hso/ImageReader.h"

using namespace cv;
using namespace std;

const int G_MAX_RESOLUTION = 848 * 800;

namespace hso
{
    class BenchmarkNode
    {
        hso::AbstractCamera *cam_;
        hso::FrameHandlerMono *vo_;
        hso::Viewer *viewer_;
        boost::thread *viewer_thread_;

    public:
        BenchmarkNode(std::string &VideoPath);
        BenchmarkNode(int imgW, int imgH, std::string &CamPath, std::string &VideoPath);
        ~BenchmarkNode();
        void runFromVideo();
        void runFromUndistortedVideo();
        void saveResult(bool stamp_valid);

    private:
        void InitSystem(int w, int h, std::string &cam_file_path);
        void InitSystem(int w, int h, double fov);

    public:
        std::string result_name_;
        std::string CamPath_;
        std::string VideoPath_;
        cv::Mat camMatrix_;
        cv::Mat distCoeffs_;
    };

    BenchmarkNode::BenchmarkNode(int imgW, int imgH, std::string &CamPath, std::string &VideoPath)
        : CamPath_(CamPath), VideoPath_(VideoPath)
    {
        InitSystem(imgW, imgH, CamPath_);
    }

    BenchmarkNode::BenchmarkNode(std::string &VideoPath)
        : VideoPath_(VideoPath)
    {

        result_name_ = "test";
    }

    BenchmarkNode::~BenchmarkNode()
    {
        delete vo_;
        delete cam_;
        delete viewer_;
        delete viewer_thread_;
    }

    void BenchmarkNode::InitSystem(int w, int h, double fov)
    {
        double focal = w / std::tan(fov / 2.);
        std::cout << " focal = " << focal << std::endl;

        cam_ = new hso::PinholeCamera(w, h, focal, focal, w / 2., h / 2.);

        vo_ = new hso::FrameHandlerMono(cam_);
        vo_->start();

        viewer_ = new hso::Viewer(vo_);
        viewer_thread_ = new boost::thread(&hso::Viewer::run, viewer_);
        viewer_thread_->detach(); // 分离子线程
    }

    void BenchmarkNode::InitSystem(int w, int h, std::string &cam_file_path)
    {
        // 读取相机参数
        cv::FileStorage fs(cam_file_path, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            throw std::invalid_argument("Invalid camera file");
        }
        fs["camera_matrix"] >> camMatrix_;
        fs["dist_coeff"] >> distCoeffs_;
        // 相机内参
        double fx = camMatrix_.at<double>(0, 0);
        double fy = camMatrix_.at<double>(1, 1);
        double cx = camMatrix_.at<double>(0, 2);
        double cy = camMatrix_.at<double>(1, 2);
        // 畸变参数
        std::cout << distCoeffs_ << std::endl;
        // double d0 = distCoeffs_.at<double>(0, 0);
        // double d1 = distCoeffs_.at<double>(1, 0);
        // double d2 = distCoeffs_.at<double>(2, 0);
        // double d3 = distCoeffs_.at<double>(3, 0);
        // double d4 = distCoeffs_.at<double>(4, 0);
        double d0 = distCoeffs_.at<double>(0, 0);
        double d1 = distCoeffs_.at<double>(0, 1);
        double d2 = distCoeffs_.at<double>(0, 2);
        double d3 = distCoeffs_.at<double>(0, 3);
        double d4 = distCoeffs_.at<double>(0, 4);
        // 初始化相机
        cam_ = new hso::PinholeCamera(w, h, fx, fy, cx, cy, d0, d1, d2, d3, d4);

        // 初始化VO
        vo_ = new hso::FrameHandlerMono(cam_);
        vo_->start();

        viewer_ = new hso::Viewer(vo_);
        viewer_thread_ = new boost::thread(&hso::Viewer::run, viewer_);
        viewer_thread_->detach(); // 分离子线程
    }

    void BenchmarkNode::runFromVideo()
    {

        // cv::VideoCapture cap(1);  // open the default camera
        cv::VideoCapture cap(VideoPath_); // open the default camera

        if (!cap.isOpened()) // check if we succeeded
            return;

        int img_id = 0;

        for (;;)
        {

            cv::Mat image;
            cap.read(image); // get a new frame from camera

            // std::cout <<" image size " << image.size() << std::endl;

            assert(!image.empty());
            img_id++;

            if (image.cols > 1000 || image.rows > 1000)
            {
                cv::resize(image, image, image.size() / 2);
            }

            // if(img_id < 800) continue;

            // cv::imshow("origin_image", image);
            // if (cv::waitKey(1) >= 0) break;

            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            /*
            cv::Mat unimg;
            cam_pinhole_->undistortImage(gray,unimg);
            vo_->addImage(unimg, 0.01*img_id);
             */
            vo_->addImage(gray, 0.01 * img_id);

            // display tracking quality
            if (vo_->lastFrame() != NULL)
            {
                std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                          << "#Features: " << vo_->lastNumObservations() << " \n";
                //<< "Proc. Time: " << vo_->lastProcessingTime()*1000 << "ms \n";
                // std::cout<<"Frame pose: "<< vo_->lastFrame()->T_f_w_ <<std::endl;

                // put a virtual  cube in front of the camera
                double axis_len = 0.2;

                Eigen::AngleAxisd rot(0.25 * M_PI, Eigen::Vector3d(1, 0, 0).normalized());
                Eigen::Matrix3d Rwl = rot.toRotationMatrix();
                Eigen::Vector3d twl(0, -0.1, 2);

                Eigen::Vector2d o = vo_->lastFrame()->w2c(Rwl * Eigen::Vector3d(0, 0, 0) + twl);
                Eigen::Vector2d x = vo_->lastFrame()->w2c(Rwl * Eigen::Vector3d(axis_len, 0, 0) + twl);
                Eigen::Vector2d y = vo_->lastFrame()->w2c(Rwl * Eigen::Vector3d(0, -axis_len, 0) + twl);
                Eigen::Vector2d z = vo_->lastFrame()->w2c(Rwl * Eigen::Vector3d(0, 0, -axis_len) + twl);

                cv::line(image, cv::Point2f(o.x(), o.y()), cv::Point2f(x.x(), x.y()), cv::Scalar(255, 0, 0), 2);
                cv::line(image, cv::Point2f(o.x(), o.y()), cv::Point2f(y.x(), y.y()), cv::Scalar(0, 255, 0), 2);
                cv::line(image, cv::Point2f(o.x(), o.y()), cv::Point2f(z.x(), z.y()), cv::Scalar(0, 0, 255), 2);

                cv::imshow("origin_image", image);
                cv::waitKey(1);
            }
        }

        cap.release();
        return;
    }

    void BenchmarkNode::runFromUndistortedVideo()
    {

        // cv::VideoCapture cap(1);  // open the default camera
        cv::VideoCapture cap(VideoPath_); // open the default camera

        if (!cap.isOpened()) // check if we succeeded
            return;

        int img_id = 0;
        bool init_flag = true;
        for (;;)
        {

            cv::Mat image;
            cap.read(image); // get a new frame from camera

            // std::cout <<" image size " << image.size() << std::endl;

            assert(!image.empty());
            img_id++;

            if (image.cols > 1000 || image.rows > 1000)
            {
                cv::resize(image, image, image.size() / 4);
            }
            if (init_flag)
            {
                init_flag = false;
                int w = image.cols;
                int h = image.rows;
                InitSystem(w, h, 90 / 57.3);
            }
            // if(img_id < 800) continue;

            // cv::imshow("origin_image", image);
            // if (cv::waitKey(1) >= 0) break;

            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            /*
            cv::Mat unimg;
            cam_pinhole_->undistortImage(gray,unimg);
            vo_->addImage(unimg, 0.01*img_id);
             */
            vo_->addImage(gray, 0.01 * img_id);

            // display tracking quality
            if (vo_->lastFrame() != NULL)
            {
                std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                          << "#Features: " << vo_->lastNumObservations() << " \n";
                //<< "Proc. Time: " << vo_->lastProcessingTime()*1000 << "ms \n";
                // std::cout<<"Frame pose: "<< vo_->lastFrame()->T_f_w_ <<std::endl;

                // put a virtual  cube in front of the camera
                double axis_len = 0.2;

                Eigen::AngleAxisd rot(0.25 * M_PI, Eigen::Vector3d(1, 0, 0).normalized());
                Eigen::Matrix3d Rwl = rot.toRotationMatrix();
                Eigen::Vector3d twl(0, -0.1, 2);

                Eigen::Vector2d o = vo_->lastFrame()->w2c(Rwl * Eigen::Vector3d(0, 0, 0) + twl);
                Eigen::Vector2d x = vo_->lastFrame()->w2c(Rwl * Eigen::Vector3d(axis_len, 0, 0) + twl);
                Eigen::Vector2d y = vo_->lastFrame()->w2c(Rwl * Eigen::Vector3d(0, -axis_len, 0) + twl);
                Eigen::Vector2d z = vo_->lastFrame()->w2c(Rwl * Eigen::Vector3d(0, 0, -axis_len) + twl);

                cv::line(image, cv::Point2f(o.x(), o.y()), cv::Point2f(x.x(), x.y()), cv::Scalar(255, 0, 0), 2);
                cv::line(image, cv::Point2f(o.x(), o.y()), cv::Point2f(y.x(), y.y()), cv::Scalar(0, 255, 0), 2);
                cv::line(image, cv::Point2f(o.x(), o.y()), cv::Point2f(z.x(), z.y()), cv::Scalar(0, 0, 255), 2);

                cv::imshow("origin_image", image);
                cv::waitKey(1);
            }
        }

        cap.release();
        return;
    }

    void BenchmarkNode::saveResult(bool stamp_valid)
    {
        // Trajectory
        std::ofstream okt("./result/" + result_name_ + ".txt");
        for (auto it = vo_->map_.keyframes_.begin(); it != vo_->map_.keyframes_.end(); ++it)
        {
            SE3 Tinv = (*it)->T_f_w_.inverse();
            if (!stamp_valid)
                okt << (*it)->id_ << " ";
            else
                okt << (*it)->m_timestamp_s << " ";

            okt << Tinv.translation()[0] << " "
                << Tinv.translation()[1] << " "
                << Tinv.translation()[2] << " "
                << Tinv.unit_quaternion().x() << " "
                << Tinv.unit_quaternion().y() << " "
                << Tinv.unit_quaternion().z() << " "
                << Tinv.unit_quaternion().w() << endl;
        }
        okt.close();
    }

} // namespace hso

int main(int argc, char **argv)
{
    std::string videoPath = "/home/my/Downloads/17.mp4";
    std::string camPath = "/home/my/Downloads/camera.yaml";
    // int imgW = 3840;
    // int imgH = 2160;
    int imgW = 1920;
    int imgH = 1080;
    printf("BenchmarkNode start.\n");
    // BenchmarkNode benchmark(imgW, imgH, camPath, videoPath);
    // benchmark.runFromVideo();
    hso::BenchmarkNode benchmark(videoPath);
    benchmark.runFromUndistortedVideo();

    printf("BenchmarkNode finished.\n");
    return 0;
}
