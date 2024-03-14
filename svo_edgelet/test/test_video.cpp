// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <svo/math_lib.h>
#include <svo/camera_model.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <iostream>

#include <svo/slamviewer.h>
#include <thread>

class BenchmarkNode
{
  svo::AbstractCamera *cam_;
  svo::PinholeCamera *cam_pinhole_;
  svo::FrameHandlerMono *vo_;

  SLAM_VIEWER::Viewer *viewer_;
  std::thread *viewer_thread_;

public:
  BenchmarkNode(std::string &VideoPath);
  BenchmarkNode(int imgW, int imgH, std::string &CamPath, std::string &VideoPath);
  ~BenchmarkNode();
  void runFromVideo();
  void runFromUndistortedVideo();

private:
  void InitSystem(int w, int h, std::string &cam_file_path);
  void InitSystem(int w, int h, double fov);

public:
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
}

void BenchmarkNode::InitSystem(int w, int h, double fov)
{
  double focal = w / std::tan(fov / 2.);
  std::cout << " focal = " << focal << std::endl;

  cam_ = new svo::PinholeCamera(w, h, focal, focal, w / 2., h / 2.);

  vo_ = new svo::FrameHandlerMono(cam_);
  vo_->start();

  viewer_ = new SLAM_VIEWER::Viewer(vo_);
  viewer_thread_ = new std::thread(&SLAM_VIEWER::Viewer::run, viewer_);
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
  cam_ = new svo::PinholeCamera(w, h, fx, fy, cx, cy, d0, d1, d2, d3, d4);

  // 初始化VO
  vo_ = new svo::FrameHandlerMono(cam_);
  vo_->start();

  viewer_ = new SLAM_VIEWER::Viewer(vo_);
  viewer_thread_ = new std::thread(&SLAM_VIEWER::Viewer::run, viewer_);
  viewer_thread_->detach(); // 分离子线程
}

BenchmarkNode::~BenchmarkNode()
{
  delete vo_;
  delete cam_;
  delete cam_pinhole_;

  delete viewer_;
  delete viewer_thread_;
}

// #define TXTREAD
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
    cv::cvtColor(image, gray, CV_BGR2GRAY);

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
    cv::cvtColor(image, gray, CV_BGR2GRAY);

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

int main(int argc, char **argv)
{
  std::string videoPath = "/home/my/Downloads/15.mp4";
  std::string camPath = "/home/my/Downloads/camera.yaml";
  // int imgW = 3840;
  // int imgH = 2160;
  int imgW = 1920;
  int imgH = 1080;
  printf("BenchmarkNode start.\n");
  // BenchmarkNode benchmark(imgW, imgH, camPath, videoPath);
  // benchmark.runFromVideo();
  BenchmarkNode benchmark(videoPath);
  benchmark.runFromUndistortedVideo();

  printf("BenchmarkNode finished.\n");
  return 0;
}
