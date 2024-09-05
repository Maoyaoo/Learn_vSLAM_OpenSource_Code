/*-------------------------------------------------------------
Copyright 2019 Wenxin Liu, Kartik Mohta, Giuseppe Loianno

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------*/

#include "sdd_vio/pinhole_camera_stereo.h"
#include <cmath>
#include "sdd_vio/utils/math_utils.h"
#include "sdd_vio/utils/calib_utils.h"
#include <ros/console.h>

#if (CV_MAJOR_VERSION == 3)
#include <opencv2/calib3d.hpp>
#endif


namespace sdd_vio {

PinholeCameraStereo::PinholeCameraStereo(const ros::NodeHandle &nh) :
    rectify_maps_initialized_(false)
{
    std::vector<double> intrinsics_leftcam;
    if( !nh.getParam("cam0/intrinsics", intrinsics_leftcam) )
        ROS_ERROR("Failed to get left camera intrisics from server.");
    std::vector<double> intrinsics_rightcam;
    if( !nh.getParam("cam1/intrinsics", intrinsics_rightcam) )
        ROS_ERROR("Failed to get right camera intrisics from server.");
    std::vector<double> distcoeff_leftcam;
    if( !nh.getParam("cam0/distortion_coeffs", distcoeff_leftcam) )
        ROS_ERROR("Failed to get left camera distortion coeffs from server.");
    std::vector<double> distcoeff_rightcam;
    if( !nh.getParam("cam1/distortion_coeffs", distcoeff_rightcam) )
        ROS_ERROR("Failed to get right camera distortion coeffs from server.");
    std::vector<double> resolution;
    if( !nh.getParam("cam0/resolution", resolution) )
        ROS_ERROR("Failed to get resolution from server.");


    cv::Mat K_l = (cv::Mat_<double>(3,3) << intrinsics_leftcam[0], 0.0, intrinsics_leftcam[2],
                                            0.0, intrinsics_leftcam[1], intrinsics_leftcam[3],
                                            0.0, 0.0, 1.0);
    cv::Mat K_r = (cv::Mat_<double>(3,3)<< intrinsics_rightcam[0], 0.0, intrinsics_rightcam[2],
                                           0.0, intrinsics_rightcam[1], intrinsics_rightcam[3],
                                           0.0, 0.0, 1.0);
    cv::Mat D_l = (cv::Mat_<double>(1, 4) << distcoeff_leftcam[0], distcoeff_leftcam[1], distcoeff_leftcam[2], distcoeff_leftcam[3]);
    cv::Mat D_r = (cv::Mat_<double>(1, 4) << distcoeff_rightcam[0], distcoeff_rightcam[1], distcoeff_rightcam[2], distcoeff_rightcam[3]);

    // get transformation from left to right camera
    cv::Mat T_rl = utils::getTransformCV(nh, "cam1/T_cn_cnm1");
    cv::Mat R = T_rl.colRange(0,3).rowRange(0,3);
    cv::Mat t = T_rl.col(3).rowRange(0,3);

    cv::Mat R_rect[2];
    cv::Matx34d P_[2];
    cv::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(resolution[0], resolution[1]), R, t,
                            R_rect[0], R_rect[1], P_[0], P_[1], Q_, cv::CALIB_ZERO_DISPARITY, 0);

    std::cout<<"rectify R_0: \n"<<R_rect[0]<<"\n";
    std::cout<<"rectify P_0: \n"<<P_[0]<<"\n";
    std::cout<<"rectify R_1: \n"<<R_rect[1]<<"\n";
    std::cout<<"rectify P_1: \n"<<P_[1]<<"\n";
    std::cout<<"Q: \n"<<Q_<<"\n";

    cv::initUndistortRectifyMap(K_l, D_l, R_rect[0], P_[0], cv::Size(resolution[0], resolution[1]), CV_32FC1, rectify_map_[0][0], rectify_map_[0][1]);
    cv::initUndistortRectifyMap(K_r, D_r, R_rect[1], P_[1], cv::Size(resolution[0], resolution[1]), CV_32FC1, rectify_map_[1][0], rectify_map_[1][1]);
    rectify_maps_initialized_ = true;//纠正map初始化标志位


    cx_rec_ = -Q_.at<double>(0,3);
    cy_rec_ = -Q_.at<double>(1,3);
    f_rec_ = Q_.at<double>(2,3);
    base_ = 1/Q_.at<double>(3,2);
    K_rec_ << f_rec_,0,cx_rec_,0,f_rec_,cy_rec_,0,0,1;
    frame_width_ = resolution[0];
    frame_height_ = resolution[1];

    ROS_INFO_STREAM("cx_rec_: " << cx_rec_);
    ROS_INFO_STREAM("cy_rec_: " << cy_rec_);
    ROS_INFO_STREAM("f_rec_: " << f_rec_);
    ROS_INFO_STREAM("base_: " << base_);
    ROS_INFO_STREAM("frame_width_: " << frame_width_);
    ROS_INFO_STREAM("frame_height_: " << frame_height_);

    matcher_sgbm_400_ = cv::StereoSGBM::create(0,    //int minDisparity
                                  32,   //int numDisparities
                                  9,    //int SADWindowSize
                                  600,  //int P1 = 0
                                  2400, //int P2 = 0
                                  20,   //int disp12MaxDiff = 0        0ms
                                  10,   //int preFilterCap = 0        -0.1ms
                                  15,    //int uniquenessRatio = 0     0.1ms
                                  200,  //int speckleWindowSize = 0     0.1ms
                                  1,   //int speckleRange = 0
                                  false);//bool fullDP = false

    matcher_sgbm_200_ = cv::StereoSGBM::create(0,    //int minDisparity
                                  16,   //int numDisparities
                                  9,    //int SADWindowSize
                                  600,  //int P1 = 0
                                  2400, //int P2 = 0
                                  20,   //int disp12MaxDiff = 0        0ms
                                  10,   //int preFilterCap = 0        -0.1ms
                                  15,    //int uniquenessRatio = 0     0.1ms
                                  200,  //int speckleWindowSize = 0     0.1ms
                                  1,   //int speckleRange = 0
                                  false);//bool fullDP = false

    //块匹配器自定义适合不同的分辨率范围-采取最短的维度
    /* block matcher custom fit for different resolution ranges - take shortest dimension */
    matcher_bm_400_ = cv::StereoBM::create(64,15);
    matcher_bm_200_ = cv::StereoBM::create(32,9);
    matcher_bm_100_ = cv::StereoBM::create(16,9);
    matcher_bm_50_ = cv::StereoBM::create(16,5);

}

//畸变矫正
void PinholeCameraStereo::undistortImage(const cv::Mat& raw0, cv::Mat& rectified0, const cv::Mat& raw1, cv::Mat& rectified1)
{
  	cv::remap(raw0, rectified0, rectify_map_[0][0], rectify_map_[0][1], cv::INTER_LINEAR);
  	cv::remap(raw1, rectified1, rectify_map_[1][0], rectify_map_[1][1], cv::INTER_LINEAR);
}

//获取视差
void PinholeCameraStereo::getDisparityMap(const cv::Mat& img0, const cv::Mat& img1, cv::Mat& disp)
{
    int dim = img0.cols;
    if (dim >= 400)
      matcher_bm_400_->compute(img0, img1, disp);
    else if (dim >= 200)
      matcher_bm_200_->compute(img0, img1, disp);
    else if (dim >= 100)
      matcher_bm_100_->compute(img0, img1, disp);
    else
      matcher_bm_50_->compute(img0, img1, disp);
}

//将像素坐标三角化为 3D 点坐标
/*feat_pixels：一个包含特征点像素坐标的向量，其中每个元素是一个 Eigen::Vector2i 类型的对象，表示特征点在图像中的 (x, y) 坐标。
disp_vec：一个浮点数向量，包含与特征点对应的视差值。
feat_3D：用于存储计算得到的 3D 点的向量，其中每个元素是一个 Eigen::Vector3f 类型的对象，表示 3D 空间中的 (X, Y, Z) 坐标。
npts：要处理的特征点数量。
ilays：用于索引摄像机内参数组的层级索引。
*/
/* given feature pixel coordinates and disparity values, triangulate 3D points */
void PinholeCameraStereo::get3DPoints(const vector_aligned<Eigen::Vector2i>& feat_pixels,
    const std::vector<float>& disp_vec, vector_aligned<Eigen::Vector3f>& feat_3D, int npts, int ilays)
{
    feat_3D.resize(npts);
    for (int i=0; i<npts; ++i)
    {
        double W = disp_vec[i]/base_;//视差/基线=逆深度
        double X = (feat_pixels[i](0)-cx_rec_pyr_[ilays])/W;
        double Y = (feat_pixels[i](1)-cy_rec_pyr_[ilays])/W;
        double Z = f_rec_pyr_[ilays]/W;//z=fb/d
        Eigen::Vector3f p_3D(X,Y,Z);
        feat_3D[i] = p_3D;
    }
}

//将给定的摄像机坐标系中的 3D 点重新投影到图像平面上，得到对应的像素坐标
/* given 3D points in camera frame (left), reproject to get image pixel coordinates */
// not Vector2i anymore but 2f, because it might be used in linear interpolation
void PinholeCameraStereo::get2DPixels(const vector_aligned<Eigen::Vector3f>& feat_3D,
    vector_aligned<Eigen::Vector2f>& feat_pixels, int npts, const int ilays)
{
    feat_pixels.resize(npts);
    for (int i=0; i<npts; ++i)
    {
        Eigen::Vector3f u = K_rec_pyr_[ilays] * feat_3D[i];//投影到图像平面
        feat_pixels[i] = sdd_vio::project2d(u);//归一化处理
    }
}

/* configure camera intrinsics for specified pyramid level */
// Notice: scale is the multiplier from current level
void PinholeCameraStereo::configPyramid()
{
    float scale;//缩放因子
    for (int i=0; i<nlays_; ++i) {
        scale = pow(0.5,(base_layer_-1+i));//每层的尺寸相对于初始层缩小0.5的幂次方。

        cx_rec_pyr_[i] = scale * cx_rec_;//每一层的图像主点坐标（通常是光学中心）。
        cy_rec_pyr_[i] = scale * cy_rec_;
        f_rec_pyr_[i] = scale * f_rec_;

        Eigen::Matrix3f scaleMat;
        scaleMat << scale,0,0,0,scale,0,0,0,1;
        K_rec_pyr_[i] = scaleMat * K_rec_;
    }
}

//初始化金字塔
void PinholeCameraStereo::initPyramid(int nlays, int base_layer)
{
    cx_rec_pyr_.resize(nlays);
    cy_rec_pyr_.resize(nlays);
    f_rec_pyr_.resize(nlays);
    K_rec_pyr_.resize(nlays);
    nlays_ = nlays;
    base_layer_ = base_layer;
}


}  // end namespace sdd_vio
