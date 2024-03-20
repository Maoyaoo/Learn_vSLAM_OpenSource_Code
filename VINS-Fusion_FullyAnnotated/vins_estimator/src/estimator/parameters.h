/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const double FOCAL_LENGTH = 460.0; // 焦距
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
// #define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX; // 关键帧选择阈值(像素)
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W; // 加速度计测量噪声标准差。加速度计偏差随机工作噪声标准差。
extern double GYR_N, GYR_W; // 陀螺仪测量噪声标准差。陀螺仪偏差随机工作噪声标准差。

extern std::vector<Eigen::Matrix3d> RIC; // IMU相机之间的旋转矩阵
extern std::vector<Eigen::Vector3d> TIC; // IMU相机之间的平移矩阵
extern Eigen::Vector3d G;                // G.z元素代表重力的大小

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME; // 求解器最大迭代时间(ms)，保证实时性
extern int NUM_ITERATIONS; // 求解器最大迭代数，保证实时性
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;         // 是否使用IMU
extern int MULTIPLE_THREAD; // 多线程
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;        // 特征跟踪，最大特征点跟踪数量
extern int MIN_DIST;       // 特征点之间最小距离
extern double F_THRESHOLD; // Ransac阈值(像素)
extern int SHOW_TRACK;     // 是否发布跟踪图像到主题
extern int FLOW_BACK;      // 是否进行前向和后向光流，提高特征跟踪精度

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
