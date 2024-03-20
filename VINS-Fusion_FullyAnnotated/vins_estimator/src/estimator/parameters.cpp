/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;        // 特征跟踪中的最大特征数
int MIN_DIST;       // 两个特征之间的最小距离
double F_THRESHOLD; // Ransac阈值(像素)
int SHOW_TRACK;
int FLOW_BACK;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    // 检查配置文件是否存在
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL)
    {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ); // cv::FileStorage对象代表着一个XML或者YAML格式的数据文件
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC; // 读取左相机话题名称
    fsSettings["image1_topic"] >> IMAGE1_TOPIC; // 读取右相机话题名称
    MAX_CNT = fsSettings["max_cnt"];            // 读取最大特征点数
    MIN_DIST = fsSettings["min_dist"];          // 读取特征点最小距离
    F_THRESHOLD = fsSettings["F_threshold"];    // 读取Ransac阈值(像素)
    SHOW_TRACK = fsSettings["show_track"];      // 读取是否发布跟踪图像到主题
    FLOW_BACK = fsSettings["flow_back"];        // 读取进行前向和后向光流，提高特征跟踪精度

    MULTIPLE_THREAD = fsSettings["multiple_thread"]; // 读取是否启用多线程

    USE_IMU = fsSettings["imu"]; // 读取是否使用IMU
    printf("USE_IMU: %d\n", USE_IMU);
    if (USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC; // 读取IMU话题
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];  // 读取加速度计测量噪声标准差
        ACC_W = fsSettings["acc_w"];  // 读取加速度计偏差随机工作噪声标准差
        GYR_N = fsSettings["gyr_n"];  // 读取陀螺仪测量噪声标准差。
        GYR_W = fsSettings["gyr_w"];  // 读取陀螺仪偏差随机工作噪声标准差。
        G.z() = fsSettings["g_norm"]; // 读取重力加速度
    }

    SOLVER_TIME = fsSettings["max_solver_time"];       // 读取求解器最大迭代时间
    NUM_ITERATIONS = fsSettings["max_num_iterations"]; // 读取求解器最大迭代=数
    MIN_PARALLAX = fsSettings["keyframe_parallax"];    // 读取关键帧选择阈值
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER; // 读取结果输出路径
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"]; // 读取是否使用标定的IMU与相机之间的外部参数。
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else
    {
        if (ESTIMATE_EXTRINSIC == 1) // 对外在参数有一个初步的猜测。我们将围绕您的初始猜测进行优化。
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0) // 有一个准确的外在参数。我们将信任下面的imu^R_cam, imu^T_cam，不要更改它。
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T; // 读取IMU和左相机之间的平移矩阵
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    NUM_OF_CAM = fsSettings["num_of_cam"]; //读取相机个数
    printf("camera number %d\n", NUM_OF_CAM);

    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;//读取左相机参数保存文件名称
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if (NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        // printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);

        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T; // 读取IMU和右相机之间的平移矩阵
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"]; //时间偏移的初始值。单位:s.读图像时钟+ td =实图像时钟(IMU时钟)
    ESTIMATE_TD = fsSettings["estimate_td"];//在线估计相机和imu之间的时间偏移
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];//读取图像高
    COL = fsSettings["image_width"]; //读取图像宽
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if (!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
