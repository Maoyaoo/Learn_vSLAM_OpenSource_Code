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

#ifndef VO_STEREO_H_
#define VO_STEREO_H_

#include "sdd_vio/pinhole_camera_stereo.h"
#include "sdd_vio/utils/timer.hpp"
#include "sdd_vio/grid.h"
#include <Eigen/Dense>
#include <boost/thread.hpp>
#include <Eigen/StdVector>

#include "opencv2/core/version.hpp"
#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace sdd_vio
{

    class VoStereo
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VoStereo(sdd_vio::PinholeCameraStereo *cam, const std::string &ns, const ros::NodeHandle &nh);
        ~VoStereo();

        enum Stage
        {
            STAGE_PAUSED,       // example featureless, turn to IMU
            STAGE_NEXT_FRAME,   // intermediate frames that are not keyframes不是关键帧的中间帧
            STAGE_NEW_KEYFRAME, // new keyframe
        };

        void reset();
        void start()
        {
            set_start_ = true;
            set_reset_ = false;
            paused_ = false;
        }
        Stage stage() const { return stage_; }

        /* visualization flags */
        // 可视化标志位
        bool visCamUpdate() const { return vis_cam_avail_; }
        bool visPtsUpdate() const { return vis_pts_avail_; }

        /* imu and gravity */
        // imu 和重力
        Eigen::Vector3f getImuBiasGyro() const { return b_w_; }
        Eigen::Vector3f getImuBiasAccel() const { return b_a_; }
        Eigen::Vector3f getGravity() const { return g_; }
        void setGravity(const Eigen::Vector3f &g) { g_ = g; }
        void setImuBiasGyro(const Eigen::Vector3f &w_b);
        void setImuBiasAccel(const Eigen::Vector3f &a_b);
        void importImuMeas(const Eigen::Matrix3f &R_meas, const Eigen::Vector3f &v_meas, const Eigen::Vector3f &p_meas, const float t_meas, const Eigen::Matrix<float, 9, 9> &Cov_meas);

        /* outputs */
        vector_aligned<Eigen::Vector3f> getVisPoints();
        Eigen::Isometry3f getVisPose() const { return T_wb_; }
        Eigen::Vector3f getVisVel() const { return v_; }

        void setInitPose(Eigen::Isometry3f const &T)
        {
            T_wb_ = T;
            T_ = T_bc_inv_ * T_wb_ * T_bc_;
            T_kf_ = T_;
            T_curr_.setIdentity();
        }

        // 主要处理函数
        /* main function for processing */
        void addImagePair(const cv::Mat &img0, const cv::Mat &img1, double timestamp, double seq);

        /* for visualization and debug purposes */
        // 用于可视化和调试的变量
        cv::Mat visual1;
        cv::Mat visual2;
        cv::Mat vis_kf, vis_curr;

    private:
        sdd_vio::PinholeCameraStereo *cam_;
        std::vector<sdd_vio::Grid *> gd_;

        /* state machine */
        Stage stage_;    //!< Current stage of the algorithm.当前算法状态
        bool set_reset_; //!< Flag that the user can set. Will reset the system before the next iteration.在下次迭代前重置系统
        bool set_start_; //!< Flag the user can set to start the system when the next image is received.
        bool paused_;

        /* visualization flags */ // 可视化标志位
        bool vis_cam_avail_;      // flag new visualization should be called
        bool vis_pts_avail_;
        bool use_opt_vis_; // visualize optimization process with opencv imshow
        bool use_opt_vis_gb_;
        bool pub_image_debug_;
        int visual_pts_layer_;   // the layer of points to visualize
        int visual_image_layer_; // visual debug image layer

        /* enabling thread */
        bool use_thread_;

        /* selecting code structure */
        bool use_ica_; // use inverse compositional approach 使用逆组合方法

        /* options for optimization (not Ceres) */
        bool use_lma_;
        bool use_huber_;

        /*choosing adaptive weights for the Gauss-Newton optimization */
        // 为高斯-牛顿优化选择自适应权值
        bool use_weights_;
        float c_;

        /* for pyramid */
        int full_layer_; // number of pyramid layers to create for tracking
        int base_layer_; // the largest layer to base tracking on 在最大的图层上进行跟踪
        int nlays_;      // number of layers used for tracking. tracking base layer up.用于跟踪的层数。向上跟踪基础层

        /* tunable keyframe statistics */
        int switch_count_;                  // when reached, switch to new keyframe
        float switch_perc_;                 // when reached, switch to new keyframe
        float switch_depth_;                // percentage of movement wrt average depth  移动WRT平均深度百分比
        float disp_range_;                  // used to chop distance in point selection based on disparity 用于基于视差的点选择中的距离切割
        std::vector<float> disp_range_pyr_; // the disparity range used for this pyramid 用于这个金字塔的视差范围
        bool layer_chop_;                   // indicator of whether necesssary to cut down the number of layers for this keyframe, due to表示是否需要减少此关键帧的层数，因为
                                            // unexpected reasons such as not enough number of points or poor disparity map意想不到的原因，如没有足够的点或差的差距地图
        int nlays_chopped_;                 // only used when layer_chop is true仅在图层剪切为真时使用
        float switch_angle_;                // rotation threshold to be reached for switching切换需要达到的旋转阈值

        /* Gauss-Newton / Levenberg-Marquardt optimization */
        int max_num_iter_; // maximum number of iterations for Gauss-Newton process高斯-牛顿过程的最大迭代次数
        bool verbose_;     // print iteration details打印迭代细节
        float lambda_;
        float up_factor_;
        float down_factor_;
        float error_scale_factor_;
        float target_derr_;

        /* feature extraction parameters */ // 特征提取参数
        int adapt_size_;                    // adaptive threshold filter size 自适应阈值滤波器大小
        int adapt_thresh_;                  // adaptive threshold cutoff value 自适应阈值截止值
        int feat_margin_;                   // margin for feature selection 特征选择边缘

        /* grid param */
        int gd_size_;
        bool use_gd_; // use grid or not

        /* camera info */
        int frameSize_x_, frameSize_y_;
        float focal_;
        std::vector<int> frameSize_x_pyr_, frameSize_y_pyr_;
        std::vector<float> focal_pyr_;

        /* keyframe statistics */
        int frame_count_;               // counting the number of frames processed since last keyframe 计算自最后一个关键帧以来处理的帧数
        float perc_;                    // percentage of points in track 跟踪点的百分比
        float avg_depth_;               // average scene depth, used to decide new keyframe 平均场景深度，用来决定新的关键帧
        std::vector<int> num_feat_pts_; // number of feature points 特征点数量

        /* convergence evaluation flags */ // 收敛评估标志
        bool tracking_lost_;
        int lost_frames_;

        /* IMU Biases */
        Eigen::Vector3f b_w_; // gyro bias //陀螺仪偏差
        Eigen::Vector3f b_a_; // accelerometer bias //加速度计偏差

        /* IMU integrated measurement */
        Eigen::Matrix3f R_meas_;
        Eigen::Vector3f v_meas_;
        Eigen::Vector3f p_meas_;
        float t_meas_;
        Eigen::Matrix<float, 9, 9> Cov_meas_;

        /* in the last keyframe */
        // std::vector<cv::Mat> G_binary_, Gx_mat_, Gy_mat_, disp_;  // matrices of Gx, Gy, disp, and binary mask
        std::vector<vector_aligned<Eigen::Vector2i>> feat_pixels_; // pixel coordinates of feature points x,y 最后一个关键帧像素坐标
        std::vector<vector_aligned<Eigen::Vector3f>> feat_3D_;     // 3D coordinates of feature points 最后一个关键帧3D坐标
        std::vector<std::vector<float>> intensities_;              // intensities of feature points 特征点的强度
        std::vector<std::vector<float>> Gx_, Gy_;                  // gradients of feature points 特征点的梯度

        /* for optimization */
        vector_aligned<Eigen::VectorXf> error_;
        vector_aligned<Eigen::VectorXf> variance_;
        vector_aligned<Eigen::VectorXf> mask_; // 0 if not using this point
        vector_aligned<Eigen::MatrixXf> J_;

        /* transformation between camera to body frame */  // 相机到主体帧之间的转换
        /* consider the first body frame as world frame */ // 将第一个主体框架视为世界框架
        Eigen::Isometry3f T_bc_;                           // p_b = T_bc_ * p_c
        Eigen::Isometry3f T_bc_inv_;                       // transpose

        /* gravity constant */
        Eigen::Vector3f g_;

        /* weight on the imu term */
        float alpha_;
        float weight_R_;
        float weight_p_;
        float weight_v_;
        float beta_;

        /* for the keyframe - relative pose to the initial camera */
        Eigen::Isometry3f T_kf_; // P_world = T_kf_ * P_kf

        /* for the current frame - relative pose to the last keyframe */
        Eigen::Isometry3f T_curr_; // P_curr = T_curr_ * P_kf

        /* for the initial camera and current camera - the transforms to update to */
        Eigen::Isometry3f T_; // P_world = T_ * P_curr

        /* States - relative pose of imu frame to the initial imu frame */ // imu帧与初始imu帧的相对姿态
        Eigen::Isometry3f T_wb_last_;                                      // last body frame
        Eigen::Isometry3f T_wb_;                                           // current body frame  T_wb_ = T_bc_ * T_ * T_bc_inv_ ; p_w = T_wb_ * p_b
        Eigen::Vector3f v_last_;                                           // velocity of last body frame in world frame
        Eigen::Vector3f v_;                                                // velocity of current body frame in world frame

        /* thread objects for image preprocessing */
        boost::thread threadPrepImg0_;
        boost::thread threadPrepImg1_;

        /* thread objects for pyramid disparity */
        boost::thread thread_pyr1_disp_, thread_pyr1_feat_, thread_tracking_;
        boost::thread thread_pyr2_, thread_pyr3_, thread_pyr4_;

        //    void optimize_ceres(const cv::Mat& img_curr, int ilay);

        /* return true if the point is in the specified boundary */
        template <typename T>
        bool inBound(T x, T y, int margin, const int ilay) const
        {
            bool inbound = true;
            if (x < T(margin) || x > T(frameSize_x_pyr_[ilay] - 1 - margin) || y < T(margin) || y > T(frameSize_y_pyr_[ilay] - 1 - margin))
                inbound = false;
            return inbound;
        }

        // 双线性插值
        /*
         *亚像素级别的精度：在处理图像时，有时需要在非整点处获取像素值，此时双线性插值能够提供足够的精度。
         *特征提取和跟踪：在跟踪算法中，可能需要在浮点坐标处提取图像梯度等信息，插值是这种情况下的常见手段。
         *图像缩放：当图像缩放到非整比例大小时，插值计算可以帮助生成平滑的缩放结果。
         */
        float interpolate_gradient(float x, float y, const cv::Mat &img) const
        {
            float i0, i1, i2, i3, i4, i5, i, x1, x2, y1, y2;
            i0 = (img.at<float>(floor(y), floor(x))); // leftupper 获取左上角的像素值（取整向下的 (x, y)）。
            i1 = (img.at<float>(ceil(y), floor(x)));  // leftlower 获取左下角的像素值（y 取整向上，x 取整向下）。
            i2 = (img.at<float>(floor(y), ceil(x)));  // rightupper 获取右上角的像素值（y 取整向下，x 取整向上）。
            i3 = (img.at<float>(ceil(y), ceil(x)));   // rightlower 获取右下角的像素值（取整向上的 (x, y)）。

            // 计算水平和垂直方向的插值权重：
            x1 = x - floor(x); // 在水平方向上的插值比例。
            x2 = ceil(x) - x;  // 水平方向的另一插值比例（补值）。
            y1 = y - floor(y); // 在垂直方向上的插值比例。
            y2 = ceil(y) - y;  // 在垂直方向上的插值比例。

            if (floor(x) == ceil(x)) // 说明 x 是整数，设置 x1 = 1.0 以避免插值计算错误。
                x1 = 1.0;
            if (floor(y) == ceil(y)) // 说明 y 是整数，设置 y1 = 1.0 以避免插值计算错误。
                y1 = 1.0;

            // 双线性插值
            // 水平插值
            i4 = x1 * i2 + x2 * i0; // 上部插值
            i5 = x1 * i3 + x2 * i1; // 下部插值
            // 垂直插值
            i = y1 * i5 + y2 * i4;
            return i;
        }

        /* return bilinear interpolated pixel intensity given inbound float pixel coordinates */
        //返回给定入站浮点像素坐标的双线性插值像素强度
        double interpolate(double x, double y, const cv::Mat &img) const
        {
            double i0, i1, i2, i3, i4, i5, i, x1, x2, y1, y2;
            i0 = (img.at<uchar>(floor(y), floor(x))); // leftupper
            i1 = (img.at<uchar>(ceil(y), floor(x)));  // leftlower
            i2 = (img.at<uchar>(floor(y), ceil(x)));  // rightupper
            i3 = (img.at<uchar>(ceil(y), ceil(x)));   // rightlower

            x1 = x - floor(x);
            x2 = ceil(x) - x;
            if (floor(x) == ceil(x))
                x1 = 1.0;
            y1 = y - floor(y);
            y2 = ceil(y) - y;
            if (floor(y) == ceil(y))
                y1 = 1.0;
            i4 = x1 * i2 + x2 * i0;
            i5 = x1 * i3 + x2 * i1;
            i = y1 * i5 + y2 * i4;
            return i;
        }

        float interpolate(float x, float y, const cv::Mat &img) const
        {
            float i0, i1, i2, i3, i4, i5, i, x1, x2, y1, y2;
            i0 = (img.at<uchar>(floor(y), floor(x))); // leftupper
            i1 = (img.at<uchar>(ceil(y), floor(x)));  // leftlower
            i2 = (img.at<uchar>(floor(y), ceil(x)));  // rightupper
            i3 = (img.at<uchar>(ceil(y), ceil(x)));   // rightlower

            x1 = x - floor(x);
            x2 = ceil(x) - x;
            if (floor(x) == ceil(x))
                x1 = 1.0;

            y1 = y - floor(y);
            y2 = ceil(y) - y;
            if (floor(y) == ceil(y))
                y1 = 1.0;

            i4 = x1 * i2 + x2 * i0;
            i5 = x1 * i3 + x2 * i1;
            i = y1 * i5 + y2 * i4;
            return i;
        }

        /* return bilinear interpolated pixel intensity given inbound float pixel coordinates */
        template <typename T>
        T interpolate(T x, T y, const cv::Mat &img) const
        {
            T i0, i1, i2, i3, i4, i5, i, x1, x2, y1, y2;
            i0 = T(img.at<uchar>(floor(y.a), floor(x.a))); // leftupper
            i1 = T(img.at<uchar>(ceil(y.a), floor(x.a)));  // leftlower
            i2 = T(img.at<uchar>(floor(y.a), ceil(x.a)));  // rightupper
            i3 = T(img.at<uchar>(ceil(y.a), ceil(x.a)));   // rightlower
            x1 = x - floor(x);
            x2 = ceil(x) - x;
            y1 = y - floor(y);
            y2 = ceil(y) - y;
            if (floor(x) == ceil(x))
                x1 = T(1.0);
            if (floor(y) == ceil(y))
                y1 = T(1.0);
            i4 = x1 * i2 + x2 * i0;
            i5 = x1 * i3 + x2 * i1;
            i = y1 * i5 + y2 * i4;
            return i;
        }

        /* two image pipeline - getting 3D points and jacobians */
        void twoImagePipeline(const std::vector<cv::Mat> &img0, const std::vector<cv::Mat> &img1);

        /* one image pipeline - tracking */
        void oneImagePipeline(const std::vector<cv::Mat> &img0);

        /* extract high gradient pixels, obtain binary mask and gradient in x and y */
        void extractFeats(const cv::Mat &img0, cv::Mat &G_binary, cv::Mat &Gx, cv::Mat &Gy, int ilay);

        void fillVectors(const cv::Mat &img0, const cv::Mat &G_binary, const cv::Mat &disp,
                         const cv::Mat &Gx, const cv::Mat &Gy, std::vector<float> &disp_vec, vector_aligned<Eigen::Vector2i> &feat_pixels,
                         std::vector<float> &intensities, std::vector<float> &Gx_vec, std::vector<float> &Gy_vec, int &num_pts, int ilay);

        /* get Jacobian for all points */
        void getJacobian(const std::vector<float> &Gx, const std::vector<float> &Gy,
                         float fx, float fy, const vector_aligned<Eigen::Vector3f> &feat_3D, Eigen::MatrixXf &J, int npts);

        /* get Jacobian for one feature point in FCA */
        void getJacobian_fca(const std::vector<float> &Gx, const std::vector<float> &Gy, const vector_aligned<Eigen::Vector3f> &feat_3D_curr, const std::vector<int> &index,
                             const Eigen::Isometry3f &T_curr, const Eigen::Isometry3f &T_wb, Eigen::MatrixXf &J, const int ilay, const int npts_in);

        /* initialize IMU Jacobian */
        void initJacobian_imu_fca(Eigen::MatrixXf &J_imu);

        /* transformation of 3D points from one frame to another */
        void transform3D(const vector_aligned<Eigen::Vector3f> &points_f1, vector_aligned<Eigen::Vector3f> &points_f2,
                         const Eigen::Isometry3f &T_21, const int npts);

        /* obtain error vector and update mask and masked jacobian and error*/
        void getError(const cv::Mat &img_curr, const Eigen::MatrixXf &J, const Eigen::Isometry3f &T_curr,
                      const vector_aligned<Eigen::Vector3f> &feat_3D, const std::vector<float> &intensities,
                      vector_aligned<Eigen::Vector3f> &feat_3D_curr, vector_aligned<Eigen::Vector2f> &feat_pixels_curr, Eigen::VectorXf &error,
                      Eigen::VectorXf &mask, const int npts, const int ilay, int &npts_in,
                      Eigen::MatrixXf &J_masked, Eigen::VectorXf &error_masked, Eigen::VectorXf &W);

        /* obtain Jacobian, error vector and huber weights for current frame using FCA */
        void getError_fca(const cv::Mat &img_curr, const cv::Mat &Gx_mat, const cv::Mat &Gy_mat, const Eigen::Isometry3f &T_curr, const Eigen::Isometry3f &T_wb,
                          Eigen::MatrixXf &J_masked, Eigen::VectorXf &error_masked, Eigen::VectorXf &W,
                          int &npts_in, const int ilay);

        /* obtain IMU error */
        void getError_imu_fca(const Eigen::Isometry3f &T_wb, const Eigen::Vector3f &v, Eigen::VectorXf &r_imu, Eigen::MatrixXf &J_imu);

        /* Gauss-Newton optimization */
        void optimize(const cv::Mat &img_curr, const Eigen::MatrixXf &J, Eigen::Isometry3f &T_curr, Eigen::VectorXf &error,
                      Eigen::VectorXf &variance, vector_aligned<Eigen::Vector2i> &feat_pixels, vector_aligned<Eigen::Vector3f> &feat_3D,
                      std::vector<float> &intensities, Eigen::VectorXf &mask, const int npts, int max_iter, int ilay);

        /* optimization for forward compositional approach (FCA) */
        void optimize_fca(const cv::Mat &img_curr, Eigen::Isometry3f &T_curr, Eigen::Isometry3f &T_wb, Eigen::Isometry3f &T, Eigen::Vector3f &v, int ilay);

        /* obtain convergence evaluation error mean */
        float get_error_mean(const Eigen::VectorXf &error_masked, const Eigen::VectorXf &r_imu);

        /* update optimization targets given delta */
        void update_optimization_step(const Eigen::Matrix<float, 9, 1> &delta, Eigen::Isometry3f &T_wb, Eigen::Vector3f &v, Eigen::Isometry3f &T, Eigen::Isometry3f &T_curr);

        /* initialize pose as identity */
        void initPose(Eigen::Vector3f &t, Eigen::Matrix3f &R, Eigen::Quaternionf &q, Eigen::Isometry3f &T);

        /* smoothing */
        void smooth();
        void smooth_global();

        /* get current pose in world for vis */
        void updatePose();

        /* set keyframe pose */
        void setKfPose();

        /* set pyramid values */
        void setPyrScale();

        /* set grid object pointers */
        void setGD();

        /* thread functions */
        void threadCreatePyr(const cv::Mat &img, std::vector<cv::Mat> &img_ready);
        void threadFunc_disp(const cv::Mat &img0, const cv::Mat &img1, cv::Mat &disp);
        void threadFunc_feat(const cv::Mat &img0, cv::Mat &G_binary, cv::Mat &Gx_mat, cv::Mat &Gy_mat);
        void threadFuncComb(const cv::Mat &img0, const cv::Mat &img1, cv::Mat &G_binary,
                            cv::Mat &disp, cv::Mat &Gx_mat, cv::Mat &Gy_mat, int ilay);
        void threadOneImagePipeline(const std::vector<cv::Mat> &img0);
    };

}

#endif
