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

#include "sdd_vio/vo_stereo.h"

#include <iostream>
#include "sdd_vio/utils/math_utils.h"
#include <algorithm>
#include <math.h>
#include <ros/ros.h>
#include <cmath>
#include <assert.h>
#include <boost/thread.hpp>
#include "sdd_vio/utils/ros_params_helper.h"
#include "sdd_vio/utils/timer.hpp"
#include <ctime>
#include <opencv2/core/eigen.hpp>

namespace sdd_vio
{

    /* extract high gradient pixels */
    // Input: left image 'img0'
    // Output: high gradient pixels mask 'G_binary', gradient mat 'Gx','Gy'
    void VoStereo::extractFeats(const cv::Mat &img0, cv::Mat &G_binary, cv::Mat &Gx, cv::Mat &Gy, int ilay)
    {
        ROS_DEBUG_STREAM("extract-Feats: ----------");

        /* get gradient - from the tracking base layer and each layer up */
        ROS_DEBUG_STREAM("extract-Feats: get gradient");
        cv::Mat Gx2, Gy2, G2, G;
        cv::Sobel(img0, Gx, CV_32F, 1, 0, 3); // 计算x方向的梯度
        cv::Sobel(img0, Gy, CV_32F, 0, 1, 3); // 计算y方向的梯度
        cv::multiply(Gx, Gx, Gx2);            // 对x方向梯度平方
        cv::multiply(Gy, Gy, Gy2);            // 对y方向梯度平方
        cv::add(Gx2, Gy2, G2);                // 将x方向和y方向的梯度平方相加
        cv::sqrt(G2, G);                      // 计算梯度幅值的平方根

        G = sdd_vio::adjustVis(G); // adjust to 0-255
        G.convertTo(G, CV_8U);     // 转换为8位图像
        // sdd_vio::showMatMinMax(G);

        /*  threshold gradient - on the base layer */
        ROS_DEBUG_STREAM("extract-Feats: threshold gradient");

        // 使用自适应阈值方法 (cv::adaptiveThreshold) 将梯度图像转换为二值图像 G_binary
        cv::adaptiveThreshold(G, G_binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, adapt_size_, adapt_thresh_);
        // double thresh_otsu;
        // thresh_otsu = cv::threshold(G, G_binary, 0, 255, cv::THRESH_OTSU);
        // std::cout << "thresh otsu: " << thresh_otsu << std::endl;
        // cv::threshold(G2, G2_binary, 254, 255, cv::THRESH_BINARY);

        // /* prune function that changes G_binary to contain only selected points */
        // 如果启用了 use_gd_，则调用修剪函数 gd_[ilay]->prune(G, G_binary) 对二值化后的图像进行修剪，以去除不符合特定条件的像素。
        ROS_DEBUG_STREAM("extract-Feats: prune function");
        if (use_gd_)
            gd_[ilay]->prune(G, G_binary); // 通过自定义的修剪函数去除不符合条件的像素

        /* set margin - for selection not too close to the boundary */
        // std::cout<<"G_Binar at 0,0: "<<G_binary.at(0,0)<<"\n";
        int nrows = G_binary.rows;
        int ncols = G_binary.cols;
        G_binary.rowRange(0, feat_margin_).setTo(cv::Scalar(0));             // 清除顶部边界
        G_binary.rowRange(nrows - feat_margin_, nrows).setTo(cv::Scalar(0)); // 清除底部边界
        G_binary.colRange(0, feat_margin_).setTo(cv::Scalar(0));             // 清除左侧边界
        G_binary.colRange(ncols - feat_margin_, ncols).setTo(cv::Scalar(0)); // 清除右侧边界
    }

    /*
    Fill out vectors given mask
        Input:
            img0: keyframe image关键帧图像（灰度图）。
            G_binary: binary mask of selected high gradient points二进制掩码图，标识高梯度点的位置。
            disp: disparity image 视差图，表示每个像素的视差值
            Gx: gradient of feature points in x direction特征点在x方向的梯度图
            Gy: gradient of feature points in y direction特征点在y方向的梯度图
            ilay: specifying the layer number图像金字塔的层号，用于根据不同层进行视差范围的选择

        Output:
            feat_pixels: the pixel coordinates of feature points//特征点的像素坐标，存储为二维向量
            disp_vec: disparity of feature points//特征点的视差值
            intensities: pixel intensity of feature points特征点的像素强度
            Gx_vec: Gx of feature points特征点在x方向的梯度
            Gy_vec: Gy of feature points特征点在y方向的梯度
            num_pts: total number of feature points有效特征点的总数量
    */
    void VoStereo::fillVectors(const cv::Mat &img0, const cv::Mat &G_binary, const cv::Mat &disp,
                               const cv::Mat &Gx, const cv::Mat &Gy, std::vector<float> &disp_vec, vector_aligned<Eigen::Vector2i> &feat_pixels,
                               std::vector<float> &intensities, std::vector<float> &Gx_vec, std::vector<float> &Gy_vec, int &num_pts, int ilay)
    {
        // 清空所有输出向量，以确保不会受到之前数据的影响
        disp_vec.clear();
        feat_pixels.resize(0);
        intensities.resize(0);
        Gx_vec.resize(0);
        Gy_vec.resize(0);

        //使用OpenCV的findNonZero函数在G_binary中找到所有高梯度点的坐标，这些点将作为候选特征点。
        std::vector<cv::Point> high_gradient_pixels;
        cv::findNonZero(G_binary, high_gradient_pixels);
        int npts = high_gradient_pixels.size();

        /* get disparity vector and filter out unwanted points */
        //遍历高梯度点并填充特征信息：
        for (int i = 0; i < npts; ++i)
        {
            //遍历每一个高梯度点，提取其像素坐标 (x, y)。
            int x = high_gradient_pixels[i].x;
            int y = high_gradient_pixels[i].y;
            //检查该点是否在允许的边界内（通过调用inBound函数）。如果不在边界内，则输出调试信息。
            if (!inBound(x, y, 0, ilay))
                std::cout << "not inbound!" << std::endl;
            //从视差图中获取当前点的视差值 d。视差值通过将int16格式的值除以16.0得到。
            float d = disp.at<int16_t>(y, x) / 16.0;
            //提取当前点在x方向和y方向的梯度值 gx 和 gy
            float gx = Gx.at<float>(y, x);
            float gy = Gy.at<float>(y, x);
            //获取当前点的像素强度值 intensity。
            float intensity = float(img0.at<uchar>(y, x));

            /* select valid disparity range */
            //筛选并存储有效的特征点：
            //检查视差值 d 是否在指定的视差范围内（由disp_range_pyr_[ilay]定义），并再次调用inBound检查点是否在允许的范围内。
            if (d > disp_range_pyr_[ilay] && inBound(x, y, 1, ilay))
            {
                //如果点满足上述条件，将视差值、像素坐标、梯度值和强度值分别存储到相应的输出向量中。
                disp_vec.push_back(d);
                Eigen::Vector2i pixel(x, y);
                feat_pixels.push_back(pixel);
                Gx_vec.push_back(gx);
                Gy_vec.push_back(gy);
                intensities.push_back(intensity);
            }
        }
        // showVecMinMax(Gx_);
        // std::cout << "valid number of feature points: " << disp_vec.size() << std::endl;
        // sdd_vio::showVecMinMax(disp_vec);
        num_pts = disp_vec.size();//统计有效特征点的数量并将其存储到num_pts中。
    }

    /*
        Input:
            Gx: gradient vector of feature points in x direction
            Gy: gradient vector of feature points in y direction
            fx: calibration param
            fy: calibration param
            feat_3D: feature point coordinates in 3D in keyframe
            npts: umber of feature points

        Output:
            J: Jacobian matrix of the feature points
    */
    void VoStereo::getJacobian(const std::vector<float> &Gx, const std::vector<float> &Gy,
                               float fx, float fy, const vector_aligned<Eigen::Vector3f> &feat_3D, Eigen::MatrixXf &J, int npts)
    {
        J.resize(npts, 6);
        Eigen::Matrix<float, 1, 6> J_i;    // Jacobian for one point
        Eigen::Matrix<float, 1, 2> J_grad; // gradient jacobian
        Eigen::Matrix<float, 2, 3> J_proj; // projection jacobian
        Eigen::Matrix<float, 3, 6> J_SE3;  // exponential jacobian
        for (int i = 0; i < npts; ++i)
        {
            /* calculate for each point the 1x6 Jacobian */
            float x = feat_3D[i](0);
            float y = feat_3D[i](1);
            float z = feat_3D[i](2);

            J_grad(0, 0) = Gx[i];
            J_grad(0, 1) = Gy[i];

            J_proj(0, 0) = fx / z;
            J_proj(1, 0) = 0;
            J_proj(0, 1) = 0;
            J_proj(1, 1) = fy / z;
            J_proj(0, 2) = -fx * x / (z * z);
            J_proj(1, 2) = -fy * y / (z * z);

            Eigen::Matrix3f npHat;
            npHat << 0, z, -y, -z, 0, x, y, -x, 0;
            J_SE3 << Eigen::Matrix3f::Identity(3, 3), npHat;

            J_i = J_grad * J_proj * J_SE3;

            // if (i==0) {
            //  std::cout<<"x,y,z:\n"<<x<<'\n'<<y<<'\n'<<z<<std::endl;
            //  std::cout<<"J_grad:\n"<<J_grad<<std::endl;
            //  std::cout<<"J_proj:\n"<<J_proj<<std::endl;
            //  std::cout<<"J_SE3:\n"<<J_SE3<<std::endl;
            //  std::cout<<"J_i:\n"<<J_i<<std::endl;
            // }

            /* assign J_i to J */
            J.block<1, 6>(i, 0) = J_i;
        }

    } // void VoStereo::getJacobian

    /*
        Input:
            img_curr: current frame image
            T_curr: current optimized transformation T between current frame and keyframe
            feat_3D: feature points in 3D in keyframe coordinates
            npts: number of feature points in total in the keyframe
            ilay: specifying the layer from image pyramid to get error from
            intensities: intensities of feature points from keyframe
            J: Jacobian calculated from keyframe on the feature points

        Output:
            feat_3D_curr: feature points in 3D in current frame coordinates
            feat_pixels_curr: feature points in current frame pixel coordinates
            npts_in: the number of points remaining in view
            mask: vector of size feat_3D, 0 if not in view, 1 if in view
            error: vector of intensity errors, size as feat_3D, only update the values when mask=1
            J_masked: of size npts_in x 6, with only entries of mask=1
            error_masked: of size npts_in x 6, with only entries of mask=1
            W: diagonal weight matrix for robust estimation Huber weights
    */
    void VoStereo::getError(const cv::Mat &img_curr, const Eigen::MatrixXf &J, const Eigen::Isometry3f &T_curr,
                            const vector_aligned<Eigen::Vector3f> &feat_3D, const std::vector<float> &intensities,
                            vector_aligned<Eigen::Vector3f> &feat_3D_curr, vector_aligned<Eigen::Vector2f> &feat_pixels_curr, Eigen::VectorXf &error,
                            Eigen::VectorXf &mask, const int npts, const int ilay, int &npts_in,
                            Eigen::MatrixXf &J_masked, Eigen::VectorXf &error_masked, Eigen::VectorXf &W)
    {

        /* Timer Template
        ros::Time t1 = ros::Time::now();
        ros::Time t2 = ros::Time::now();
        ros::Duration duration = t2 - t1;
        std::cout << "Time for sorting: " << duration.toSec() << std::endl;
        */

        ROS_DEBUG_STREAM("get-Error: --------- ");

        /* transformation of feat_3D into current camera frame and obtain feat_3D_curr */
        transform3D(feat_3D, feat_3D_curr, T_curr, npts);

        /* for debug - will print out info if nan values found */
        bool feat3dhasnan = false;
        bool feat3dcurrhasnan = false;
        for (int i = 0; i < npts; ++i)
        {
            if (std::isnan(feat_3D_curr[i](2)))
                feat3dcurrhasnan = true;
            if (std::isnan(feat_3D[i](2)))
                feat3dhasnan = true;
        }
        if (feat3dhasnan == true)
            std::cout << "feat_3D has nan" << std::endl;
        if (feat3dcurrhasnan == true)
        {
            std::cout << "feat_3D_curr has nan" << std::endl;
            // std::cout<<"T_curr: "<<'\n'<<T_curr<<std::endl;
        }
        /* for debug */

        /* reproject into current camera image */
        cam_->get2DPixels(feat_3D_curr, feat_pixels_curr, npts, ilay);

        /* update error vector and mask */
        npts_in = 0;
        for (int i = 0; i < npts; ++i)
        {
            if (inBound(feat_pixels_curr[i](0), feat_pixels_curr[i](1), 0, ilay))
            {
                error[i] = intensities[i] - interpolate(feat_pixels_curr[i](0), feat_pixels_curr[i](1), img_curr);
                npts_in = npts_in + 1;
                mask[i] = 1;
            }
            else
            {
                mask[i] = 0; // mark the point as out of bound and not to consider
            }
        }

        /* fill in the masked Jacobian */
        J_masked.resize(npts_in, 6);
        error_masked.resize(npts_in);
        int ind = 0;
        for (int i = 0; i < npts; ++i)
        {
            if (mask[i] != 0)
            {
                J_masked.row(ind) = J.row(i);
                error_masked[ind] = error[i];
                ind++;
            }
        }

        Eigen::VectorXf r = error_masked;
        // Eigen::VectorXf r_abs = error_masked.cwiseAbs();

        /* sort to find median - impossible for time */
        // std::sort(r.data(),r.data()+r.size());
        // std::sort(r_abs.data(),r_abs.data()+r_abs.size());
        // int mid = (int)(npts_in/2);
        // float r_med = r[mid-1];
        // float r_abs_med = r_abs[mid-1];
        // Eigen::VectorXf r_hat = error_masked.array() - r_med;  // normalize by removing the median (original order)
        // float sigma = 1.485 * r_abs_med;  // median absolute deviation

        float r_mean = r.mean();
        Eigen::VectorXf r_hat = error_masked.array() - r_mean;          // normalize by removing the mean (original order)
        float sigma = sqrt(((r_hat.array()).square()).sum() / npts_in); // standard deviation
        // std::cout << "sigma: " << sigma << std::endl;

        W.resize(npts_in);
        for (int i = 0; i < npts_in; ++i)
        {
            if (fabs(r_hat[i]) < 1.345 * sigma)
                W[i] = 1;
            else
                W[i] = (1.345 * sigma) / fabs(r_hat[i]);
        }

        /* for debug */
        if (J_masked.maxCoeff() > J.maxCoeff())
        {
            std::cout << "ind: " << ind << std::endl;
            std::cout << "npts_in: " << npts_in << std::endl;
            std::cout << "mask_sum: " << mask.sum() << std::endl;
        }
        /* for debug */
    }

    /*
        Input:
            img_curr: current image
            当前帧图像。
            J: Jacobian calculated from last keyframe of size npts x 6
            从最后一个关键帧计算得到的雅可比矩阵（Jacobian），大小为 npts x 6，其中 npts 是关键帧中特征点的数量。
            feat_pixels:
            2D 像素坐标，表示关键帧中的特征点位置。
            feat_3D: feature points in 3D in keyframe coordinates
            3D 特征点，表示在关键帧坐标系中的特征点位置。
            intensities: pixel intensities of feat_3D points
            特征点的像素强度值。
            npts: number of feature points in keyframe
            关键帧中特征点的数量。
            max_iter: max number of iterations specified for the optimizer
            优化器的最大迭代次数。
            ilay: specifying the layer in image pyramid
            图像金字塔中的层数，指定当前的图像层。

        Updates:
            T_curr: current optimized transformation from keyframe to current frame
            优化后的从关键帧到当前帧的变换矩阵（位姿）。
            error: intensity errors of size npts x 1
            强度误差向量，大小为 npts x 1，记录了每个特征点的强度误差。
            variance:
            误差的方差，用于衡量误差的分布。
            mask: 0 if not in current view, 1 if in view
            遮罩矩阵，标记特征点是否在当前视野中。0 表示不在视野中，1 表示在视野中。
    */
    void VoStereo::optimize(const cv::Mat &img_curr, const Eigen::MatrixXf &J, Eigen::Isometry3f &T_curr, Eigen::VectorXf &error,
                            Eigen::VectorXf &variance, vector_aligned<Eigen::Vector2i> &feat_pixels, vector_aligned<Eigen::Vector3f> &feat_3D,
                            std::vector<float> &intensities, Eigen::VectorXf &mask, const int npts, int max_iter, int ilay)
    {
        ROS_DEBUG_STREAM("op-timize: ----------- ");

        int npts_in; // number of points remained in view当前视野中的点数量。
        perc_ = 1.;  // percentage of points remained in view在视野中的点占总点数的百分比。
        float err, err_mean = 0, err_init, err_last, err_diff = 0;

        vector_aligned<Eigen::Vector3f> feat_3D_curr;     // 3D points in current camera's frame
        vector_aligned<Eigen::Vector2f> feat_pixels_curr; // 2D pixel coordinates in current image

        /* masked Jacobian matrix and error vector - remove points outside projection */
        Eigen::MatrixXf J_masked;
        Eigen::VectorXf error_masked;
        Eigen::VectorXf W; // diagonal weight matrix for Huber estimation
        Eigen::MatrixXf J_masked_weighted;

        /* Levenberg Marquardt */
        float lambda = lambda_; // initial lambda is equal to the initial one set  初始 lambda 值用于控制优化的步长。
        // 设置增大和减小 lambda 的因子 (up_factor_ 和 down_factor_)。
        float up = up_factor_;
        float down = down_factor_;
        bool if_break = false;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            // vis_curr = vis_curr.clone();
            // vis_curr = vis_curr;

            /* calculate error vector and update mask given initial transformation T_curr */
            if (iter == 0)
            {

                /*
                Calculate errors and Jacobians for optimization
                    Inputs:
                        img_curr
                        J
                        T_curr
                        feat_3D
                        intensities
                        npts
                        ilay
                    Outputs:
                        npts_in: for statistics
                        J_masked: for Hessian calculation
                        error_masked: for Hessian calculation,
                        feat_pixels_curr: for visualization
                    Recalculate in every iteration:
                        error
                        mask
                        feat_3D_curr
                */
                getError(img_curr, J, T_curr, feat_3D, intensities, feat_3D_curr, feat_pixels_curr, error, mask,
                         npts, ilay, npts_in, J_masked, error_masked, W);
                ROS_DEBUG_STREAM("op-timize: initial error obtained ");

                /* Monitoring
                    getting statistics for setting criteria for end of iterations:
                    percentage of points in view and current average error value */
                perc_ = 100 * npts_in / npts;
                err = error_masked.transpose() * error_masked; // sum of square errors
                err_mean = sqrt(err / npts_in);                // NOTE: not technically mean, but mean on squares then do sqrt
                err_init = err_mean;

                // draw blue circles on vis_curr of all the points
                if (use_opt_vis_)
                {
                    for (int i = 0; i < npts; ++i)
                    {
                        cv::Point p(feat_pixels_curr[i](0), feat_pixels_curr[i](1));
                        cv::circle(vis_curr, p, 1, cv::Scalar(255, 0, 0), 0.5);
                    }
                }
            }

            /* convert weight vector W to diagonal matrix */
            // J_masked_weighted.resize(npts_in,6);
            // for (int i=0; i<npts_in; ++i) {
            //     J_masked_weighted.row(i) = J_masked.row(i) * W[i];
            // }

            /* get H = J^t * W * J */
            Eigen::MatrixXf H = J_masked.transpose() * W.asDiagonal() * J_masked;
            // Eigen::MatrixXf H = J_masked_weighted.transpose() * J_masked;

            /* get b = J^t * w * e */
            Eigen::VectorXf b = J_masked.transpose() * (W.asDiagonal() * error_masked);
            // Eigen::VectorXf b = J_masked_weighted.transpose() * error_masked;

            ROS_DEBUG_STREAM("op-timize: Hessian obtained. ");

            /* Monitoring */
            err_last = err_mean;

            /* LMA */
            float mult = 1 + lambda;
            bool ill = true; // if ill conditioned (error increase)
            Eigen::Isometry3f T_temp;
            int lmit = 0; // number of iterations while ill=true

            while ((ill == true) && (iter < max_iter) && (lmit < 5))
            {
                /* Get H based on LMA current damping parameter, multiply diagonal by mult */
                for (int par = 0; par < 6; par++)
                    H(par, par) = H(par, par) * mult;

                /* get delta increment */
                Eigen::VectorXf delta;
                delta = -H.ldlt().solve(b);

                /* for debug - delta is valid */
                if (std::isnan(delta(0) + delta(1) + delta(2) + delta(3) + delta(4) + delta(5)))
                {
                    std::cout << "delta:" << '\n'
                              << delta << std::endl;
                    std::cout << "H" << '\n'
                              << H << std::endl;
                    std::cout << "b" << '\n'
                              << b << std::endl;
                    showEigenInfo(J_masked);
                    showEigenInfo(error_masked);
                    showEigenInfo(J);
                    showEigenInfo(error);
                }
                /* for debug */

                /* update current transformation */
                T_temp = T_curr * (exp_SE3(delta).inverse());

                /* calculate error vector and update mask given transformation */
                getError(img_curr, J, T_temp, feat_3D, intensities, feat_3D_curr, feat_pixels_curr, error, mask,
                         npts, ilay, npts_in, J_masked, error_masked, W);

                perc_ = 100 * npts_in / npts;
                err = error_masked.transpose() * error_masked;
                err_mean = sqrt(err / npts_in);

                /* get error increment */
                err_diff = err_mean - err_last;
                ill = (err_diff > 0); // if error larger than before, set ill = true, try increase lambda.

                /* terminal output */
                if (verbose_)
                {
                    std::cout << "iteration = " << iter << ", lambda = " << lambda << ", err = " << err_mean << ", derr = " << err_diff << std::endl;
                }

                /* if ill=true, higher lambda by up factor, count as one iteration */
                if (ill)
                {
                    mult = (1 + lambda * up) / (1 + lambda);
                    lambda *= up;
                    iter++;
                }

                lmit++;
            }

            /* update T with the one that makes error smaller or the last try */
            /* error and jacobians are already updated in the loop */
            T_curr = T_temp;

            /* if error doesn't get higher for quite some time, this term will bring lambda down eventually */
            if (lambda > 1e-8) // make sure doesn't overflow
                lambda *= (1 / down);

            /* if LM iterations didn't decrease the error, stop */
            if (ill)
            {
                // ROS_WARN_STREAM("bad convergence!");
                if_break = true;
            }

            /* if error is decreased below target or reach max iterations */
            if (-err_diff < target_derr_ || iter == max_iter - 1)
                if_break = true;

            /* end of optimization */
            if (if_break)
            {

                ROS_INFO_STREAM("iteration " << iter << "--------");
                ROS_INFO_STREAM("err/initial err: " << err_mean << "/" << err_init);

                // optimized position, draw green circles
                if (use_opt_vis_)
                {
                    for (int i = 0; i < npts; ++i)
                    {
                        if (inBound(feat_pixels_curr[i](0), feat_pixels_curr[i](1), 0, ilay))
                        {
                            cv::Point p(feat_pixels_curr[i](0), feat_pixels_curr[i](1));
                            cv::circle(vis_curr, p, 1, cv::Scalar(0, 255, 0), 0.5);
                        }
                    }
                    cv::imshow("current features", vis_curr);
                    cv::waitKey(1);
                }
                break;
            }

        } // for (int iter=0;

    } // void VoStereo::op-timize

    /*
        Optimization for state x given current frame image using Ceres-Solver

        x: 6x1 state vector. first three elements in so(3) of R_curr_, last three elements same as t_curr_
        Ceres is using AutoDiff to find the Jacobians of photometric errors with respect to state x

        Input:
            img_curr: current left camera image
            ilay: the layer that it is on (e.g. there're two layers in total, then 0 is the larger and 1 is the smaller)
        Updates:
            R_curr_: 3x3 rotation matrix from keyframe to current frame
            t_curr_: 3x1 translation vector from keyframe to current frame
            T_curr_: 4x4 SE3 transformation matrix from keyframe to current frame
    */
    //    void VoStereo::optimize_ceres(const cv::Mat& img_curr, int ilay)
    //    {
    //        // The variable to solve for with its initial value.
    //        Eigen::Matrix<double,3,1> phi_init = log_SO3<double>(R_curr_.cast<double>());
    //        double x[6] = {phi_init(0), phi_init(1), phi_init(2), t_curr_(0), t_curr_(1), t_curr_(2)};

    //        // Build the problem.
    //        ceres::Problem problem;

    //        // Set up the cost function (also known as residual). This uses
    //        // auto-differentiation to obtain the derivative (jacobian).
    //        ceres::CostFunction* cost_function =
    //            new ceres::AutoDiffCostFunction<pixelIntensityError, ceres::DYNAMIC, 6>(new pixelIntensityError(this, &img_curr, ilay), num_feat_pts_[ilay]);
    //        problem.AddResidualBlock(cost_function, NULL, x);

    //        // Run the solver!
    //        ceres::Solver::Options options;
    //        options.minimizer_progress_to_stdout = false;
    //        options.max_num_iterations = 30;
    //        ceres::Solver::Summary summary;
    //        ceres::Solve(options, &problem, &summary);

    //        std::cout << summary.BriefReport() << "\n";
    ////        std::cout << summary.FullReport() << "\n";

    //        /* construct R and t from x */
    //        Eigen::Matrix<float,3,1> phi, p;  // state variables os so3 and translation
    //        phi(0) = x[0]; phi(1) = x[1]; phi(2) = x[2];
    //        p(0) = x[3]; p(1) = x[4]; p(2) = x[5];
    //        Eigen::Matrix<float,3,3> R_curr = exp_SO3<float>(phi);

    //        /* assign R_curr_ and t_curr_, T_curr_ with optimized values */
    //        R_curr_ = R_curr;
    //        t_curr_ = p;
    //        T_curr_ = TfromRt(R_curr, p);
    //        rot_curr_ = phi.norm();

    //    }

}
