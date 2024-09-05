#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

Estimator estimator;

std::condition_variable con; // 用于在线程之间进行事件通知协调
double current_time = -1;

// 三个回调函数存数据buf
// 队列imu_buf、feature_buf、relo_buf是被多线程共享的，
// 因而在回调函数将相应的msg放入buf或进行pop时，需要设置互斥锁m_buf，在操作前lock()，操作后unlock()。
// 在feature_callback和imu_callback中还设置了条件锁，在完成将msg放入buf的操作后唤醒作用于process线程中的获取观测值数据的函数。
// 在imu_callback中还通过lock_guard的方式构造互斥锁m_state，能够自动加锁、解锁
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf; // 重定位
int sum_of_wait = 0;

// 互斥量，锁
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;

// IM U项[P,Q,B,Ba,Bg,a,g]
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

// 初始化feature、imu、last_imu_t
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

// 通过当前IMU来积分得到当前位置
//  从IMU测量值imu_msg和上一个PVQ递推得到下一个tmp_Q，tmp_P，tmp_V，(位置、速度、旋转)
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 1.先取当前imu时间戳
    double t = imu_msg->header.stamp.toSec();
    // 如果imu第一个时间戳
    if (init_imu) // init_imu=1表示第一个IMU数据
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time; // 时间间隔
    latest_time = t;             // 更新上一次imu时间戳

    // 2.线性加速度和角速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg; // 中值积分的角速度
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); // 中值积分的加速度

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// imu_buf中PVQ递推更新
// 从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
// 因为imu的频率比图像频率要高很多，在getMeasurements(）将图像和imu时间对齐后，imu_buf中还会存在剩余的imu数据
// 对imu_buf中剩余的所有imu_msg进行PVQ递推
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

// 对imu和图像数据对齐组合，直到把缓存中Feature数据或者IMU数据读取完，才能跳出函数，返回数据。
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        // IMU数据队列或者feature数据的buf是空的，直接返回
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        // 对齐标准：最新IMU数据时间戳 大于 最旧Feature时间戳，保证了有足够的IMU数据
        //  最新IMU的时间戳 小于 最旧的图像帧，说明IMU数据太少，返回空值，等待IMU刷新
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            // ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // 3.对齐标准：最旧IMU时间戳 小于 最旧的Feature时间戳，保证了feature帧数据之前一定要有IMU数据
        // 开头，只有图像，后来突然来了IMU，但是第一帧图像太旧了，就弃第一帧图像。即相机帧太旧了，要丢弃。
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        // 4.图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        // 理想：IMU最旧的在图像帧最旧的之前，说明IMU数据足够
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        // 这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布PVQ，header
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 1. 判断时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec(); // 获取最新的imu时间戳

    // 2.将imu_msg保存到imu_buf，记得加锁解锁
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();

    // 3.唤醒作用于process线程中的获取观测值数据的函数
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        // 构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state);

        // 4. 递推得到IMU的PQV
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        // 5.发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// feature回调函数，将feature_msg放入feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    // 1、 判断是否第一帧，光流没有速度
    if (!init_feature)
    {
        // skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    // 2.feature_msg加入到feature_buf中，记得加锁解锁
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    // 3、唤醒作用于process线程中的获取观测值数据的函数
    con.notify_one();
}

// restart回调函数，清空feature-buffnimu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        // 1.清空feature和imu的buf，记得对m_uf加锁解锁
        m_buf.lock();
        while (!feature_buf.empty())
            feature_buf.pop();
        while (!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();

        // 2.估计器重置，参数重置，记得对m_eastimator加锁解锁
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        // 3.时间重置
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// 重定位回调函数，将points_msg放入relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    // printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        // 多个imu数据+1帧特征点数据组成一个measurements数据结构体
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

        // 对IMU和图像特征数据进行对齐并且组合
        // 在提取measurements时互斥锁m_buf会锁住，此时无法接收数据;等待measurements上面两个接收数据完成就会被唤醒
        // 回调函数接收数据，接受完一组数据后唤醒提取数据的线程；提取数据线程提取完数据后，回调函数继续接收数据。
        // 作者在这里用了一个lambda表达式，也就是说，在return里面的部分是false时，保持上锁状态，继续配对数据；
        // 如果return里面时true，说明配对完成，释放锁，measurements完成了，以供后续使用。
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 { return (measurements = getMeasurements()).size() != 0; }); // 调用wait函数，先解锁lk，然后判断lambda的返回值
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements) // 遍历获取的Feature和IMU测量值组成的measurement
        {
            // 2.1 IMU 预积分
            auto img_msg = measurement.second; // 获取measurement中的Feature
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first) // 遍历获取measurement中对齐的imu
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                // 发送IMU数据进行预积分
                if (t <= img_t) // imu时间在feature时间之前
                {
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time; // 两个imu时间间隔
                    ROS_ASSERT(dt >= 0);
                    current_time = t;                    // 更新当前imu时间
                    dx = imu_msg->linear_acceleration.x; // 线速度
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z; // 角速度
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    // imu预积分,通过中值积分得到当前PQV作为优化初值
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    // printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else // imu时间在feature时间之后,需要做一个简单的线性插值
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;

                    // IMU预积分
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    // printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // 2.2 重定位
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            // 在relo_buf中取出最后一个重定位帧，拿出其中的信息并执行setReloFrame()
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                // 重定位
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            // 2.3 处理图像
            // 建立每个特征点的<feature_id,<camera_id,[x,y,z,u,v,vx,vy]>>的map，索引为feature_id
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                // 遍历当前帧中的每一个特征点！
                int v = img_msg->channels[0].values[i] + 0.5; // 四舍五入技巧（传入的特征点编码p_id * NUM_OF_CAM + i）
                int feature_id = v / NUM_OF_CAM;              // 获取特征点id
                int camera_id = v % NUM_OF_CAM;               // 获取相机号
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];//第i个特征点的u坐标
                double p_v = img_msg->channels[2].values[i];//第i个特征点的v坐标
                double velocity_x = img_msg->channels[3].values[i];//第i个特征点的x向速度
                double velocity_y = img_msg->channels[4].values[i];//第i个特征点的y向速度
                ROS_ASSERT(z == 1);//判断是否归一化了
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            // 处理图像特征，初始化，紧耦合的非线性优化(处理1帧特征点)
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            // 给RVIZ发送topic
            pubOdometry(estimator, header);   //"odometry" 里程计信息PQV
            pubKeyPoses(estimator, header);   //"key_poses" 关键点三维坐标
            pubCameraPose(estimator, header); //"camera_pose" 相机位姿
            pubPointCloud(estimator, header); //"history_cloud" 点云信息
            pubTF(estimator, header);         //"extrinsic" 相机到IMU的外参
            pubKeyframe(estimator);           //"keyframe_point"、"keyframe_pose" 关键帧位姿和点云
            if (relo_msg != NULL)
                pubRelocalization(estimator); //"relo_relative_pose" 重定位位姿
            // ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();

        // 3. 更新IMU参数
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) // 非线性优化
            update();                                                   // 更新IMU参数[P,Q,V,ba,bg,a,g]
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    // 1.ROS初始化设置节点vins_estimator，设置句柄n
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    // 2.读取参数，设置估计器参数
    readParameters(n);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // 注册可视化相关的发布话题（用于RVIZ显示的Topic）
    registerPub(n);

    // 订阅IMU数据
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    // 订阅特征点数据
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // 订阅复位话题
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    // 订阅重定位匹配得到的数据
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    // 起一个单独线程（VIO主线程）
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
