#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

// 判断跟踪的特征点是否在图像边界内
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x); // cvRound()：返回跟参数最接近的整数值，即四舍五入；
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 去除无法跟踪的特征点。status为0的点直接跳过，否则v[j++]=v[i]留下来，最后v.resize(j)根据最新的j安排内存。
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}


// 对跟踪到的特征点，按照被追踪到的次数排序并依次选点，
// 使用mask进行类似非极大抑制，半径为30，去掉密集点，使特征点分布均匀。
void FeatureTracker::setMask()
{
    // 如果是鱼眼镜头直接clone即可，否则创建空白板
    if (FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

    // 倾向于留下被追踪时间很长的特征点
    // 构造(cnt，pts，id)序列，（追踪次数，当前特征点坐标，id）
    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 对光流跟踪到的特征点forw_pts，按照被跟踪到的次数cnt从大到小排序（lambda表达式）
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; }); // a.first指的追踪次数track_cnt

    // 清空cnt，pts，id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255) // 这个特征点对应的mask值为255，表明点是白色的，还没占有
        {
            // 则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);

            // 在mask中将当前特征点周围半径为MIN_DIST的区域设置为0（变成黑色），后面不再选取该区域内的点（使跟踪点不集中在一个区域上）
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 添将新检测到的特征点n_pts，ID初始化-1，跟踪次数1
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);      // 新提取的特征点id初始化为-1
        track_cnt.push_back(1); // 新提取的特征点被跟踪的次数初始化为1
    }
}

/**
 * @brief
 *
 * @param[in] _img 输入图像
 * @param[in] _cur_time 图像的时间戳
 * 1、图像均衡化预处理
 * 2、光流追踪
 * 3、提取新的特征点（如果发布）
 * 4、所有特征点去畸变，计算速度
 */
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    // 1.如果EQUALIZE=1，表示太亮或太暗，进行直方图均衡化处理
    if (EQUALIZE)
    {
        // 自适应直方图均衡
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    // 2. 判断当前帧图像forw_img是否为空，为空，说明当前是第一次读入图像数据
    if (forw_img.empty())
    {
        // 将读入的图像赋给当前帧forw_img，同时还赋给prev_img、cur_img
        //首帧处理
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        // 否则，说明之前就已经有图像读入，只需要更新当前帧forw_img的数据
        forw_img = img;
    }

    // 此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();

    // 前一帧有特征点,进行光流跟踪（首帧不做处理）
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        // 3. 调用cv::calcOpticalFlowPyrLK()对前一帧的特征点cur_pts进行LK金字塔光流跟踪，得到forw_pts
        // status标记了从前一帧cur_img到forw_img特征点的跟踪状态，无法被追踪到的点标记为0
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i])) // 将当前帧跟踪的位于图像边界外的点标记为0（通过图像边界剔除outlier）
                status[i] = 0;

        // 4. 根据status,把跟踪失败的点剔除
        // 记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        // prev_pts和cur_pts中的特征点是一一对应的
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);

        //将光流跟踪后的点的ids和跟踪次数track_cnt，根据跟踪的状态(status)进行重组
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    // 5. 光流追踪成功,特征点被成功跟踪的次数就加1
    // 更新跟踪次数，数值代表被追踪的次数，数值越大，说明被追踪的就越久
    for (auto &n : track_cnt)
        n++;

    // PUB_THIS_FRAME=1 需要发布特征点
    if (PUB_THIS_FRAME)
    {
        // 6. rejectWithF()通过基本矩阵剔除outliers(对极约束)，首帧不处理
        rejectWithF();

        // 7. setMask()：该函数先排序跟踪次数多的点，然后使用圆形mask均匀了特征点，
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        // 8. 寻找新的特征点 goodFeaturesToTrack()
        ROS_DEBUG("detect feature begins");
        TicToc t_t;

        // 计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            //在mask中不为0的区域检测新的特征点
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            /**
             *void cv::goodFeaturesToTrack(    
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double  minDistance,            返回角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点(mask的白色区域)
             *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04                Harris角点检测需要的k值
             *)
             */
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        // 9. addPoints()向forw_pts添加新的追踪点
        ROS_DEBUG("add feature begins");
        TicToc t_a;

        // 添将新检测到的特征点n_pts添加到forw_pts中，id初始化-1,track_cnt初始化为1.
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    // 10. 更新帧、特征点
    // 当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;

    // 把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    cur_img = forw_img;
    cur_pts = forw_pts;

    // 11. 根据不同的相机模型去畸变矫正和转换到归一化坐标系上，计算速度
    //这个函数干了2件事，第一个是获取forw时刻去畸变的归一化坐标(这个是要发布到rosmsg里的points数据)，另一个是获取forw时刻像素运动速度
    undistortedPoints();
    prev_time = cur_time;
}

// 通过F矩阵去除outliers
// 首先把特征点转化到归一化相机坐标系，然后计算F矩阵，再根据status清除为0的特征点。
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8) // 当前帧（追踪上）特征点数量足够多
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;

        // 1.遍历所有特征点，转化为归一化相机坐标系
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++) // 遍历上一帧所有特征点
        {
            Eigen::Vector3d tmp_p;
            // 对于PINHOLE（针孔相机）可将像素坐标转换到归一化平面并去畸变。(去畸变然后投影到相机系)
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // 这里用一个虚拟相机，原因同样参考https://github.com/HKUST-Aerial-Robotics/VINS-Mono/issues/48
            // 这里有个好处就是对F_THRESHOLD和相机无关
            // 投影到虚拟相机的像素坐标系
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        // 2. 调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算F矩阵，需要归一化相机系，z=1
        //这两幅图像之间的基础矩阵。基础矩阵在立体视觉、相机标定和3D重建等计算机视觉任务中非常重要。
        //通过使用 RANSAC 算法，它能够有效地过滤掉噪声和错误匹配点对，确保基础矩阵的准确性。
        //status 中的值为 1 表示对应的点对是内点，为 0 则表示是外点。
        //函数传入的是归一化坐标，那么得到的是本质矩阵E，如果传入的是像素坐标，那么得到的是基础矩阵。
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        // 3. 根据status删除一些特征点
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

// 更新特征点id
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

// 读取相机内参
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

// 显示去畸变矫正后的特征点,name为图像帧名称
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

// 对特征点的图像坐标去畸变矫正，并计算每个角点的速度
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    // cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());

    // 1.归一化相机坐标系
    for (unsigned int i = 0; i < cur_pts.size(); i++) // 遍历所有特征点
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;

        // 根据不同的相机模型将二维坐标转换到归一化相机三维坐标系
        m_camera->liftProjective(a, b);

        // 再延伸到深度归一化平面上
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    // 2.计算每个特征点的速度到pts_velocity
    // caculate points velocity
    if (!prev_un_pts_map.empty()) // 2.1 地图不是空的判断是否新的点
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1) // 2.2 通过id判断不是最新的点
            {
                std::map<int, cv::Point2f>::iterator it; // map的迭代器
                it = prev_un_pts_map.find(ids[i]);       // 找到对应的id

                if (it != prev_un_pts_map.end()) // 2.3 在地图中寻找是否出现过id判断是否最新点
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt; // 当前帧-地图点上一帧
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y)); // 之前出现过，push_back即可
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0)); // 之前没出现过，先放进去但是速度为0
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0)); // 是最新的点，速度为0
            }
        }
    }
    else // 如果prev_un_pts_map是空的，速度是0
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    // 更新地图
    prev_un_pts_map = cur_un_pts_map;
}
