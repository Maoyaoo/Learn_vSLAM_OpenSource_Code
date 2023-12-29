#include "svo/slamviewer.h"
#include "svo/frame.h"
#include "svo/feature.h"
#include "svo/point.h"
#include "svo/depth_filter.h"

#include "pangolin/gl/gltext.h"

namespace SLAM_VIEWER
{

  Viewer::Viewer(svo::FrameHandlerMono *vo) : _vo(vo)
  {

    mbFinished = false;
    mbFinished2 = false;
    mViewpointX = 0;
    mViewpointY = -0.7;
    mViewpointZ = -1.8;
    mViewpointF = 500;

    mKeyFrameSize = 0.05;
    mKeyFrameLineWidth = 1.0;
    mCameraSize = 0.08;
    mCameraLineWidth = 3.0;

    mPointSize = 5.0;
  }

  Viewer::~Viewer()
  {
    SetFinish();

    while (!CheckFinish2())
      usleep(5000);
  }

  bool Viewer::CheckFinish()
  {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
  }

  bool Viewer::CheckFinish2()
  {
    std::unique_lock<std::mutex> lock(mMutexFinish2);
    return mbFinished2;
  }

  void Viewer::SetFinish()
  {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
  }

  void Viewer::DrawKeyFrames(const bool bDrawKF)
  {

    const float &w = mKeyFrameSize;
    const float h = w * 0.75;
    const float z = w * 0.6;

    // const std::vector<cv::Mat> pos = svo_msg_->getAllPose();
    svo::FramePtr lastframe = _vo->lastFrame();
    if (lastframe == NULL || lastframe->id_ == _drawedframeID)
    {
      // return;
    }
    else // save new pose
    {
      _drawedframeID = lastframe->id_;
      _CurrentPoseTwc = lastframe->T_f_w_.inverse();
      _pos.push_back(_CurrentPoseTwc);
    }

    if (_pos.empty())
      return;

    /*
      int n = pos.size();

      if(bDrawKF)
      {
          for(size_t i = 0; i<n;i++)
          {
            cv::Mat Twc = pos[i].t();

            //std::cout<<Twc<<std::endl;

            glPushMatrix();
            glMultMatrixd(Twc.ptr<GLdouble>(0));
            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();


          }
      }
    */
    glPointSize(2);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);
    for (size_t i = 0; i < _pos.size(); i++)
    {
      Sophus::SE3d Twc = _pos[i];
      glVertex3d(Twc.translation()[0], Twc.translation()[1], Twc.translation()[2]);
    }
    glEnd();
  }

  void Viewer::DrawMapRegionPoints()
  {

    // if(_drawedframeID == 0) return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 1.0);//绘制蓝色点

    //获取最后一帧的附近关键帧集合
    std::vector<std::pair<svo::FramePtr, size_t>> overlap_kfs = _vo->overlap_kfs();
    auto it_frame = overlap_kfs.begin();
    for (size_t i = 0; i < overlap_kfs.size(); ++i, ++it_frame)
    {
      svo::FramePtr frame = it_frame->first;
      //循环每张图像的特征列表。
      for (svo::Features::iterator it = frame->fts_.begin(); it != frame->fts_.end(); ++it)
      {
        if ((*it)->point == NULL)
          continue;
        Eigen::Vector3d Pw = (*it)->point->pos_;
        glVertex3f(Pw[0], Pw[1], Pw[2]);
      }
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0); //绘制红色点
    svo::FramePtr lastframe = _vo->lastFrame();
    for (svo::Features::iterator it = lastframe->fts_.begin(); it != lastframe->fts_.end(); ++it)
    {

      if ((*it)->point == NULL)
        continue;
      Eigen::Vector3d Pw = (*it)->point->pos_;
      glVertex3f(Pw[0], Pw[1], Pw[2]);
    }
    glEnd();
  }

  void Viewer::DrawMapSeeds()
  {
    svo::DepthFilter *df = _vo->depthFilter();
    std::list<svo::Seed> seeds;
    df->getAllSeedsCopy(seeds);

    glLineWidth(mKeyFrameLineWidth);
    glColor3f(0.0f, 1.0f, 1.0f); //青色线  
    glBegin(GL_LINES);
    for (std::list<svo::Seed>::iterator it = seeds.begin(); it != seeds.end(); ++it)
    {
      Eigen::Vector3d xyz_min(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0 / (it->mu + std::sqrt(it->sigma2)))));
      float d = std::max(it->mu - std::sqrt(it->sigma2), 0.000001f);
      Eigen::Vector3d xyz_max(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0 / d)));
      // if((it->mu * it->mu - it->sigma2) > 0)
      {
        glVertex3f(xyz_min[0], xyz_min[1], xyz_min[2]);
        glVertex3f(xyz_max[0], xyz_max[1], xyz_max[2]);
      }
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.5, 1.0, 0.0);
    for (std::list<svo::Seed>::iterator it = seeds.begin(); it != seeds.end(); ++it)
    {

      Eigen::Vector3d xyz_mu(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0 / (it->mu))));
      glVertex3f(xyz_mu[0], xyz_mu[1], xyz_mu[2]);
    }
    glEnd();
  }

  void Viewer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
  {
    const float &w = mCameraSize;
    const float h = w * 0.75;
    const float z = w * 0.6;

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(Twc.m);
#else
    glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
  }

  void Viewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
  {
    if (_drawedframeID != 0) // we have new pose
    {

      Eigen::Matrix3d Rwc = _CurrentPoseTwc.rotationMatrix();
      Eigen::Vector3d twc = _CurrentPoseTwc.translation();

      M.m[0] = Rwc(0, 0);
      M.m[1] = Rwc(1, 0);
      M.m[2] = Rwc(2, 0);
      M.m[3] = 0.0;

      M.m[4] = Rwc(0, 1);
      M.m[5] = Rwc(1, 1);
      M.m[6] = Rwc(2, 1);
      M.m[7] = 0.0;

      M.m[8] = Rwc(0, 2);
      M.m[9] = Rwc(1, 2);
      M.m[10] = Rwc(2, 2);
      M.m[11] = 0.0;

      M.m[12] = twc[0];
      M.m[13] = twc[1];
      M.m[14] = twc[2];
      M.m[15] = 1.0;
    }
    else
      M.SetIdentity();
  }

  void Viewer::run()
  {

    mbFinished = false;
    pangolin::CreateWindowAndBind("map and trajectory viewer", 1024, 768); // 创建一个 Pangolin 窗口，用于显示地图和轨迹等信息

    glEnable(GL_DEPTH_TEST);                           // 启用深度测试。，绘制的像素只有在它们的深度值比之前已经绘制的像素更接近观察者时才会被绘制。
    glEnable(GL_BLEND);                                // 启用颜色混合。颜色混合是一种将新绘制的像素颜色与已有像素颜色进行组合的技术，常用于实现半透明效果
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 设置颜色混合函数。这里使用的是一种常见的混合方式，即源颜色通过源颜色的 alpha 值进行混合。

    // 创建了一个名为 "menu" 的面板（panel），该面板用于容纳一组用户界面元素。
    // SetBounds 函数设置了面板的位置和大小。在这里，它的左上角位于窗口的左上角（坐标(0.0, 0.0)），右下角的纵坐标是窗口底部，横坐标距离底部 175 个像素。
    // 这些变量和面板的创建可以用于在图形界面中添加交互元素，例如复选框，用于控制程序中的一些参数或设置
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", false, true);
    pangolin::Var<bool> menuShowSeeds("menu.Show Seeds", false, true);
    pangolin::Var<bool> menuClose("menu.Close", false, false);

    // std::string ss = "PaoPaoRobot";
    // pangolin::GlText txt = pangolin::GlFont::I().Text(ss.c_str());

    // Define Camera Render Object (for view / scene browsing)
    // 定义相机渲染对象(用于视图/场景浏览)
    // 定义了相机的视图和投影矩阵。在渲染时，你可以将这个渲染状态对象传递给 Pangolin 的渲染函数，以设置相应的 OpenGL 状态。
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    // 添加命名的OpenGL视窗到窗口，并提供3D处理器
    // d_cam 就是一个与 Pangolin 渲染窗口相关联的显示对象，用于显示三维场景。这个对象可以在渲染窗口中用于显示相机视图、地图、轨迹等内容。
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    // 这里的 Twc 可以表示相机在世界坐标系中的变换矩阵。通过 SetIdentity() 函数，将 Twc 设置为单位矩阵，表示相机在初始时刻位于世界坐标系的原点，方向与世界坐标系一致。
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    bool bFollow = true;
    while (!CheckFinish())
    {

      usleep(10000);//延时10毫秒
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//使用 OpenGL 清除颜色缓冲区和深度缓冲区

      // mCurrentPose = svo_msg_->getCurrentPose();
      GetCurrentOpenGLCameraMatrix(Twc);//获取当前相机坐标
      // std::cout<<Twc<<std::endl;

      if (menuFollowCamera && bFollow)
      {
        s_cam.Follow(Twc);
      }
      else if (menuFollowCamera && !bFollow)
      {
        s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
        s_cam.Follow(Twc);
        bFollow = true;
      }
      else if (!menuFollowCamera && bFollow)
      {
        bFollow = false;
      }

      d_cam.Activate(s_cam);
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

      DrawCurrentCamera(Twc);

      // txt.DrawWindow(200,700);

      if (menuShowKeyFrames)
      {
        DrawKeyFrames(menuShowKeyFrames);
      }

      if (menuShowPoints)
      {
        DrawMapRegionPoints();
        // DrawLastFramePoints();
      }
      if (menuShowSeeds)
      {
        DrawMapSeeds();
      }
      pangolin::FinishFrame();

      if (menuClose)
      {
        SetFinish();
      }
    }

    pangolin::BindToContext("SVO: trajactory viewer");
    std::cout << "pangolin close" << std::endl;

    {
      std::unique_lock<std::mutex> lock(mMutexFinish2);
      mbFinished2 = true;
    }
  }

}
