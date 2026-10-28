#ifndef VISUAL_FRONTEND_H_
#define VISUAL_FRONTEND_H_

#include "utils/common_utils.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <boost/noncopyable.hpp>
#include "camera/abstract_camera.h"

class VisualPoint : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  const int id;
  int num_track = 0;
  int type_track = 0;
  Eigen::Vector3d pos_;
  bool if_valid = true;
  Eigen::Vector2d m_img_pt_in_last_frame;

  VisualPoint(const Eigen::Vector3d &pos, const int id_)
      : id(id_), pos_(pos) {};
  ~VisualPoint() {};
};

class VisualFrontend
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int grid_size = 5;
  vk::AbstractCamera *cam;
  std::vector<int> grid_num;
  std::vector<float> scan_value;

  int width, height, grid_n_width, grid_n_height, length;
  int border;

  bool if_first_frame = true;
  bool plot_flag;

  Eigen::aligned_vector<Eigen::Vector3d> append_voxel_points;
  Eigen::aligned_vector<Eigen::Vector2d> append_img_points;
  Eigen::Quaterniond q_f_w_;
  Eigen::Vector3d t_f_w_;
  bool has_frame_ = false;
  cv::Mat img_rgb;
  cv::Mat img_prev;

  cv::Mat gray_scratch_;
  std::vector<cv::Mat> previous_pyramid_, current_pyramid_;
  int previous_pyramid_max_level_ = -1;
  std::vector<uchar> flow_status_, reverse_status_;
  std::vector<float> flow_error_;
  std::vector<cv::Point2f> reverse_points_;

  cv::Mat noted_img2;

  int MIN_DIST = 30;
  const int MAX_CNT = 400;
  std::vector<int> m_old_ids;
  std::vector<cv::Point2f> m_last_tracked_pts, m_current_tracked_pts;
  std::vector<VisualPointPtr> m_rgb_pts_ptr_vec_in_last_frame;
  std::map<VisualPointPtr, cv::Point2f> m_map_rgb_pts_in_last_frame_pos;
  int tracking_rows_ = 0;
  int tracking_cols_ = 0;

  double max_allow_err = 20;

  int cnt_id = 0;
  double thresh_lidar_shiTomasi = 3;

  const cv::Scalar color_map[4] = {
      cv::Scalar(255,   0,   0),
      cv::Scalar(  0, 255,   0),
      cv::Scalar(  0,   0, 255),
      cv::Scalar(  255,   0, 255)
  };  

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN,
    TYPE_TRACK
  };

  VisualFrontend();
  ~VisualFrontend();
  VisualFrontend(const VisualFrontend&) = delete;
  VisualFrontend& operator=(const VisualFrontend&) = delete;

  void updateAndAppendTrackPts();
  void generateVisualPoints(cv::Mat img, const Eigen::aligned_vector<pcl::PointXYZINormal>& pg_w,
      bool render_debug = true);
  void initialize();
  void resetGrid();
  void initNewFrame(cv::Mat& latest_img, double scale);

  Eigen::Vector2d w2c(const Eigen::Vector3d &xyz_w) const;
  Eigen::Vector3d w2f(const Eigen::Vector3d &xyz_w) const;

  Eigen::Vector3f getInterpolatedPixel(cv::Mat img, Eigen::Vector2d pc);
  template <typename T>
  void reduceVector(std::vector<T> &v, const std::vector<uchar>& status);
  double distance(const cv::Point2f& pt1, const cv::Point2f& pt2);
  void trackImg();
  void updateLastTrackingVectorAndIds();
  void setMask();
  void applyTrackingResults();

private:
  int gridIndex(const Eigen::Vector2d& pixel) const;
};

#endif
