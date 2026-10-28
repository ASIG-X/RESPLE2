/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "utils/common_utils.h"
#include <math.h>
#include <pcl/common/io.h>
#include <unistd.h>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

typedef struct VoxelMapConfig
{
  double max_voxel_size_;
  int max_layer_;
  std::vector<int> layer_init_num_;
  size_t max_points_num_;
  double planner_threshold_;
  double sigma_num_;

  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;
} VoxelMapConfig;

typedef struct VoxelPlane
{
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  float radius_ = 0;
  float d_ = 0;
  bool is_plane_ = false;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  int points_size_ = 0;
  bool is_init_ = false;
  int id_ = 0;
  bool is_update_ = false;
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelPlane;

struct PlaneQueryRecord {
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  double radius_squared;
  float offset;
  static constexpr uint32_t not_pending = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t pending_end = not_pending - 1;
  uint32_t pending_next = not_pending;
  const VoxelPlane* plane;
  Eigen::Matrix3d normal_covariance;
  Eigen::Vector3d covariance_cross;
  double covariance_constant;

  explicit PlaneQueryRecord(const VoxelPlane& source);
};

class VOXEL_LOCATION
{
public:
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }

  static VOXEL_LOCATION fromPoint(const Eigen::Vector3d& point, double voxel_size)
  {
    if (!point.allFinite() || !std::isfinite(voxel_size) || voxel_size <= 0.0)
      throw std::invalid_argument("Voxel coordinates must be finite and voxel size must be positive");
    int64_t index[3];
    const double lower = static_cast<double>(std::numeric_limits<int64_t>::min());
    for (int axis = 0; axis < 3; ++axis) {
      const double cell = std::floor(point[axis] / voxel_size);
      if (cell < lower || cell >= -lower)
        throw std::out_of_range("Voxel coordinate exceeds int64_t range");
      index[axis] = static_cast<int64_t>(cell);
    }
    return VOXEL_LOCATION(index[0], index[1], index[2]);
  }
};

namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
}

class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  std::vector<pointWithVar> temp_points_;
  VoxelPlane *plane_ptr_;
  int layer_;
  int octo_state_;
  VoxelOctoTree *leaves_[8];
  double voxel_center_[3];
  std::vector<int> layer_init_num_;
  float quater_length_;
  float planer_threshold_;
  size_t points_size_threshold_;
  int update_size_threshold_;
  size_t max_points_num_;
  int max_layer_;
  int new_points_;
  bool init_octo_;
  bool update_enable_;

  VoxelOctoTree(int max_layer, int layer, size_t points_size_threshold, size_t max_points_num,
      float planer_threshold, VoxelOctoTree* search_root = nullptr)
      : layer_(layer), planer_threshold_(planer_threshold), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num), max_layer_(max_layer),
        search_root_(search_root ? search_root : this)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new VoxelPlane;
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane,
      bool defer_plane_refresh = false);
  void init_octo_tree(bool defer_plane_refresh = false);
  void cut_octo_tree(bool defer_plane_refresh = false);
  void UpdateOctoTree(const pointWithVar &pv,
      bool defer_plane_refresh = false);

  void preparePlaneSearchCache(int max_layer);
  void invalidatePlaneSearchCache() {
    if (search_root_->plane_search_cache_) search_root_->plane_search_cache_->valid = false;
  }
  const std::vector<PlaneQueryRecord>* cachedSearchPlanes(int layer, int max_layer) const {
    const auto* cache = plane_search_cache_.get();
    return search_root_ == this && layer == 0 && cache && cache->valid &&
        cache->pending_head == PlaneQueryRecord::pending_end && cache->max_layer == max_layer
        ? &cache->planes : nullptr;
  }

  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);

private:
  friend class VoxelMapManager;
  struct PlaneSearchCache {
    std::vector<PlaneQueryRecord> planes;
    int max_layer = 0;
    bool valid = false;
    bool queued = false;
    uint32_t pending_head = PlaneQueryRecord::pending_end;
  };
  VoxelOctoTree* search_root_ = this;
  std::unique_ptr<PlaneSearchCache> plane_search_cache_;
  size_t search_record_index_ = std::numeric_limits<size_t>::max();
  size_t update_batch_index_ = std::numeric_limits<size_t>::max();
  bool acceptPointByRange(const pointWithVar& point);
  void allocateChild(int leafnum, const int xyz[3]);
  void refreshPlaneSearchRecord(const VoxelPlane* plane, bool defer_plane_refresh);
  void flushPlaneSearchRecords();
  void discardPendingPlaneRefreshes();
  void collectSearchPlanes(int layer, int max_layer, VoxelOctoTree* root,
      std::vector<PlaneQueryRecord>& planes);
};

void loadVoxelConfig(rclcpp::Node::SharedPtr& nh, VoxelMapConfig &voxel_config);

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelMapConfig config_setting_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;

  Eigen::Vector3d last_slide_position = {0,0,0};

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
  {
  };

  void BuildVoxelMap(const pcl::PointCloud<pcl::PointXYZINormal>& feats_down_world_, const pcl::PointCloud<pcl::PointXYZINormal>& feats_down_body_,
      const Eigen::Quaterniond& q, const Eigen::Vector3d& pos, const Eigen::Matrix3d& cov_pose, double dept_err_, double beam_err_, Eigen::Matrix3d extR_);

  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points,
      int num_threads = 1);

  void build_single_residual(PointData &pv, const VoxelOctoTree *current_octo, const int current_layer,
      bool &is_sucess, double &prob);
  void build_single_residual_nocov(const Eigen::Vector3d& p_w, PointData &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess);

  const VoxelPlane* findPlane(const Eigen::Vector3d& point, const Eigen::Matrix3d& point_var,
      const VoxelOctoTree* octo, int layer, double& probability) const;

  void mapSliding(const Eigen::Vector3d& position);
  void clearMemOutOfMap(int64_t x_max, int64_t x_min, int64_t y_max, int64_t y_min, int64_t z_max, int64_t z_min);

private:
  struct UpdateRootJob {
    VoxelOctoTree* root;
    size_t first, last;
    bool failed = false;
  };
  std::vector<UpdateRootJob> update_jobs_;
  std::vector<size_t> update_next_;
  void updateVoxelMapParallel(const std::vector<pointWithVar>& input_points,
      int num_threads);
  void build_single_residual_impl(const Eigen::Vector3d& point, const Eigen::Matrix3d& point_var,
      PointData& result, const VoxelOctoTree* octo, int layer, bool& success, double& probability) const;
  void find_best_plane(const Eigen::Vector3d& point, const Eigen::Matrix3d& point_var,
      const VoxelOctoTree* octo, int layer, const VoxelPlane*& best, double& probability) const;
  void score_plane(const Eigen::Vector3d& point, const Eigen::Matrix3d& point_var,
      const PlaneQueryRecord& query, const VoxelPlane*& best, double& probability) const;
};

#endif
