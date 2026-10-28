/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"
#include <cassert>
#include <array>
#include <atomic>
#include <exception>
#include <omp.h>

namespace {
std::atomic<int> next_voxel_plane_id{0};

struct PendingPlaneId {
  size_t point_index, ordinal;
  VoxelPlane* plane;
};
struct PlaneIdRecorder {
  size_t point_index = 0;
  size_t committed = 0;
  std::vector<PendingPlaneId> pending;
};
thread_local PlaneIdRecorder* plane_id_recorder = nullptr;

struct PlaneIdScope {
  PlaneIdRecorder* previous = plane_id_recorder;
  explicit PlaneIdScope(PlaneIdRecorder& recorder) { plane_id_recorder = &recorder; }
  ~PlaneIdScope() { plane_id_recorder = previous; }
  PlaneIdScope(const PlaneIdScope&) = delete;
  PlaneIdScope& operator=(const PlaneIdScope&) = delete;
};

void initializePlaneId(VoxelPlane& plane) {
  if (plane_id_recorder) {
    auto& r = *plane_id_recorder;
    r.pending.push_back({r.point_index, r.pending.size(), &plane});
  } else {
    plane.id_ = next_voxel_plane_id.fetch_add(1, std::memory_order_relaxed);
  }
  plane.is_init_ = true;
}
}

PlaneQueryRecord::PlaneQueryRecord(const VoxelPlane& source)
    : center(source.center_), normal(source.normal_), offset(source.d_), plane(&source),
      normal_covariance(source.plane_var_.topLeftCorner<3, 3>())
{
  const double radius = 3.0 * source.radius_;
  radius_squared = radius * radius;
  covariance_cross = source.plane_var_.topRightCorner<3, 3>() * normal +
      source.plane_var_.bottomLeftCorner<3, 3>().transpose() * normal;
  covariance_constant = normal.dot(source.plane_var_.bottomRightCorner<3, 3>() * normal);
}

void loadVoxelConfig(rclcpp::Node::SharedPtr& nh, VoxelMapConfig &voxel_config)
{
  voxel_config.max_layer_ = CommonUtils::readParam<int>(nh, "max_layer", 1);
  voxel_config.max_voxel_size_ = CommonUtils::readParam<double>(nh, "voxel_size", 0.5);
  voxel_config.planner_threshold_ = CommonUtils::readParam<double>(nh, "min_eigen_value", 0.01);
  voxel_config.sigma_num_ = CommonUtils::readParam<double>(nh, "sigma_num", 3);
  std::vector<int64_t> layer_init_num = CommonUtils::readParam<std::vector<int64_t>>(nh, "layer_init_num", std::vector<long int>{5, 5, 5, 5, 5});

  voxel_config.layer_init_num_.resize(layer_init_num.size());
  for (size_t i = 0; i < layer_init_num.size(); i++) {
    voxel_config.layer_init_num_[i] = int(layer_init_num[i]);
  }
  voxel_config.max_points_num_ = CommonUtils::readParam<int>(nh, "max_points_num", 50);

  voxel_config.map_sliding_en = CommonUtils::readParam<bool>(nh, "map_sliding_en", false);
  voxel_config.half_map_size = CommonUtils::readParam<int>(nh, "half_map_size", 100);
  voxel_config.sliding_thresh = CommonUtils::readParam<double>(nh, "sliding_thresh", 8.0);
}

void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane,
    bool defer_plane_refresh)
{
  struct PlaneFitScope {
    VoxelOctoTree* owner;
    const VoxelPlane* plane;
    bool was_plane;
    bool defer_plane_refresh;
    ~PlaneFitScope() {
      const bool changed = was_plane != plane->is_plane_;
      if (changed) owner->invalidatePlaneSearchCache();
      else if (plane->is_plane_) owner->refreshPlaneSearchRecord(plane, defer_plane_refresh);
    }
  } fit{this, plane, plane->is_plane_, defer_plane_refresh};
  plane->is_plane_ = false;
  plane->is_update_ = true;
  plane->plane_var_.setZero();
  plane->covariance_.setZero();
  plane->center_.setZero();
  plane->normal_.setZero();
  plane->x_normal_.setZero();
  plane->y_normal_.setZero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  plane->d_ = 0;
  plane->min_eigen_value_ = plane->mid_eigen_value_ = plane->max_eigen_value_ = 0;
  if (points.size() < 3) return;

  for (const auto& pv : points) {
    if (!pv.point_w.allFinite() || !pv.var.allFinite()) return;
    plane->center_ += pv.point_w;
  }
  const double count = static_cast<double>(points.size());
  plane->center_ /= count;
  for (const auto& pv : points) {
    const Eigen::Vector3d delta = pv.point_w - plane->center_;
    plane->covariance_.noalias() += delta * delta.transpose();
  }
  plane->covariance_ /= count;
  if (!plane->covariance_.allFinite()) return;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(plane->covariance_);
  if (solver.info() != Eigen::Success) return;
  const Eigen::Vector3d eigenvalues = solver.eigenvalues();
  const Eigen::Matrix3d eigenvectors = solver.eigenvectors();
  if (!eigenvalues.allFinite() || !eigenvectors.allFinite()) return;
  const double gap_tolerance = std::max(1e-12, 1e-6 * eigenvalues[2]);
  if (eigenvalues[0] < -gap_tolerance || eigenvalues[0] >= planer_threshold_ ||
      eigenvalues[1] - eigenvalues[0] <= gap_tolerance) return;

  Eigen::Matrix<double, 6, 6> plane_var = Eigen::Matrix<double, 6, 6>::Zero();
  for (const auto& pv : points) {
    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
    for (int m = 1; m < 3; ++m) {
      F.row(m) = (pv.point_w - plane->center_).transpose() /
          (count * (eigenvalues[0] - eigenvalues[m])) *
          (eigenvectors.col(m) * eigenvectors.col(0).transpose() +
           eigenvectors.col(0) * eigenvectors.col(m).transpose());
    }
    Eigen::Matrix<double, 6, 3> J;
    J.topRows<3>() = eigenvectors * F;
    J.bottomRows<3>() = Eigen::Matrix3d::Identity() / count;
    plane_var.noalias() += J * pv.var * J.transpose();
  }
  if (!plane_var.allFinite()) return;

  plane->plane_var_ = plane_var;
  plane->normal_ = eigenvectors.col(0);
  plane->y_normal_ = eigenvectors.col(1);
  plane->x_normal_ = eigenvectors.col(2);
  plane->min_eigen_value_ = std::max(0.0, eigenvalues[0]);
  plane->mid_eigen_value_ = eigenvalues[1];
  plane->max_eigen_value_ = eigenvalues[2];
  plane->radius_ = std::sqrt(eigenvalues[2]);
  plane->d_ = -plane->normal_.dot(plane->center_);
  plane->is_plane_ = true;
  if (!plane->is_init_) {
    initializePlaneId(*plane);
  }
}

void VoxelOctoTree::init_octo_tree(bool defer_plane_refresh)
{
  if (temp_points_.size() > points_size_threshold_)
  {
    init_plane(temp_points_, plane_ptr_, defer_plane_refresh);
    if (plane_ptr_->is_plane_ == true)
    {
      octo_state_ = 0;
      if (temp_points_.size() > max_points_num_)
      {
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    }
    else
    {
      octo_state_ = 1;
      cut_octo_tree(defer_plane_refresh);
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::cut_octo_tree(bool defer_plane_refresh)
{
  if (layer_ >= max_layer_)
  {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      allocateChild(leafnum, xyz);
    }
    if (leaves_[leafnum]->acceptPointByRange(temp_points_[i])) {
      leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
      leaves_[leafnum]->new_points_++;
    }    
  }
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        leaves_[i]->init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_, defer_plane_refresh);
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cut_octo_tree(defer_plane_refresh);
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv, bool defer_plane_refresh)
{
  if (!init_octo_)
  {
    if (acceptPointByRange(pv)) {
      new_points_++;
      temp_points_.push_back(pv);
      if (temp_points_.size() > points_size_threshold_) { init_octo_tree(defer_plane_refresh); }
    }     
  }
  else
  {
    if (plane_ptr_->is_plane_)
    {
      if (update_enable_)
      {
        if (acceptPointByRange(pv)) {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_ || temp_points_.size() >= max_points_num_)
          {
            init_plane(temp_points_, plane_ptr_, defer_plane_refresh);
            new_points_ = 0;
            if (!plane_ptr_->is_plane_ && layer_ < max_layer_) {
              octo_state_ = 1;
              cut_octo_tree(defer_plane_refresh);
              return;
            }
          }
          if (temp_points_.size() >= max_points_num_)
          {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }     
      }
    }
    else
    {
      if (layer_ < max_layer_)
      {
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateOctoTree(pv, defer_plane_refresh); }
        else
        {
          allocateChild(leafnum, xyz);
          leaves_[leafnum]->UpdateOctoTree(pv, defer_plane_refresh);
        }
      }
      else
      {
        if (update_enable_)
        {
          if (acceptPointByRange(pv)) {
            new_points_++;
            temp_points_.push_back(pv);
            if (new_points_ > update_size_threshold_ || temp_points_.size() >= max_points_num_)
            {
              init_plane(temp_points_, plane_ptr_, defer_plane_refresh);
              new_points_ = 0;
            }
            if (temp_points_.size() >= max_points_num_)
            {
              update_enable_ = false;
              std::vector<pointWithVar>().swap(temp_points_);
              new_points_ = 0;
            }
          }   
        }
      }
    }
  }
}

bool VoxelOctoTree::acceptPointByRange(const pointWithVar& point)
{
  return CommonUtils::decideToAddByRange(temp_points_, point, quater_length_);
}

void VoxelOctoTree::allocateChild(int leafnum, const int xyz[3])
{
  auto child = std::make_unique<VoxelOctoTree>(max_layer_, layer_ + 1,
      layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_, search_root_);
  child->layer_init_num_ = layer_init_num_;
  child->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
  child->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
  child->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
  child->quater_length_ = quater_length_ / 2;
  leaves_[leafnum] = child.release();
}

void VoxelOctoTree::collectSearchPlanes(int layer, int max_layer, VoxelOctoTree* root,
    std::vector<PlaneQueryRecord>& planes)
{
  search_root_ = root;
  search_record_index_ = std::numeric_limits<size_t>::max();
  if (plane_ptr_->is_plane_) {
    search_record_index_ = planes.size();
    planes.emplace_back(*plane_ptr_);
  } else if (layer < max_layer) {
    for (auto* child : leaves_)
      if (child) child->collectSearchPlanes(layer + 1, max_layer, root, planes);
  }
}

void VoxelOctoTree::refreshPlaneSearchRecord(const VoxelPlane* plane,
    bool defer_plane_refresh)
{
  auto* cache = search_root_->plane_search_cache_.get();
  if (!cache || !cache->valid) return;
  size_t index = cache->planes.size();
  if (plane == plane_ptr_ && search_record_index_ < cache->planes.size() &&
      cache->planes[search_record_index_].plane == plane) {
    index = search_record_index_;
  } else {
    for (size_t i = 0; i < cache->planes.size(); ++i)
      if (cache->planes[i].plane == plane) { index = i; break; }
  }
  if (index == cache->planes.size()) return;
  auto& entry = cache->planes[index];
  if (defer_plane_refresh) {
    if (index >= PlaneQueryRecord::pending_end) {
      cache->valid = false;
    } else if (entry.pending_next == PlaneQueryRecord::not_pending) {
      entry.pending_next = cache->pending_head;
      cache->pending_head = static_cast<uint32_t>(index);
    }
    return;
  }
  const uint32_t pending_next = entry.pending_next;
  entry = PlaneQueryRecord(*plane);
  entry.pending_next = pending_next;
}

void VoxelOctoTree::flushPlaneSearchRecords()
{
  auto& cache = *plane_search_cache_;
  if (cache.pending_head == PlaneQueryRecord::pending_end) return;
  while (cache.pending_head != PlaneQueryRecord::pending_end) {
    auto& entry = cache.planes[cache.pending_head];
    cache.pending_head = entry.pending_next;
    entry = PlaneQueryRecord(*entry.plane);
  }
}

void VoxelOctoTree::discardPendingPlaneRefreshes()
{
  if (!plane_search_cache_) return;
  auto& cache = *plane_search_cache_;
  while (cache.pending_head != PlaneQueryRecord::pending_end) {
    auto& entry = cache.planes[cache.pending_head];
    cache.pending_head = entry.pending_next;
    entry.pending_next = PlaneQueryRecord::not_pending;
  }
}

void VoxelOctoTree::preparePlaneSearchCache(int max_layer)
{
  if (search_root_ != this) return;
  if (!plane_search_cache_) plane_search_cache_ = std::make_unique<PlaneSearchCache>();
  auto& cache = *plane_search_cache_;
  if (cache.valid && cache.max_layer == max_layer) {
    flushPlaneSearchRecords();
    return;
  }
  cache.valid = false;
  cache.pending_head = PlaneQueryRecord::pending_end;
  cache.planes.clear();
  collectSearchPlanes(0, max_layer, this, cache.planes);
  cache.max_layer = max_layer;
  cache.valid = true;
}

VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw)
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw) : this;
}

void VoxelMapManager::BuildVoxelMap(const pcl::PointCloud<pcl::PointXYZINormal>& feats_down_world_, const pcl::PointCloud<pcl::PointXYZINormal>& feats_down_body_,
  const Eigen::Quaterniond& q, const Eigen::Vector3d& pos, const Eigen::Matrix3d& cov_pose, double dept_err_, double beam_err_, Eigen::Matrix3d extR_)
{
  const double voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;

  std::vector<pointWithVar> input_points;

  for (size_t i = 0; i < feats_down_world_.size(); i++)
  {
    pointWithVar pv;
    pv.point_w << feats_down_world_.points[i].x, feats_down_world_.points[i].y, feats_down_world_.points[i].z;
    pv.reflectivity = feats_down_world_.points[i].curvature;
    Eigen::Vector3d point_this(feats_down_body_.points[i].x, feats_down_body_.points[i].y, feats_down_body_.points[i].z);
    Eigen::Matrix3d var;
    CommonUtils::calcBodyCov(point_this, dept_err_, beam_err_, var);
    var = (q * extR_) * var * (q * extR_).transpose() + cov_pose;
    pv.var = var;
    input_points.push_back(pv);
  }

  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    const VOXEL_LOCATION position = VOXEL_LOCATION::fromPoint(p_v.point_w, voxel_size);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      if (iter->second->acceptPointByRange(p_v)) {
        voxel_map_[position]->temp_points_.push_back(p_v);
        voxel_map_[position]->new_points_++;
      }
    }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
    }
  }
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter)
  {
    iter->second->init_octo_tree();
    iter->second->preparePlaneSearchCache(config_setting_.max_layer_);
  }
}

void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points,
    int num_threads)
{
  if (num_threads > 1 && input_points.size() >= 256) {
    updateVoxelMapParallel(input_points, num_threads);
    return;
  }
  const double voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  const std::vector<int> &layer_init_num = config_setting_.layer_init_num_;
  std::vector<VoxelOctoTree*> changed_roots;
  VoxelOctoTree* active_root = nullptr;
  struct RootCacheEntry {
    VOXEL_LOCATION position;
    VoxelOctoTree* root = nullptr;
  };
  std::array<RootCacheEntry, 64> root_cache{};
  const auto cache_slot = [](const VOXEL_LOCATION& key) {
    const uint64_t mixed = static_cast<uint64_t>(key.x) * UINT64_C(0x9e3779b185ebca87) ^
        static_cast<uint64_t>(key.y) * UINT64_C(0xc2b2ae3d27d4eb4f) ^
        static_cast<uint64_t>(key.z) * UINT64_C(0x165667b19e3779f9);
    return static_cast<size_t>((mixed ^ (mixed >> 32)) & 63);
  };
  try {
    const size_t plsize = input_points.size();
    for (size_t i = 0; i < plsize; i++)
    {
      const pointWithVar &p_v = input_points[i];
      const VOXEL_LOCATION position = VOXEL_LOCATION::fromPoint(p_v.point_w, voxel_size);
      auto& cache_entry = root_cache[cache_slot(position)];
      const bool cache_hit = cache_entry.root && position == cache_entry.position;
      VoxelOctoTree* root = nullptr;
      if (cache_hit) {
        root = cache_entry.root;
      } else {
        const auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) root = iter->second;
      }
      if (!root) {
        auto octo_tree = std::make_unique<VoxelOctoTree>(max_layer, 0, layer_init_num[0],
            max_points_num, planer_threshold);
        octo_tree->layer_init_num_ = layer_init_num;
        octo_tree->quater_length_ = voxel_size / 4;
        octo_tree->voxel_center_[0] = (0.5 + position.x) * voxel_size;
        octo_tree->voxel_center_[1] = (0.5 + position.y) * voxel_size;
        octo_tree->voxel_center_[2] = (0.5 + position.z) * voxel_size;
        root = octo_tree.get();
        voxel_map_.emplace(position, root);
        octo_tree.release();
      }
      active_root = root;
      if (!cache_hit) {
        cache_entry.position = position;
        cache_entry.root = root;
      }
      root->UpdateOctoTree(p_v, true);
      if (!root->plane_search_cache_)
        root->plane_search_cache_ = std::make_unique<VoxelOctoTree::PlaneSearchCache>();
      auto& cache = *root->plane_search_cache_;
      if ((!cache.valid || cache.max_layer != max_layer ||
           cache.pending_head != PlaneQueryRecord::pending_end) && !cache.queued) {
        changed_roots.push_back(root);
        cache.queued = true;
      }
    }
    for (auto* root : changed_roots) {
      auto& cache = *root->plane_search_cache_;
      cache.queued = false;
      root->preparePlaneSearchCache(max_layer);
    }
  } catch (...) {
    const auto abandon = [](VoxelOctoTree* root) {
      if (!root || !root->plane_search_cache_) return;
      root->plane_search_cache_->valid = false;
      root->discardPendingPlaneRefreshes();
      root->plane_search_cache_->queued = false;
    };
    for (auto* root : changed_roots) abandon(root);
    abandon(active_root);
    throw;
  }
}

void VoxelMapManager::updateVoxelMapParallel(const std::vector<pointWithVar>& input_points,
    int num_threads)
{
  auto& jobs = update_jobs_;
  auto& next = update_next_;
  jobs.clear();
  const size_t end = input_points.size();
  next.resize(end);
  struct Worker {
    PlaneIdRecorder ids;
    std::exception_ptr error;
  };
  const auto abandon = [](VoxelOctoTree* root) {
    if (!root || !root->plane_search_cache_) return;
    root->plane_search_cache_->valid = false;
    root->discardPendingPlaneRefreshes();
    root->plane_search_cache_->queued = false;
  };
  try {
    struct RootCacheEntry { VOXEL_LOCATION position; VoxelOctoTree* root = nullptr; };
    std::array<RootCacheEntry, 64> root_cache{};
    const auto cache_slot = [](const VOXEL_LOCATION& key) {
      const uint64_t mixed = static_cast<uint64_t>(key.x) * UINT64_C(0x9e3779b185ebca87) ^
          static_cast<uint64_t>(key.y) * UINT64_C(0xc2b2ae3d27d4eb4f) ^
          static_cast<uint64_t>(key.z) * UINT64_C(0x165667b19e3779f9);
      return static_cast<size_t>((mixed ^ (mixed >> 32)) & 63);
    };
    const double voxel_size = config_setting_.max_voxel_size_;
    for (size_t i = 0; i < end; ++i) {
      const auto position = VOXEL_LOCATION::fromPoint(input_points[i].point_w, voxel_size);
      auto& entry = root_cache[cache_slot(position)];
      const bool hit = entry.root && entry.position == position;
      VoxelOctoTree* root = hit ? entry.root : nullptr;
      if (!hit) {
        const auto it = voxel_map_.find(position);
        if (it != voxel_map_.end()) root = it->second;
      }
      if (!root) {
        auto octo = std::make_unique<VoxelOctoTree>(config_setting_.max_layer_, 0,
            config_setting_.layer_init_num_[0], config_setting_.max_points_num_,
            config_setting_.planner_threshold_);
        octo->layer_init_num_ = config_setting_.layer_init_num_;
        octo->quater_length_ = voxel_size / 4;
        octo->voxel_center_[0] = (0.5 + position.x) * voxel_size;
        octo->voxel_center_[1] = (0.5 + position.y) * voxel_size;
        octo->voxel_center_[2] = (0.5 + position.z) * voxel_size;
        root = octo.get();
        voxel_map_.emplace(position, root);
        octo.release();
      }
      entry = {position, root};
      size_t job_index = root->update_batch_index_;
      next[i] = end;
      if (job_index >= jobs.size() || jobs[job_index].root != root) {
        job_index = jobs.size();
        jobs.push_back({root, i, i, false});
        root->update_batch_index_ = job_index;
      } else {
        next[jobs[job_index].last] = i;
        jobs[job_index].last = i;
      }
    }
    const int worker_count = std::min(num_threads, static_cast<int>(std::min(
        jobs.size(), static_cast<size_t>(std::numeric_limits<int>::max()))));
    std::vector<Worker> workers(worker_count);
    #pragma omp parallel if(worker_count > 1) num_threads(worker_count)
    {
      auto& worker = workers[omp_get_thread_num()];
      PlaneIdScope id_scope(worker.ids);
      #pragma omp for schedule(dynamic, 8)
      for (size_t j = 0; j < jobs.size(); ++j) {
        auto& job = jobs[j];
        try {
          for (size_t i = job.first; i != end; i = next[i]) {
            worker.ids.point_index = i;
            job.root->UpdateOctoTree(input_points[i], true);
          }
          if (!job.root->plane_search_cache_)
            job.root->plane_search_cache_ = std::make_unique<VoxelOctoTree::PlaneSearchCache>();
        } catch (...) {
          job.failed = true;
          if (!worker.error) worker.error = std::current_exception();
        }
      }
      #pragma omp for schedule(dynamic, 8)
      for (size_t j = 0; j < jobs.size(); ++j) {
        auto& job = jobs[j];
        if (job.failed) continue;
        try {
          auto& cache = *job.root->plane_search_cache_;
          cache.queued = false;
          const bool rebuild = !cache.valid || cache.max_layer != config_setting_.max_layer_;
          if (rebuild || cache.pending_head != PlaneQueryRecord::pending_end) {
            job.root->preparePlaneSearchCache(config_setting_.max_layer_);
          }
        } catch (...) {
          job.failed = true;
          if (!worker.error) worker.error = std::current_exception();
        }
      }
    }

    const auto earlier = [](const PendingPlaneId& a, const PendingPlaneId& b) {
      return a.point_index < b.point_index ||
          (a.point_index == b.point_index && a.ordinal < b.ordinal);
    };
    for (auto& worker : workers)
      std::sort(worker.ids.pending.begin(), worker.ids.pending.end(), earlier);
    for (;;) {
      PlaneIdRecorder* first = nullptr;
      for (auto& worker : workers) {
        auto& ids = worker.ids;
        if (ids.committed < ids.pending.size() &&
            (!first || earlier(ids.pending[ids.committed], first->pending[first->committed]))) first = &ids;
      }
      if (!first) break;
      first->pending[first->committed++].plane->id_ =
          next_voxel_plane_id.fetch_add(1, std::memory_order_relaxed);
    }
    for (const auto& worker : workers) if (worker.error) std::rethrow_exception(worker.error);
  } catch (...) {
    for (const auto& job : jobs) abandon(job.root);
    throw;
  }
}

void VoxelMapManager::build_single_residual(PointData &pv, const VoxelOctoTree *current_octo,
    const int current_layer, bool &is_sucess, double &prob)
{
  const Eigen::Vector3d point(pv.pt_w.x, pv.pt_w.y, pv.pt_w.z);
  build_single_residual_impl(point, pv.w_var, pv, current_octo, current_layer, is_sucess, prob);
}

void VoxelMapManager::build_single_residual_nocov(const Eigen::Vector3d& point, PointData &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess)
{
  double probability = 0.0;
  build_single_residual_impl(point, 0.01 * Eigen::Matrix3d::Identity(), pv,
      current_octo, current_layer, is_sucess, probability);
}

const VoxelPlane* VoxelMapManager::findPlane(const Eigen::Vector3d& point,
    const Eigen::Matrix3d& point_var, const VoxelOctoTree* octo, int layer,
    double& probability) const
{
  if (!octo || !point.allFinite() || !point_var.allFinite()) return nullptr;
  const VoxelPlane* best = nullptr;
  if (const auto* planes = octo->cachedSearchPlanes(layer, config_setting_.max_layer_)) {
    for (const auto& plane : *planes) score_plane(point, point_var, plane, best, probability);
  } else {
    find_best_plane(point, point_var, octo, layer, best, probability);
  }
  return best;
}

void VoxelMapManager::build_single_residual_impl(const Eigen::Vector3d& point, const Eigen::Matrix3d& point_var,
    PointData& result, const VoxelOctoTree* octo, int layer, bool& success, double& probability) const
{
  const VoxelPlane* best = findPlane(point, point_var, octo, layer, probability);
  if (best) {
    success = true;
    result.plane_var_ = best->plane_var_;
    result.normvec = best->normal_;
    result.center_ = best->center_;
    result.dist = best->d_;
  }
}

void VoxelMapManager::find_best_plane(const Eigen::Vector3d& point, const Eigen::Matrix3d& point_var,
    const VoxelOctoTree* octo, int layer, const VoxelPlane*& best, double& probability) const
{
  const VoxelPlane& plane = *octo->plane_ptr_;
  if (!plane.is_plane_) {
    if (layer < config_setting_.max_layer_) {
      for (const auto* child : octo->leaves_) {
        if (child) find_best_plane(point, point_var, child, layer + 1, best, probability);
      }
    }
    return;
  }
  score_plane(point, point_var, PlaneQueryRecord(plane), best, probability);
}

void VoxelMapManager::score_plane(const Eigen::Vector3d& point, const Eigen::Matrix3d& point_var,
    const PlaneQueryRecord& query, const VoxelPlane*& best, double& probability) const
{
  const Eigen::Vector3d delta = point - query.center;
  const Eigen::Vector3d tangential = delta - query.normal * query.normal.dot(delta);
  if (!(tangential.squaredNorm() <= query.radius_squared)) {
    return;
  }
  const double distance = std::abs(query.normal.dot(point) + query.offset);
  const double quadratic = delta.dot(query.normal_covariance * delta);
  const double linear = delta.dot(query.covariance_cross);
  const double plane_variance = quadratic - linear + query.covariance_constant;
  const double point_variance = query.normal.dot(point_var * query.normal);
  double variance = plane_variance + point_variance;
  const double cancellation_scale = std::abs(quadratic) + std::abs(linear) +
      std::abs(query.covariance_constant);
  constexpr double roundoff_guard = 32.0 * std::numeric_limits<double>::epsilon();
  if (!std::isfinite(variance) || variance <= 0.0 || plane_variance < 0.0 ||
      (cancellation_scale > 0.0 &&
       std::abs(plane_variance) <= roundoff_guard * cancellation_scale)) {
    Eigen::Matrix<double, 1, 6> J;
    J.head<3>() = delta.transpose();
    J.tail<3>() = -query.normal.transpose();
    variance = J * query.plane->plane_var_ * J.transpose();
    variance += point_variance;
  }
  if (!std::isfinite(variance) || variance <= 0.0) {
    return;
  }
  const double sigma = std::sqrt(variance);
  if (!(distance < config_setting_.sigma_num_ * sigma)) {
    return;
  }
  if (probability > 0.0 && 1.0 / sigma <= probability) {
    return;
  }
  const double candidate = std::exp(-0.5 * distance * distance / variance) / sigma;
  if (std::isfinite(candidate) && candidate > probability) {
    probability = candidate;
    best = query.plane;
  }
}

void VoxelMapManager::mapSliding(const Eigen::Vector3d& position)
{
  if (!position.allFinite()) throw std::invalid_argument("Map sliding requires a finite position");
  if ((position - last_slide_position).norm() < config_setting_.sliding_thresh) return;

  const VOXEL_LOCATION center = VOXEL_LOCATION::fromPoint(position, config_setting_.max_voxel_size_);
  const int64_t margin = config_setting_.half_map_size;
  if (margin < 0) throw std::invalid_argument("Map half-size must be nonnegative");
  for (int64_t index : {center.x, center.y, center.z}) {
    if (index < std::numeric_limits<int64_t>::min() + margin ||
        index > std::numeric_limits<int64_t>::max() - margin)
      throw std::out_of_range("Sliding map bounds exceed int64_t range");
  }
  clearMemOutOfMap(center.x + margin, center.x - margin,
                  center.y + margin, center.y - margin,
                  center.z + margin, center.z - margin);
  last_slide_position = position;
}

void VoxelMapManager::clearMemOutOfMap(int64_t x_max, int64_t x_min, int64_t y_max, int64_t y_min, int64_t z_max, int64_t z_min)
{
  for (auto it = voxel_map_.begin(); it != voxel_map_.end(); )
  {
    const VOXEL_LOCATION& loc = it->first;
    bool should_remove = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min || loc.z > z_max || loc.z < z_min;
    if (should_remove){
      delete it->second;
      it = voxel_map_.erase(it);
    } else {
      ++it;
    }
  }
}
