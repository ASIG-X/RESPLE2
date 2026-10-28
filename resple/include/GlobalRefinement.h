#pragma once
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <Eigen/Core>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rclcpp/timer.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/path.hpp>

#include <rclcpp/service.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <pcl/kdtree/kdtree_flann.h>

#include "utils/common_utils.h"
#include "camera/ImgPreprocessing.h"
#include "SplineState.h"
#include "utils/global_map_publisher.h"
#include <pcl/filters/voxel_grid.h>

extern SplineState spline_global;

struct Linearizer;

class GlobalEstimator
{
  public:
    Eigen::aligned_vector<LoopData> lc_in_opt;
    SplineState spline_opt;

    GlobalEstimator();
    void initOptParam(rclcpp::Node::SharedPtr &nh);
    bool process(const Eigen::aligned_vector<LoopData>& loop_closures);
    void getDrift(Eigen::Quaterniond& q, Eigen::Vector3d& t);
    void lock_spline();
    void unlock_spline();
    std::shared_lock<std::shared_mutex> splineReadLock() {
        return std::shared_lock<std::shared_mutex>(m_spline);
    }
    std::unique_lock<std::shared_mutex> splineWriteLock() {
        return std::unique_lock<std::shared_mutex>(m_spline);
    }
    void lock_spline_shared();
    void unlock_spline_shared();
  private:
    int64_t opt_start_t_ns;
    Eigen::aligned_vector<LoopData> odom_meas;
    std::map<std::pair<int64_t, int64_t>, LoopData> control_point_odom_cache;
    OptParameters param;
    Eigen::Quaterniond rot_drift = Eigen::Quaterniond::Identity();
    Eigen::Vector3d trans_drift = Eigen::Vector3d::Zero();

    struct DriftState {
        Eigen::Quaterniond rot = Eigen::Quaterniond::Identity();
        Eigen::Vector3d trans = Eigen::Vector3d::Zero();
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    std::shared_ptr<const DriftState> drift_snapshot = std::make_shared<const DriftState>();

    std::shared_mutex m_spline;

    int max_iter = 20;

    Eigen::VectorXd jacobian_scaling;
    size_t global_state_size;
    double radius = 1e4;
    double max_radius = 1e16;
    double min_radius = 1e-32;
    double decrease_factor = 2.0;

    const double min_lm_diagonal = 1e-6;
    const double max_lm_diagonal = 1e32;
    const double min_relative_decrease = 1e-3;

    const int max_consecutive_nonmonotonic_steps_ = 0;
    double minimum_cost_;
    double current_cost_;
    double reference_cost_;
    double candidate_cost_;
    double accumulated_reference_model_cost_change_;
    double accumulated_candidate_model_cost_change_;
    int num_consecutive_nonmonotonic_steps_;

    const double gradient_tolerance = 1e-6;
    const double cost_tolerance = 1e-6;
    const double translation_step_tolerance = 1e-8;
    const double rotation_step_tolerance = 1e-8;

    bool optimization();
    void updateControlPointOdometryMeasurements();
    void getStateVector(Eigen::VectorXd *x, const Eigen::VectorXd *delta_x = nullptr);
    bool globalOptimize(const int iter, bool& terminated, Linearizer& lopt);
    void applyIncFull(Eigen::VectorXd& inc_full);
    bool isStepSmall(const Eigen::VectorXd& inc_full) const;
    double stepQuality(const double cost, const double model_cost_change) const;
    void stepAccepted(const double cost, const double model_cost_change);

};

extern LiDARKFs<pcl::PointXYZINormal> keyframes;
extern GlobalMapPublisher global_map_pub;

class PoseVisualization
{
public:
    PoseVisualization() : line_width(0.05) {};

    void setColor(float r, float g, float b, float a = 1.0) {
        line_color.r = r;
        line_color.g = g;
        line_color.b = b;
        line_color.a = a;
    }

    void setLineWidth(double lw) {
        line_width = lw;
    }

    void pubCorresp(const Eigen::Vector3d& p, const Eigen::Vector3d& norm_vec, const std_msgs::msg::Header& header, int idx,
                    visualization_msgs::msg::MarkerArray& marker_array) const
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.id = idx;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = line_width;
        marker.pose.position.x = 0.0;
        marker.pose.position.y = 0.0;
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        geometry_msgs::msg::Point p0, p1;
        eigen2Point(p, p0);
        eigen2Point(norm_vec + p, p1);
        marker.points.push_back(p0);
        marker.points.push_back(p1);
        marker.colors.push_back(line_color);
        marker.colors.push_back(line_color);
        marker_array.markers.push_back(marker);
    }

private:
    std_msgs::msg::ColorRGBA line_color;
    double line_width;

    void eigen2Point(const Eigen::Vector3d& v, geometry_msgs::msg::Point& p) const
    {
        p.x = v.x();
        p.y = v.y();
        p.z = v.z();
    }
};

class LoopDetection
{
  public:
    GlobalEstimator gest;
    std::mutex m_lc_buff;
    Eigen::aligned_vector<LoopData> lc_buf;  

    LoopDetection();
    void init(rclcpp::Node::SharedPtr &nh);
    void correction();
    bool consumePathCorrection() { return path_correction_pending.exchange(false); }
    void requestStop() { if_finished.store(true); }
  private:
    bool rgb_mapping_enabled = false;
    std::atomic<bool> path_correction_pending{false};
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr client_reset;    
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_update_global_map;
    std::mutex m_map_update;

    PoseVisualization lc_vis;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_lc;
    const std::string odom_id = "world";
    std::atomic<bool> if_finished{false};
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_drift;

    struct MarkerSnapshot {
        Eigen::aligned_vector<LoopData> loop_closures;
    };
    std::shared_ptr<const MarkerSnapshot> marker_snapshot;
    rclcpp::TimerBase::SharedPtr marker_publish_timer;

    void updateGlobalMap(const Eigen::aligned_vector<PoseData>& poses,
                         const std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr>& clouds,
                         const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr>& rgb_clouds);
    void updateGlobalMapService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                const std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void resetService();
    void correctKeyframeUpdateMap();
    void publishLoopClosureConnection(const Eigen::aligned_vector<LoopData>& loop_closure_connection) const;
};
