#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <thread>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <limits>
#include <deque>
#include <string>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/int64.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include "livox_ros_driver/msg/custom_msg.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "livox_interfaces/msg/custom_msg.hpp"
#include "estimate_msgs/msg/estimate.hpp"
#include "estimate_msgs/msg/loop_closure.hpp"
#include "SplineState.h"
#include "GlobalRefinement.h"
#include "utils/global_map_publisher.h"
#include "utils/math_tools.h"
#include "utils/common_utils.h"

LiDARKFs<pcl::PointXYZINormal> keyframes;
SplineState spline_global;
GlobalMapPublisher global_map_pub;

template<typename PointType>
class MappingBase
{
  public:

    std::mutex mtx;    
    LidarConfig lidar;

    MappingBase(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : lidar(lidar_config)
    {
        const bool if_custom_ds = CommonUtils::readParam<bool>(nh, "if_custom_ds", false);
        ds_filter_each_scan.setUseCustom(if_custom_ds);
        ds_filter_kf_scan.setUseCustom(if_custom_ds);
        ds_filter_kf_rgb.setUseCustom(if_custom_ds);
        double ds_gm_scan_voxel = CommonUtils::readParam<double>(nh, "ds_gm_scan_voxel", 0.1);
        if (!std::isfinite(ds_gm_scan_voxel) || ds_gm_scan_voxel < 0.0) {
            throw std::invalid_argument("ds_gm_scan_voxel must be finite and nonnegative (0 disables downsampling)");
        }
        downsample_scans = ds_gm_scan_voxel > 0.0;
        if (downsample_scans) {
            ds_filter_each_scan.setLeafSize(ds_gm_scan_voxel, ds_gm_scan_voxel, ds_gm_scan_voxel);
        }
        pc_last.reset(new typename pcl::PointCloud<PointType>());
        pc_last_ds.reset(new typename pcl::PointCloud<PointType>());

        if_loop_closure = CommonUtils::readParam<bool>(nh, "if_loop_closure");
        if (if_loop_closure) {
            keyframes.thresh_kf_translation = CommonUtils::readParam<double>(nh, "thresh_kf_translation");
            double ds_kf_voxel = CommonUtils::readParam<double>(nh, "ds_kf_voxel");
            ds_filter_kf_scan.setLeafSize(ds_kf_voxel, ds_kf_voxel, ds_kf_voxel);
            ds_filter_kf_rgb.setLeafSize(ds_kf_voxel, ds_kf_voxel, ds_kf_voxel);
            accumulated_scans_ds.reset(new typename pcl::PointCloud<PointType>());
        } 
    }

    void processScan(SplineState* spl, const int64_t spl_window_st_ns,
                     const std::function<void()>& lock_spline,
                     const std::function<void()>& unlock_spline,
                     RgbMapColorizer* rgb_colorizer = nullptr)
    {
        int64_t t_end_ns = 0;
        rclcpp::Rate rate(100);
        if (pc_L_buff.empty()) {
            rate.sleep();
        }
        while (!pc_L_buff.empty()) {
            mtx.lock();
            const auto& scan = pc_L_buff.front();
            if (scan.empty()) {
                pc_L_buff.pop_front();
                mtx.unlock();
                continue;
            }
            float last_offset_ms = 0.0f;
            for (const auto& point : scan)
                if (std::isfinite(point.intensity)) last_offset_ms = std::max(last_offset_ms, point.intensity);
            t_end_ns = static_cast<int64_t>(scan.header.stamp) + CommonUtils::ms2ns(last_offset_ms);
            if (t_end_ns < spl->minTimeNs()) {
                pc_L_buff.pop_front();
                mtx.unlock();
            } else if (t_end_ns <= spl_window_st_ns) {
                typename pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>());

                lock_spline();
                transformCloud(pc_L_buff.front(), spl, pc);
                auto colored = rgb_colorizer ? rgb_colorizer->colorize(*pc, spl, spl_window_st_ns) : nullptr;
                if (if_loop_closure && int64_t(pc_L_buff.front().header.stamp) > spl->minTimeNs()) {
                    keyframes.m_frame_buff.lock();
                    accumulateKF(spl, pc_L_buff.front().header.stamp, pc, colored);
                    keyframes.m_frame_buff.unlock();
                }
                unlock_spline();
                pc_L_buff.pop_front();
                mtx.unlock();
                if (rgb_colorizer && colored && !colored->empty()) {
                    rgb_colorizer->publish(colored);
                } else if (!pc->points.empty()) {
                    global_map_pub.submit(pc);
                }
            } else if (int64_t(pc_L_buff.front().header.stamp) <= spl_window_st_ns) {
                typename pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>());
                typename pcl::PointCloud<PointType> pc_remain;

                lock_spline();
                splitAndTransformCloud(pc_L_buff.front(), spl, spl_window_st_ns, pc, pc_remain);
                auto colored = rgb_colorizer ? rgb_colorizer->colorize(*pc, spl, spl_window_st_ns) : nullptr;
                if (if_loop_closure && int64_t(pc_L_buff.front().header.stamp) > spl->minTimeNs()) {
                    keyframes.m_frame_buff.lock();
                    accumulateKF(spl, pc_L_buff.front().header.stamp, pc, colored);
                    keyframes.m_frame_buff.unlock();
                }
                unlock_spline();
                if (pc_remain.empty()) {
                    pc_L_buff.pop_front();
                } else {
                    pc_L_buff.front() = pc_remain;
                }
                mtx.unlock();
                if (rgb_colorizer && colored && !colored->empty()) {
                    rgb_colorizer->publish(colored);
                } else if (!pc->points.empty()) {
                    global_map_pub.submit(pc);
                }
                break;
            } else {
                mtx.unlock();
                rate.sleep();
                break;
            }
        }
    }

    int64_t earliestPendingScanTime(int64_t boundary_ns)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto& scan : pc_L_buff) {
            boundary_ns = std::min(boundary_ns, static_cast<int64_t>(scan.header.stamp));
        }
        return boundary_ns;
    }

  private:
    resple2::VoxelDownsampler<PointType> ds_filter_kf_scan;
    resple2::VoxelDownsampler<pcl::PointXYZRGB> ds_filter_kf_rgb;
    typename pcl::PointCloud<PointType>::Ptr accumulated_scans_ds;

    PointType transformPoint(int64_t time_ns, const SplineState* spl, const PointType& pt_in) const
    {     
        Eigen::Quaterniond q_itp;
        Eigen::Vector3d t_itp;
        spl->itpQuaternionFromDelta(time_ns, &q_itp);
        t_itp = spl->itpPosition(time_ns);
        Eigen::Vector3d p_body(pt_in.x, pt_in.y, pt_in.z);
        Eigen::Vector3d p_imu(lidar.q_bl * p_body + lidar.t_bl);
        Eigen::Vector3d p_global(q_itp * p_imu + t_itp);
        PointType point_world;
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = pt_in.intensity;
        point_world.curvature = pt_in.curvature;
        return point_world;
    }    

    void transformCloud(const typename pcl::PointCloud<PointType>& pc_in, SplineState* spl,
                       typename pcl::PointCloud<PointType>::Ptr pc_out) const
    {
        int64_t time_begin = rclcpp::Time(pc_in.header.stamp).nanoseconds();
        pc_out->header = pc_in.header;
        pc_out->points.reserve(pc_in.size());
        for (size_t i = 0; i < pc_in.size(); i++) {
            const PointType& pt = pc_in.points[i];
            if (!pcl::isFinite(pt))
                continue;
            int64_t t_ns = int64_t(pt.intensity * float(1e6)) + time_begin;
            if (t_ns >= spl->minTimeNs() && t_ns <= spl->maxTimeNs()) {
                pc_out->push_back(transformPoint(t_ns, spl, pt));
            }
        }
        pc_out->width = pc_out->points.size();
        pc_out->height = 1;
        pc_out->is_dense = false;
    }

    void splitAndTransformCloud(const typename pcl::PointCloud<PointType>& pc_in, SplineState* spl,
                       int64_t boundary_ns, typename pcl::PointCloud<PointType>::Ptr pc_out,
                       typename pcl::PointCloud<PointType>& pc_remain) const
    {
        int64_t time_begin = rclcpp::Time(pc_in.header.stamp).nanoseconds();
        pc_out->header = pc_in.header;
        pc_out->points.reserve(pc_in.size());
        pc_remain.header = pc_in.header;
        pc_remain.points.reserve(pc_in.size());
        for (size_t i = 0; i < pc_in.size(); i++) {
            const PointType& pt = pc_in.points[i];
            int64_t t_ns = int64_t(pt.intensity * float(1e6)) + time_begin;
            if (t_ns > boundary_ns) {
                pc_remain.push_back(pt);
                continue;
            }
            if (!pcl::isFinite(pt))
                continue;
            if (t_ns >= spl->minTimeNs() && t_ns <= spl->maxTimeNs()) {
                pc_out->push_back(transformPoint(t_ns, spl, pt));
            }
        }
        pc_out->width = pc_out->points.size();
        pc_out->height = 1;
        pc_out->is_dense = false;
        pc_remain.width = pc_remain.points.size();
        pc_remain.height = 1;
        pc_remain.is_dense = pc_in.is_dense;
    }

    bool decideKeyframe(SplineState* spl, int64_t frame_time_ns)
    {
        Eigen::Quaterniond orient_interp;
        Eigen::Vector3d t_interp = spl->itpPosition(frame_time_ns);
        spl->itpQuaternionFromDelta(frame_time_ns, &orient_interp);
        const PoseData odom_pose_curr(frame_time_ns, orient_interp, t_interp);

        keyframes.accumulated_odoms.push_back(odom_pose_curr);
        if (!keyframes.has_previous_odom) {
            keyframes.odom_pose_prev = odom_pose_curr;
            keyframes.has_previous_odom = true;
            return false;
        }
        keyframes.kf_distance_accumulated += (odom_pose_curr.pos - keyframes.odom_pose_prev.pos).norm();
        keyframes.odom_pose_prev = odom_pose_curr;
        if (keyframes.kf_distance_accumulated > keyframes.thresh_kf_translation) {
            keyframes.kf_distance_accumulated = 0;
            return true;
        } else {
            return false;
        }
    }    

    void accumulateKF(SplineState* spl, int64_t frame_time_ns,
                      typename pcl::PointCloud<PointType>::Ptr pc,
                      const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& colored)
    {
        *keyframes.accumulated_scans += *pc;
        if (colored) *keyframes.accumulated_rgb += *colored;
        if (decideKeyframe(spl, frame_time_ns)) {
            int idx_near = keyframes.accumulated_odoms.size()/2;
            PoseData central_point = keyframes.accumulated_odoms[idx_near];

            Eigen::Quaterniond q_wi = central_point.orient;
            Eigen::Vector3d t_wi = central_point.pos;
            accumulated_scans_ds->clear();
            ds_filter_kf_scan.setInputCloud(keyframes.accumulated_scans);
            ds_filter_kf_scan.filter(*accumulated_scans_ds);

            Eigen::Quaternionf q_iw = q_wi.inverse().cast<float>();
            Eigen::Vector3f t_iw = -(q_wi.inverse()*t_wi).cast<float>();
            typename pcl::PointCloud<PointType>::Ptr transf_scans_body = CommonUtils::transformCloud(accumulated_scans_ds, q_iw, t_iw);
            auto rgb_body = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            if (!keyframes.accumulated_rgb->empty()) {
                pcl::PointCloud<pcl::PointXYZRGB> rgb_downsampled;
                ds_filter_kf_rgb.setInputCloud(keyframes.accumulated_rgb);
                ds_filter_kf_rgb.filter(rgb_downsampled);
                pcl::transformPointCloud(rgb_downsampled, *rgb_body, t_iw, q_iw);
            }
            rgb_body->header.stamp = central_point.time_ns;
            keyframes.keyframePoses.push_back(central_point);
            keyframes.keyframeLaserClouds.push_back(transf_scans_body);
            keyframes.keyframeRgbClouds.push_back(rgb_body);
            keyframes.accumulated_odoms.clear();
            keyframes.accumulated_scans->clear();
            keyframes.accumulated_rgb->clear();
        }
    }

  protected:
    void prepareVisualizationScan()
    {
        if (downsample_scans) {
            ds_filter_each_scan.setInputCloud(pc_last);
            pc_last_ds->clear();
            ds_filter_each_scan.filter(*pc_last_ds);
        } else {
            *pc_last_ds = *pc_last;
        }
    }

    bool downsample_scans = true;
    std::deque<typename pcl::PointCloud<PointType>> pc_L_buff;
    typename pcl::PointCloud<PointType>::Ptr pc_last;
    typename pcl::PointCloud<PointType>::Ptr pc_last_ds;
    resple2::VoxelDownsampler<PointType> ds_filter_each_scan;
    const std::string frame_id = "body";
    const std::string odom_id = "world";
    bool if_loop_closure;    

};

class OusterBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  OusterBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config)
    {
        pc_subscription_ouster = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&OusterBuff::ousterLidarCallback, this, std::placeholders::_1));         
        double lidar_time_offset = CommonUtils::readParam<double>(nh, "lidar_time_offset", 0.0);
        time_offset = 1e9*lidar_time_offset;        
    }

    void ousterLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr ouster_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<ouster_ros::Point>::Ptr pc_last_ouster(new pcl::PointCloud<ouster_ros::Point>());
        pcl::fromROSMsg(*ouster_msg_in, *pc_last_ouster);
        size_t plsize = pc_last_ouster->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        for (uint i = 1; i < plsize; i++) {
            pt.x = pc_last_ouster->points[i].x;
            pt.y = pc_last_ouster->points[i].y;
            pt.z = pc_last_ouster->points[i].z;
            pt.intensity = float (pc_last_ouster->points[i].t) / float (1e6);
            pt.curvature = 0.1 * pc_last_ouster->points[i].intensity;
            if (pt.curvature < 0.0) continue;

            if (pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }      
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(ouster_msg_in->header.stamp).nanoseconds() - time_offset;
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        prepareVisualizationScan();
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(ouster_msg_in->header.stamp).nanoseconds() - time_offset;
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_ouster;
    int64_t time_offset = 0;
};

class RobosenseBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  RobosenseBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config)
    {
        pc_subscription_rs = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&RobosenseBuff::RSLidarCallback, this, std::placeholders::_1));              
    }

    void RSLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr rs_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<rs_ros::Point>::Ptr pc_last_rs(new pcl::PointCloud<rs_ros::Point>());
        pcl::fromROSMsg(*rs_msg_in, *pc_last_rs);
        size_t plsize = pc_last_rs->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        double time_begin_s = pc_last_rs->points[0].timestamp;        
        for (uint i = 1; i < plsize; i++) {
            pt.x = pc_last_rs->points[i].x;
            pt.y = pc_last_rs->points[i].y;
            pt.z = pc_last_rs->points[i].z;
            pt.intensity = float ((pc_last_rs->points[i].timestamp - time_begin_s) * 1e3);
            pt.curvature = pc_last_rs->points[i].intensity;
            if (pt.curvature < 0.0) continue;

            if (pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(rs_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        prepareVisualizationScan();
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(rs_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_rs;
};

class Mid70AviaBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  Mid70AviaBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config)
    {
        pc_subscription_livox = nh->create_subscription<livox_ros_driver::msg::CustomMsg>(
            this->lidar.topic, 100, std::bind(&Mid70AviaBuff::livoxLidarCallback, this, std::placeholders::_1));
    }

    void livoxLidarCallback(const livox_ros_driver::msg::CustomMsg::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        for (int i = 1; i < plsize; i++) {
            if ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00) {
                pt.x = livox_msg_in->points[i].x;
                pt.y = livox_msg_in->points[i].y;
                pt.z = livox_msg_in->points[i].z;
                pt.normal_x = 0;
                pt.normal_y = 0;
                pt.normal_z = 0;
                pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6);
                pt.curvature = livox_msg_in->points[i].reflectivity;
                if (pt.curvature < 0.0) continue;
                pc_last->points.push_back(pt);
            }
        } 
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        prepareVisualizationScan();
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<livox_ros_driver::msg::CustomMsg>::SharedPtr pc_subscription_livox;
};

class HAP360Buff : public MappingBase<pcl::PointXYZINormal>
{
public:
    HAP360Buff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config)
    {
        pc_subscription_livox = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            this->lidar.topic, 100, std::bind(&HAP360Buff::livoxLidarCallback, this, std::placeholders::_1));
        double lidar_time_offset = CommonUtils::readParam<double>(nh, "lidar_time_offset", 0.0);
        time_offset = 1e9*lidar_time_offset;               
    }

    void livoxLidarCallback(livox_ros_driver2::msg::CustomMsg::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        for (int i = 1; i < plsize; i++) {
            if ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00) {
                pt.x = livox_msg_in->points[i].x;
                pt.y = livox_msg_in->points[i].y;
                pt.z = livox_msg_in->points[i].z;
                pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6);
                pt.curvature = livox_msg_in->points[i].reflectivity;
                if (pt.curvature < 0.0) continue;
                pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds() - time_offset;
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        prepareVisualizationScan();
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds() - time_offset;
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr pc_subscription_livox;
    int64_t time_offset;
};

class AviaRespleBuff : public MappingBase<pcl::PointXYZINormal>
{
public:
    AviaRespleBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config)
    {
        pc_subscription_livox = nh->create_subscription<livox_interfaces::msg::CustomMsg>(
            this->lidar.topic, 100, std::bind(&AviaRespleBuff::livoxLidarCallback, this, std::placeholders::_1));
    }

    void livoxLidarCallback(livox_interfaces::msg::CustomMsg::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        for (int i = 1; i < plsize; i++) {
            if ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00) {
                pt.x = livox_msg_in->points[i].x;
                pt.y = livox_msg_in->points[i].y;
                pt.z = livox_msg_in->points[i].z;
                pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6);
                pt.curvature = livox_msg_in->points[i].reflectivity;
                if (pt.curvature < 0.0) continue;
                pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        prepareVisualizationScan();
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<livox_interfaces::msg::CustomMsg>::SharedPtr pc_subscription_livox;
};

class HesaiBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  HesaiBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config)
    {
        pc_subscription_hesai = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&HesaiBuff::hesaiLidarCallback, this, std::placeholders::_1));
        double lidar_time_offset = CommonUtils::readParam<double>(nh, "lidar_time_offset", 0.0);
        time_offset = 1e9*lidar_time_offset;                
    }

    void hesaiLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr hesai_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<hesai_ros::Point>::Ptr pc_last_hesai(new pcl::PointCloud<hesai_ros::Point>());
        pcl::fromROSMsg(*hesai_msg_in, *pc_last_hesai);
        size_t plsize = pc_last_hesai->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(hesai_msg_in->header.stamp)  - rclcpp::Duration(0, time_offset);
        pcl::PointXYZINormal pt;
        for (unsigned int i = 0; i < plsize; ++i) {
            pt.x = pc_last_hesai->points[i].x;
            pt.y = pc_last_hesai->points[i].y;
            pt.z = pc_last_hesai->points[i].z;
            double timestamp_s;
            double timestamp_ns = std::modf(pc_last_hesai->points[i].timestamp  - double(time_offset)*1e-9, &timestamp_s);                
            rclcpp::Time timestamp_ros(static_cast<int32_t>(timestamp_s), static_cast<int32_t>(timestamp_ns * 1.0e9),
                rcl_clock_type_t::RCL_ROS_TIME);
            pt.intensity = (timestamp_ros - timestamp_begin).seconds() * 1.0e3; 
            pt.curvature = pc_last_hesai->points[i].intensity;
            if (pt.curvature < 0.0) continue;
            if (pt.intensity >= 0 &&
                pt.x * pt.x + pt.y * pt.y + pt.z * pt.z > lidar.blind * lidar.blind) {
                this->pc_last->points.push_back(pt);
            }
        }        
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(hesai_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        prepareVisualizationScan();
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(hesai_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_hesai;
    int64_t time_offset;
};

class Mid360BoxiBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  Mid360BoxiBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config)
    {
        pc_subscription_mid360 = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&Mid360BoxiBuff::mid360BoxiCallback, this, std::placeholders::_1));
    }

    void mid360BoxiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<livox_mid360_boxi::Point>::Ptr pc_last_livox(new pcl::PointCloud<livox_mid360_boxi::Point>());
        pcl::fromROSMsg(*livox_msg_in, *pc_last_livox);
        size_t plsize = pc_last_livox->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(livox_msg_in->header.stamp);
        pcl::PointXYZINormal pt;
        for (uint i = 0; i < plsize; i++) {
            pt.x = pc_last_livox->points[i].x;
            pt.y = pc_last_livox->points[i].y;
            pt.z = pc_last_livox->points[i].z;
            rclcpp::Time timestamp_ros(static_cast<int64_t>(pc_last_livox->points[i].timestamp),
                rcl_clock_type_t::RCL_ROS_TIME);
            pt.intensity = (timestamp_ros - timestamp_begin).seconds() * 1.0e3; 
            pt.curvature = pc_last_livox->points[i].intensity;
            if (pt.curvature < 0.0) continue;

            if (pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        prepareVisualizationScan();
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_mid360;
};

class OdinBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  OdinBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config)
    {
        pc_subscription_odin = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&OdinBuff::odinCallback, this, std::placeholders::_1));
    }

    void odinCallback(const sensor_msgs::msg::PointCloud2::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<ls_ros::Point>::Ptr pc_last_livox(new pcl::PointCloud<ls_ros::Point>());
        pcl::fromROSMsg(*livox_msg_in, *pc_last_livox);
        size_t plsize = pc_last_livox->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(livox_msg_in->header.stamp);
        pcl::PointXYZINormal pt;
        for (uint i = 0; i < plsize; i++) {
            pt.x = pc_last_livox->points[i].x;
            pt.y = pc_last_livox->points[i].y;
            pt.z = pc_last_livox->points[i].z;
            pt.intensity = pc_last_livox->points[i].offset_time * 1.0e3;
            pt.curvature = static_cast<float>(pc_last_livox->points[i].intensity);
            if (pt.curvature < 0.0) continue;

            if (pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        prepareVisualizationScan();
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_odin;
};

class Mapping
{

public:

Mapping(rclcpp::Node::SharedPtr &nh, std::vector<MappingBase<pcl::PointXYZINormal>*>& mappings)
    {
        sub_start = nh->create_subscription<std_msgs::msg::Int64>("/start_time", 100, std::bind(&Mapping::startCallBack, this, std::placeholders::_1));
        spl_window_st_ns = 0;
        sub_est = nh->create_subscription<estimate_msgs::msg::Estimate>("/est_window", 30000, std::bind(&Mapping::getEstCallback, this, std::placeholders::_1));
        pub_path = nh->create_publisher<nav_msgs::msg::Path>("traj_path", 100);
        opt_old_path.header.frame_id = odom_id;
        const double path_dur = CommonUtils::readParam<double>(nh, "path_dur", 1000000.0);
        if (!std::isfinite(path_dur) || path_dur <= 0.0) {
            throw std::invalid_argument("path_dur must be finite and positive");
        }
        if (path_dur * 1e9 >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument("path_dur is too large to represent in nanoseconds");
        }
        path_dur_ns = static_cast<int64_t>(path_dur * 1e9);
        vis_maps = mappings;
        const bool if_visual = CommonUtils::readParam<bool>(nh, "if_visual", false);
        const bool legacy_rgb = CommonUtils::readParam<bool>(nh, "if_rbg_map", false);
        const bool if_rgb = CommonUtils::readParam<bool>(nh, "if_rgb", legacy_rgb);
        if (if_visual && if_rgb) rgb_colorizer.reset(new RgbMapColorizer(nh));
        constexpr double traj_itp_hz = 20.0;
        dt_ns = 1e9/traj_itp_hz;        
        pub_odom = nh->create_publisher<nav_msgs::msg::Odometry>("odometry", 500);
        br = std::make_shared<tf2_ros::TransformBroadcaster>(nh);
        if_lc = CommonUtils::readParam<bool>(nh, "if_loop_closure");   
        if (if_lc) {
            lcdetect.init(nh);
            sub_loop_closure = nh->create_subscription<estimate_msgs::msg::LoopClosure>(
                "/loop_closure_tranformation", 10, std::bind(&Mapping::loopClosureCallback, this, std::placeholders::_1));
            thresh_tf_dist = CommonUtils::readParam<double>(nh, "thresh_tf_dist");
        }          
        auto camera_names = CommonUtils::readParam<std::vector<std::string>>(nh, "cameras", std::vector<std::string>());
        if (!camera_names.empty()) {
            std::string cam_prefix = camera_names[0] + ".";
            std::string camera_topic = CommonUtils::readParam<std::string>(nh, cam_prefix + "topic_camera", "none");
            if (camera_topic != "none") {
                std::vector<double> q_cb_v = CommonUtils::readParam<std::vector<double>>(nh, cam_prefix + "q_cb");
                Eigen::Quaterniond q_cb = Eigen::Quaterniond(q_cb_v.at(0), q_cb_v.at(1), q_cb_v.at(2), q_cb_v.at(3)).normalized();
                Eigen::Vector3d t_cb = CommonUtils::readVector3d(nh, cam_prefix + "t_cb");

                T_ci = Eigen::Matrix4d::Identity();
                T_ci.topLeftCorner<3, 3>() = q_cb.toRotationMatrix();
                T_ci.topRightCorner<3, 1>() = t_cb;
                T_ic = T_ci.inverse();

            }
        }
    }

    ~Mapping()
    {
        global_map_pub.stop();
    };

    void process() {
        rclcpp::Rate rate(1000);
        int64_t num_knot = 0;
        auto last_path_publish = std::chrono::steady_clock::now();
        while (!if_finished) {
            const bool rebuild_path = if_init_succeed && lcdetect.consumePathCorrection();
            int64_t current_num_knots = 0;
            {
                auto spline_lock = lcdetect.gest.splineReadLock();
                current_num_knots = spline_global.numKnots();
            }
            if (if_init_succeed && (rebuild_path || current_num_knots > num_knot)) {
                publishPath(rebuild_path);
                last_path_publish = std::chrono::steady_clock::now();
                pubOdom();
                num_knot = current_num_knots;
            } else {
                rate.sleep();
            }
            if (!opt_old_path.poses.empty() &&
                std::chrono::steady_clock::now() - last_path_publish >= std::chrono::seconds(1)) {
                pub_path->publish(opt_old_path);
                last_path_publish = std::chrono::steady_clock::now();
            }
            if (!if_init_succeed) {
                rate.sleep();
                continue;
            }
            if (if_finished) {
                break;
            }                 
        }
    }

    void requestStop()
    {
        {
            std::lock_guard<std::mutex> lock(m_est_buf);
            if_finished = true;
        }
        lcdetect.requestStop();
        cv_est.notify_all();
    }        

    void processEstThread()
    {
        while (true) {          
            std::deque<estimate_msgs::msg::Estimate::SharedPtr> local_buf;
            {
                std::unique_lock<std::mutex> lk(m_est_buf);
                cv_est.wait(lk, [this]{ return !est_buf.empty() || if_finished; });
                local_buf.swap(est_buf);
            }
            if (local_buf.empty() && if_finished) {
                break;
            }               
            while (!local_buf.empty()) {
                processEst(local_buf.front());
                local_buf.pop_front();
            }       
        }
    }    

    void correction()
    {
        if (if_lc) {
            lcdetect.correction();
        } else {
            return;
        }
    }

private:
    std::string node_name = "Mapping";
    std::atomic<int64_t> spl_window_st_ns;
    int64_t dt_ns;  
    rclcpp::Subscription<estimate_msgs::msg::Estimate>::SharedPtr sub_est;
    rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr sub_start;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    nav_msgs::msg::Path opt_old_path;
    int64_t next_path_time_ns = 0;
    int64_t path_dur_ns;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
    std::vector<MappingBase<pcl::PointXYZINormal>*> vis_maps;
    std::unique_ptr<RgbMapColorizer> rgb_colorizer;
    const std::string frame_id = "body";
    const std::string odom_id = "world";
    std::shared_ptr<tf2_ros::TransformBroadcaster> br;
    bool if_init_succeed = false;

    bool if_lc = false;
    LoopDetection lcdetect;
    Eigen::Vector3d trans_drift = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rot_drift =Eigen::Quaterniond::Identity();    

    rclcpp::Subscription<estimate_msgs::msg::LoopClosure>::SharedPtr sub_loop_closure;
    std::deque<estimate_msgs::msg::Estimate::SharedPtr> est_buf;
    std::mutex m_est_buf;
    std::condition_variable cv_est;
    std::atomic<bool> if_finished{false};
    Eigen::Matrix4d T_ci;
    Eigen::Matrix4d T_ic;
    double thresh_tf_dist;

    void getEstCallback(const estimate_msgs::msg::Estimate::SharedPtr est_msg)
    {
        if (!if_init_succeed) return;
        std::lock_guard<std::mutex> lk(m_est_buf);
        est_buf.push_back(est_msg);
        cv_est.notify_one();
    }

    void processEst(const estimate_msgs::msg::Estimate::SharedPtr est_msg)
    {
        const auto& spline_msg = est_msg->spline;
        if (spline_msg.knots.empty() || spline_msg.idles.size() != 3 ||
            spline_msg.start_idx > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return;
        }
        for (size_t i = 1; i < spline_msg.knots.size(); ++i) {
            if (spline_msg.knots[i].tk <= spline_msg.knots[i - 1].tk) {
                return;
            }
        }

        SplineState spline_w;
        spline_w.init(spline_msg.dt, 0, spline_msg.start_t, spline_msg.start_idx);
        for (const auto& knot : spline_msg.knots) {
            Eigen::Vector3d pos(knot.position.x, knot.position.y, knot.position.z);
            Eigen::Vector3d quat_del(knot.orientation_del.x, knot.orientation_del.y, knot.orientation_del.z);
            spline_w.addOneStateKnot(pos, quat_del, knot.tk);
        }
        Eigen::Quaterniond q_idle0 = Eigen::Quaterniond(spline_msg.start_q.w, spline_msg.start_q.x, spline_msg.start_q.y, spline_msg.start_q.z);
        for (int i = 0; i < 3; i++) {
            estimate_msgs::msg::Knot idle = spline_msg.idles[i];
            Eigen::Vector3d t_idle(idle.position.x, idle.position.y, idle.position.z);
            Eigen::Vector3d quat_idle(idle.orientation_del.x, idle.orientation_del.y, idle.orientation_del.z);
            spline_w.setIdles(i, t_idle, quat_idle, q_idle0);
        }

        auto spline_lock = lcdetect.gest.splineWriteLock();
        if (if_lc) {
            lcdetect.gest.getDrift(rot_drift, trans_drift);   
            for (int64_t i = 0; i < spline_w.numKnots(); i++) {
                Eigen::Vector3d t_knot = spline_w.getKnotPos(i);
                Eigen::Quaterniond q_knot = spline_w.getKnotOrt(i);
                q_knot = rot_drift * q_knot;
                t_knot = rot_drift * t_knot + trans_drift;
                spline_w.setOneStateKnot(i, q_knot, t_knot);
            }
            for (int i = 0; i < 3; i++) {
                Eigen::Vector3d t_knot = spline_w.getKnotPosIdle(i);
                Eigen::Quaterniond q_knot = spline_w.getKnotOrtIdle(i);
                q_knot = rot_drift * q_knot;
                t_knot = rot_drift * t_knot + trans_drift;
                spline_w.setIdles(i, q_knot, t_knot);
            }
        }

        std::string merge_error;
        if (!spline_global.updateKnots(&spline_w, merge_error)) {
            return;
        }
        spl_window_st_ns.store(spline_global.maxTimeNs(), std::memory_order_relaxed);
        spline_lock.unlock();
        for (const auto vis_map : vis_maps) {
            vis_map->processScan(&spline_global, spl_window_st_ns.load(std::memory_order_relaxed),
                [this]{ lcdetect.gest.lock_spline_shared(); },
                [this]{ lcdetect.gest.unlock_spline_shared(); }, rgb_colorizer.get());
        }
        if (rgb_colorizer) {
            int64_t earliest_pending_ns = spl_window_st_ns.load(std::memory_order_relaxed);
            for (const auto vis_map : vis_maps) {
                earliest_pending_ns = vis_map->earliestPendingScanTime(earliest_pending_ns);
            }
            rgb_colorizer->discardBefore(earliest_pending_ns);
        }                         
    }

    void loopClosureCallback(const estimate_msgs::msg::LoopClosure::SharedPtr lc_msg)
    {   
        Eigen::Vector3d t_lc_qc(lc_msg->transform.position.x, lc_msg->transform.position.y, lc_msg->transform.position.z);
        Eigen::Quaterniond q_lc_qc(lc_msg->transform.orientation.w, lc_msg->transform.orientation.x, lc_msg->transform.orientation.y, lc_msg->transform.orientation.z);
        Eigen::Matrix4d T_lc_qc = Eigen::Matrix4d::Identity();
        T_lc_qc.topLeftCorner<3, 3>() = q_lc_qc.toRotationMatrix();
        T_lc_qc.topRightCorner<3, 1>() = t_lc_qc;

        Eigen::Matrix4d T_li_qi = T_ic * T_lc_qc * T_ci;
        Eigen::Vector3d t_li_qi = T_li_qi.topRightCorner<3, 1>();
        Eigen::Quaterniond q_li_qi(T_li_qi.topLeftCorner<3, 3>());

        double dis_norm = t_li_qi.norm();
        if (dis_norm > thresh_tf_dist) {
            return;
        }

        LoopData loop_pair(lc_msg->t_ref_ns, lc_msg->t_cur_ns, q_li_qi, t_li_qi, 0, 0);
        lcdetect.m_lc_buff.lock();
        lcdetect.lc_buf.push_back(loop_pair);
        lcdetect.m_lc_buff.unlock();
    }

    void pubOdom()
    {
        if (opt_old_path.poses.empty()) {
            return;
        }
        nav_msgs::msg::Odometry odom_msg;
        geometry_msgs::msg::PoseStamped odom_pose = opt_old_path.poses.back();
        odom_msg.header.stamp = rclcpp::Time(odom_pose.header.stamp);
        odom_msg.header.frame_id = odom_id;
        odom_msg.child_frame_id = frame_id;
        odom_msg.pose.pose = odom_pose.pose;
        pub_odom->publish(odom_msg);
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = odom_msg.header.stamp;
        transformStamped.header.frame_id = odom_id;
        transformStamped.child_frame_id = frame_id;
        transformStamped.transform.translation.x = odom_pose.pose.position.x;
        transformStamped.transform.translation.y = odom_pose.pose.position.y;
        transformStamped.transform.translation.z = odom_pose.pose.position.z;
        transformStamped.transform.rotation = odom_pose.pose.orientation;
        br->sendTransform(transformStamped);
        transformStamped.header.stamp = odom_msg.header.stamp;
        transformStamped.header.frame_id = odom_id;
        transformStamped.child_frame_id = "imu_frame";
        transformStamped.transform.translation.x = odom_pose.pose.position.x;
        transformStamped.transform.translation.y = odom_pose.pose.position.y;
        transformStamped.transform.translation.z = odom_pose.pose.position.z;
        transformStamped.transform.rotation.w = odom_pose.pose.orientation.w;
        transformStamped.transform.rotation.x = odom_pose.pose.orientation.x;
        transformStamped.transform.rotation.y = odom_pose.pose.orientation.y;
        transformStamped.transform.rotation.z = odom_pose.pose.orientation.z;
        br->sendTransform(transformStamped);

        int id_lidar = 0;
        for (const auto vis_map : vis_maps) {
            Eigen::Quaterniond q_wi(odom_pose.pose.orientation.w, odom_pose.pose.orientation.x, odom_pose.pose.orientation.y, odom_pose.pose.orientation.z);
            Eigen::Vector3d t_wi(odom_pose.pose.position.x, odom_pose.pose.position.y, odom_pose.pose.position.z);
            Eigen::Quaterniond q_wl = q_wi * vis_map->lidar.q_bl;
            Eigen::Vector3d t_wl = q_wi * vis_map->lidar.t_bl + t_wi;
            std::string lidar_str = "lidar" + std::to_string(id_lidar) + "_frame";
            transformStamped.header.stamp = odom_msg.header.stamp;
            transformStamped.header.frame_id = odom_id;
            transformStamped.child_frame_id = lidar_str;
            transformStamped.transform.translation.x = t_wl.x();
            transformStamped.transform.translation.y = t_wl.y();
            transformStamped.transform.translation.z = t_wl.z();
            transformStamped.transform.rotation.w = q_wl.w();
            transformStamped.transform.rotation.x = q_wl.x();
            transformStamped.transform.rotation.y = q_wl.y();
            transformStamped.transform.rotation.z = q_wl.z();
            br->sendTransform(transformStamped);
            id_lidar++;
        }                 
    }

    void startCallBack(const std_msgs::msg::Int64::SharedPtr start_time_msg)
    {
        int64_t bag_start_time = start_time_msg->data;
        if (if_lc) {
            spline_global.init(0, 0, bag_start_time, 0);
        } else {
            spline_global.init(0, 0, bag_start_time, 0);
        }        
        if_init_succeed = true;
    }

    void publishPath(bool rebuild = false) {
        auto spline_lock = lcdetect.gest.splineReadLock();
        if (!if_init_succeed || spline_global.numKnots() <= 4) {
            return;
        }
        if (rebuild || opt_old_path.poses.empty()) {
            opt_old_path.poses.clear();
            next_path_time_ns = spline_global.minTimeNs();
        }
        const int64_t max_t = std::min(spl_window_st_ns.load(std::memory_order_relaxed),
                                       spline_global.maxTimeNs());
        while (next_path_time_ns < max_t) {
            const int64_t t_ns = next_path_time_ns;
            Eigen::Quaterniond orient_interp;
            Eigen::Vector3d t_interp;
            spline_global.itpQuaternionFromDelta(t_ns, &orient_interp);
            t_interp = spline_global.itpPosition(t_ns);
            auto pose = CommonUtils::pose2msg(t_ns, t_interp, orient_interp);
            pose.header.frame_id = odom_id;
            opt_old_path.poses.push_back(pose);
            next_path_time_ns += dt_ns;
        }
        spline_lock.unlock();
        if (!opt_old_path.poses.empty()) {
            opt_old_path.header.stamp = opt_old_path.poses.back().header.stamp;
            const int64_t latest_t_ns = rclcpp::Time(opt_old_path.header.stamp).nanoseconds();
            const auto first_retained = std::find_if(
                opt_old_path.poses.begin(), opt_old_path.poses.end(),
                [this, latest_t_ns](const geometry_msgs::msg::PoseStamped& pose) {
                    return latest_t_ns - rclcpp::Time(pose.header.stamp).nanoseconds() <= path_dur_ns;
                });
            opt_old_path.poses.erase(opt_old_path.poses.begin(), first_retained);
        }
        pub_path->publish(opt_old_path);
    }         

};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto nh = rclcpp::Node::make_shared("Mapping");
    resple2::configureOpenCvThreads(nh);
    global_map_pub.start(argc, argv);
    std::vector<LidarConfig> lidars;
    auto lidar_names = nh->declare_parameter<std::vector<std::string>>("lidars", std::vector<std::string>());
    (void)nh->get_parameter({"lidars"}, lidar_names);
    if (lidar_names.empty()) {
        lidars.emplace_back(nh, "");
    } else {
        for (const auto& lidar_name : lidar_names) {
            lidars.emplace_back(nh, lidar_name + ".");
        }
    }
    std::vector<MappingBase<pcl::PointXYZINormal>*> buffs;
    for (const auto& lidar : lidars) {
        if (!lidar.type.compare("Ouster")) {
            buffs.push_back(new OusterBuff(nh, lidar));
        } else if (!lidar.type.compare("Mid70Avia")) {
            buffs.push_back(new Mid70AviaBuff(nh, lidar));
        } else if (!lidar.type.compare("HAP360")) {
            buffs.push_back(new HAP360Buff(nh, lidar));    
        } else if (!lidar.type.compare("AviaResple")) {
            buffs.push_back(new AviaRespleBuff(nh, lidar));
        } else if (!lidar.type.compare("Hesai")) {
            buffs.push_back(new HesaiBuff(nh, lidar));
        } else if (!lidar.type.compare("Mid360Boxi")) {
            buffs.push_back(new Mid360BoxiBuff(nh, lidar));
        } else if (!lidar.type.compare("Odin")) {
            buffs.push_back(new OdinBuff(nh, lidar));
        } else if (!lidar.type.compare("Robosense")) {
            buffs.push_back(new RobosenseBuff(nh, lidar));             
        } else {
            exit(1);
        }
    }
    Mapping mapping(nh, buffs);
    std::thread mappingThread{&Mapping::process, &mapping};
    std::thread estThread{&Mapping::processEstThread, &mapping};
    std::thread globalOpt{&Mapping::correction, &mapping};
    rclcpp::Rate rate(500);
    while (rclcpp::ok()) {
        rclcpp::spin_some(nh);
        rate.sleep();
    }

    mapping.requestStop();
    mappingThread.join();
    estThread.join();
    globalOpt.join();
    rclcpp::shutdown();
}
