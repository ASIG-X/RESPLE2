/*
This file is part of RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement.
 
Copyright (c) 2026 Ziyu Cao

Developer: Ziyu Cao <ziyu.cao@liu.se>

For commercial use, please contact me at <ziyu.cao@liu.se> or Kailai Li at <kailai.li@liu.se>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file, 
which is included as part of this source code package.
*/

#include <rclcpp/rclcpp.hpp>
#include "utils/geometry_batch.h"
#include <std_msgs/msg/int64.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <thread>
#include <mutex>
#include <omp.h>
#include <cstring>
#include <cmath>
#include <array>
#include <stdexcept>

#include <pcl/point_types.h>
#ifdef ROS_DISTRO_HUMBLE
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif
#include "livox_interfaces/msg/custom_msg.hpp"
#include "livox_ros_driver/msg/custom_msg.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "estimate_msgs/msg/spline.hpp"
#include "estimate_msgs/msg/estimate.hpp"
#include "anymal_msgs/msg/anymal_state.hpp"
#include <unitree_msgs/msg/low_state_stamped.hpp>
#include "camera/ImgPreprocessing.h"
#include "utils/math_tools.h"
#include "Estimator.h"
#include "utils/common_utils.h"

int NUM_OF_THREAD = 5;

class RESPLE2
{

public:
    RESPLE2(rclcpp::Node::SharedPtr& nh) 
    {
        readParameters(nh);
        std::string imu_type = CommonUtils::readParam<std::string>(nh, "topic_imu");
        auto imu_qos = rclcpp::SensorDataQoS().keep_last(2000000);
        if (CommonUtils::readParam<bool>(nh, "imu_reliable_qos", false))
            imu_qos.reliable();
        sub_imu = nh->create_subscription<sensor_msgs::msg::Imu>(imu_type, imu_qos, std::bind(&RESPLE2::getImuCallback, this, std::placeholders::_1));
        pub_est = nh->create_publisher<estimate_msgs::msg::Estimate>("est_window", 50);
        pub_start_time = nh->create_publisher<std_msgs::msg::Int64>("start_time", 50);
        pub_cur_scan = nh->create_publisher<sensor_msgs::msg::PointCloud2>("current_scan", 2);
        br = std::make_shared<tf2_ros::TransformBroadcaster>(nh);        
        pc_frame.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
        lidar_names = nh->declare_parameter<std::vector<std::string>>("lidars", std::vector<std::string>());
        (void)nh->get_parameter({"lidars"}, lidar_names);
        if (lidar_names.empty()) {
            LidarConfig lidar(nh, "");
            lidars.emplace(lidar.type, lidar);
            meas_data.lidars_data.emplace(std::piecewise_construct, std::make_tuple(lidar.type), std::make_tuple());
        } else {
            for (const auto& lidar_name : lidar_names) {
                LidarConfig lidar(nh, lidar_name + ".");
                lidars.emplace(lidar.type, lidar);
                meas_data.lidars_data.emplace(std::piecewise_construct, std::make_tuple(lidar.type), std::make_tuple());
            }
        }    
        for (const auto& [lidar_name, lidar] : lidars) {
            if (!lidar.type.compare("Ouster")) {
                sub_ouster = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000, std::bind(&RESPLE2::ousterLidarCallback<ouster_ros::Point>, this, std::placeholders::_1));
            } else if (!lidar.type.compare("Mid70Avia")) {
                sub_livox = nh->create_subscription<livox_ros_driver::msg::CustomMsg>(
                        lidar.topic, 200000, std::bind(&RESPLE2::livoxLidarCallback, this, std::placeholders::_1));
            } else if (!lidar.type.compare("HAP360")) {
                sub_livox2 = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                        lidar.topic, 200000, std::bind(&RESPLE2::livoxLidar2Callback, this, std::placeholders::_1));
            } else if (!lidar.type.compare("AviaResple")) {
                sub_livox_avia = nh->create_subscription<livox_interfaces::msg::CustomMsg>(
                        lidar.topic, 200000, std::bind(&RESPLE2::livoxAVIACallback, this, std::placeholders::_1));
            } else if (!lidar.type.compare("Hesai")) {
                sub_hesai = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000, std::bind(&RESPLE2::hesaiLidarCallback, this, std::placeholders::_1));
            } else if (!lidar.type.compare("Mid360Boxi")) {
                sub_livox_mid360_boxi = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000, std::bind(&RESPLE2::livoxMid360BoxiCallback, this, std::placeholders::_1));
            } else if (!lidar.type.compare("Odin")) {
                sub_odin = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000, std::bind(&RESPLE2::odinCallback, this, std::placeholders::_1));                   
            }
        }        
        if (param.if_kin) {
            const std::string urdf_filename = model_path;
            pinocchio::urdf::buildModel(urdf_filename, pmodel);
            pdata = std::make_shared<pinocchio::Data>(pmodel); 

            std::string topic_kin = CommonUtils::readParam<std::string>(nh, "topic_kin");
            if (!topic_kin.compare("/anymal/state_estimator/anymal_state")) {
                sub_anym = nh->create_subscription<anymal_msgs::msg::AnymalState>(
                                topic_kin, 200000, std::bind(&RESPLE2::getKinematicAnymalCallback, this, std::placeholders::_1));
                kinematics->map_foot_id = {
                    {"LF_FOOT", 0},
                    {"RF_FOOT", 1},
                    {"LH_FOOT", 2},
                    {"RH_FOOT", 3}    
                };                                  
            } else {
                rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(
                    rmw_qos_profile_sensor_data));
                qos.keep_last(200000);

                sub_go2 = nh->create_subscription<unitree_msgs::msg::LowStateStamped>(
                    topic_kin,
                    qos,
                    std::bind(&RESPLE2::getKinematicLowSStampedCallback,
                            this,
                            std::placeholders::_1));   
                kinematics->map_foot_id = {
                    {"FL_foot", 0},
                    {"FR_foot", 1},
                    {"RL_foot", 2},
                    {"RR_foot", 3}     
                };                          
            }        
            pub_foot_pos = nh->create_publisher<sensor_msgs::msg::PointCloud2>("foot_pos", 20);         
        }        
        if (if_visual) { 
            if (camera_data.if_compressed) {
                sub_cam_comp = nh->create_subscription<sensor_msgs::msg::CompressedImage>(
                            camera_data.topic, 200000, std::bind(&RESPLE2::getCameraCompCallback, this, std::placeholders::_1));
            } else {
                sub_cam = nh->create_subscription<sensor_msgs::msg::Image>(
                            camera_data.topic, 200000, std::bind(&RESPLE2::getCameraCallback, this, std::placeholders::_1));
            }
            if (camera_data.if_camera_info) {
                if (img_preprocess->cam_model == "PolynomialCamera")
                    throw std::invalid_argument("PolynomialCamera requires YAML calibration (if_camera_info: false)");
                sub_cam_info = nh->create_subscription<sensor_msgs::msg::CameraInfo>(
                            camera_data.topic_info, 200000, std::bind(&RESPLE2::cameraInfoCallback, this, std::placeholders::_1));
            } else {
                img_preprocess->init(nh, camera_data.pref, visual_frontend->cam);
            }
            pub_feature_image = nh->create_publisher<sensor_msgs::msg::Image>("feature_image", 2);
            pub_img_scale_ = CommonUtils::readParam<double>(nh, "pub_img_scale", 1.0);
            pub_cam_poses = nh->create_publisher<nav_msgs::msg::Odometry>("/camera/poses_inworld", 500);                 
        }               
        pub_scan_world = nh->create_publisher<sensor_msgs::msg::PointCloud2>("scan_world", 2);
    }

    ~RESPLE2(){};

    void processData()
    {
        rclcpp::Rate rate(1000);
        int64_t max_spl_knots = 0;
        int64_t next_lidar_end_time = 0;
        bool if_process_next_lidar = true;
        int64_t cnt_imu = 0;

        Eigen::aligned_vector<sensor_msgs::msg::Imu::SharedPtr> imu_buff_msg;
        Eigen::aligned_deque<ImgData> img_buff_msg;
        Eigen::aligned_deque<KinImuData> kin_buff_msg;
        imu_buff_msg.reserve(256);
        resple2::BufferTransferSchedule transfer_schedule;
        resple2::VoxelFilterSelection voxel_selection;

        while (rclcpp::ok()) {
            for (auto& [lidar_name, lidar_data] : meas_data.lidars_data) {
                Eigen::aligned_vector<pcl::PointXYZINormal> incoming_points;
                int64_t time_begin = 0;
                bool have_frame = false;
                {
                    std::unique_lock<std::mutex> lock(lidar_data.mtx_pc, std::try_to_lock);
                    if (lock.owns_lock() && !lidar_data.t_buff.empty()) {
                        incoming_points.swap(lidar_data.pc_buff.front());
                        lidar_data.pc_buff.pop_front();
                        time_begin = lidar_data.t_buff.front();
                        lidar_data.t_buff.pop_front();
                        have_frame = true;
                    }
                }
                if (have_frame) {
                    const size_t map_begin = vpts_body.size();
                    pc_frame->points.swap(incoming_points);
                    nan_indices.clear();
                    pcl::removeNaNFromPointCloud(*pc_frame, *pc_frame, nan_indices);                  
                    LidarConfig& lidar = lidars.at(lidar_name);
                    const CommonUtils::BodyCovarianceNoise body_noise(lidar.dept_err, lidar.beam_err);
                    const auto append_map_point = [&](const pcl::PointXYZINormal& point) {
                        vpts_body.emplace_back(
                            point, time_begin, lidar.q_bl, lidar.t_bl, body_noise);
                    };
                    const auto append_measurement_point = [&](const pcl::PointXYZINormal& point) {
                        lidar_data.pt_buff.emplace_back(
                            point, time_begin, lidar.q_bl, lidar.t_bl,
                            lidar.w_pt, body_noise);
                    };
                    const auto append_reusing_covariance = [&](const pcl::PointXYZINormal& point, size_t map_index) {
                        lidar_data.pt_buff.emplace_back(
                            point, time_begin, lidar.q_bl, lidar.t_bl,
                            lidar.w_pt, vpts_body[map_index].body_var);
                    };
                    const bool filter_map = lidar.if_dense_map || !if_init_map;
                    lidar.ds_filter_body.setInputCloud(pc_frame);
                    if (filter_map) ds_filter_map.setInputCloud(pc_frame);
                    const bool direct_buffers = filter_map &&
                        ds_filter_map.selectBothInTimeOrder(lidar.ds_filter_body, voxel_selection);
                    if (direct_buffers) {
                        const bool scan_uses_map = lidar.if_dense_map &&
                            voxel_selection.counts[1] < static_cast<size_t>(param.num_points_upd);
                        for (size_t i = 0; i < voxel_selection.mask.size(); ++i)
                            if (voxel_selection.mask[i] & resple2::VoxelFilterSelection::first)
                                append_map_point(pc_frame->points[i]);
                        size_t map_index = map_begin;
                        const uint8_t scan_mask = scan_uses_map ? resple2::VoxelFilterSelection::first
                                                               : resple2::VoxelFilterSelection::second;
                        for (size_t i = 0; i < voxel_selection.mask.size(); ++i) {
                            const auto selected = voxel_selection.mask[i];
                            const bool in_map = selected & resple2::VoxelFilterSelection::first;
                            if (selected & scan_mask) {
                                if (in_map) append_reusing_covariance(pc_frame->points[i], map_index);
                                else append_measurement_point(pc_frame->points[i]);
                            }
                            if (in_map) ++map_index;
                        }
                    } else {
                        pc_last_ds->clear();
                        if (filter_map) {
                            pc_map_ds->clear();
                            ds_filter_map.filterBoth(lidar.ds_filter_body, *pc_map_ds, *pc_last_ds);
                            if (!std::is_sorted(pc_map_ds->points.begin(), pc_map_ds->points.end(), &CommonUtils::time_list))
                                std::sort(pc_map_ds->points.begin(), pc_map_ds->points.end(), &CommonUtils::time_list);
                            for (const auto& point : *pc_map_ds) {
                                append_map_point(point);
                            }
                        } else {
                            lidar.ds_filter_body.filter(*pc_last_ds);
                        }
                        if (lidar.if_dense_map && pc_last_ds->points.size() < (size_t) param.num_points_upd) {
                            size_t map_index = map_begin;
                            for (const auto& point : *pc_map_ds) {
                                append_reusing_covariance(point, map_index++);
                            }
                        } else {
                            sort(pc_last_ds->points.begin(), pc_last_ds->points.end(), &CommonUtils::time_list);
                            for (const auto& point : *pc_last_ds) {
                                append_measurement_point(point);
                            }
                        }
                    }
                    if (if_process_next_lidar && !lidar_data.pt_buff.empty()) {
                        next_lidar_end_time = lidar_data.pt_buff.back().time_ns;
                        if_process_next_lidar = false;
                    }
                }
            }
            const bool initializing = !(if_init_filter && if_init_map);
            const bool require_frame_coverage = initializing || !geometry_batch_.enabled;
            if (require_frame_coverage &&
               ((last_imu_ns < next_lidar_end_time) ||
                (if_visual && last_img_ns < next_lidar_end_time) ||
                (param.if_kin && last_kin_ns < next_lidar_end_time))) {
                rate.sleep();
                continue;
            }

            if_process_next_lidar = true;
            const auto transfer_now = resple2::BufferTransferSchedule::Clock::now();
            const bool transfer_due = require_frame_coverage || transfer_schedule.due(transfer_now);
            if (transfer_due) {
                transfer_schedule.attempted(transfer_now);
                imu_handoff_.take(meas_data.m_buff, meas_data.imu_int_buff,
                                  imu_buff_msg, require_frame_coverage);
                if (if_visual)
                    img_handoff_.take(m_img_buff_, meas_data.img_int_buff,
                                      img_buff_msg, require_frame_coverage);
                if (param.if_kin)
                    kin_handoff_.take(meas_data.m_imu_buff, meas_data.kin_imu_int_buff,
                                      kin_buff_msg, require_frame_coverage);
            }
            if (!imu_buff_msg.empty()) {
                for (size_t i = 0; i < imu_buff_msg.size(); i++) {
                    const auto& imu_msg = imu_buff_msg[i];
                    int64_t t_ns = rclcpp::Time(imu_msg->header.stamp).nanoseconds();
                    Eigen::Vector3d acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                    if (acc_ratio) acc *= 9.81;
                    Eigen::Vector3d gyro(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                    if (cnt_imu % 2 == 0) {
                        meas_data.imu_pred.emplace_back(t_ns, gyro, acc);
                    } else {
                        meas_data.imu_buff.emplace_back(t_ns, gyro, acc);
                    }
                    cnt_imu++;
                }
                imu_buff_msg.clear();
            }
            if (!img_buff_msg.empty()) {
                meas_data.img_buff.insert(meas_data.img_buff.end(), img_buff_msg.begin(), img_buff_msg.end());
                img_buff_msg.clear();
            }
            if (!kin_buff_msg.empty()) {
                for (size_t i = 0; i < kin_buff_msg.size(); i++) {
                    meas_data.kin_imu_buff.push_back(kin_buff_msg[i]);
                }
                kin_buff_msg.clear();
            }

            const bool initialized = initialization();
            if (!initialized) {
                rate.sleep();
                continue;
            }

            while (rclcpp::ok()) {
                const bool collected = collectMeasurements();
                if (!collected) {
                    std::this_thread::sleep_until(transfer_schedule.deadline());
                    break;
                }
                estimator.propRCPNU(max_time_ns);

                meas_data.removeOldMeas(spline->getKnotTimeNs(spline->numKnots() - 2));

                trackImageMeasurements();
                estimator.updateIEKF(voxelmap_manager, visual_frontend, camera_data);
                if (estimator.num_tot_eff < 10) {
                    spline->removeOneTk();
                    estimator.updateIntState();
                    continue;
                }
                publishFootPositions();
                pubCameraPoses();

                transformLiDARToWorld();

                bool if_new_img = (!meas_data.img_meas.empty());
                meas_data.removeOldMeas(spline->getKnotTimeNs(spline->numKnots() - 1));

                publishFrameWorld();
                voxelmap_manager->UpdateVoxelMap(pv_list, NUM_OF_THREAD);
                pv_list.clear();
                if(voxelmap_manager->config_setting_.map_sliding_en) {
                    voxelmap_manager->mapSliding(spline->itpPosition(spline->maxTimeNs()));
                }
                if (if_visual && if_new_img) {
                    const bool render_debug = pub_feature_image->get_subscription_count() > 0 ||
                        pub_feature_image->get_intra_process_subscription_count() > 0;
                    visual_frontend->generateVisualPoints(latest_img, pc_world_visual.points, render_debug);
                    if (render_debug) publishImgRgb(visual_frontend->noted_img2, pub_feature_image);
                    pc_world.points.clear();
                    pc_world_visual.clear();
                } else if (!if_visual) {
                    pc_world.points.clear();
                }

                if (spline->numKnots() > max_spl_knots) {
                    estimate_msgs::msg::Spline spline_msg;
                    spline->getSplineMsg(spline_msg, std::max(int(max_spl_knots-1),0));
                    estimate_msgs::msg::Estimate est_msg;
                    est_msg.spline = spline_msg;
                    est_msg.if_full_window.data = (spline->numKnots() >= 4);
                    est_msg.runtime.data = 0;
                    pub_est->publish(est_msg);
                    max_spl_knots = spline->numKnots();
                }
                spline->removeOneTk();
                estimator.updateIntState();
            }            
        }
    }    

    void transformLiDARToWorld()
    {
        const Eigen::Matrix<double, 24, 24>& cov_rcp = estimator.getCovRCP();
        const bool use_map_points = !vpts_body.empty();
        size_t num_pts = 0;
        if (use_map_points) {
            while (!vpts_body.empty() && vpts_body.front().time_ns <= spline->minTimeNs()) {
                vpts_body.pop_front();
            }
            while (num_pts < vpts_body.size() && vpts_body[num_pts].time_ns <= spline->maxTimeNs()) {
                ++num_pts;
            }
        } else {
            num_pts = meas_data.pt_meas.size();
        }
        const size_t size_prev = pc_world.points.size();
        const size_t visual_size_prev = pc_world_visual.points.size();

        constexpr size_t EXTRA = 5000;
        if (pc_world.points.capacity() < size_prev + num_pts) {
            pc_world.points.reserve(size_prev + num_pts + EXTRA);
        }
        if (if_visual) {
            if (pc_world_visual.points.capacity() < visual_size_prev + num_pts) {
                pc_world_visual.points.reserve(visual_size_prev + num_pts + EXTRA);
            }
            pc_world_visual.points.resize(visual_size_prev + num_pts);
        }
        if (pv_list.capacity() < num_pts) {
            pv_list.reserve(num_pts + EXTRA);
        }
        pc_world.points.resize(size_prev + num_pts);
        pv_list.resize(num_pts);

        const auto transform_points = [&](auto& points) {
            using BufferedPoint = typename std::decay_t<decltype(points)>::value_type;
            #pragma omp parallel num_threads(NUM_OF_THREAD)
            {
                int64_t last_t_ns = -1;
                Association::SplinePoseCache pose_cache;
                Association::PoseCovarianceCache pose_covariance;
                unsigned int points_at_pose = 0;
                #pragma omp for schedule(static)
                for (size_t i = 0; i < num_pts; i++) {
                    auto& point = points[i];
                    const auto& input = [&]() -> const auto& {
                        if constexpr (std::is_same_v<BufferedPoint, MapPointData>) return point;
                        else return point.input();
                    }();
                    if (input.time_ns != last_t_ns) {
                        pose_cache = Association::evalSpline(input.time_ns, spline);
                        last_t_ns = input.time_ns;
                        pose_covariance.valid = false;
                        points_at_pose = 0;
                    }
                    if (++points_at_pose == 3 && i + 1 < num_pts) {
                        const int64_t next_time_ns = [&]() {
                            if constexpr (std::is_same_v<BufferedPoint, MapPointData>) return points[i + 1].time_ns;
                            else return points[i + 1].input().time_ns;
                        }();
                        if (next_time_ns == input.time_ns) pose_covariance.prepare(pose_cache, cov_rcp);
                    }
                    const auto store_world = [&](const pcl::PointXYZINormal& world, const Eigen::Matrix3d& variance) {
                        pc_world.points[size_prev + i] = world;
                        pv_list[i].point_w = Eigen::Vector3d(world.x, world.y, world.z);
                        pv_list[i].var = variance;
                        pv_list[i].reflectivity = input.pt.curvature;
                        if (if_visual) {
                            pc_world_visual.points[visual_size_prev + i] = world;
                        }
                    };
                    if constexpr (std::is_same_v<BufferedPoint, MapPointData>) {
                        pcl::PointXYZINormal world;
                        Eigen::Matrix3d variance, cov_pose;
                        Association::pointBodyToWorld(pose_cache, input.pt, world, input.t_bl, input.q_bl,
                            input.body_var, variance, cov_pose, cov_rcp, nullptr, &pose_covariance);
                        store_world(world, variance);
                    } else {
                        Eigen::Matrix3d cov_pose;
                        Association::pointBodyToWorld(pose_cache, input.pt, point.pt_w, input.t_bl, input.q_bl,
                            input.body_var, point.w_var, cov_pose, cov_rcp, nullptr, &pose_covariance);
                        store_world(point.pt_w, point.w_var);
                    }
                }
            }
        };
        if (use_map_points) {
            transform_points(vpts_body);
            vpts_body.pop_front(num_pts);
        } else {
            transform_points(meas_data.pt_meas);
        }
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_ouster;
    rclcpp::Subscription<livox_ros_driver::msg::CustomMsg>::SharedPtr sub_livox;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_livox2;
    rclcpp::Subscription<livox_interfaces::msg::CustomMsg>::SharedPtr sub_livox_avia;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_hesai;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_livox_mid360_boxi;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_odin;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_rs;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cur_scan;
    rclcpp::Publisher<estimate_msgs::msg::Estimate>::SharedPtr pub_est;
    rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr pub_start_time;
    std::shared_ptr<tf2_ros::TransformBroadcaster> br;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_world;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_feature_image;
    double pub_img_scale_ = 1.0;  
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_cam_poses;  

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_cam;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_cam_comp;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_cam_info;
    rclcpp::Subscription<unitree_msgs::msg::LowStateStamped>::SharedPtr sub_go2;
    rclcpp::Subscription<anymal_msgs::msg::AnymalState>::SharedPtr sub_anym;    
    CameraConfig camera_data;
    bool if_visual = false;
    std::shared_ptr<VisualFrontend> visual_frontend;
    ImgPreprocessing* img_preprocess;
    cv::Mat latest_img;
    resple2::ReusablePointBuffer<MapPointData> vpts_body;
    std::vector<int> nan_indices;
    std::vector<std::string> lidar_names;

    const std::string frame_id = "body";
    const std::string odom_id = "world";    

    std::map<std::string, LidarConfig> lidars;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_frame;
    resple2::VoxelDownsampler<pcl::PointXYZINormal> ds_filter_map;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last_ds;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_map_ds;
    pcl::PointCloud<pcl::PointXYZINormal> pc_world;
    pcl::PointCloud<pcl::PointXYZINormal> pc_world_visual;
    int point_filter_num = 1;
    int64_t time_offset = 0;
    std::atomic<int64_t> last_imu_ns{0};
    std::atomic<int64_t> last_img_ns{0};
    std::mutex m_img_buff_;
    resple2::BufferHandoff imu_handoff_, img_handoff_, kin_handoff_;
    int64_t max_time_ns = 0;

    std::shared_ptr<Kinematics> kinematics;
    std::atomic<int64_t> last_kin_ns{0};
    int64_t kin_time_offset_ns = 0;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_foot_pos;
    pinocchio::Model pmodel;
    std::shared_ptr<pinocchio::Data> pdata;    
    std::string model_path;    

    bool if_init_map = false;

    GeometryBatch::Config geometry_batch_;
    struct GeometryCandidate {
        int64_t time_ns;
        size_t lidar_id;
        const BufferedLidarPoint* point;
    };
    std::vector<GeometryCandidate> geometry_candidates_;
    struct GeometryEvaluation {
        ImuPredictionPreview::Pose pose;
        const VoxelPlane* plane = nullptr;
        double variance = 0.0;
    };
    Eigen::aligned_vector<GeometryEvaluation> geometry_evaluations_;
    struct KnotTriggers {
        bool position = false;
        bool rotation = false;
        bool velocity = false;
        bool motion() const { return position || rotation || velocity; }

    };
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
    MeasData meas_data;
    bool acc_ratio;

    bool if_init_filter = false;
    Estimator estimator;
    SplineState* spline;
    double cov_P0 = 0.02;
    double cov_RCP_pos_old = 0.02;
    double cov_RCP_ort_old = 0.02;
    double cov_RCP_pos_new = 0.1;    
    double cov_RCP_ort_new = 0.1;    
    double cov_sys_pos = 0.1;    
    double cov_sys_ort = 0.01;    
    Parameters param;
    int64_t dt_ns;

    bool if_loop_closure;
    pcl::PointCloud<pcl::PointXYZI> laserCloudWorld;
    int64_t last_pub_ns = 0;

    std::shared_ptr<VoxelMapManager> voxelmap_manager;
    std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
    std::vector<pointWithVar> pv_list;

    Eigen::Quaterniond q_WI = Eigen::Quaterniond::Identity();
    Eigen::Vector3d p_WI = Eigen::Vector3d::Zero();   

    void readParameters(rclcpp::Node::SharedPtr &nh)
    {
        param.nn_thresh = CommonUtils::readParam<double>(nh, "nn_thresh");
        acc_ratio = CommonUtils::readParam<bool>(nh, "acc_ratio");
        const auto ba_initial = CommonUtils::readParam<std::vector<double>>(nh, "bias_acc_initial_var", {});
        const auto bg_initial = CommonUtils::readParam<std::vector<double>>(nh, "bias_gyro_initial_var", {});
        const auto ba_psd = CommonUtils::readParam<std::vector<double>>(nh, "bias_acc_rw_psd", {});
        const auto bg_psd = CommonUtils::readParam<std::vector<double>>(nh, "bias_gyro_rw_psd", {});
        auto& noise = param.bias_noise;
        noise.continuous = !ba_initial.empty() || !bg_initial.empty() || !ba_psd.empty() || !bg_psd.empty();
        if (noise.continuous) {
            noise.initial_acc = BiasNoise::readDiagonal(ba_initial, "bias_acc_initial_var");
            noise.initial_gyro = BiasNoise::readDiagonal(bg_initial, "bias_gyro_initial_var");
            noise.drift_acc = BiasNoise::readDiagonal(ba_psd, "bias_acc_rw_psd");
            noise.drift_gyro = BiasNoise::readDiagonal(bg_psd, "bias_gyro_rw_psd");
        } else {
            noise.initial_acc = BiasNoise::readDiagonal(
                CommonUtils::readParam<std::vector<double>>(nh, "cov_ba"), "cov_ba");
            noise.initial_gyro = BiasNoise::readDiagonal(
                CommonUtils::readParam<std::vector<double>>(nh, "cov_bg"), "cov_bg");
            noise.drift_acc = noise.initial_acc;
            noise.drift_gyro = noise.initial_gyro;
        }
        std::vector<double> acc_var = CommonUtils::readParam<std::vector<double>>(nh, "cov_acc");
        param.cov_acc << acc_var.at(0), acc_var.at(1), acc_var.at(2);
        param.acc_huber_delta = CommonUtils::readParam<double>(nh, "acc_huber_delta", 0.0);
        ImuRobustLoss::validateConfig(param.acc_huber_delta, param.cov_acc);
        std::vector<double> gyro_var = CommonUtils::readParam<std::vector<double>>(nh, "cov_gyro");
        param.cov_gyro << gyro_var.at(0), gyro_var.at(1), gyro_var.at(2);

        dt_ns = 10000000; // 10 ms initial knot interval (100 Hz).
        double dt_s = double(dt_ns) * 1e-9;
        cov_P0 = CommonUtils::readParam<double>(nh, "cov_P0");
        cov_P0 *= (dt_s*dt_s);
        cov_RCP_pos_old = CommonUtils::readParam<double>(nh, "cov_RCP_pos_old");
        cov_RCP_ort_old = CommonUtils::readParam<double>(nh, "cov_RCP_ort_old");
        cov_RCP_pos_new = CommonUtils::readParam<double>(nh, "cov_RCP_pos_new");
        cov_RCP_ort_new = CommonUtils::readParam<double>(nh, "cov_RCP_ort_new");
        double std_pos = CommonUtils::readParam<double>(nh, "std_sys_pos");
        double std_ort = CommonUtils::readParam<double>(nh, "std_sys_ort");
        cov_sys_pos = std_pos*std_pos;
        cov_sys_ort = std_ort*std_ort;   
        param.coeff_cov = CommonUtils::readParam<double>(nh, "coeff_cov", 10);
        param.lidar_cov_floor = CommonUtils::readParam<double>(nh, "lidar_cov_floor", 0.001);
        if (!std::isfinite(param.lidar_cov_floor) || param.lidar_cov_floor < 0.0) {
            throw std::invalid_argument("lidar_cov_floor must be finite and positive (variance in m^2)");
        }

        point_filter_num = CommonUtils::readParam<int>(nh, "point_filter_num");
        param.num_points_upd = CommonUtils::readParam<int>(nh, "num_points_upd");
        estimator.n_iter = CommonUtils::readParam<int>(nh, "n_iter");
        pc_last_ds.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        pc_map_ds.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        double lidar_time_offset = CommonUtils::readParam<double>(nh, "lidar_time_offset", 0.0);
        time_offset = 1e9*lidar_time_offset;
        ds_filter_map.setUseCustom(CommonUtils::readParam<bool>(nh, "if_custom_ds", true));

        NUM_OF_THREAD = CommonUtils::readParam<int>(nh, "num_threads", 5);
        param.max_dt_ns = 1e9 / CommonUtils::readParam<int>(nh, "min_knot_hz");
        param.num_imu_upd = CommonUtils::readParam<int>(nh, "num_imu_upd");
        std::vector<double> pred_acc_cov = CommonUtils::readParam<std::vector<double>>(nh, "pred_acc_cov");
        param.pred_acc_cov << pred_acc_cov.at(0), pred_acc_cov.at(1), pred_acc_cov.at(2);
        std::vector<double> pred_gyro_cov = CommonUtils::readParam<std::vector<double>>(nh, "pred_gyro_cov");
        param.pred_gyro_cov << pred_gyro_cov.at(0), pred_gyro_cov.at(1), pred_gyro_cov.at(2);
        param.thresh_p = CommonUtils::readParam<double>(nh, "thresh_p");
        param.thresh_del = CommonUtils::readParam<double>(nh, "thresh_del");
        param.thresh_dv = CommonUtils::readParam<double>(nh, "thresh_dv", 0.0);
        param.dynamic_min_dt_ms = CommonUtils::readParam<double>(nh, "dynamic_min_dt_ms", 10.0);
        if (!std::isfinite(param.thresh_dv) || param.thresh_dv < 0.0)
            throw std::invalid_argument("thresh_dv must be finite and >= 0 (0 disables the velocity-change trigger)");
        if (!std::isfinite(param.dynamic_min_dt_ms) || param.dynamic_min_dt_ms <= 0.0 ||
            (param.thresh_dv > 0.0 && param.dynamic_min_dt_ms * 1e6 > param.max_dt_ns))
            throw std::invalid_argument("dynamic_min_dt_ms must be finite and > 0, and not exceed the maximum knot interval when thresh_dv is enabled");
        geometry_batch_.enabled = CommonUtils::readParam<bool>(nh, "geometry_batch.enabled", false);
        if (geometry_batch_.enabled) {
            geometry_batch_.first_points = param.num_points_upd;
            geometry_batch_.check_stride = CommonUtils::readParam<int>(nh, "geometry_batch.check_stride", 64);
            geometry_batch_.min_information = CommonUtils::readParam<double>(nh, "geometry_batch.min_information", 10000.0);
            geometry_batch_.min_eigen_ratio = CommonUtils::readParam<double>(nh, "geometry_batch.min_eigen_ratio", 0.01);
            geometry_batch_.validate(param.max_dt_ns);

            if (param.num_imu_upd < 1) throw std::invalid_argument("geometry_batch requires num_imu_upd >= 1");
        }
        VoxelMapConfig voxel_config;
        loadVoxelConfig(nh, voxel_config);
        voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));  
        float ds_map_size = voxel_config.max_voxel_size_/5.0;
        ds_filter_map.setLeafSize(ds_map_size, ds_map_size, ds_map_size);

        if_visual = CommonUtils::readParam<bool>(nh, "if_visual");
        param.if_visual = if_visual;
        if (if_visual) {
            auto cam_names = nh->declare_parameter<std::vector<std::string>>("cameras", std::vector<std::string>());
            camera_data.loadParameters(nh, cam_names[0] + ".");            
            visual_frontend.reset(new VisualFrontend());
            visual_frontend->grid_n_height = CommonUtils::readParam<int>(nh, "grid_n_height");
            visual_frontend->width = camera_data.width;
            visual_frontend->height = camera_data.height;
            visual_frontend->max_allow_err = CommonUtils::readParam<double>(nh, "max_allow_err");
            param.img_point_cov = CommonUtils::readParam<double>(nh, "img_point_cov"); 
            param.img_huber_loss = CommonUtils::readParam<double>(nh, "img_huber_loss"); 
            visual_frontend->initialize();
            img_preprocess = new ImgPreprocessing(nh, cam_names[0] + ".");
            if_loop_closure = CommonUtils::readParam<bool>(nh, "if_loop_closure");  
        } else {
            if_loop_closure = false;
        }
        param.if_kin =  CommonUtils::readParam<bool>(nh, "if_kin");
        if (param.if_kin) {
            kin_time_offset_ns = CommonUtils::readParam<int64_t>(nh, "kin_time_offset_ns", 0);
            Kinematics::Config kinematics_config;
            kinematics_config.contact_force_threshold_up = CommonUtils::readParam<double>(nh, "contact_force_threshold_up", 0);
            kinematics_config.contact_force_threshold_down = CommonUtils::readParam<double>(nh, "contact_force_threshold_down", 0);
            kinematics.reset(new Kinematics(kinematics_config));  
            param.cov_kin_vel = CommonUtils::readParam<double>(nh, "cov_kin_vel");
            std::vector<double> q_br_v;
            q_br_v = CommonUtils::readParam<std::vector<double>>(nh, "q_br", {1.0, 0.0, 0.0, 0.0});            
            param.q_br = Eigen::Quaterniond(q_br_v[0], q_br_v[1], q_br_v[2], q_br_v[3]);             
            param.t_br = CommonUtils::readVector3d(nh, "t_br");
            model_path = CommonUtils::readParam<std::string>(nh, "model_path");
            if (!model_path.empty() && model_path.front() != '/') {
                model_path = std::string(RESPLENU_SOURCE_DIR) + "/" + model_path;
            }
            param.cov_kin_pt2pl = CommonUtils::readParam<double>(nh, "cov_kin_pt2pl");
        }     
    }

    void initFilter(int64_t start_t_ns, Eigen::Vector3d t_init = Eigen::Vector3d::Zero(), Eigen::Quaterniond q_init = Eigen::Quaterniond::Identity())
    {
        Eigen::Matrix<double, 24, 24> cov_RCPs = cov_P0 * Eigen::Matrix<double, 24, 24>::Identity();
        Eigen::Matrix<double, 6, 6> Q_block_old = Eigen::Matrix<double, 6, 6>::Zero();
        Q_block_old.topLeftCorner<3, 3>() = cov_RCP_pos_old*cov_sys_pos *Eigen::Matrix3d::Identity();
        Q_block_old.bottomRightCorner<3, 3>() = cov_RCP_ort_old*cov_sys_ort *Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 6, 6> Q_block_new = Eigen::Matrix<double, 6, 6>::Zero();
        Q_block_new.topLeftCorner<3, 3>() = cov_RCP_pos_new*cov_sys_pos *Eigen::Matrix3d::Identity();
        Q_block_new.bottomRightCorner<3, 3>() = cov_RCP_ort_new*cov_sys_ort *Eigen::Matrix3d::Identity();   

        constexpr int base_size = 30;
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(base_size, base_size);
        Eigen::MatrixXd cov_x = Eigen::MatrixXd::Zero(base_size, base_size);
        for (int i = 0; i < 2; ++i) {
            Q.block<6, 6>(i*6, i*6) = Q_block_old;
        }
        Q.block<6, 6>(12, 12) = Q_block_old;   
        Q.block<6, 6>(18, 18) = Q_block_new;   

        cov_x.topLeftCorner<24, 24>() = cov_RCPs;
        cov_x.block<3, 3>(24, 24) = param.bias_noise.initial_acc.asDiagonal();
        cov_x.block<3, 3>(27, 27) = param.bias_noise.initial_gyro.asDiagonal();
        estimator.setState(dt_ns, start_t_ns, t_init, q_init, Q, cov_x, &param);
        spline = estimator.getSpline();
        estimator.meas_data = &meas_data;      
    }   

    void getImuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
    {
        imu_handoff_.receive(meas_data.m_buff, [&] {
            meas_data.imu_int_buff.push_back(imu_msg);
        });
        last_imu_ns = rclcpp::Time(imu_msg->header.stamp).nanoseconds();
    }    

    template<typename T>
    void ousterLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr ouster_msg_in)
    {
        std::string name = "Ouster";
        const LidarConfig& lidar = lidars.at(name);        
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());
        typename pcl::PointCloud<T>::Ptr pc_last_ouster(new typename pcl::PointCloud<T>());
        pcl::fromROSMsg(*ouster_msg_in, *pc_last_ouster);
        size_t plsize = pc_last_ouster->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        int64_t time_begin = rclcpp::Time(ouster_msg_in->header.stamp).nanoseconds() - time_offset;
        static int64_t last_t_ns = time_begin;
        int64_t max_ofs_ns = 0;
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                pt.x = pc_last_ouster->points[i].x;
                pt.y = pc_last_ouster->points[i].y;
                pt.z = pc_last_ouster->points[i].z;
                pt.intensity = float (pc_last_ouster->points[i].t) / float (1e6);
                pt.curvature = pc_last_ouster->points[i].intensity;
                if (pt.intensity >= 0 && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && pc_last_ouster->points[i].t + time_begin > last_t_ns) {
                    pc_last->points.push_back(pt);
                    int64_t ofs = pc_last_ouster->points[i].t;
                    max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                }
            }
        }
        LidarData& lidar_buffs = meas_data.lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();        
        last_t_ns = time_begin + max_ofs_ns;
    }        

    void livoxLidarCallback(const livox_ros_driver::msg::CustomMsg::SharedPtr livox_msg_in)
    {
        std::string name = "Mid70Avia";
        const LidarConfig& lidar = lidars.at(name);   
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());     
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        int64_t time_begin = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        static int64_t last_t_ns = time_begin;
        int64_t max_ofs_ns = 0;
        int valid_point_num = 0;
        pcl::PointXYZINormal pt_pre;
        pt_pre.x = livox_msg_in->points[0].x;
        pt_pre.y = livox_msg_in->points[0].y;
        pt_pre.z = livox_msg_in->points[0].z;
        int N_SCAN_LINES = lidar.scan_line;
        float blind = lidar.blind;        
        for (int i = 1; i < plsize; ++i) {
            if ((livox_msg_in->points[i].line < N_SCAN_LINES) && ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00)) {
                valid_point_num++;
                if (valid_point_num % point_filter_num == 0 && livox_msg_in->points[i].x > 0.7) {
                    pcl::PointXYZINormal pt;
                    pt.x = livox_msg_in->points[i].x;
                    pt.y = livox_msg_in->points[i].y;
                    pt.z = livox_msg_in->points[i].z;
                    pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6);
                    if (fabs(pt.intensity - pt_pre.intensity) >= 1.0) {
                        pt.intensity = pt_pre.intensity + 0.004166667f;
                    }
                    pt.curvature = livox_msg_in->points[i].reflectivity;
                    if (pt.intensity >= 0 && ((abs(pt.x - pt_pre.x) > 1e-7) || (abs(pt.y - pt_pre.y) > 1e-7) || (abs(pt.z - pt_pre.z) > 1e-7))
                                            && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind)&& livox_msg_in->points[i].offset_time + time_begin > last_t_ns) {
                        int64_t ofs = livox_msg_in->points[i].offset_time;
                        max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                        pc_last->points.push_back(pt);
                    }
                    pt_pre = pt;
                }

            } 
        }
        LidarData& lidar_buffs = meas_data.lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();
        last_t_ns = time_begin + max_ofs_ns;
    }    

    void livoxLidar2Callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr livox_msg_in)
    {
        std::string name = "HAP360";
        const LidarConfig& lidar = lidars.at(name);     
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());        
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        int64_t time_begin = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds() - time_offset;
        static int64_t last_t_ns = time_begin;
        int64_t max_ofs_ns = 0;
        int valid_point_num = 0;
        pcl::PointXYZINormal pt_pre;
        pt_pre.x = livox_msg_in->points[0].x;
        pt_pre.y = livox_msg_in->points[0].y;
        pt_pre.z = livox_msg_in->points[0].z;
        int N_SCAN_LINES = lidar.scan_line;
        float blind = lidar.blind;          
        for (int i = 1; i < plsize; ++i) {
            if ((livox_msg_in->points[i].line < N_SCAN_LINES) && ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00)) {
                valid_point_num++;
                if (valid_point_num % point_filter_num == 0) {
                    pcl::PointXYZINormal pt;
                    pt.x = livox_msg_in->points[i].x;
                    pt.y = livox_msg_in->points[i].y;
                    pt.z = livox_msg_in->points[i].z;
                    pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6); 
                    pt.curvature = livox_msg_in->points[i].reflectivity;
                    if (pt.intensity >= 0 && ((abs(pt.x - pt_pre.x) > 1e-7) || (abs(pt.y - pt_pre.y) > 1e-7) || (abs(pt.z - pt_pre.z) > 1e-7))
                                            && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && livox_msg_in->points[i].offset_time + time_begin > last_t_ns) {
                        int64_t ofs = livox_msg_in->points[i].offset_time;
                        max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                        pc_last->points.push_back(pt);
                    }
                    pt_pre = pt;
                }
            } 
        }
        LidarData& lidar_buffs = meas_data.lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();        
        last_t_ns = time_begin + max_ofs_ns;
    }

     void livoxAVIACallback(const livox_interfaces::msg::CustomMsg::SharedPtr livox_msg_in)
     {
        std::string name = "AviaResple";
        const LidarConfig& lidar = lidars.at(name);       
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());      
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        int64_t time_begin = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        static int64_t last_t_ns = time_begin;
        int64_t max_ofs_ns = 0;
        int valid_point_num = 0;
        pcl::PointXYZINormal pt_pre;
        pt_pre.x = livox_msg_in->points[0].x;
        pt_pre.y = livox_msg_in->points[0].y;
        pt_pre.z = livox_msg_in->points[0].z;
        int N_SCAN_LINES = lidar.scan_line;
        float blind = lidar.blind;           
        for (int i = 1; i < plsize; ++i) {
            if ((livox_msg_in->points[i].line < N_SCAN_LINES) && ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00) && livox_msg_in->points[i].offset_time + time_begin > last_t_ns) {
                valid_point_num++;
                if (valid_point_num % point_filter_num == 0) {
                    pcl::PointXYZINormal pt;
                    pt.x = livox_msg_in->points[i].x;
                    pt.y = livox_msg_in->points[i].y;
                    pt.z = livox_msg_in->points[i].z;
                    pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6); 
                    pt.curvature = livox_msg_in->points[i].reflectivity;
                    if (pt.intensity >= 0 && ((abs(pt.x - pt_pre.x) > 1e-7) || (abs(pt.y - pt_pre.y) > 1e-7) ||
                                            (abs(pt.z - pt_pre.z) > 1e-7))
                                            && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind)) {
                        int64_t ofs = livox_msg_in->points[i].offset_time;
                        max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                        pc_last->points.push_back(pt);
                    }
                    pt_pre = pt;
                }
            }
        }
        LidarData& lidar_buffs = meas_data.lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();
        last_t_ns = time_begin + max_ofs_ns;
     }        

    void hesaiLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr hesai_msg_in)
	{
        std::string name = "Hesai";
        const LidarConfig& lidar = lidars.at(name);    
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());    
        pcl::PointCloud<hesai_ros::Point>::Ptr pc_last_hesai(new pcl::PointCloud<hesai_ros::Point>());
        pcl::fromROSMsg(*hesai_msg_in, *pc_last_hesai);
        size_t plsize = pc_last_hesai->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(hesai_msg_in->header.stamp)  - rclcpp::Duration(0, time_offset);
        int64_t time_begin = timestamp_begin.nanoseconds();
        static int64_t last_t_ns = time_begin;
        int64_t max_ofs_ns = 0;        
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                pt.x = pc_last_hesai->points[i].x;
                pt.y = pc_last_hesai->points[i].y;
                pt.z = pc_last_hesai->points[i].z;
                double timestamp_s;
                double timestamp_ns = std::modf(pc_last_hesai->points[i].timestamp  - double(time_offset)*1e-9, &timestamp_s);                
                rclcpp::Time timestamp_ros(static_cast<int32_t>(timestamp_s), static_cast<int32_t>(timestamp_ns * 1.0e9),
                    rcl_clock_type_t::RCL_ROS_TIME);
                pt.intensity = (timestamp_ros - timestamp_begin).seconds() * 1.0e3; 
                pt.curvature = pc_last_hesai->points[i].intensity;
                if (pt.intensity >= 0 && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && CommonUtils::ms2ns(pt.intensity) + time_begin > last_t_ns) {
                    int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                    max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                    pc_last->points.push_back(pt);
                }
            }
        }
        LidarData& lidar_buffs = meas_data.lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();        
        last_t_ns = time_begin + max_ofs_ns;
	}     

    void livoxMid360BoxiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr livox_msg_in)
	{
        std::string name = "Mid360Boxi";
        const LidarConfig& lidar = lidars.at(name);   
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());     
        pcl::PointCloud<livox_mid360_boxi::Point>::Ptr pc_last_livox(new pcl::PointCloud<livox_mid360_boxi::Point>());
        pcl::fromROSMsg(*livox_msg_in, *pc_last_livox);
        size_t plsize = pc_last_livox->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(livox_msg_in->header.stamp);
        int64_t time_begin = timestamp_begin.nanoseconds();
        static int64_t last_t_ns = time_begin;   
        int64_t max_ofs_ns = 0;         
        pcl::PointXYZINormal pt;
        int N_SCAN_LINES = lidar.scan_line;
        float blind = lidar.blind;
        for (unsigned int i = 0; i < plsize; ++i) {
            if ((pc_last_livox->points[i].line < N_SCAN_LINES) && ((pc_last_livox->points[i].tag & 0x30) == 0x10 || (pc_last_livox->points[i].tag & 0x30) == 0x00)) {
                if (i % point_filter_num == 0) {
                    pt.x = pc_last_livox->points[i].x;
                    pt.y = pc_last_livox->points[i].y;
                    pt.z = pc_last_livox->points[i].z;
                    rclcpp::Time timestamp_ros(static_cast<int64_t>(pc_last_livox->points[i].timestamp),
                        rcl_clock_type_t::RCL_ROS_TIME);
                    pt.intensity = (timestamp_ros - timestamp_begin).seconds() * 1.0e3;
                    pt.curvature = pc_last_livox->points[i].intensity;
                    if (pt.intensity >= 0 && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && CommonUtils::ms2ns(pt.intensity) + time_begin > last_t_ns) {
                        int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                        max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                        pc_last->points.push_back(pt);
                    }
                }
            }
        }
        LidarData& lidar_buffs = meas_data.lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();     
        last_t_ns = time_begin + max_ofs_ns;
	}

    void odinCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::string name = "Odin";
        const LidarConfig& lidar = lidars.at(name);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());
        pcl::PointCloud<ls_ros::Point>::Ptr pc_last_livox(new pcl::PointCloud<ls_ros::Point>());
        pcl::fromROSMsg(*msg, *pc_last_livox);
        size_t plsize = pc_last_livox->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(msg->header.stamp);
        int64_t time_begin = timestamp_begin.nanoseconds();
        static int64_t last_t_ns = time_begin;
        int64_t max_ofs_ns = 0;
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                pt.x = pc_last_livox->points[i].x;
                pt.y = pc_last_livox->points[i].y;
                pt.z = pc_last_livox->points[i].z;
                pt.intensity = pc_last_livox->points[i].offset_time * 1.0e3;
                pt.curvature = static_cast<float>(pc_last_livox->points[i].intensity);
                if (pt.intensity >= 0 && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && CommonUtils::ms2ns(pt.intensity) + time_begin > last_t_ns) {
                    int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                    max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                    pc_last->points.push_back(pt);
                }
            }
        }
        LidarData& lidar_buffs = meas_data.lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();
        last_t_ns = time_begin + max_ofs_ns;
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (img_preprocess->undist_maps_initialized) return;

        cv::Mat K(3, 3, CV_64F, (void*)msg->k.data());
        cv::Mat D(msg->d.size(), 1, CV_64F, (void*)msg->d.data());
        try {
            img_preprocess->init(K, D, visual_frontend->cam);
        } catch (const std::invalid_argument&) {
        }
    }        

    void getCameraCallback(const sensor_msgs::msg::Image::SharedPtr camera_msg_in)
    {
        const int64_t img_t_ns = camera_data.correctTimestampNs(rclcpp::Time(camera_msg_in->header.stamp).nanoseconds());
        if (!img_preprocess->undist_maps_initialized) return;
        cv::Mat img_get = cv_bridge::toCvCopy(camera_msg_in, sensor_msgs::image_encodings::BGR8 )->image;
        if (camera_data.if_vignette_correct || camera_data.if_undist) {
            img_preprocess->preprocess(img_get);
        }
        img_handoff_.receive(m_img_buff_, [&] {
            meas_data.img_int_buff.emplace_back(img_t_ns, img_get);
        });
        last_img_ns = img_t_ns;
    }

    void getCameraCompCallback(const sensor_msgs::msg::CompressedImage::SharedPtr camera_msg_in)
    {
        const int64_t img_t_ns = camera_data.correctTimestampNs(rclcpp::Time(camera_msg_in->header.stamp).nanoseconds());
        if (!img_preprocess->undist_maps_initialized) return;
        cv::Mat img_get = cv_bridge::toCvCopy(camera_msg_in, sensor_msgs::image_encodings::BGR8 )->image;
        if (camera_data.if_vignette_correct || camera_data.if_undist) {
            img_preprocess->preprocess(img_get);
        }
        img_handoff_.receive(m_img_buff_, [&] {
            meas_data.img_int_buff.emplace_back(img_t_ns, img_get);
        });
        last_img_ns = img_t_ns;      
    }    

    void getKinematicAnymalCallback(const anymal_msgs::msg::AnymalState::SharedPtr msg)
    {
        static int legInContactForNSteps_[4] = {0};
        int64_t timestamp = rclcpp::Time(msg->header.stamp).nanoseconds();

        KinImuData kin_imu_meas;
        kin_imu_meas.time_ns = timestamp;
        int legInContactDebounceThreshold_ = 3;
        std::map<std::string, bool> contact_;
        for (int legIndex = 0; legIndex < 4; legIndex++) {
            std::string jname = msg->contacts[legIndex].name;
            bool in_contact = msg->contacts[legIndex].state;
            if (in_contact) {
                ++legInContactForNSteps_[legIndex];
                if (legInContactForNSteps_[legIndex] >= legInContactDebounceThreshold_) {
                    contact_[jname] = true;
                } else {
                    contact_[jname] = false;
                }                
            } else {
                legInContactForNSteps_[legIndex] = 0;
                contact_[jname] = false;
            }
        }
        for (const auto& it : kinematics->map_foot_id) {
            const std::string& foot_name = it.first;
            int idx = it.second;
            kin_imu_meas.contact_[idx] = contact_[foot_name];
        }

        std::map<std::string, double> foot_angle;
        std::map<std::string, double> foot_angle_vel;
        for (int i = 0; i < 12; ++i) {
            std::string jname = msg->joints.name[i];
            foot_angle[jname] = msg->joints.position[i];
            foot_angle_vel[jname] = msg->joints.velocity[i];
        }
        kinematics->caculateFootPosVel(foot_angle, foot_angle_vel, kin_imu_meas.foot_pos_, kin_imu_meas.foot_vel_, pmodel, *pdata);  
        kin_handoff_.receive(meas_data.m_imu_buff, [&] {
            meas_data.kin_imu_int_buff.push_back(kin_imu_meas);
        });
        last_kin_ns = timestamp;
    }

    void getKinematicLowSStampedCallback(const unitree_msgs::msg::LowStateStamped::SharedPtr msg)
    {
        int64_t timestamp = rclcpp::Time(msg->header.stamp).nanoseconds();

        KinImuData kin_imu_meas;

        kin_imu_meas.time_ns = timestamp + kin_time_offset_ns;     

        kin_imu_meas.contact_[0] = kinematics->contacts_[0].update(msg->foot_force[1]);
        kin_imu_meas.contact_[1] = kinematics->contacts_[1].update(msg->foot_force[0]);
        kin_imu_meas.contact_[2] = kinematics->contacts_[2].update(msg->foot_force[3]);
        kin_imu_meas.contact_[3] = kinematics->contacts_[3].update(msg->foot_force[2]);

        std::map<std::string, double> foot_angle;
        std::map<std::string, double> foot_angle_vel;
        for (int i = 0; i < 12; ++i) {
            std::string jname = kinematics->go2_motor_names[i];
            foot_angle[jname] = msg->motor_state[i].q;
            foot_angle_vel[jname] = msg->motor_state[i].dq;
        }

        kinematics->caculateFootPosVel(foot_angle, foot_angle_vel, kin_imu_meas.foot_pos_, kin_imu_meas.foot_vel_, pmodel, *pdata);  

        kin_handoff_.receive(meas_data.m_imu_buff, [&] {
            meas_data.kin_imu_int_buff.push_back(kin_imu_meas);
        });
        last_kin_ns = kin_imu_meas.time_ns;
    }       

    void publishImgRgb(cv::Mat img_rgb, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img)
    {
        int64_t t_ns = spline->maxTimeNs();
        if (t_ns < 0) {
            return;
        }        
        std_msgs::msg::Header header;
        header.stamp = rclcpp::Time(t_ns);
        if (pub_img_scale_ < 1.0) {
            cv::resize(img_rgb, img_rgb, cv::Size(), pub_img_scale_, pub_img_scale_, cv::INTER_LINEAR);
        }
        auto msg = cv_bridge::CvImage(header, "bgr8", img_rgb).toImageMsg();
        pub_img->publish(*msg);
    }        

    void publishFrameWorld() 
    {
        size_t size = pc_world.points.size();
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(pc_world, laserCloudmsg);
        laserCloudmsg.header.frame_id = odom_id;
        pub_scan_world->publish(laserCloudmsg);
        if (if_loop_closure) {
            size_t cur_size = laserCloudWorld.points.size();
            laserCloudWorld.points.resize(cur_size + pc_world.points.size());
            for (size_t i = 0; i < size; i++) {
                laserCloudWorld.points[cur_size + i].x = pc_world.points[i].x;
                laserCloudWorld.points[cur_size + i].y = pc_world.points[i].y;
                laserCloudWorld.points[cur_size + i].z = pc_world.points[i].z;
                laserCloudWorld.points[cur_size + i].intensity = pc_world.points[i].curvature; 
            }
            int64_t max_t_ns = spline->maxTimeNs();
            if (max_t_ns >= 0 && max_t_ns - last_pub_ns > 1e9/20) {
                pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
                laserCloudmsg.header.stamp = rclcpp::Time(max_t_ns);
                laserCloudmsg.header.frame_id = odom_id;
                pub_cur_scan->publish(laserCloudmsg);
                laserCloudWorld.clear();
                last_pub_ns = max_t_ns;
            }
        }
    }   

    void pubCameraPoses()
    {
        if (!if_loop_closure) return;
        for (const auto& img : meas_data.img_meas) {
            int64_t t_ns = img.time_ns;

            Eigen::Quaterniond q_wb;
            spline->itpQuaternionFromDelta(t_ns, &q_wb);
            Eigen::Vector3d t_wb = spline->itpPosition(t_ns);
            Eigen::Vector3d t_wc = q_wb * camera_data.t_bc + t_wb;
            Eigen::Quaterniond q_wc = q_wb * camera_data.q_bc;            

            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = rclcpp::Time(t_ns);
            odom_msg.header.frame_id = odom_id;
            odom_msg.child_frame_id = "camera";
            odom_msg.pose.pose.position.x = t_wc.x();
            odom_msg.pose.pose.position.y = t_wc.y();
            odom_msg.pose.pose.position.z = t_wc.z();
            odom_msg.pose.pose.orientation.w = q_wc.w();
            odom_msg.pose.pose.orientation.x = q_wc.x();
            odom_msg.pose.pose.orientation.y = q_wc.y();
            odom_msg.pose.pose.orientation.z = q_wc.z();
            pub_cam_poses->publish(odom_msg);
        }
    }    

    bool initialization()
    {
        if (if_init_filter && if_init_map) {
            return true;
        } 
        for (const auto& [lidar_name, lidar_data] : meas_data.lidars_data) {
            if (lidar_data.pt_buff.empty()) {
                return false;
            }
        }
        int64_t start_t_ns = std::numeric_limits<int64_t>::max();
        for (const auto& [lidar_name, lidar_data] : meas_data.lidars_data) {
            start_t_ns = std::min(start_t_ns, std::max(lidar_data.pt_buff.front().time_ns, int64_t(0)));
        }        
        last_pub_ns = start_t_ns;
        if (!if_init_filter) {
            Eigen::Vector3d gravity_sum(0, 0, 0);
            Eigen::Vector3d bg_sum(0, 0, 0);
            int buff_size = meas_data.imu_buff.size();
            constexpr int n_imu = 15;
            if (buff_size < n_imu) {
                return false;
            }
            for (int i = 0; i < n_imu; i++) {
                gravity_sum += meas_data.imu_buff.at(i).accel;
                bg_sum += meas_data.imu_buff.at(i).gyro;
            }
            while (!meas_data.imu_buff.empty() && meas_data.imu_buff.front().time_ns < start_t_ns) {
                meas_data.imu_buff.pop_front();
            }
            gravity_sum /= n_imu;
            bg_sum /= n_imu;
            Eigen::Vector3d gravity_ave = gravity_sum.normalized() * 9.81;
            Eigen::Matrix3d R0 = CommonUtils::g2R(gravity_ave);
            double yaw = CommonUtils::R2ypr(R0).x();
            R0 = CommonUtils::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
            Eigen::Quaterniond q0(R0);
            q_WI = Quater::positify(q0);
            param.gravity = q_WI * gravity_ave;
            estimator.setBg(bg_sum);
            initFilter(start_t_ns, p_WI, q_WI);
            if_init_filter = true;            
            std_msgs::msg::Int64 start_time;
            start_time.data = start_t_ns;
            pub_start_time->publish(start_time);    
        }
        if (!if_init_map) {
            estimator.addInitialProcessNoise();
            pc_world.clear();
            pcl::PointCloud<pcl::PointXYZINormal> pc_body;
                for (size_t i = 0; i < vpts_body.size(); i++) {
                    if (vpts_body[i].time_ns < start_t_ns + 1e8) {
                        pcl::PointXYZINormal pt_w;
                        Association::pointBodyToWorld(p_WI, q_WI, vpts_body[i].pt,
                            pt_w, vpts_body[i].t_bl, vpts_body[i].q_bl);
                        pc_world.points.push_back(pt_w);
                        pc_body.points.push_back(vpts_body[i].pt);
                    } else {
                        break;
                    }
                }
            Eigen::Matrix3d cov = 0.01 * Eigen::Matrix3d::Identity();
            const auto& first_lid = lidars.begin()->second;
            voxelmap_manager->BuildVoxelMap(pc_world, pc_body, q_WI, p_WI, cov, first_lid.dept_err, first_lid.beam_err, first_lid.q_bl.toRotationMatrix());                 
            pc_world.clear();
            if_init_map = true;
        }       
        return false; 
    }

    bool collectMeasurements()
    {
        int64_t next_t_ns = spline->getKnotTimeNs(spline->numKnots() - 1)
            + int64_t(0.04*1e9);
        if (geometry_batch_.enabled) next_t_ns = spline->getKnotTimeNs(spline->numKnots() - 1);
        for (const auto& [lidar_name, lidar_data] : meas_data.lidars_data) {
            if (lidar_data.pt_buff.empty()) {
                return false;
            }
            if ((!geometry_batch_.enabled && lidar_data.pt_buff.size() <= size_t(param.num_points_upd)) || lidar_data.pt_buff.back().time_ns < next_t_ns) {
                return false;
            }             
        }    
        if (meas_data.imu_buff.empty() || meas_data.imu_buff.back().time_ns < next_t_ns) {
            return false;
        }        
        if (if_visual && (meas_data.img_buff.empty() || meas_data.img_buff.back().time_ns < next_t_ns)) {
            return false;
        }
        if (param.if_kin && (meas_data.kin_imu_buff.empty() || meas_data.kin_imu_buff.back().time_ns < next_t_ns)) {
            return false;
        }                      
        max_time_ns = determineKnotNs();
        if (max_time_ns == 0) {
            return false;
        }
        loadMeasurement(max_time_ns);
        if (max_time_ns + 1 > spline->getKnotTimeNs(spline->numKnots() - 1)) {
            Eigen::aligned_deque<ImuData> imu_pred;
            while (!meas_data.imu_pred.empty() && meas_data.imu_pred.front().time_ns <= max_time_ns) {    
                imu_pred.push_back(meas_data.imu_pred.front());
                meas_data.imu_pred.pop_front();
            }
            estimator.propagateIntState(max_time_ns + 1, imu_pred, param.gravity);
        }                  
        return true;
    }    

    void loadMeasurement(const int64_t max_t_ns)
    {
        for (auto& [lidar_name, lidar_data] : meas_data.lidars_data) {
            size_t count = 0;
            while (count < lidar_data.pt_buff.size() && lidar_data.pt_buff[count].time_ns <= max_t_ns)
                ++count;
            meas_data.pt_meas.append_from(lidar_data.pt_buff, count);
        }         
        while (!meas_data.imu_buff.empty() && meas_data.imu_buff.front().time_ns <= max_t_ns) {
            meas_data.imu_meas.emplace_back(meas_data.imu_buff.front());
            meas_data.imu_buff.pop_front();
        }
        if ((int) meas_data.imu_meas.size() > 3) {
            size_t n = meas_data.imu_meas.size();
            size_t n_ds = (size_t) 3;
            Eigen::aligned_deque<ImuData> imu_meas_ds;
            for (size_t i = 0; i < n_ds; i++) {
                size_t idx = (n_ds > 1) ? (i * (n - 1)) / (n_ds - 1) : (n - 1);
                imu_meas_ds.push_back(meas_data.imu_meas[idx]);
            }
            meas_data.imu_meas = std::move(imu_meas_ds);
        }
        if (if_visual) {
            while (!meas_data.img_buff.empty() && meas_data.img_buff.front().time_ns <= max_t_ns) {
                meas_data.img_meas.emplace_back(meas_data.img_buff.front());
                meas_data.img_buff.pop_front();
            } 
        }  
        if (param.if_kin) {
            while (!meas_data.kin_imu_buff.empty() && meas_data.kin_imu_buff.front().time_ns <= max_t_ns) {
                meas_data.kin_meas.emplace_back(meas_data.kin_imu_buff.front());
                meas_data.kin_imu_buff.pop_front();
            } 
        }
    }

    bool velocityChangeExceeded(int64_t start_ns, int64_t time_ns,
        const Eigen::Vector3d& start_velocity, const Eigen::Vector3d& velocity) const
    {
        const double dt_ms = (time_ns - start_ns) * 1e-6;
        return param.thresh_dv > 0.0 && dt_ms >= param.dynamic_min_dt_ms &&
            (velocity - start_velocity).norm() > param.thresh_dv;
    }

    template <class Buffer>
    static void discardBefore(Buffer& buffer, int64_t time_ns)
    {
        while (!buffer.empty() && buffer.front().time_ns < time_ns)
            buffer.pop_front();
    }

    template <bool SkipOldSamples>
    int extendKnotForImuCount(int64_t start_ns, int64_t& cutoff_ns, int64_t& imu_time_ns) const
    {
        int count = 0;
        for (const auto& imu : meas_data.imu_buff) {
            if constexpr (SkipOldSamples) {
                if (imu.time_ns < start_ns) continue;
            }
            if (count >= param.num_imu_upd) break;
            if (imu.time_ns >= start_ns) ++count;
            cutoff_ns = std::max(cutoff_ns, imu.time_ns);
            imu_time_ns = imu.time_ns;
        }
        return count;
    }

    KnotTriggers evaluateMotionTriggers(int64_t start_ns, int64_t time_ns,
        const Eigen::Vector3d& start_velocity, const Eigen::Vector3d& velocity,
        double position_change, double rotation_change) const
    {
        KnotTriggers triggers;
        triggers.velocity = velocityChangeExceeded(start_ns, time_ns, start_velocity, velocity);
        triggers.position = position_change > param.thresh_p;
        triggers.rotation = rotation_change > param.thresh_del;
        return triggers;
    }

    int64_t determineInitialKnotNs(bool& if_init)
    {
        const int64_t cutoff_ns = meas_data.lidars_data.empty()
            ? std::numeric_limits<int64_t>::max()
            : spline->getKnotTimeNs(spline->numKnots() - 1) - 1;
        if (meas_data.imu_buff.size() < (size_t) param.num_imu_upd ||
            meas_data.imu_buff.back().time_ns <= cutoff_ns)
            return 0;

        discardBefore(meas_data.imu_buff, spline->minTimeNs());
        discardBefore(meas_data.imu_pred, spline->minTimeNs());
        discardBefore(meas_data.imu_buff, spline->getKnotTimeNs(spline->numKnots() - 2));
        if (if_visual) {
            discardBefore(meas_data.img_buff, spline->minTimeNs());
            discardBefore(meas_data.img_buff, spline->getKnotTimeNs(spline->numKnots() - 2));
        }
        if (param.if_kin) {
            discardBefore(meas_data.kin_imu_buff, spline->minTimeNs());
            discardBefore(meas_data.kin_imu_buff, spline->getKnotTimeNs(spline->numKnots() - 2));
        }
        if_init = false;
        return cutoff_ns;
    }

    int64_t determineKnotNs()
    {
        static bool if_init = true;
        if (if_init) return determineInitialKnotNs(if_init);

        if (!geometry_batch_.enabled) {
            int64_t max_time_ns = std::numeric_limits<int64_t>::max();
            int64_t imu_time_ns = 0;
            for (auto& [lidar_name, lidar_data] : meas_data.lidars_data) {
                int64_t meas_dt_front = lidar_data.pt_buff.front().time_ns;
                for (int i = 0; i < (int) lidar_data.pt_buff.size(); i++) {
                    if (i >= param.num_points_upd || lidar_data.pt_buff[i].time_ns - meas_dt_front >= param.max_dt_ns) {
                        break;
                    } else {
                        max_time_ns = lidar_data.pt_buff[i].time_ns;
                    }
                }
            }
            if (meas_data.imu_buff.size() < (size_t) param.num_imu_upd ||
                meas_data.imu_buff.back().time_ns <= max_time_ns)
                return 0;

            discardBefore(meas_data.imu_buff, spline->minTimeNs());
            discardBefore(meas_data.imu_pred, spline->minTimeNs());
            const int64_t interval_start_ns = spline->getKnotTimeNs(spline->numKnots() - 1);
            discardBefore(meas_data.imu_pred, interval_start_ns);
            discardBefore(meas_data.imu_buff, interval_start_ns);
            extendKnotForImuCount<false>(
                interval_start_ns, max_time_ns, imu_time_ns);

            const Eigen::Vector3d v0 = estimator.int_state.vk;
            while (!meas_data.imu_pred.empty() && meas_data.imu_pred.front().time_ns <= max_time_ns) {
                const auto& imu_data = meas_data.imu_pred.front();
                if (estimator.int_state.tk < imu_data.time_ns) {
                    Eigen::aligned_deque<ImuData> imu_pred;
                    imu_pred.push_back(imu_data);
                    estimator.propagateIntState(imu_data.time_ns, imu_pred, param.gravity);
                    Eigen::Vector3d p0 = spline->getKnotPos(spline->numKnots() - 1);
                    const KnotTriggers triggers = evaluateMotionTriggers(
                        interval_start_ns, estimator.int_state.tk, v0, estimator.int_state.vk,
                        (estimator.int_state.pk - p0).norm(), estimator.int_state.delk.norm());
                    if (triggers.motion()) {
                        max_time_ns = imu_data.time_ns;
                        break;
                    }
                }
                max_time_ns = std::max(max_time_ns, imu_data.time_ns);
                meas_data.imu_pred.pop_front();
            }
            if (if_visual) discardBefore(meas_data.img_buff, spline->minTimeNs());
            if (param.if_kin) discardBefore(meas_data.kin_imu_buff, spline->minTimeNs());
            return max_time_ns;
        }

        const int64_t start_ns = spline->getKnotTimeNs(spline->numKnots() - 1);
        if (meas_data.imu_pred.empty() || meas_data.imu_buff.empty()) return 0;
        int64_t available_ns = std::min(meas_data.imu_pred.back().time_ns,
                                       meas_data.imu_buff.back().time_ns) - 1;
        for (const auto& item : meas_data.lidars_data) {
            if (item.second.pt_buff.empty()) return 0;
            available_ns = std::min(available_ns, item.second.pt_buff.back().time_ns);
        }
        if (if_visual) {
            if (meas_data.img_buff.empty()) return 0;
            available_ns = std::min(available_ns, meas_data.img_buff.back().time_ns);
        }
        if (param.if_kin) {
            if (meas_data.kin_imu_buff.empty()) return 0;
            available_ns = std::min(available_ns, meas_data.kin_imu_buff.back().time_ns);
        }
        if (available_ns < start_ns) return 0;
        const int64_t time_limit_ns = start_ns + param.max_dt_ns - 1;
        int64_t limit_ns = time_limit_ns;
        std::string limit_reason = "time_limit";
        const int64_t min_interval_ns = start_ns + int64_t(geometry_batch_.min_dt_ms * 1e6) - 1;
        int64_t earliest_ns = min_interval_ns;
        int64_t imu_ready_ns = start_ns;
        const int imu_count = extendKnotForImuCount<true>(start_ns, earliest_ns, imu_ready_ns);
        if (imu_count < param.num_imu_upd && available_ns < limit_ns) return 0;

        auto preview = estimator.makePredictionPreview();
        if (preview.timeNs() != start_ns)
            throw std::logic_error("Geometry preview and spline knot timestamps differ");
        if (!preview.hasSample()) {
            for (const auto& imu : meas_data.imu_pred) {
                if (imu.time_ns >= start_ns) { preview.seed(imu.accel, imu.gyro); break; }
            }
        }
        if (!preview.hasSample()) return 0;
        const auto origin = preview.at(start_ns);
        auto motion_preview = preview;
        KnotTriggers limit_triggers;
        const auto motionLimitReason = [&](const ImuPredictionPreview::Pose& pose) -> const char* {
            limit_triggers = evaluateMotionTriggers(origin.time_ns, pose.time_ns, origin.v, pose.v,
                (pose.p - origin.p).norm(), 0.5 * origin.q.angularDistance(pose.q));
            const bool control_point_change = limit_triggers.position || limit_triggers.rotation;
            if (limit_triggers.velocity && control_point_change) return "velocity_and_motion_limit";
            if (limit_triggers.velocity) return "velocity_limit";
            if (control_point_change) return "motion_limit";
            return nullptr;
        };
        for (const auto& imu : meas_data.imu_pred) {
            if (imu.time_ns < start_ns) continue;
            if (imu.time_ns > std::min(limit_ns, available_ns) + 1) break;
            motion_preview.pushSample(imu.time_ns, imu.accel, imu.gyro);
            if (const char* reason = motionLimitReason(motion_preview.at(imu.time_ns))) {
                limit_ns = imu.time_ns - 1;
                limit_reason = reason;
                break;
            }
        }
        if (limit_reason == "time_limit" && available_ns >= limit_ns) {
            if (const char* reason = motionLimitReason(motion_preview.at(limit_ns + 1)))
                limit_reason = reason;
        }

        using Candidate = GeometryCandidate;
        auto& candidates = geometry_candidates_;
        candidates.clear();
        size_t lidar_id = 0;
        for (const auto& item : meas_data.lidars_data) {
            for (const auto& point : item.second.pt_buff) {
                if (point.time_ns < start_ns) continue;
                if (point.time_ns > std::min(limit_ns, available_ns)) break;
                candidates.push_back({point.time_ns, lidar_id, &point});
            }
            ++lidar_id;
        }
        const auto earlier = [](const Candidate& a, const Candidate& b) { return a.time_ns < b.time_ns; };
        if (!std::is_sorted(candidates.begin(), candidates.end(), earlier))
            std::stable_sort(candidates.begin(), candidates.end(), earlier);
        size_t imu_index = 0;
        bool has_pose = false;
        auto pose = origin;
        struct RotationCache {
            bool valid = false;
            int64_t time_ns = 0;
            Eigen::Quaterniond q_bl;
            Eigen::Matrix3d R_wl;
        };
        struct Worker {
            Association::PreviewLookupCache lookup_cache;
            Eigen::aligned_vector<RotationCache> rotations;
        };
        const int worker_count = std::max(1, NUM_OF_THREAD);
        std::vector<Worker> workers(worker_count);
        for (auto& worker : workers) worker.rotations.resize(meas_data.lidars_data.size());
        auto& evaluations = geometry_evaluations_;
        size_t prepared_begin = 0;
        const double preview_variance = geometry_batch_.preview_position_std * geometry_batch_.preview_position_std;
        const auto prepare = [&](size_t begin, size_t end) {
            prepared_begin = begin;
            evaluations.resize(end - begin);
            for (size_t i = begin; i < end; ++i) {
                const auto& candidate = candidates[i];
                if (!has_pose || pose.time_ns != candidate.time_ns) {
                    while (imu_index < meas_data.imu_pred.size() &&
                           meas_data.imu_pred[imu_index].time_ns <= candidate.time_ns) {
                        const auto& imu = meas_data.imu_pred[imu_index++];
                        preview.pushSample(imu.time_ns, imu.accel, imu.gyro);
                    }
                    pose = preview.at(candidate.time_ns);
                    has_pose = true;
                }
                evaluations[i - begin].pose = pose;
                evaluations[i - begin].plane = nullptr;
            }
            #pragma omp parallel if(end - begin >= 256 && worker_count > 1) num_threads(worker_count)
            {
                auto& worker = workers[omp_get_thread_num()];
                const auto evaluate_point = [&](size_t i) {
                    const auto& candidate = candidates[i];
                    auto& result = evaluations[i - begin];
                    const auto& pose = result.pose;
                    const auto& source = *candidate.point;
                    Eigen::Vector3d point_w, query_point;
                    Eigen::Matrix3d point_var;
                    auto& rotation = worker.rotations[candidate.lidar_id];
                    {
                        if (!rotation.valid || rotation.time_ns != candidate.time_ns ||
                            std::memcmp(rotation.q_bl.coeffs().data(), source.q_bl.coeffs().data(), 4 * sizeof(double)) != 0) {
                            rotation.R_wl = (pose.q * source.q_bl).toRotationMatrix();
                            rotation.q_bl = source.q_bl;
                            rotation.time_ns = candidate.time_ns;
                            rotation.valid = true;
                        }
                        const Eigen::Matrix3d& R_wl = rotation.R_wl;
                        point_w = pose.p + pose.q * (source.q_bl * source.pt_b + source.t_bl);
                        if (!point_w.allFinite()) return;
                        query_point = point_w.cast<float>().cast<double>();
                        point_var = R_wl * source.body_var * R_wl.transpose() +
                            preview_variance * Eigen::Matrix3d::Identity();
                    }
                    const Eigen::Matrix3d& R_wl = rotation.R_wl;
                    const VoxelPlane* plane = Association::findPlaneAtWorld(voxelmap_manager,
                        query_point, point_var, &worker.lookup_cache);
                    if (!plane) return;
                    double variance;
                    {
                        const double residual = plane->normal_.dot(point_w) + plane->d_;
                        if (!std::isfinite(residual) || std::abs(residual) >= param.nn_thresh) return;
                        Eigen::Matrix<double, 1, 6> J_plane;
                        J_plane.head<3>() = (point_w - plane->center_).transpose();
                        J_plane.tail<3>() = -plane->normal_.transpose();
                        const Eigen::Vector3d normal_l = R_wl.transpose() * plane->normal_;
                        variance = param.lidar_cov_floor +
                            (J_plane * plane->plane_var_ * J_plane.transpose())(0, 0) +
                            normal_l.dot(source.body_var * normal_l);
                    }
                    result.plane = plane;
                    result.variance = variance;
                };
                #pragma omp for schedule(static)
                for (size_t i = begin; i < end; ++i) evaluate_point(i);
            }
        };
        const auto decision = GeometryBatch::chooseCutoff(candidates, earliest_ns, limit_ns,
            available_ns, limit_reason, geometry_batch_, [&](const Candidate& candidate, GeometryBatch::GeometryInformationAccumulator& batch) {
                const size_t i = &candidate - candidates.data();
                batch.addPoint();
                const auto& result = evaluations[i - prepared_begin];
                if (result.plane) batch.addMatch(result.plane->normal_, result.variance);
            }, prepare);
        if (!decision) return 0;

        discardBefore(meas_data.imu_buff, start_ns);
        discardBefore(meas_data.imu_pred, start_ns);
        return decision->cutoff_ns;
    }

    void trackImageMeasurements()
    {  
        if (meas_data.img_meas.empty()) {
            return;
        }        
        for (size_t i = 0; i < meas_data.img_meas.size(); i++) {
            ImgData& img_data = meas_data.img_meas[i];
            latest_img = img_data.img;      
            visual_frontend->initNewFrame(latest_img, camera_data.scale);
            estimator.updateFramePose(img_data.time_ns, camera_data, *visual_frontend);
            visual_frontend->trackImg();
            img_data.m_rgb_pts_ptr_vec_in_last_frame = visual_frontend->m_rgb_pts_ptr_vec_in_last_frame;
            img_data.m_last_tracked_pts = visual_frontend->m_last_tracked_pts;
        }
    }

    void publishFootPositions()
    {
        if (meas_data.kin_meas.empty()) {
            return;
        }
        const KinImuData& kin_data = meas_data.kin_meas.back();
        Eigen::Matrix<double, 12, 1> v_foot;
        v_foot = getFootWorld(kin_data.time_ns, kin_data.foot_pos_);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr foot_pts(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (size_t i = 0; i < 4; i++) {
            pcl::PointXYZRGB ptxyz;
            Eigen::Vector3d foot = v_foot.segment<3>(i*3);
            ptxyz.x = foot.x();
            ptxyz.y = foot.y();
            ptxyz.z = foot.z();     
            if (kin_data.contact_[i]) { 
                ptxyz.r = 0;
                ptxyz.g = 255;
                ptxyz.b = 0;      
            } else {
                ptxyz.r = 255;
                ptxyz.g = 0;
                ptxyz.b = 0;     
            }          
            foot_pts->points.push_back(ptxyz);
        }
        sensor_msgs::msg::PointCloud2 points_msg;
        pcl::toROSMsg(*foot_pts, points_msg);    
        points_msg.header.frame_id = odom_id;        
        pub_foot_pos->publish(points_msg);
    }

    Eigen::Matrix<double, 12, 1> getFootWorld(int64_t time_ns, const double (&foot_pos_)[4][3])
    {
        Eigen::Matrix<double, 12, 1> v_foot;
        Eigen::Quaterniond q_itp;
        spline->itpQuaternionFromDelta(time_ns, &q_itp);
        Eigen::Vector3d p_itp = spline->itpPosition(time_ns);
        Eigen::Quaterniond q_wr = q_itp * param.q_br;
        Eigen::Vector3d p_wr = q_itp * param.t_br + p_itp;
        for (int k = 0; k < 4; k++) {
            Eigen::Vector3d foot_pos(foot_pos_[k][0], foot_pos_[k][1], foot_pos_[k][2]);
            v_foot.segment(k*3, 3) = q_wr * foot_pos + p_wr;
        }      

        return v_foot;
    }    

};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto nh = rclcpp::Node::make_shared("RESPLE2");
    resple2::configureOpenCvThreads(nh);
    RESPLE2 resple(nh);
    rclcpp::Rate rate(2000);
    std::thread opt{&RESPLE2::processData, &resple};
    while (rclcpp::ok()) {
        rclcpp::spin_some(nh);
        rate.sleep();
    }
    opt.join();
    rclcpp::shutdown();
}
