/*
This file is part of RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement.
 
Copyright (c) 2026 Ziyu Cao

Developer: Ziyu Cao <ziyu.cao@liu.se>

For commercial use, please contact me at <ziyu.cao@liu.se> or Kailai Li at <kailai.li@liu.se>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file, 
which is included as part of this source code package.
*/

#include "GlobalRefinement.h"
#include "Linearizer.h"

#include <algorithm>
#include <chrono>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

LoopDetection::LoopDetection() {};

void LoopDetection::init(rclcpp::Node::SharedPtr &nh) {
    const bool if_visual = CommonUtils::readParam<bool>(nh, "if_visual", false);
    const bool legacy_rgb = CommonUtils::readParam<bool>(nh, "if_rbg_map", false);
    const bool if_rgb = CommonUtils::readParam<bool>(nh, "if_rgb", legacy_rgb);
    rgb_mapping_enabled = if_visual && if_rgb;

    pub_lc = nh->create_publisher<visualization_msgs::msg::MarkerArray>("loop_closure", 100);   
    client_reset = nh->create_client<std_srvs::srv::Empty>("/rviz2/reset_time"); 
    pub_drift = nh->create_publisher<geometry_msgs::msg::PoseStamped>("odom_drift", 2);
    srv_update_global_map = nh->create_service<std_srvs::srv::Trigger>(
        "/update_global_map", std::bind(&LoopDetection::updateGlobalMapService, this,
            std::placeholders::_1, std::placeholders::_2));

    gest.initOptParam(nh);
    lc_vis.setColor(255.0/255.0,0.0/255.0,255.0/255.0);
    lc_vis.setLineWidth(0.1);
    marker_publish_timer = nh->create_wall_timer(std::chrono::seconds(1), [this]() {
        const auto snapshot = std::atomic_load(&marker_snapshot);
        if (!snapshot) return;

        if (!snapshot->loop_closures.empty()) {
            publishLoopClosureConnection(snapshot->loop_closures);
        }
    });
}

void LoopDetection::correction()
{
    while (rclcpp::ok() && !if_finished) {
        Eigen::aligned_vector<LoopData> pending_lc;
        {
            std::lock_guard<std::mutex> lock(m_lc_buff);
            if (!lc_buf.empty() && gest.lc_in_opt.size() != lc_buf.size()) {
                pending_lc = lc_buf;
            }
        }
        if (pending_lc.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        const bool corrected = gest.process(pending_lc);
        if (corrected) {
            resetService();
            correctKeyframeUpdateMap();
            path_correction_pending.store(true);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

void LoopDetection::resetService()
{
    auto request = std::make_shared<std_srvs::srv::Empty::Request>();
    auto result = client_reset->async_send_request(request);
    rclcpp::Rate rate(100);
    while (!result.valid()) {
        rate.sleep();
    }
}

void LoopDetection::correctKeyframeUpdateMap()
{
    std::lock_guard<std::mutex> map_lock(m_map_update);

    Eigen::aligned_vector<PoseData> kf_poses;
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> kf_clouds;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> kf_rgb_clouds;
    {
        std::lock_guard<std::mutex> lk(keyframes.m_frame_buff);
        kf_poses.assign(keyframes.keyframePoses.begin(), keyframes.keyframePoses.end());
        kf_clouds = keyframes.keyframeLaserClouds;
        kf_rgb_clouds = keyframes.keyframeRgbClouds;
    }
    const size_t n_kf = kf_poses.size();

    for (auto& it : kf_poses) {
        int64_t pose_time_ns = it.time_ns;
        it.pos = spline_global.itpPosition(pose_time_ns);
        spline_global.itpQuaternionFromQuat(pose_time_ns, &it.orient);
    }

    updateGlobalMap(kf_poses, kf_clouds, kf_rgb_clouds);

    Eigen::Quaterniond rot_drift;
    Eigen::Vector3d trans_drift;
    gest.getDrift(rot_drift, trans_drift);
    {
        std::lock_guard<std::mutex> lk(keyframes.m_frame_buff);
        for (size_t i = 0; i < n_kf && i < keyframes.keyframePoses.size(); i++) {
            keyframes.keyframePoses[i].pos = kf_poses[i].pos;
            keyframes.keyframePoses[i].orient = kf_poses[i].orient;
        }
        keyframes.reset(rot_drift, trans_drift);
    }

    auto snapshot = std::make_shared<MarkerSnapshot>();
    snapshot->loop_closures = gest.lc_in_opt;
    std::atomic_store(&marker_snapshot,
                      std::shared_ptr<const MarkerSnapshot>(std::move(snapshot)));

    geometry_msgs::msg::PoseStamped drift_msg;
    drift_msg.header.stamp = rclcpp::Time(spline_global.maxTimeNs());
    drift_msg.pose.position.x = trans_drift.x();
    drift_msg.pose.position.y = trans_drift.y();
    drift_msg.pose.position.z = trans_drift.z();

    drift_msg.pose.orientation.w = rot_drift.w();
    drift_msg.pose.orientation.x = rot_drift.x();
    drift_msg.pose.orientation.y = rot_drift.y();
    drift_msg.pose.orientation.z = rot_drift.z();
    pub_drift->publish(drift_msg);
}

void LoopDetection::updateGlobalMapService(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    const std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    response->success = false;
    if (if_finished.load()) {
        response->message = "Mapping is stopping; the global map cannot be updated.";
        return;
    }

    try {
        std::lock_guard<std::mutex> map_lock(m_map_update);
        Eigen::aligned_vector<PoseData> kf_poses;
        std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> kf_clouds;
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> kf_rgb_clouds;
        {
            std::lock_guard<std::mutex> lock(keyframes.m_frame_buff);
            kf_poses = keyframes.keyframePoses;
            kf_clouds = keyframes.keyframeLaserClouds;
            kf_rgb_clouds = keyframes.keyframeRgbClouds;
        }
        if (kf_poses.empty()) {
            response->message = "No keyframes are available to build the global map.";
            return;
        }

        updateGlobalMap(kf_poses, kf_clouds, kf_rgb_clouds);
        response->success = true;
        response->message = "Global map rebuilt from " + std::to_string(kf_poses.size())
            + " keyframes and queued for publication.";
    } catch (const std::exception& error) {
        response->message = std::string("Global map update failed: ") + error.what();
    }
}

void LoopDetection::updateGlobalMap(const Eigen::aligned_vector<PoseData>& poses,
                                      const std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr>& clouds,
                                      const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr>& rgb_clouds)
{
    const uint64_t stamp_ns = poses.empty() ? 0 : poses.back().time_ns;
    if (!rgb_mapping_enabled) {
        auto global_map = pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>();
        for (size_t i = 0; i < poses.size() && i < clouds.size(); i++) {
            const auto& pose_new = poses[i];
            *global_map += *(CommonUtils::transformCloud<double>(clouds[i], pose_new.orient, pose_new.pos));
        }
        global_map->header.stamp = stamp_ns;
        global_map_pub.submit(global_map,              true);
    }

    auto global_rgb = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    for (size_t i = 0; i < poses.size() && i < rgb_clouds.size(); ++i) {
        if (!rgb_clouds[i] || rgb_clouds[i]->empty()) continue;
        pcl::PointCloud<pcl::PointXYZRGB> world_rgb;
        pcl::transformPointCloud(*rgb_clouds[i], world_rgb, poses[i].pos, poses[i].orient);
        *global_rgb += world_rgb;
    }
    global_rgb->header.stamp = stamp_ns;
    global_map_pub.submitCorrectedRgb(global_rgb);
}

void LoopDetection::publishLoopClosureConnection(const Eigen::aligned_vector<LoopData>& loop_closure_connection) const
{
    visualization_msgs::msg::MarkerArray loop_closure_msg;
    std_msgs::msg::Header header;
    header.frame_id = odom_id;
    int id = 0;
    for (const auto& it : loop_closure_connection) {
        Eigen::Vector3d pos_i = it.posi;
        Eigen::Vector3d pos_j = it.posj;
        lc_vis.pubCorresp(pos_i, pos_j - pos_i, header, id, loop_closure_msg);
        id++;
    }
    pub_lc->publish(loop_closure_msg);
}

GlobalEstimator::GlobalEstimator() {};

void GlobalEstimator::initOptParam(rclcpp::Node::SharedPtr &nh)
{
    max_iter = std::max(1, CommonUtils::readParam<int>(nh, "global_max_iter", 20));
    param.w_lc = CommonUtils::readParam<double>(nh, "w_lc");
    param.w_odom = CommonUtils::readParam<double>(nh, "w_odom", 1.0);
    param.if_loss_func = CommonUtils::readParam<bool>(nh, "if_loss_func");
    param.loss_func_thresh = CommonUtils::readParam<double>(nh, "loss_func_thresh");

}

bool GlobalEstimator::process(const Eigen::aligned_vector<LoopData>& loop_closures) {
    if (loop_closures.empty()) return false;
    int64_t min_t = loop_closures.back().time_i_ns;
    int64_t max_t = loop_closures.back().time_j_ns;
    for (const auto& it : loop_closures) {
        min_t = it.time_i_ns < min_t ? it.time_i_ns : min_t;
        max_t = it.time_j_ns > max_t ? it.time_j_ns : max_t;
    }
    int start_cp_idx, end_cp_idx;
    {
        auto spline_lock = splineReadLock();
        if (spline_global.numKnots() < 10 ||
            min_t < spline_global.minTimeNs() || max_t > spline_global.maxTimeNs()) {
            return false;
        }
        end_cp_idx = spline_global.findLeftCP(max_t) + 1;
        if (end_cp_idx + 2 >= spline_global.numKnots()) return false;

        min_t = std::max(min_t - 1, spline_global.minTimeNs());
        start_cp_idx = std::max(spline_global.findLeftCP(min_t) - 3, 0);
        spline_opt.clearSpline();
        spline_opt.init(&spline_global, start_cp_idx, end_cp_idx);
    }
    lc_in_opt = loop_closures;
    opt_start_t_ns = std::min(min_t, spline_opt.getKnotTimeNs(3));
    updateControlPointOdometryMeasurements();
    const bool optimized = optimization();
    if (!optimized) {
        return false;
    }

    lock_spline();

    spline_global.updateKnots(&spline_opt, &rot_drift, &trans_drift);
    std::atomic_store(&drift_snapshot, std::make_shared<const DriftState>(DriftState{rot_drift, trans_drift}));
    spline_global.computeDeltas();
    unlock_spline();
    for (auto& it : lc_in_opt) {
        it.posi = spline_global.itpPosition(it.time_i_ns);
        Eigen::Quaterniond orti;
        spline_global.itpQuaternionFromQuat(it.time_i_ns, &orti);
        it.posj = orti*it.translation + it.posi;
    }

    return true;
}

void GlobalEstimator::getDrift(Eigen::Quaterniond& q, Eigen::Vector3d& t) {
    std::shared_ptr<const DriftState> snap = std::atomic_load(&drift_snapshot);
    q = snap->rot;
    t = snap->trans;
};

void GlobalEstimator::lock_spline() { m_spline.lock(); }

void GlobalEstimator::unlock_spline() { m_spline.unlock(); }

void GlobalEstimator::lock_spline_shared() { m_spline.lock_shared(); }

void GlobalEstimator::unlock_spline_shared() { m_spline.unlock_shared(); }

bool GlobalEstimator::optimization()
{
    bool converged = false;
    bool terminated = false;
    int opt_iter = 0;

    radius = 1e4;
    decrease_factor = 2.0;
    int num_knots = spline_opt.numKnots();

    const size_t hess_size = Linearizer::POSE_SIZE * num_knots;
    global_state_size = (Linearizer::POSE_SIZE + 1) * num_knots;

    Linearizer lopt(hess_size, &spline_opt, &param, true);
    while (!converged && !terminated && opt_iter < max_iter) {
        converged = globalOptimize(opt_iter, terminated, lopt);
        opt_iter++;
    }
    if (terminated) {
        return false;
    }
    return true;
}

void GlobalEstimator::updateControlPointOdometryMeasurements()
{
    odom_meas.clear();
    const int n_knots = static_cast<int>(spline_opt.numKnots());
    if (n_knots < 2) return;
    odom_meas.reserve(n_knots - 1);
    for (int i = 0; i + 1 < n_knots; ++i) {
        const auto key = std::make_pair(spline_opt.getKnotTimeNs(i),
                                        spline_opt.getKnotTimeNs(i + 1));
        auto it = control_point_odom_cache.find(key);
        if (it == control_point_odom_cache.end()) {
            const Eigen::Quaterniond q_iw = spline_opt.getKnotOrt(i).inverse();
            Eigen::Quaterniond q_ij = q_iw * spline_opt.getKnotOrt(i + 1);
            Eigen::Vector3d p_ij = q_iw * (spline_opt.getKnotPos(i + 1) - spline_opt.getKnotPos(i));
            it = control_point_odom_cache.emplace(
                key, LoopData(key.first, key.second, q_ij, p_ij, i, i + 1)).first;
        }
        odom_meas.push_back(it->second);
        odom_meas.back().idx_i = i;
        odom_meas.back().idx_j = i + 1;
    }
}

void GlobalEstimator::getStateVector(Eigen::VectorXd *x, const Eigen::VectorXd *delta_x)
{
    x->setZero(global_state_size);
    size_t x_offset = 0;
    for (int64_t i = 0; i < spline_opt.numKnots(); i++) {
        x->segment(x_offset, 3) = spline_opt.getKnotPos(i);
        x_offset += 3;
        x->segment(x_offset, 4) = Eigen::Map<Eigen::VectorXd>(spline_opt.getKnotOrt(i).coeffs().data(), 4);
        x_offset += 4;
    }
    if (delta_x != nullptr) {
        x_offset = 0;
        int64_t x_offset_inc = 0;
        for (int64_t i = 0; i < spline_opt.numKnots(); i++) {
            x->segment(x_offset, 3) += delta_x->segment(x_offset_inc, 3);
            x_offset += 3;
            x_offset_inc += 3;
            Eigen::Quaterniond q_inc;
            Quater::exp(delta_x->segment(x_offset_inc, 3), q_inc);
            x->segment(x_offset, 4) = Eigen::Map<Eigen::VectorXd>((spline_opt.getKnotOrt(i) * q_inc).coeffs().data(), 4);
            x_offset += 4;
            x_offset_inc += 3;
        }
    }
}

bool GlobalEstimator::globalOptimize(const int iter, bool& terminated, Linearizer& lopt)
{
    {
        lopt.reset();
        lopt(lc_in_opt);
        lopt(odom_meas, param.w_odom, true);
    }

    Eigen::VectorXd Hdiag;
    Eigen::VectorXd Hdiag_scaled;
    {
        lopt.setupSolver();
        Hdiag = lopt.accum.Hdiagonal();
        if (!std::isfinite(lopt.error) || !Hdiag.allFinite() ||
            (Hdiag.array() < 0.0).any() || !lopt.accum.getB().allFinite()) {
            terminated = true;
            return false;
        }
        jacobian_scaling = 1.0 / (1.0 + Hdiag.array().sqrt());
        if (iter == 0) {
            minimum_cost_ = lopt.error;
            current_cost_ = lopt.error;
            reference_cost_ = lopt.error;
            candidate_cost_ = lopt.error;
            accumulated_reference_model_cost_change_ = 0;
            accumulated_candidate_model_cost_change_ = 0;
            num_consecutive_nonmonotonic_steps_ = 0;
        }
        lopt.accum.scaleSystem(jacobian_scaling);

        const double gradient_max_norm = lopt.accum.getB().lpNorm<Eigen::Infinity>();
        if (gradient_max_norm <= gradient_tolerance) {
            return true;
        }

        Hdiag_scaled = Hdiag.cwiseMax(min_lm_diagonal).cwiseMin(max_lm_diagonal);
        Hdiag_scaled.array() *= jacobian_scaling.array() * jacobian_scaling.array();
    }
    while (!terminated) {
        if (!std::isfinite(radius) || radius <= min_radius) {
            terminated = true;
            return false;
        }
        const Eigen::VectorXd Hdiag_lambda = Hdiag_scaled * (1.0 / radius);
        Eigen::VectorXd inc_full;
        {
            inc_full = -lopt.accum.solve(&Hdiag_lambda);
        }
        if (!inc_full.allFinite()) {
            radius /= decrease_factor;
            decrease_factor *= 2.0;
            continue;
        }
        const Eigen::VectorXd t0 = lopt.accum.getB() + lopt.accum.getH().selfadjointView<Eigen::Lower>() * inc_full * 0.5;
        const double model_cost_change = -2.0 * inc_full.dot(t0);

        inc_full = (inc_full.array() * jacobian_scaling.array()).matrix();
        if (!inc_full.allFinite() || !std::isfinite(model_cost_change) || model_cost_change <= 0.0) {
            radius /= decrease_factor;
            decrease_factor *= 2.0;
            continue;
        }
        Eigen::aligned_deque<Eigen::Vector3d> knots_trans_backup;
        Eigen::aligned_deque<Eigen::Quaterniond> knots_rot_backup;
        spline_opt.getAllStateKnots(knots_trans_backup, knots_rot_backup);
        applyIncFull(inc_full);
        ComputeErrorSplineOpt eopt(&spline_opt, &param);
        eopt(lc_in_opt);
        eopt(odom_meas, param.w_odom, true);
        Eigen::VectorXd temp_state_vector;
        getStateVector(&temp_state_vector);
        if (!std::isfinite(eopt.error) || !temp_state_vector.allFinite()) {
            spline_opt.setAllKnots(knots_trans_backup, knots_rot_backup);
            radius /= decrease_factor;
            decrease_factor *= 2.0;
            continue;
        }
        const double f_diff = lopt.error - eopt.error;
        const double step_quality = stepQuality(eopt.error, model_cost_change);
        if (!std::isfinite(step_quality) || step_quality < min_relative_decrease || f_diff < 0.0) {
            spline_opt.setAllKnots(knots_trans_backup, knots_rot_backup);
            radius /= decrease_factor;
            decrease_factor *= 2.0;
            continue;
        }

        radius /= std::max(1.0 / 3.0, 1.0 - pow(2.0 * step_quality - 1.0, 3));
        radius = std::min(max_radius, radius);
        decrease_factor = 2.0;
        stepAccepted(eopt.error, model_cost_change);
        if (f_diff <= lopt.error * cost_tolerance) {
            return true;
        }
        if (isStepSmall(inc_full)) {
            return true;
        }
        return false;
    }
    return false;
}

bool GlobalEstimator::isStepSmall(const Eigen::VectorXd& inc_full) const
{
    if (!inc_full.allFinite()) return false;
    const int64_t num_knots = spline_opt.numKnots();
    for (int64_t i = 0; i < num_knots; i++) {
        const int64_t offset = Linearizer::POSE_SIZE * i;
        if (inc_full.segment<3>(offset).norm() > translation_step_tolerance ||
            2.0 * inc_full.segment<3>(offset + Linearizer::ROT_OFFSET).norm() > rotation_step_tolerance) {
            return false;
        }
    }
    return true;
}

void GlobalEstimator::applyIncFull(Eigen::VectorXd& inc_full)
{
    int64_t num_knots = spline_opt.numKnots();
    for (int64_t i = 0; i < num_knots; i++) {
        Eigen::Matrix<double, 6, 1> inc = inc_full.segment<Linearizer::POSE_SIZE>(Linearizer::POSE_SIZE * i);
        spline_opt.applyPoseInc(i, inc);
    }

    spline_opt.checkQuaternionControlPoints();
    spline_opt.computeDeltas();
}

double GlobalEstimator::stepQuality(const double cost, const double model_cost_change) const
{
    if (!std::isfinite(cost) || cost >= std::numeric_limits<double>::max() ||
        !std::isfinite(model_cost_change) || model_cost_change <= 0.0) {
        return std::numeric_limits<double>::lowest();
    }

    const double relative_decrease = (current_cost_ - cost) / model_cost_change;
    const double historical_relative_decrease =
            (reference_cost_ - cost) /
            (accumulated_reference_model_cost_change_ + model_cost_change);
    if (!std::isfinite(relative_decrease) || !std::isfinite(historical_relative_decrease)) {
        return std::numeric_limits<double>::lowest();
    }
    return std::max(relative_decrease, historical_relative_decrease);
}

void GlobalEstimator::stepAccepted(const double cost, const double model_cost_change)
{
    current_cost_ = cost;
    accumulated_candidate_model_cost_change_ += model_cost_change;
    accumulated_reference_model_cost_change_ += model_cost_change;

    if (current_cost_ < minimum_cost_) {
        minimum_cost_ = current_cost_;
        num_consecutive_nonmonotonic_steps_ = 0;
        candidate_cost_ = current_cost_;
        accumulated_candidate_model_cost_change_ = 0.0;
    } else {
        ++num_consecutive_nonmonotonic_steps_;
        if (current_cost_ > candidate_cost_) {
            candidate_cost_ = current_cost_;
            accumulated_candidate_model_cost_change_ = 0.0;
        }
    }

    if (num_consecutive_nonmonotonic_steps_ ==
        max_consecutive_nonmonotonic_steps_) {
        reference_cost_ = candidate_cost_;
        accumulated_reference_model_cost_change_ =
                accumulated_candidate_model_cost_change_;
    }
}
