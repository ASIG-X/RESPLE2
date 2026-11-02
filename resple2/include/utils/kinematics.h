/*
This file is part of RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement.
 
Copyright (c) 2026 Ziyu Cao

Developer: Ziyu Cao <ziyu.cao@liu.se>

For commercial use, please contact me at <ziyu.cao@liu.se> or Kailai Li at <kailai.li@liu.se>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file, 
which is included as part of this source code package.
*/

#pragma once

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <Eigen/Core>
#include <array>
#include <cstring>
#include <map>
#include <string>
#include <vector>

class ContactDetector {
    double T_on_, T_off_;
    bool in_contact_{true};

   public:
    ContactDetector(double ton, double toff) : T_on_(ton), T_off_(toff) {}
    bool update(double val) {
        if (!in_contact_ && val > T_on_) {
            in_contact_ = true;
        } else if (in_contact_ && val < T_off_) {
            in_contact_ = false;
        }
        return in_contact_;
    }
};

class Kinematics {
   public:
    struct Config {
        double contact_force_threshold_up;
        double contact_force_threshold_down;
    };

    explicit Kinematics(const Config& config)
        : contacts_(std::vector<ContactDetector>(
              4, ContactDetector(config.contact_force_threshold_down, config.contact_force_threshold_up))) {}

    void caculateFootPosVel(std::map<std::string, double>& foot_angle, std::map<std::string, double>& foot_angle_vel,
                                double (&foot_pos)[4][3], double (&foot_vel)[4][3], pinocchio::Model& pmodel, pinocchio::Data& pdata);

    std::vector<ContactDetector> contacts_;
    std::map<std::string, int> map_foot_id;

    const std::array<std::string, 12> go2_motor_names = {
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint"
    };

};

inline void Kinematics::caculateFootPosVel(std::map<std::string, double>& foot_angle, std::map<std::string, double>& foot_angle_vel,
                            double (&foot_pos)[4][3], double (&foot_vel)[4][3], pinocchio::Model& pmodel, pinocchio::Data& pdata) {
    Eigen::VectorXd q = Eigen::VectorXd::Zero(pmodel.nq);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(pmodel.nv);

    for (pinocchio::JointIndex joint_id = 1; joint_id < (pinocchio::JointIndex)pmodel.njoints; ++joint_id) {
        std::string joint_name = pmodel.names[joint_id];
        q[joint_id - 1] = foot_angle[joint_name];
        v[joint_id - 1] = foot_angle_vel[joint_name];
    }
    forwardKinematics(pmodel, pdata, q, v);
    pinocchio::updateFramePlacements(pmodel, pdata);
    Eigen::Matrix<double,3,4> foot_pos_mat;
    Eigen::Matrix<double,3,4> foot_vel_mat;
    for (const auto& it : map_foot_id) {
        const std::string& foot_name = it.first;
        int idx = it.second;
        auto foot_id = pmodel.getFrameId(foot_name);

        foot_pos_mat.col(idx) = pdata.oMf[foot_id].translation();
        foot_vel_mat.col(idx) = pinocchio::getFrameVelocity(pmodel, pdata, foot_id, pinocchio::LOCAL_WORLD_ALIGNED).linear();
    }
    for (int idx = 0; idx < 4; ++idx) {
        std::memcpy(&foot_pos[idx][0], foot_pos_mat.col(idx).data(), 3 * sizeof(double));
        std::memcpy(&foot_vel[idx][0], foot_vel_mat.col(idx).data(), 3 * sizeof(double));
    }
}
