

#pragma once
#define BOOST_VARIANT_LIMIT_TYPES 50
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <boost/variant.hpp>

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <boost/math/special_functions/sinc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

class Quater
{
public:
    static void Qleft(const Eigen::Quaterniond &q, Eigen::Matrix4d &ans)
    {
        double q1 = q.w();
        double q2 = q.x();
        double q3 = q.y();
        double q4 = q.z();
        ans << q1, -q2, -q3, -q4,
                q2,  q1, -q4,  q3,
                q3,  q4,  q1, -q2,
                q4, -q3,  q2,  q1;
    }

    static void Qright(const Eigen::Quaterniond &q, Eigen::Matrix4d &ans)
    {
        double q1 = q.w();
        double q2 = q.x();
        double q3 = q.y();
        double q4 = q.z();
        ans << q1, -q2, -q3, -q4,
                q2,  q1,  q4, -q3,
                q3, -q4,  q1,  q2,
                q4,  q3, -q2,  q1;
    }

    static void exp(const Eigen::Vector3d &v, Eigen::Quaterniond& q)
    {
        double v_norm = v.norm();
        q.w() = std::cos(v_norm);
        q.vec() = boost::math::sinc_pi(v_norm) * v;

    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        Eigen::Quaternion<typename Derived::Scalar> p(q.w(), q.x(), q.y(), q.z());
        return q.w() >= (typename Derived::Scalar)(0.0) ? p : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
    }

    static void log(const Eigen::Quaterniond &q, Eigen::Vector3d& v)
    {
        Eigen::Quaterniond qn = q.normalized();
        Eigen::Vector3d rv = qn.vec();
        double norm_rv2 = rv.squaredNorm();
        double norm_rv = std::sqrt(norm_rv2);
        const double sign = qn.w() < 0.0 ? -1.0 : 1.0;
        double scale;
        if (norm_rv < 1e-4) {
            scale = sign * (1.0 + norm_rv2 * (1.0/6.0 + norm_rv2 * (3.0/40.0 + norm_rv2 * 5.0/112.0)));
        } else {
            scale = sign * std::atan2(norm_rv, std::abs(qn.w())) / norm_rv;
        }
        v = scale * rv;
    }    

    static void dexp(const Eigen::Vector3d& v, Eigen::Quaterniond& q, Eigen::Matrix<double,4,3>& J)
    {
        double v_norm = v.norm();
        const double v_norm2 = v_norm * v_norm;
        double sinc;
        double tmp;
        if (v_norm < 1e-4) {
            sinc = 1.0 + v_norm2 * (-1.0/6.0 + v_norm2 * (1.0/120.0 - v_norm2/5040.0));
            q.w() = 1.0 + v_norm2 * (-1.0/2.0 + v_norm2 * (1.0/24.0 - v_norm2/720.0));
            tmp = -1.0/3.0 + v_norm2 * (1.0/30.0 - v_norm2/840.0);
        } else {
            sinc = boost::math::sinc_pi(v_norm);
            q.w() = std::cos(v_norm);
            tmp = (q.w() - sinc) / v_norm2;
        }
        q.vec() = sinc * v;
        double v1 = v(0);
        double v2 = v(1);
        double v3 = v(2);
        double tmp_v1 = tmp * v1;
        double tmp_v2 = tmp * v2;
        double tmp_v11 = tmp_v1 * v1 + sinc;
        double tmp_v12 = tmp_v1 * v2;
        double tmp_v13 = tmp_v1 * v3;
        double tmp_v22 = tmp_v2 * v2 + sinc;
        double tmp_v23 = tmp_v2 * v3;
        double tmp_v33 = tmp * v3 * v3 + sinc;
        J << -v1 * sinc, -v2 * sinc, -v3 * sinc,
                tmp_v11, tmp_v12, tmp_v13,
                tmp_v12, tmp_v22, tmp_v23,
                tmp_v13, tmp_v23, tmp_v33;
    }

    static void dlog(const Eigen::Quaterniond &r, Eigen::Vector3d& v, Eigen::Matrix<double, 3, 4>& J)
    {
        const double r_norm = r.norm();
        Eigen::Quaterniond rn;
        rn.coeffs() = r.coeffs() / r_norm;
        double rnw = rn.w();
        double r1 = rn.x();
        double r2 = rn.y();
        double r3 = rn.z();
        double r11 = r1 * r1;
        double r22 = r2 * r2;
        double r33 = r3 * r3;
        double norm_rv2 = r11 + r22 + r33;
        double norm_rv = sqrt(norm_rv2);
        const double sign = rnw < 0.0 ? -1.0 : 1.0;
        double tmp_diag;
        double tmp;
        if (norm_rv < 1e-4) {
            tmp_diag = sign * (1.0 + norm_rv2 * (1.0/6.0 + norm_rv2 * (3.0/40.0 + norm_rv2 * 5.0/112.0)));
            tmp = sign * (-2.0/3.0 + norm_rv2 * (-1.0/5.0 - norm_rv2 * 3.0/28.0));
        } else {
            tmp_diag = sign * std::atan2(norm_rv, std::abs(rnw)) / norm_rv;
            tmp = (rnw - tmp_diag) / norm_rv2;
        }
        double r12 = r1 * r2;
        double r13 = r1 * r3;
        double r23 = r2 * r3;
        J << -r1, r11 * tmp + tmp_diag, r12 * tmp, r13 * tmp,
             -r2, r12 * tmp, r22 * tmp + tmp_diag, r23 * tmp,
             -r3, r13 * tmp, r23 * tmp, r33 * tmp + tmp_diag;
        v = tmp_diag * rn.vec();
        J /= r_norm;
    }    

    static void drot(const Eigen::Vector3d& v,const Eigen::Quaterniond& q, Eigen::Matrix<double, 3, 4> &J)
    {
        double qw = q.w();
        double qx = q.x();
        double qy = q.y();
        double qz = q.z();
        double v1 = v(0);
        double v2 = v(1);
        double v3 = v(2);
        Eigen::Vector3d vec;
        vec << (v1 * qw + v2 * qz - v3 * qy) * 2,
                (v2 * qw - v1 * qz + v3 * qx) * 2,
                (v3 * qw + v1 * qy - v2 * qx) * 2;
        double tmp = (v1 * qx + v2 * qy + v3 * qz) * 2;
        J << vec(0), tmp, -vec(2), vec(1),
                vec(1), vec(2), tmp, -vec(0),
                vec(2), -vec(1), vec(0), tmp;
    }

    static void drotInv(const Eigen::Vector3d& v,const Eigen::Quaterniond& q, Eigen::Matrix<double, 3, 4> &J)
    {
        double qw = q.w();
        double qx = q.x();
        double qy = q.y();
        double qz = q.z();
        double v1 = v(0);
        double v2 = v(1);
        double v3 = v(2);
        Eigen::Vector3d vec;
        vec << (v1 * qw - v2 * qz + v3 * qy) * 2,
                (v2 * qw + v1 * qz - v3 * qx) * 2,
                (v3 * qw - v1 * qy + v2 * qx) * 2;
        double tmp = (v1 * qx + v2 * qy + v3 * qz)*2;
        J << vec(0), tmp, vec(2), -vec(1),
                vec(1), -vec(2), tmp, vec(0),
                vec(2), vec(1), -vec(0), tmp;
    }

    static Eigen::Vector3d warpDelta(const Eigen::Vector3d& del)
    {

        double angle = del.norm();
        Eigen::Vector3d dir = del.normalized();
        angle = std::fmod(angle, 2*M_PI);
        angle -= M_PI;

        if (angle > M_PI/2) {
            angle = - M_PI + angle;
        } else if (angle < -M_PI/2) {
            angle = M_PI + angle;
        }
        Eigen::Vector3d out_del = angle * dir;
        return out_del;
    }
};

class RobustLossCorrector {
 public:
    RobustLossCorrector(double sq_norm, const double rho[3])
    {
        sqrt_rho1_ = sqrt(rho[1]);
        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
            return;
        }
        const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
        const double alpha = 1.0 - sqrt(D);
        residual_scaling_ = sqrt_rho1_ / (1 - alpha);
        alpha_sq_norm_ = alpha / sq_norm;
    }

    template <int rows>
    void CorrectResiduals(Eigen::Matrix<double, rows, 1>& residuals)
    {
        residuals *= residual_scaling_;
    }

    template <int rows, int cols>
    void CorrectJacobian(const Eigen::Matrix<double, rows, 1>& residuals, Eigen::Matrix<double, rows, cols>& jacobian){

        if (alpha_sq_norm_ == 0.0) {
            jacobian *= sqrt_rho1_;
            return;
        }

        for (int c = 0; c < cols; ++c) {
            double r_transpose_j = 0.0;
            for (int r = 0; r < rows; ++r) {
                r_transpose_j += jacobian(r, c) * residuals[r];
            }

            for (int r = 0; r < rows; ++r) {
                jacobian(r, c) = sqrt_rho1_ * (jacobian(r, c) - alpha_sq_norm_ * residuals[r] * r_transpose_j);
            }
        }

    }

  private:
    double sqrt_rho1_;
    double residual_scaling_;
    double alpha_sq_norm_;
};

inline void huberLoss(const double huber_scale, const double e2, double rho[3])
{
    double r = sqrt(e2);
    double huber_scale2 = huber_scale * huber_scale;
    rho[0] = huber_scale2 * ( 2.0 * r / huber_scale - 1.0);
    rho[1] = std::max(std::numeric_limits<double>::min(), huber_scale / r);
    rho[2] = - rho[1] / ( 2.0 * e2 );
}

namespace ImuRobustLoss {

inline void validateConfig(double delta, const Eigen::Vector3d& variance) {
    if (!std::isfinite(delta) || delta < 0.0) {
        throw std::invalid_argument("acc_huber_delta must be finite and nonnegative (0 disables it)");
    }
    if (!variance.allFinite() || (variance.array() <= 0.0).any()) {
        throw std::invalid_argument("cov_acc must contain finite, strictly positive variances");
    }
}

inline double accelWeight(const Eigen::Vector3d& residual,
                          const Eigen::Vector3d& inverse_variance, double delta) {
    if (delta == 0.0) return 1.0;
    const double norm = residual.cwiseProduct(inverse_variance.cwiseSqrt()).stableNorm();
    return norm > delta ? delta / norm : 1.0;
}

}

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
