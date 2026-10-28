/*
This file is part of RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement.
 
Copyright (c) 2026 Ziyu Cao

Developer: Ziyu Cao <ziyu.cao@liu.se>

For commercial use, please contact me at <ziyu.cao@liu.se> or Kailai Li at <kailai.li@liu.se>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file, 
which is included as part of this source code package.
*/

#pragma once

#include "utils/common_utils.h"
#include "utils/math_tools.h"
#include "estimate_msgs/msg/spline.hpp"
#include "estimate_msgs/msg/knot.hpp"
#include <string>

template <class MatT>
struct JacobianStruct {
    size_t start_idx;
    std::vector<MatT> d_val_d_knot;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef JacobianStruct<double> Jacobian;
typedef JacobianStruct<Eigen::Matrix<double, 4, 3>> Jacobian43;
typedef JacobianStruct<Eigen::Matrix3d> Jacobian33;
typedef JacobianStruct<Eigen::Matrix<double, 3, 6>> Jacobian36;

struct PoseJacobian
{
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    Jacobian Jp;
    Jacobian43 Jr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class SplineState
{

  public:

    SplineState() {};

    void init(int64_t dt_ns_, int num_knot_, int64_t start_t_ns_, int start_i_ = 0,
              Eigen::Vector3d t0 = Eigen::Vector3d::Zero(), Eigen::Quaterniond q0 = Eigen::Quaterniond::Identity())
    {
        dt_ns = dt_ns_;
        start_t_ns = start_t_ns_;
        num_knot = num_knot_;
        start_i = start_i_;
        t_idle = {t0, t0, t0};
        ort_delta_idle = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
        q_idle = {q0, q0, q0};
    }

    void init(const SplineState* other, int start_i_, int end_i)
    {
        dt_ns = other->dt_ns;
        start_t_ns = other->getKnotTimeNs(start_i_);
        num_knot = 0;
        start_i = start_i_;
        if (start_i_ >= 3) {
            q_idle = {other->q_knots[start_i_ - 3], other->q_knots[start_i_ - 2], other->q_knots[start_i_ - 1]};
            t_idle = {other->t_knots[start_i_ - 3], other->t_knots[start_i_ - 2], other->t_knots[start_i_ - 1]};
            ort_delta_idle = {other->ort_delta[start_i_ - 3], other->ort_delta[start_i_ - 2], other->ort_delta[start_i_ - 1]};
        } else if (start_i_ == 2) {
            q_idle = {other->q_idle[2], other->q_knots[0], other->q_knots[1]};
            t_idle = {other->t_idle[2], other->t_knots[0], other->t_knots[1]};
            ort_delta_idle = {other->ort_delta_idle[2], other->ort_delta[0], other->ort_delta[1]};
        } else if (start_i_ == 1) {
            q_idle = {other->q_idle[1], other->q_idle[2], other->q_knots[0]};
            t_idle = {other->t_idle[1], other->t_idle[2], other->t_knots[0]};
            ort_delta_idle = {other->ort_delta_idle[1], other->ort_delta_idle[2], other->ort_delta[0]};
        } else if (start_i_ == 0) {
            q_idle = other->q_idle;
            t_idle = other->t_idle;
            ort_delta_idle = other->ort_delta_idle;
        }
        for (int i = 0; i < end_i + 1 - start_i_; i++) {
            addOneStateKnot(other->q_knots[start_i_ + i], other->t_knots[start_i_ + i], other->getKnotTimeNs(start_i_ + i));
        }
        const auto& source_times = other->getAllTK();
        for (size_t offset = 1; offset <= 2; ++offset) {
            const size_t source_idx = static_cast<size_t>(end_i) + offset;
            const int64_t support_time = source_idx < source_times.size()
                ? source_times[source_idx] : tk_ns.back();
            tk_ns.push_back(support_time);
        }
    }

    void clearSpline()
    {
        q_knots.clear();
        t_knots.clear();
        ort_delta.clear();
        tk_ns.clear();
        num_knot = 0;
    }    

    void setTimeIntervalNs(int64_t t_ns)
    {
        dt_ns = t_ns;
    }    

    Eigen::Matrix<double, 24, 1> getRCPs()
    {
        Eigen::Matrix<double, 24, 1> RCPs;
        for (int i = 0; i < 4; i++) {
            RCPs.block<3, 1>(i*6, 0) = t_knots[num_knot - 4 + i];
            RCPs.block<3, 1>(i*6+3, 0) = ort_delta[num_knot - 4 + i];
        }      
        return RCPs;
    }

    void updateRCPs(const Eigen::Matrix<double, 24, 1>& cp)
    {
        for (int i = 0; i < 4; i++) {
            t_knots[num_knot - 4 + i] = cp.block<3, 1>(i*6, 0);
            ort_delta[num_knot - 4 + i] = cp.block<3, 1>(i*6 + 3, 0);
            if (ort_delta[num_knot - 4 + i].norm() > M_PI/2) {
                ort_delta[num_knot - 4 + i] = Quater::warpDelta(ort_delta[num_knot - 4 + i]);
            }            
            Eigen::Quaterniond q_del;
            Quater::exp(ort_delta[num_knot - 4 + i], q_del);
            if (int(num_knot) - 4 + i > 0) {
                q_knots[num_knot - 4 + i] = q_knots[num_knot - 5 + i] * q_del;
            } else if (int(num_knot) - 4 + i == 0) {
                q_knots[num_knot - 4 + i] = q_idle[2] * q_del;
            }
        }
    }    

    void addOneStateKnot(Eigen::Vector3d pos, Eigen::Vector3d ort_del, int64_t time_ns)
    {
        t_knots.push_back(pos);
        ort_delta.push_back(ort_del);
        Eigen::Quaterniond ort;
        Eigen::Quaterniond q_del;
        Quater::exp(ort_del, q_del);
        if (num_knot > 0) {
            ort = q_knots.back() * q_del;
        } else {
            ort = q_idle[2] * q_del;
        }
        q_knots.push_back(ort);
        num_knot++;
        if (size_t(num_knot) > tk_ns.size()) {
            addOneTk(time_ns);
        } 
    }    

    void addOneStateKnot(Eigen::Quaterniond q, Eigen::Vector3d pos, int64_t time_ns)
    {
        if (abs(q.norm() - 1) > 1e-5) {
            q.normalize();
        }   
        Eigen::Quaterniond q0;
        if (num_knot > 0) {
            q0 = q_knots[num_knot - 1];
        } else {
            q0 = q_idle[2];
        }
        if (q0.dot(q) < 0) {
            q = Eigen::Quaterniond(-q.w(), -q.x(), -q.y(), -q.z());
        }
        Eigen::Quaterniond delta_q = q0.inverse() * q;   
        q_knots.push_back(q);
        t_knots.push_back(pos);
        ort_delta.push_back(Eigen::Vector3d::Zero());
        Quater::log(delta_q, ort_delta.back());
        num_knot++;
        addOneTk(time_ns);
    }    

    void addOneTk(int64_t time_ns, bool if_dup = false)
    {
        if (!if_dup) {
            tk_ns.push_back(time_ns);
        } else {
            tk_ns.push_back(tk_ns.back());
            tk_ns.push_back(tk_ns.back());
        }
    }

    void addSupportKnotTime(int64_t time_ns)
    {
        tk_ns.push_back(time_ns);
    }

    void removeOneTk()
    {
        while (tk_ns.size() > size_t(num_knot)) {
            tk_ns.pop_back();
        }
    }

    void setIdles(int idx, const Eigen::Vector3d& t, const Eigen::Vector3d& ort_del, const Eigen::Quaterniond& q_idle0)
    {
        t_idle[idx] = t;
        ort_delta_idle[idx] = ort_del;
        Eigen::Quaterniond q_del;
        Quater::exp(ort_del, q_del);        
        if (idx == 2) {
            q_knots[0] = q_idle[2] * q_del;
        } else if (idx == 1) {
            q_idle[2] = q_idle[1] * q_del;
        } else if (idx == 0) {
            q_idle[0] = q_idle0;
            q_idle[1] = q_idle[0] * q_del;
        }
    }    

    void setIdles(int idx, const Eigen::Quaterniond& q, const Eigen::Vector3d& t)
    {
        t_idle[idx] = t;
        q_idle[idx] = q;
    }       

    void getAllStateKnots(Eigen::aligned_deque<Eigen::Vector3d>& knots_trans,
        Eigen::aligned_deque<Eigen::Quaterniond>& knots_rot)
    {
        knots_trans = t_knots;
        knots_rot = q_knots;
    }

    void setAllKnots(Eigen::aligned_deque<Eigen::Vector3d>& knots_trans,
        Eigen::aligned_deque<Eigen::Quaterniond>& knots_rot)
    {
        t_knots = knots_trans;
        q_knots = knots_rot;
        computeDeltas();
    }

    void setOneStateKnot(int i, const Eigen::Vector3d& pos, const Eigen::Vector3d& ort_del)
    {
        t_knots[i] = pos;

        ort_delta[i] = ort_del;
        Eigen::Quaterniond q_del;
        Quater::exp(ort_del, q_del);
        if (i > 0) {
            q_knots[i] = q_knots[i - 1] * q_del;
        } else if (i == 0) {
            q_knots[i] = q_idle[2] * q_del;
        }
    }    

    void setOneStateKnot(int i, Eigen::Quaterniond q, Eigen::Vector3d pos)
    {
        t_knots[i] = pos;

        q_knots[i] = q;
    }    

    bool updateKnots(const SplineState* other, std::string& error)
    {
        error.clear();
        const int64_t num_knots_other = other->numKnots();
        const int64_t first = other->start_i;
        auto reject = [&error](const std::string& reason) {
            error = reason;
            return false;
        };
        if (first < 0 || num_knots_other <= 0) {
            return reject("invalid or empty knot window");
        }
        if (first > num_knot) {
            return reject("missing control points [" + std::to_string(num_knot) + ", " +
                          std::to_string(first) + "); incoming window starts at " +
                          std::to_string(first));
        }
        if (tk_ns.size() < size_t(num_knot) ||
            other->tk_ns.size() < size_t(num_knots_other)) {
            return reject("control-point/timestamp count mismatch");
        }
        for (int64_t i = 0; i < num_knots_other; ++i) {
            const int64_t index = first + i;
            const int64_t time = other->getKnotTimeNs(i);
            if (index < num_knot && tk_ns[index] != time) {
                return reject("timestamp mismatch at control point " + std::to_string(index) +
                              ": stored=" + std::to_string(tk_ns[index]) +
                              " incoming=" + std::to_string(time));
            }
            if (i > 0 || first > 0) {
                const int64_t previous = i > 0 ? other->getKnotTimeNs(i - 1) : tk_ns[first - 1];
                if (time <= previous) {
                    return reject("non-increasing timestamp at control point " + std::to_string(index) +
                                  ": previous=" + std::to_string(previous) +
                                  " incoming=" + std::to_string(time));
                }
            }
        }
        if (first + num_knots_other <= num_knot) {
            return reject("stale knot window: end=" + std::to_string(first + num_knots_other) +
                          " current_count=" + std::to_string(num_knot));
        }

        removeOneTk();
        setTimeIntervalNs(other->dt_ns);
        if (num_knot <= num_knots_other) {
            q_idle = other->q_idle;
            ort_delta_idle = other->ort_delta_idle;
            t_idle = other->t_idle;
        }

        for (int64_t i = 0; i < num_knots_other; i++) {
            if (i + other->start_i < num_knot) {
                setOneStateKnot(i + other->start_i, other->t_knots[i], other->ort_delta[i]);
            } else {
                addOneStateKnot(other->t_knots[i], other->ort_delta[i], other->getKnotTimeNs(i));
            }
        }
        addOneTk(tk_ns.back(), true);
        return true;
    }

    void updateKnots(SplineState* other, Eigen::Quaterniond* q_drift, Eigen::Vector3d* t_drift)
    {
        int64_t num_knots_other = other->numKnots();
        int64_t end_idx_other = other->start_i + num_knots_other - 1;
        Eigen::Quaterniond ref_ort_old;
        Eigen::Vector3d ref_trans_old;
        Eigen::Quaterniond ref_ort_new;
        Eigen::Vector3d ref_trans_new;
        if (q_drift) {
            ref_ort_old = q_knots[end_idx_other];
            ref_trans_old = t_knots[end_idx_other];
            ref_ort_new = other->q_knots[num_knots_other - 1];
            ref_trans_new = other->t_knots[num_knots_other - 1];
            Eigen::Quaterniond delta_q = ref_ort_new * ref_ort_old.inverse();
            (*q_drift) = delta_q * (*q_drift);
            (*t_drift) = ref_trans_new + delta_q * (*t_drift - ref_trans_old);
        }
        if (num_knot <= end_idx_other + 1) {
            for (int64_t i = 0; i < num_knots_other; i++) {
                if (i + other->start_i < num_knot) {
                    setOneStateKnot(i + other->start_i, other->q_knots[i],other->t_knots[i]);
                } else {
                    addOneStateKnot(other->q_knots[i],other->t_knots[i], other->getKnotTimeNs(i));
                }
            }
            if (num_knot <= num_knots_other) {
                q_idle = other->q_idle;
                t_idle = other->t_idle;
                ort_delta_idle = other->ort_delta_idle;
            }
        } else {
            Eigen::aligned_vector<Eigen::Quaterniond> rel_ort;
            Eigen::aligned_vector<Eigen::Vector3d> rel_trans;

            for (int64_t i = end_idx_other + 1; i  < num_knot; i++) {
                rel_ort.push_back(ref_ort_old.inverse() * q_knots[i]);
                rel_trans.push_back(ref_ort_old.inverse() * (t_knots[i] - ref_trans_old));
            }
            for (int64_t i = 0; i < num_knots_other; i++) {
                setOneStateKnot(i + other->start_i, other->q_knots[i],other->t_knots[i]);
            }
            for (size_t i = 0; i < rel_ort.size(); i++) {
                setOneStateKnot(i + end_idx_other + 1, ref_ort_new * rel_ort[i],
                                ref_ort_new * rel_trans[i] + ref_trans_new);
            }
        }
    }    

    void applyPoseInc(int i, const Eigen::Matrix<double, 6, 1> &inc)
    {
        t_knots[i] += inc.head<3>();
        Eigen::Quaterniond q_inc;
        Quater::exp(inc.tail<3>(), q_inc);
        q_knots[i] *= q_inc;
    }    

    void checkQuaternionControlPoints()
    {
        if (num_knot > 1) {
            for (size_t i = 1; i < (size_t) num_knot; i++) {
                Eigen::Quaterniond q1 = q_knots[i];
                if (q_knots[i - 1].dot(q1) < 0) {
                    q_knots[i] = Eigen::Quaterniond(-q1.w(), -q1.x(), -q1.y(), -q1.z());
                }
            }
        }
    }    

    int64_t getKnotTimeNs(size_t i) const
    {
        return tk_ns[i];
    }

    int64_t getKnotTimeIntervalNs() const
    {
        return dt_ns;
    }

    Eigen::Vector3d getKnotPos(size_t i) const
    {
        return t_knots[i];
    }

    Eigen::Quaterniond getKnotOrt(size_t i) const
    {
        return q_knots.at(i);
    }    

    Eigen::Vector3d getKnotOrtDel(size_t i) const
    {
        return ort_delta.at(i);
    }  

    Eigen::Vector3d getKnotPosIdle(size_t i) const
    {
        return t_idle[i];
    }

    Eigen::Quaterniond getKnotOrtIdle(size_t i) const
    {
        return q_idle[i];
    }       

    const std::deque<int64_t>& getAllTK() const
    {
        return tk_ns;
    }

    int64_t maxTimeNs() const
    {
        if (num_knot <= 1 || tk_ns.size() < 3) {
           return start_t_ns;
        } else {
            return tk_ns[tk_ns.size() - 3] - 1;
        }
    }

    int64_t minTimeNs() const
    {
        if (tk_ns.size() < 3) {
            return start_t_ns;
        }
        return tk_ns[2];
    }

    int64_t numKnots() const
    {
        return num_knot;
    }

    void computeDeltas()
    {
        for (size_t i = 1; i < q_idle.size(); i++) {
            Quater::log(q_idle[i - 1].inverse() * q_idle[i], ort_delta_idle[i]);
        }
        ort_delta.resize(q_knots.size());
        for (size_t i = 0; i < q_knots.size(); i++) {
            const Eigen::Quaterniond& previous = i == 0 ? q_idle.back() : q_knots[i - 1];
            Quater::log(previous.inverse() * q_knots[i], ort_delta[i]);
        }
    }

    template <int Derivative = 0>
    Eigen::Vector3d itpPosition (int64_t time_ns, Jacobian* J = nullptr) const
    {
        return itpEuclidean<Eigen::Vector3d, Derivative>(time_ns, t_idle, t_knots, J);
    }

    void itpQuaternionFromDelta(int64_t t_ns, Eigen::Quaterniond* q_out = nullptr,
                       Eigen::Vector3d* w_out = nullptr, Jacobian43* J_q = nullptr, Jacobian33* J_w = nullptr) const
    {
        double u;
        int64_t idx0;
        int idx_r;
        std::array<Eigen::Vector3d, 4> t_delta;
        prepareInterpolation(t_ns, ort_delta_idle, ort_delta, idx0, u, t_delta, idx_r);
        Eigen::Vector4d p;
        Eigen::Vector4d coeff;
        baseCoeffsWithTime<0>(p, u);
        {
            double inv_dt_nu;
            Eigen::Matrix4d cumulative_blending_matrix_nu = computeBlendingMatrixNU<true>(idx_r - 1, inv_dt_nu);
            coeff = cumulative_blending_matrix_nu * p;
        }

        Eigen::Quaterniond cp0;
        if (idx_r > 3) {
            cp0 = q_knots[idx0-1];
        } else if (idx_r == 3) {
            cp0 = q_idle[2];
        } else if (idx_r == 2) {
            cp0 = q_idle[1];
        } else if (idx_r == 1) {
            cp0 = q_idle[0];
        }

        Eigen::Vector3d t_delta_scale[4];
        Eigen::Quaterniond q_delta_scale[4];
        Eigen::Quaterniond q_itps[4];
        Eigen::Vector3d w_itps[4];
        Eigen::Vector4d dcoeff;
        if (J_q || J_w) {
            Eigen::Matrix<double, 4, 3> dexp_dt[4];
            t_delta_scale[0] = t_delta[0] * coeff[0];
            t_delta_scale[1] = t_delta[1] * coeff[1];
            t_delta_scale[2] = t_delta[2] * coeff[2];
            t_delta_scale[3] = t_delta[3] * coeff[3];
            Quater::dexp(t_delta_scale[0], q_delta_scale[0], dexp_dt[0]);
            Quater::dexp(t_delta_scale[1], q_delta_scale[1], dexp_dt[1]);
            Quater::dexp(t_delta_scale[2], q_delta_scale[2], dexp_dt[2]);
            Quater::dexp(t_delta_scale[3], q_delta_scale[3], dexp_dt[3]);
            int size_J = std::min(idx_r + 1, 4);
            Eigen::Quaterniond q_r_all[4];
            q_r_all[3] = Eigen::Quaterniond::Identity();
            for (int i = 2; i >= 0; i-- ) {
                q_r_all[i] = q_delta_scale[i+1] * q_r_all[i+1];
            }
            if (J_q) {
                q_itps[0] = cp0 * q_delta_scale[0];
                q_itps[1] = q_itps[0] * q_delta_scale[1];
                q_itps[2] = q_itps[1] * q_delta_scale[2];
                q_itps[3] = q_itps[2] * q_delta_scale[3];
                q_itps[3].normalize();
                *q_out = q_itps[3];
                Eigen::Matrix4d Q_l_all[4];
                Quater::Qleft(cp0, Q_l_all[0]);
                Quater::Qleft(q_itps[0], Q_l_all[1]);
                Quater::Qleft(q_itps[1], Q_l_all[2]);
                Quater::Qleft(q_itps[2], Q_l_all[3]);
                J_q->d_val_d_knot.resize(size_J);
                J_q->start_idx = idx0;
                for (int i = size_J - 1; i >= 0; i--) {
                    Eigen::Matrix4d Q_r_all;
                    Quater::Qright(q_r_all[4 - size_J + i], Q_r_all);
                    J_q->d_val_d_knot[i].noalias() = coeff[4 - size_J + i] * Q_r_all * Q_l_all[4 - size_J + i] * dexp_dt[4 - size_J + i];
                }
            }
            if (J_w) {
                baseCoeffsWithTime<1>(p, u);
                {
                    double inv_dt_nu;
                    Eigen::Matrix4d cumulative_blending_matrix_nu = computeBlendingMatrixNU<true>(idx_r - 1, inv_dt_nu);
                    dcoeff = inv_dt_nu * cumulative_blending_matrix_nu * p;
                }
                w_itps[0].setZero();
                w_itps[1] = 2 * dcoeff[1] * t_delta[1];
                w_itps[2] = q_delta_scale[2].inverse() * w_itps[1] + 2 * dcoeff[2] * t_delta[2];
                w_itps[3] = q_delta_scale[3].inverse() * w_itps[2] + 2 * dcoeff[3] * t_delta[3];
                *w_out = w_itps[3];
                Eigen::Matrix<double, 3, 4> drot_dq[2];
                Quater::drot(w_itps[1], q_delta_scale[2], drot_dq[0]);
                Quater::drot(w_itps[2], q_delta_scale[3], drot_dq[1]);
                J_w->d_val_d_knot.resize(size_J);
                J_w->start_idx = idx0;
                J_w->d_val_d_knot[0].setZero();  
                if (size_J == 4) {
                    J_w->d_val_d_knot[1] = 2 * dcoeff[1] * q_delta_scale[3].inverse().toRotationMatrix() * q_delta_scale[2].inverse().toRotationMatrix();
                    Eigen::Matrix3d tmp = coeff[2] * drot_dq[0] * dexp_dt[2];
                    J_w->d_val_d_knot[2] = q_delta_scale[3].inverse().toRotationMatrix() * (tmp + 2 * dcoeff[2] * Eigen::Matrix3d::Identity());
                    J_w->d_val_d_knot[3] = coeff[3] * drot_dq[1] * dexp_dt[3] + 2 * dcoeff[3] * Eigen::Matrix3d::Identity();
                } else if (size_J == 3) {
                    J_w->d_val_d_knot[0] = 2 * dcoeff[1] * q_delta_scale[3].inverse().toRotationMatrix() * q_delta_scale[2].inverse().toRotationMatrix();
                    Eigen::Matrix3d tmp = coeff[2] * drot_dq[0] * dexp_dt[2];
                    J_w->d_val_d_knot[1] = q_delta_scale[3].inverse().toRotationMatrix() * (tmp + 2 * dcoeff[2] * Eigen::Matrix3d::Identity());
                    J_w->d_val_d_knot[2] = coeff[3] * drot_dq[1] * dexp_dt[3] + 2 * dcoeff[3] * Eigen::Matrix3d::Identity();
                }
            }
        } else {
            t_delta_scale[0] = t_delta[0] * coeff[0];
            t_delta_scale[1] = t_delta[1] * coeff[1];
            t_delta_scale[2] = t_delta[2] * coeff[2];
            t_delta_scale[3] = t_delta[3] * coeff[3];
            Quater::exp(t_delta_scale[0], q_delta_scale[0]);
            Quater::exp(t_delta_scale[1], q_delta_scale[1]);
            Quater::exp(t_delta_scale[2], q_delta_scale[2]);
            Quater::exp(t_delta_scale[3], q_delta_scale[3]);
            if (q_out) {
                q_itps[0] = cp0 * q_delta_scale[0];
                q_itps[1] = q_itps[0] * q_delta_scale[1];
                q_itps[2] = q_itps[1] * q_delta_scale[2];
                q_itps[3] = q_itps[2] * q_delta_scale[3];
                q_itps[3].normalize();
                *q_out = q_itps[3];
            }
            if (w_out) {
                baseCoeffsWithTime<1>(p, u);
                {
                    double inv_dt_nu;
                    Eigen::Matrix4d cumulative_blending_matrix_nu = computeBlendingMatrixNU<true>(idx_r - 1, inv_dt_nu);
                    dcoeff = inv_dt_nu * cumulative_blending_matrix_nu * p;
                }

                w_itps[0].setZero();
                w_itps[1] = 2 * dcoeff[1] * t_delta[1];
                w_itps[2] = q_delta_scale[2].inverse() * w_itps[1] + 2 * dcoeff[2] * t_delta[2];
                w_itps[3] = q_delta_scale[3].inverse() * w_itps[2] + 2 * dcoeff[3] * t_delta[3];
                *w_out = w_itps[3];
            }
        }
    }  

    void itpQuaternionFromQuat(int64_t t_ns, Eigen::Quaterniond* q_out = nullptr,
                       Eigen::Vector3d* w_out = nullptr, Jacobian43* J_q = nullptr, Jacobian33* J_w = nullptr) const
    {
        double u;
        int64_t idx0;
        int idx_r;
        std::array<Eigen::Quaterniond, 4> cps;
        prepareInterpolation(t_ns, q_idle, q_knots, idx0, u, cps, idx_r);
        Eigen::Vector4d p;
        Eigen::Vector4d coeff;
        baseCoeffsWithTime<0>(p, u);
        {
            double inv_dt_nu;
            Eigen::Matrix4d cumulative_blending_matrix_nu = computeBlendingMatrixNU<true>(idx_r - 1, inv_dt_nu);
            coeff = cumulative_blending_matrix_nu * p;
        }
        Eigen::Quaterniond q_delta[3];
        q_delta[0] = cps[0].inverse() * cps[1];
        q_delta[1] = cps[1].inverse() * cps[2];
        q_delta[2] = cps[2].inverse() * cps[3];
        Eigen::Vector3d t_delta[3];
        Eigen::Vector3d t_delta_scale[3];
        Eigen::Quaterniond q_delta_scale[3];
        Eigen::Quaterniond q_itps[4];
        Eigen::Vector3d w_itps[4];
        Eigen::Vector4d dcoeff;
        if (J_q || J_w) {
            Eigen::Matrix<double, 3, 4> dlog_dq[3];
            Eigen::Matrix<double, 4, 3> dexp_dt[3];
            Quater::dlog(q_delta[0], t_delta[0], dlog_dq[0]);
            Quater::dlog(q_delta[1], t_delta[1], dlog_dq[1]);
            Quater::dlog(q_delta[2], t_delta[2], dlog_dq[2]);
            t_delta_scale[0] = t_delta[0] * coeff[1];
            t_delta_scale[1] = t_delta[1] * coeff[2];
            t_delta_scale[2] = t_delta[2] * coeff[3];
            Quater::dexp(t_delta_scale[0], q_delta_scale[0], dexp_dt[0]);
            Quater::dexp(t_delta_scale[1], q_delta_scale[1], dexp_dt[1]);
            Quater::dexp(t_delta_scale[2], q_delta_scale[2], dexp_dt[2]);
            int size_J = std::min(idx_r + 1, 4);
            Eigen::Matrix4d d_X_d_dj[3];
            Eigen::Matrix<double, 3, 4> d_r_d_dj[3];
            Eigen::Quaterniond q_r_all[4];
            q_r_all[3] = Eigen::Quaterniond::Identity();
            for (int i = 2; i >= 0; i-- ) {
                q_r_all[i] = q_delta_scale[i] * q_r_all[i+1];
            }
            Eigen::Matrix4d Q_l[size_J - 1];
            Eigen::Matrix4d Q_r[size_J - 1];
            for (int i = 0; i < size_J - 1; i++) {
                Quater::Qleft(q_delta[4 - size_J + i], Q_l[i]);
                Quater::Qright(q_delta[4 - size_J + i], Q_r[i]);
            }
            if (J_q) {
                q_itps[0] = cps[0];
                q_itps[1] = q_itps[0] * q_delta_scale[0];
                q_itps[2] = q_itps[1] * q_delta_scale[1];
                q_itps[3] = q_itps[2] * q_delta_scale[2];
                *q_out = q_itps[3];
                Eigen::Matrix4d Q_l_all[3];
                Quater::Qleft(q_itps[0], Q_l_all[0]);
                Quater::Qleft(q_itps[1], Q_l_all[1]);
                Quater::Qleft(q_itps[2], Q_l_all[2]);
                for (int i = 2; i >= 0; i--) {
                    Eigen::Matrix4d Q_r_all;
                    Quater::Qright(q_r_all[i+1], Q_r_all);
                    d_X_d_dj[i].noalias() = coeff[i + 1] * Q_r_all * Q_l_all[i] * dexp_dt[i] * dlog_dq[i];
                }
                J_q->d_val_d_knot.resize(size_J);
                for (int i = 0; i < size_J; i++) {
                    J_q->d_val_d_knot[i].setZero();
                }
                for (int i = 0; i < size_J - 1; i++) {
                    J_q->d_val_d_knot[i].noalias() -= d_X_d_dj[4 - size_J + i] * Q_r[i].rightCols(3);
                    J_q->d_val_d_knot[i + 1].noalias() += d_X_d_dj[4 - size_J + i] * Q_l[i].rightCols(3);
                }
                J_q->start_idx = idx0;
                if (size_J == 4) {
                    Eigen::Matrix4d Q_r_all;
                    Eigen::Matrix4d Q0_left;
                    Quater::Qright(q_r_all[0], Q_r_all);
                    Quater::Qleft(cps[0], Q0_left);
                    J_q->d_val_d_knot[0].noalias() += Q_r_all * Q0_left.rightCols(3);
                } else {
                    Eigen::Matrix4d Q_left;
                    Quater::Qleft(q_delta[3 - size_J], Q_left);
                    J_q->d_val_d_knot[0].noalias() += d_X_d_dj[3 - size_J] * Q_left.rightCols(3);
                }
            }
            if (J_w) {
                baseCoeffsWithTime<1>(p, u);
                {
                    double inv_dt_nu;
                    Eigen::Matrix4d cumulative_blending_matrix_nu = computeBlendingMatrixNU<true>(idx_r - 1, inv_dt_nu);
                    dcoeff = inv_dt_nu * cumulative_blending_matrix_nu * p;
                }
                w_itps[0].setZero();
                w_itps[1] = 2 * dcoeff[1] * t_delta[0];
                w_itps[2] = q_delta_scale[1].inverse() * w_itps[1] + 2 * dcoeff[2] * t_delta[1];
                w_itps[3] = q_delta_scale[2].inverse() * w_itps[2] + 2 * dcoeff[3] * t_delta[2];
                *w_out = w_itps[3];
                Eigen::Matrix<double, 3, 4> drot_dq[3];
                Quater::drot(w_itps[0], q_delta_scale[0], drot_dq[0]);
                Quater::drot(w_itps[1], q_delta_scale[1], drot_dq[1]);
                Quater::drot(w_itps[2], q_delta_scale[2], drot_dq[2]);
                for (int i = 2; i >= 0; i--) {
                    Eigen::Matrix3d d_vel_d_dj = coeff[i + 1] * drot_dq[i] * dexp_dt[i];
                    d_vel_d_dj.noalias() += 2 * dcoeff[i + 1] * Eigen::Matrix3d::Identity();
                    d_r_d_dj[i].noalias() = q_r_all[i+1].inverse().toRotationMatrix() * d_vel_d_dj * dlog_dq[i];
                }
                J_w->d_val_d_knot.resize(size_J);
                for (int i = 0; i < size_J; i++) {
                    J_w->d_val_d_knot[i].setZero();
                }
                for (int i = 0; i < size_J - 1; i++) {
                    J_w->d_val_d_knot[i].noalias() -= d_r_d_dj[4 - size_J + i] * Q_r[i].rightCols(3);
                    J_w->d_val_d_knot[i + 1].noalias() += d_r_d_dj[4 - size_J + i] * Q_l[i].rightCols(3);
                }
                J_w->start_idx = idx0;
                if (size_J != 4) {
                    Eigen::Matrix4d Q_left;
                    Quater::Qleft(q_delta[4 - size_J - 1], Q_left);
                    J_w->d_val_d_knot[0].noalias() += d_r_d_dj[3 - size_J] *  Q_left.rightCols(3);
                }
            }
        } else {
            Quater::log(q_delta[0], t_delta[0]);
            Quater::log(q_delta[1], t_delta[1]);
            Quater::log(q_delta[2], t_delta[2]);
            t_delta_scale[0] = t_delta[0] * coeff[1];
            t_delta_scale[1] = t_delta[1] * coeff[2];
            t_delta_scale[2] = t_delta[2] * coeff[3];
            Quater::exp(t_delta_scale[0], q_delta_scale[0]);
            Quater::exp(t_delta_scale[1], q_delta_scale[1]);
            Quater::exp(t_delta_scale[2], q_delta_scale[2]);
            if (q_out) {
                q_itps[0] = cps[0];
                q_itps[1] = q_itps[0] * q_delta_scale[0];
                q_itps[2] = q_itps[1] * q_delta_scale[1];
                q_itps[3] = q_itps[2] * q_delta_scale[2];
                q_itps[3].normalize();
                *q_out = q_itps[3];
            }
            if (w_out) {
                baseCoeffsWithTime<1>(p, u);
                {
                    double inv_dt_nu;
                    Eigen::Matrix4d cumulative_blending_matrix_nu = computeBlendingMatrixNU<true>(idx_r - 1, inv_dt_nu);
                    dcoeff = inv_dt_nu * cumulative_blending_matrix_nu * p;
                }

                w_itps[0].setZero();
                w_itps[1] = 2 * dcoeff[1] * t_delta[0];
                w_itps[2] = q_delta_scale[1].inverse() * w_itps[1] + 2 * dcoeff[2] * t_delta[1];
                w_itps[3] = q_delta_scale[2].inverse() * w_itps[2] + 2 * dcoeff[3] * t_delta[2];
                *w_out = w_itps[3];
            }
        }
    }

    void getSplineMsg(estimate_msgs::msg::Spline& spline_msg, const int start_i)
    {
        static int last_start_idx = 0;
        spline_msg.dt = dt_ns;
        int sidx = std::min(std::max((int)(num_knot - 5), 0), start_i);
        spline_msg.start_idx = std::min(sidx, last_start_idx+1);
        spline_msg.start_t = getKnotTimeNs(spline_msg.start_idx);
        for (size_t i = spline_msg.start_idx; i < (size_t) num_knot; i++) {
            estimate_msgs::msg::Knot knot_msg;
            knot_msg.position.x = t_knots[i].x();
            knot_msg.position.y = t_knots[i].y();
            knot_msg.position.z = t_knots[i].z();
            knot_msg.orientation_del.x = ort_delta[i].x();
            knot_msg.orientation_del.y = ort_delta[i].y();
            knot_msg.orientation_del.z = ort_delta[i].z();
            knot_msg.tk = getKnotTimeNs(i);
            spline_msg.knots.push_back(knot_msg);
        }
        for (int i = 0; i < 3; i++) {
            estimate_msgs::msg::Knot idle_msg;
            idle_msg.position.x = t_idle[i].x();
            idle_msg.position.y = t_idle[i].y();
            idle_msg.position.z = t_idle[i].z();
            idle_msg.orientation_del.x = ort_delta_idle[i].x();
            idle_msg.orientation_del.y = ort_delta_idle[i].y();
            idle_msg.orientation_del.z = ort_delta_idle[i].z();
            spline_msg.idles.push_back(idle_msg);
        }
        spline_msg.start_q.w = q_idle[0].w();
        spline_msg.start_q.x = q_idle[0].x();
        spline_msg.start_q.y = q_idle[0].y();
        spline_msg.start_q.z = q_idle[0].z();        
        if (num_knot < 5) {
            spline_msg.start_idx = 0;

        } 
        last_start_idx = std::max((int)spline_msg.start_idx, (int)(num_knot - 5));

    }  

    template <bool _Cumulative = false>
    Eigen::Matrix4d computeBlendingMatrixNU(int64_t idx_l, double& inv_dt_nu) const
    {

        int ofs = 0;
        double dc = static_cast<double>(tk_ns[idx_l+1+ofs] - tk_ns[idx_l+ofs]);
        double db = static_cast<double>(tk_ns[idx_l+1+ofs] - tk_ns[idx_l-1+ofs]);
        double da = static_cast<double>(tk_ns[idx_l+1+ofs] - tk_ns[idx_l-2+ofs]);
        double cb = static_cast<double>(tk_ns[idx_l+ofs] - tk_ns[idx_l-1+ofs]);
        double eb = static_cast<double>(tk_ns[idx_l+2+ofs] - tk_ns[idx_l-1+ofs]);
        double fc = static_cast<double>(tk_ns[idx_l+3+ofs] - tk_ns[idx_l+ofs]);
        double ec = static_cast<double>(tk_ns[idx_l+2+ofs] - tk_ns[idx_l+ofs]);

        inv_dt_nu = 1e9 / static_cast<double>(tk_ns[idx_l+1] - tk_ns[idx_l]);

        double mu0 = dc*dc/db/da;
        double mu10 = cb*cb/eb/db;
        double mu11 = dc*cb/eb/db;
        double mu12 = dc*dc/eb/db;
        double mu2 = dc*dc/fc/ec;
        double mu32 = dc*dc/ec/eb;
        double m32 = -mu12-mu2-mu32;

        Eigen::Matrix4d m;
        m << mu0, -3*mu0, 3*mu0, -mu0,
            1-mu0-mu10, 3*mu0-3*mu11, -3*mu0-3*mu12, mu0-m32-mu2,
            mu10, 3*mu11, 3*mu12, m32,
            0, 0, 0, mu2;

        if (_Cumulative) {
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    m.row(i) += m.row(j);
                }
            }
        }

        return m;
    }        

    int findLeftCP(const int64_t t_ns) const
    {
        int idx_l = 0;
        auto it_knot = std::upper_bound(tk_ns.begin(), tk_ns.end(), t_ns);
        --it_knot;
        idx_l = static_cast<int>(std::distance(tk_ns.begin(), it_knot));
        return idx_l;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:

    int64_t dt_ns;

    std::deque<int64_t> tk_ns;

    int64_t num_knot;
    int64_t start_i;
    int64_t start_t_ns;

    std::array<Eigen::Vector3d, 3> t_idle;
    std::array<Eigen::Vector3d, 3> ort_delta_idle;
    std::array<Eigen::Quaterniond, 3> q_idle;

    Eigen::aligned_deque<Eigen::Vector3d> t_knots;
    Eigen::aligned_deque<Eigen::Quaterniond> q_knots;
    Eigen::aligned_deque<Eigen::Vector3d> ort_delta;   

    template <typename _KnotT, int Derivative = 0>
    _KnotT itpEuclidean(int64_t t_ns, const std::array<_KnotT, 3>& knots_idle,
                        const Eigen::aligned_deque<_KnotT>& knots, Jacobian* J = nullptr) const
    {
        double u;
        int64_t idx0;
        int idx_r;
        std::array<_KnotT,4> cps;
        prepareInterpolation<_KnotT>(t_ns, knots_idle, knots, idx0, u, cps, idx_r);
        Eigen::Vector4d p, coeff;
        baseCoeffsWithTime<Derivative>(p, u);
        {
            double inv_dt_nu;
            Eigen::Matrix4d blending_matrix_nu = computeBlendingMatrixNU(idx_r - 1, inv_dt_nu);
            double pow_inv_dt_nu = std::pow(inv_dt_nu, Derivative);
            coeff = pow_inv_dt_nu * (blending_matrix_nu * p);
        }
        _KnotT res_out = coeff[0] * cps.at(0) + coeff[1] * cps.at(1) + coeff[2] * cps.at(2) + coeff[3] * cps.at(3);
        if (J) {
            int size_J = std::min(idx_r + 1, 4);
            J->d_val_d_knot.resize(size_J);
            for (int i = 0; i < size_J; i++) {
                J->d_val_d_knot[i] = coeff[4 - size_J + i];
            }
            J->start_idx = idx0;
        }
        return res_out;
    }  

    template<typename _KnotT>
    void prepareInterpolation(int64_t t_ns, const std::array<_KnotT, 3>& knots_idle,
                              const Eigen::aligned_deque<_KnotT>& knots, int64_t& idx0, double& u,
                              std::array<_KnotT,4>& cps, int& idx_r) const
    {
        int idx_l = 0;
        int64_t t_ns_l = 0;
        int64_t dt_ns_i = dt_ns;
        auto it_knot = std::upper_bound(tk_ns.begin(), tk_ns.end(), t_ns);
        --it_knot;
        t_ns_l = *it_knot;
        idx_l = static_cast<int>(std::distance(tk_ns.begin(), it_knot));
        dt_ns_i = tk_ns[idx_l + 1] - t_ns_l;
        idx_r = idx_l + 1;
        idx0 = std::max(idx_l - 2, 0);

        for (int i = 0; i < 2 - idx_l; i++) {
            cps[i] = knots_idle[i + idx_l + 1];
        }
        int idx_window = std::max(0, 2 - idx_l);
        for (int i = 0; i < std::min(idx_l + 2, 4); i++) {
            cps[i + idx_window] = knots[idx0 + i];
        }
        u = (t_ns - t_ns_l) / double(dt_ns_i);
    }

    template <int Derivative, class Derived>
    static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& res_const, double t)
    {
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 4);
        Eigen::MatrixBase<Derived>& res = const_cast<Eigen::MatrixBase<Derived>&>(res_const);
        res.setZero();
        res[Derivative] = base_coefficients(Derivative, Derivative);
        double ti = t;
        for (int j = Derivative + 1; j < 4; j++) {
            res[j] = base_coefficients(Derivative, j) * ti;
            ti = ti * t;
        }
    }

    static Eigen::Matrix4d computeBaseCoefficients()
    {
        Eigen::Matrix4d base_coeff;
        base_coeff.setZero();
        base_coeff.row(0).setOnes();
        int order = 3;
        for (int n = 1; n < 4; n++) {
            for (int i = 3 - order; i < 4; i++) {
                base_coeff(n, i) = (order - 3 + i) * base_coeff(n - 1, i);
            }
            order--;
        }
        return base_coeff;
    }

    inline static const Eigen::Matrix4d base_coefficients = computeBaseCoefficients();
};
