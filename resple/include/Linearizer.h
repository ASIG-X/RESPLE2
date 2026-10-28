#pragma once

#include "Accumulator.h"
#include "SplineState.h"
#include "utils/math_tools.h"
#include <pcl/point_cloud.h>

class Residuals
{

public:

    static PoseJacobian controlPointPose(const SplineState* spline, size_t index,
                                        bool with_jacobians = false)
    {
        PoseJacobian pose;
        pose.q = spline->getKnotOrt(index);
        pose.t = spline->getKnotPos(index);
        pose.Jp.start_idx = pose.Jr.start_idx = index;
        if (with_jacobians) {
            pose.Jp.d_val_d_knot = {1.0};
            Eigen::Matrix4d Q_left;
            Quater::Qleft(pose.q, Q_left);
            pose.Jr.d_val_d_knot = {Q_left.rightCols<3>()};
        }
        return pose;
    }

    static Eigen::Matrix<double, 6, 1> odometryResidual(const PoseJacobian& pose_i, const PoseJacobian& pose_j,
        const Eigen::Quaterniond& q_ij_meas, const Eigen::Vector3d& it_ji_meas,
        Eigen::aligned_vector<std::pair<size_t, Eigen::Matrix<double, 3, 6>>>* vJp = nullptr,
        Eigen::aligned_vector<std::pair<size_t, Eigen::Matrix3d>>* vJr = nullptr)
    {
        Eigen::Matrix<double, 6, 1> residual;
        Eigen::Quaterniond q_wi = pose_i.q;
        Eigen::Quaterniond q_iw = q_wi.inverse();
        Eigen::Quaterniond q_wj = pose_j.q;
        Eigen::Quaterniond q_ij = q_iw * q_wj;

        Eigen::Quaterniond q_ij_meas_pos = q_ij_meas;
        if (q_ij_meas.dot(q_ij) < 0) {
            q_ij_meas_pos = Quater::positify(q_ij_meas);
        }
        Eigen::Quaterniond q_delta = q_ij_meas_pos.inverse() * q_ij;
        Eigen::Vector3d r_q = q_delta.vec();
        q_iw.normalize();
        Eigen::Matrix3d R_iw = q_iw.toRotationMatrix();

        Eigen::Vector3d wt_ij = pose_j.t - pose_i.t;
        Eigen::Vector3d it_ij = R_iw * wt_ij;
        Eigen::Vector3d r_t = it_ij - it_ji_meas;

        residual.head<3>() = r_t;
        residual.tail<3>() = r_q;
        if (vJr) {
            Eigen::Matrix<double, 3, 4> dLog;
            dLog.setZero();
            dLog.rightCols<3>() = Eigen::Matrix3d::Identity();
            Eigen::Matrix4d qiw_left, q_ij_measT_left;
            Quater::Qleft(q_iw, qiw_left);
            Quater::Qleft(q_ij_meas_pos.inverse(), q_ij_measT_left);

            Eigen::Matrix4d qwj_right;
            Quater::Qright(q_wj, qwj_right);

            Eigen::Vector4d Q(1.0, -1.0, -1.0, -1.0);

            Eigen::Matrix<double, 3, 4> drot;
            Quater::drot(wt_ij, q_wi, drot);

            Eigen::Matrix3d R_tmp = R_iw;
            Eigen::Matrix<double, 3, 4> jq_tmp = dLog * q_ij_measT_left;
            Eigen::Matrix<double, 3, 4> R_tmp2 = drot;

            size_t start_idx_i = pose_i.Jp.start_idx;
            size_t start_idx_j = pose_j.Jp.start_idx;

            size_t Ni = pose_i.Jp.d_val_d_knot.size();
            size_t Nj = pose_j.Jp.d_val_d_knot.size();

            size_t end_idx_i = start_idx_i + Ni - 1;
            Jacobian33 J_orient_i, J_orient_j;
            J_orient_i.start_idx = start_idx_i;
            J_orient_j.start_idx = start_idx_j;

            Jacobian36 J_pos_i, J_pos_j;
            J_pos_i.start_idx = start_idx_i;
            J_pos_j.start_idx = start_idx_j;
            J_orient_i.d_val_d_knot.resize(pose_i.Jp.d_val_d_knot.size());
            J_pos_i.d_val_d_knot.resize(pose_i.Jp.d_val_d_knot.size());
            J_orient_j.d_val_d_knot.resize(pose_j.Jp.d_val_d_knot.size());
            J_pos_j.d_val_d_knot.resize(pose_j.Jp.d_val_d_knot.size());
            Eigen::Matrix<double, 3, 4> tmp_orti = jq_tmp * qwj_right * Q.asDiagonal();
            for (size_t i = 0; i < Ni; i ++) {
                J_orient_i.d_val_d_knot[i] = tmp_orti * pose_i.Jr.d_val_d_knot[i];
                J_pos_i.d_val_d_knot[i].template leftCols<3>() = - R_tmp * pose_i.Jp.d_val_d_knot[i];
                J_pos_i.d_val_d_knot[i].template rightCols<3>() = R_tmp2 * pose_i.Jr.d_val_d_knot[i];
            }
            Eigen::Matrix<double, 3, 4> tmp_ortj = jq_tmp * qiw_left;
            for (size_t i = 0; i < Nj; i ++) {
                J_orient_j.d_val_d_knot[i] = tmp_ortj * pose_j.Jr.d_val_d_knot[i];
                J_pos_j.d_val_d_knot[i].template leftCols<3>() = R_tmp * pose_j.Jp.d_val_d_knot[i];
                J_pos_j.d_val_d_knot[i].template rightCols<3>().setZero();
            }
            size_t n_overlap = end_idx_i >= start_idx_j ? (end_idx_i - start_idx_j + 1) : 0;
            size_t overlap_begin = Ni - n_overlap;
            size_t num_J = Ni + Nj - n_overlap;
            vJr->resize( num_J);
            vJp->resize( num_J);
            for (size_t i = 0; i < Ni; i++) {
                (*vJr)[i].first = J_orient_i.start_idx + i;
                (*vJr)[i].second = J_orient_i.d_val_d_knot[i];

                (*vJp)[i].first = J_pos_i.start_idx + i;
                (*vJp)[i].second = J_pos_i.d_val_d_knot[i];

                if (n_overlap > 0 && i >= overlap_begin) {
                    (*vJr)[i].second += J_orient_j.d_val_d_knot[i - overlap_begin];
                    (*vJp)[i].second += J_pos_j.d_val_d_knot[i - overlap_begin];
                }
            }
            for (size_t i = 0; i < Nj - n_overlap; i++) {
                (*vJr)[i + Ni].first = J_orient_j.start_idx + i + n_overlap;
                (*vJr)[i + Ni].second = J_orient_j.d_val_d_knot.at(n_overlap + i);

                (*vJp)[i + Ni].first = J_pos_j.start_idx + i + n_overlap;
                (*vJp)[i + Ni].second = J_pos_j.d_val_d_knot.at(n_overlap + i);
            }
        }
        return residual;
    }

};

struct Linearizer
{
    static const int POSE_SIZE = 6;
    static const int ROT_SIZE = 3;
    static const int ROT_OFFSET = 3;
    size_t set_fixed = 3;

    static constexpr int num_threads = 5;
    SparseHashAccumulator accum;
    std::array<SparseHashAccumulator, num_threads> accum_per_thread;

    double error;

    SplineState* spline;
    const OptParameters* param;
    const bool pose_fixed;

    Linearizer(size_t _opt_size, SplineState* spl, const OptParameters* par, bool _pose_fixed)
        : spline(spl), param(par), pose_fixed(_pose_fixed)
    {
        accum.reset(_opt_size);
        for (auto& local_accum : accum_per_thread) local_accum.reset(_opt_size);
        reset();
    }

    void reset() {
        accum.setZero();
        for (auto& local_accum : accum_per_thread) local_accum.setZero();
        error = 0;

    }

    void setupSolver() {
        for (const auto& local_accum : accum_per_thread) accum.join(local_accum);
        accum.setup_solver();
    }

    ~Linearizer() {}

    void operator()(const Eigen::aligned_vector<LoopData>& loop_data) {
        (*this)(loop_data, param->w_lc);
    }

    void operator()(const Eigen::aligned_vector<LoopData>& loop_data, double weight,
                    bool on_control_points = false) {
        if (loop_data.empty() || weight == 0.0) return;
        const double sqrt_weight = std::sqrt(weight);

        std::unordered_map<int64_t, int> time_to_idx;
        Eigen::aligned_vector<PoseJacobian> jac_cache;
        if (on_control_points) {
            jac_cache.resize(spline->numKnots());
            for (size_t i = 0; i < jac_cache.size(); ++i) {
                jac_cache[i] = Residuals::controlPointPose(spline, i, true);
            }
        } else {
            std::vector<int64_t> unique_times;
            unique_times.reserve(loop_data.size() * 2);
            for (const auto& lc : loop_data) {
                unique_times.push_back(lc.time_i_ns);
                unique_times.push_back(lc.time_j_ns);
            }
            std::sort(unique_times.begin(), unique_times.end());
            unique_times.erase(std::unique(unique_times.begin(), unique_times.end()), unique_times.end());
            time_to_idx.reserve(unique_times.size());
            jac_cache.resize(unique_times.size());
            for (size_t i = 0; i < unique_times.size(); i++) {
                time_to_idx[unique_times[i]] = static_cast<int>(i);
                spline->itpQuaternionFromQuat(unique_times[i], &jac_cache[i].q, nullptr, &jac_cache[i].Jr);
                jac_cache[i].t = spline->itpPosition(unique_times[i], &jac_cache[i].Jp);
            }
        }

        double sum_error = 0.0;
        #pragma omp parallel for num_threads(num_threads) if(loop_data.size() >= 64) reduction(+: sum_error) schedule(static)
        for (size_t k = 0; k < loop_data.size(); k++) {
            const auto& lc = loop_data[k];
            const PoseJacobian& pose_jac_i = jac_cache[on_control_points ? lc.idx_i : time_to_idx.at(lc.time_i_ns)];
            const PoseJacobian& pose_jac_j = jac_cache[on_control_points ? lc.idx_j : time_to_idx.at(lc.time_j_ns)];

            Eigen::aligned_vector<std::pair<size_t, Eigen::Matrix<double, 3, 6>>> vJp;
            Eigen::aligned_vector<std::pair<size_t, Eigen::Matrix3d>> vJr;
            Eigen::Matrix<double, 6, 1> residual = Residuals::odometryResidual(
                pose_jac_i, pose_jac_j, lc.rotation, lc.translation, &vJp, &vJr);
            residual *= sqrt_weight;
            Eigen::Vector3d rt = residual.head<3>();
            Eigen::Vector3d rq = residual.tail<3>();
            double e2t = rt.squaredNorm();
            double e2q = rq.squaredNorm();
            for (size_t i = 0; i < vJp.size(); i++) {
                vJp[i].second *= sqrt_weight;
                vJr[i].second *= sqrt_weight;
            }
            if (param->if_loss_func && e2t > param->loss_func_thresh * param->loss_func_thresh) {
                double rho[3];
                huberLoss(param->loss_func_thresh, e2t, rho);
                RobustLossCorrector corrector(e2t, rho);
                for (size_t i = 0; i < vJp.size(); i++)
                    corrector.template CorrectJacobian<3, 6>(rt, vJp[i].second);
                corrector.template CorrectResiduals<3>(rt);
                sum_error += rho[0];
            } else {
                sum_error += e2t;
            }
            if (param->if_loss_func && e2q > param->loss_func_thresh * param->loss_func_thresh) {
                double rho[3];
                huberLoss(param->loss_func_thresh, e2q, rho);
                RobustLossCorrector corrector(e2q, rho);
                for (size_t i = 0; i < vJr.size(); i++)
                    corrector.template CorrectJacobian<3, 3>(rq, vJr[i].second);
                corrector.template CorrectResiduals<3>(rq);
                sum_error += rho[0];
            } else {
                sum_error += e2q;
            }

            SparseHashAccumulator& local_accum = accum_per_thread[omp_get_thread_num()];
            for (size_t i = 0; i < vJr.size(); i++) {
                size_t start_i = vJr[i].first * POSE_SIZE + ROT_OFFSET;
                if (pose_fixed && start_i < set_fixed * POSE_SIZE) continue;
                for (size_t j = 0; j <= i; j++) {
                    size_t start_j = vJr[j].first * POSE_SIZE + ROT_OFFSET;
                    if (pose_fixed && start_j < set_fixed * POSE_SIZE) continue;
                    local_accum.template addH<ROT_SIZE, ROT_SIZE>(
                        start_i, start_j, vJr[i].second.transpose() * vJr[j].second);
                }
                local_accum.template addB<ROT_SIZE>(start_i, vJr[i].second.transpose() * rq);
            }
            for (size_t i = 0; i < vJp.size(); i++) {
                size_t start_i = vJp[i].first * POSE_SIZE;
                if (pose_fixed && start_i < set_fixed * POSE_SIZE) continue;
                for (size_t j = 0; j <= i; j++) {
                    size_t start_j = vJp[j].first * POSE_SIZE;
                    if (pose_fixed && start_j < set_fixed * POSE_SIZE) continue;
                    local_accum.template addH<POSE_SIZE, POSE_SIZE>(
                        start_i, start_j, vJp[i].second.transpose() * vJp[j].second);
                }
                local_accum.template addB<POSE_SIZE>(start_i, vJp[i].second.transpose() * rt);
            }
        }

        error   += sum_error;
    }

    double addToHessian(Eigen::Vector3d residual, Jacobian36& J_pt, size_t set_fixed,
                          const Eigen::Matrix3d& cov, SparseHashAccumulator& accum_thread) const
    {
        size_t num_Ji = J_pt.d_val_d_knot.size();
        double e2 = residual.transpose() * cov * residual;
        for (size_t i = 0; i < num_Ji; i++) {
            size_t start_i = (J_pt.start_idx + i) * POSE_SIZE;
            if (pose_fixed && start_i < set_fixed * POSE_SIZE) {
                continue;
            }
            Eigen::Matrix<double, 6, 3> tmp = J_pt.d_val_d_knot[i].transpose() * cov.topLeftCorner<3, 3>();
            for (size_t j = 0; j <= i; j++) {
                size_t start_j = (J_pt.start_idx + j) * POSE_SIZE;
                if (pose_fixed && start_j < set_fixed * POSE_SIZE) {
                    continue;
                }
                accum_thread.addH<POSE_SIZE, POSE_SIZE>(start_i, start_j,  tmp * J_pt.d_val_d_knot[j]);
            }
            accum_thread.addB<POSE_SIZE>(start_i, tmp * residual);
        }
        return e2;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ComputeErrorSplineOpt
{
    double error;
    SplineState* spline;
    const OptParameters* param;

    ComputeErrorSplineOpt(SplineState* spl, const OptParameters* par)
        : spline(spl), param(par)
    {
        error = 0;

    }

    ~ComputeErrorSplineOpt() {}

    void operator()(const Eigen::aligned_vector<LoopData>& loop_data) {
        (*this)(loop_data, param->w_lc);
    }

    void operator()(const Eigen::aligned_vector<LoopData>& loop_data, double weight,
                    bool on_control_points = false) {
        if (loop_data.empty() || weight == 0.0) return;
        const double sqrt_weight = std::sqrt(weight);
        double sum_error = 0.0;
        #pragma omp parallel num_threads(Linearizer::num_threads) if(loop_data.size() >= 64) reduction(+: sum_error)
        {
            bool have_previous = false;
            int64_t previous_j_ns = 0;
            PoseJacobian previous_j;
            #pragma omp for schedule(static)
            for (size_t k = 0; k < loop_data.size(); ++k) {
                const auto& lc = loop_data[k];
                PoseJacobian pose_jac_i;
                PoseJacobian pose_jac_j;
                if (on_control_points) {
                    pose_jac_i = Residuals::controlPointPose(spline, lc.idx_i);
                    pose_jac_j = Residuals::controlPointPose(spline, lc.idx_j);
                } else {
                    if (have_previous && previous_j_ns == lc.time_i_ns) {
                        pose_jac_i.q = previous_j.q;
                        pose_jac_i.t = previous_j.t;
                    } else {
                        spline->itpQuaternionFromQuat(lc.time_i_ns, &pose_jac_i.q);
                        pose_jac_i.t = spline->itpPosition(lc.time_i_ns);
                    }
                    spline->itpQuaternionFromQuat(lc.time_j_ns, &pose_jac_j.q);
                    pose_jac_j.t = spline->itpPosition(lc.time_j_ns);
                    previous_j.q = pose_jac_j.q;
                    previous_j.t = pose_jac_j.t;
                    previous_j_ns = lc.time_j_ns;
                    have_previous = true;
                }

                Eigen::Matrix<double, 6, 1> residual = Residuals::odometryResidual(pose_jac_i, pose_jac_j,
                                                                                   lc.rotation, lc.translation);
                residual *= sqrt_weight;
                Eigen::Vector3d rt = residual.head<3>();
                Eigen::Vector3d rq = residual.tail<3>();
                double e2t = rt.squaredNorm();
                double e2q = rq.squaredNorm();

                if (param->if_loss_func && e2t > param->loss_func_thresh * param->loss_func_thresh) {
                    double rho[3];
                    huberLoss(param->loss_func_thresh, e2t, rho);
                    sum_error += rho[0];
                } else {
                    sum_error += e2t;
                }
                if (param->if_loss_func && e2q > param->loss_func_thresh * param->loss_func_thresh) {
                    double rho[3];
                    huberLoss(param->loss_func_thresh, e2q, rho);
                    sum_error += rho[0];
                } else {
                    sum_error += e2q;
                }
            }
        }
        error += sum_error;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
