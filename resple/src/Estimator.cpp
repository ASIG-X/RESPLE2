
#include "Estimator.h"

#include "utils/math_tools.h"
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <limits>

namespace {
Eigen::Vector3d endpointDelta(Eigen::Quaterniond q)
{
    q.normalize();
    if (q.w() < 0.0) q.coeffs() *= -1.0;
    const double s = q.vec().norm();
    if (s == 0.0) return Eigen::Vector3d::Zero();
    return (std::atan2(s, q.w()) / s) * q.vec();
}

Eigen::Matrix<double, 3, 9> endpointDeltaJacobian(
    const Eigen::Matrix<double, 6, 1>& old_deltas,
    const Eigen::Matrix<double, 6, 1>& new_deltas,
    const Eigen::Vector3d& imu_delta, const Eigen::Matrix<double, 6, 6>& Jr)
{
    Eigen::Quaterniond oa, ob, na, nb, di;
    Eigen::Matrix<double, 4, 3> doa, dob, dna, dnb, ddi;
    Quater::dexp(old_deltas.head<3>(), oa, doa);
    Quater::dexp(old_deltas.tail<3>(), ob, dob);
    Quater::dexp(new_deltas.head<3>(), na, dna);
    Quater::dexp(new_deltas.tail<3>(), nb, dnb);
    Quater::dexp(imu_delta, di, ddi);
    Eigen::Matrix4d right_ob, left_oa, right_nb, left_na;
    Quater::Qright(ob, right_ob); Quater::Qleft(oa, left_oa);
    Quater::Qright(nb, right_nb); Quater::Qleft(na, left_na);
    Eigen::Matrix<double, 4, 6> d_old, d_new;
    d_old << right_ob * doa, left_oa * dob;
    d_new << right_nb * dna, left_na * dnb;
    d_new = (d_new * Jr).eval();
    const Eigen::Quaterniond old_pair = oa * ob;
    const Eigen::Quaterniond new_pair = na * nb;
    Eigen::Quaterniond relative = new_pair.conjugate() * old_pair * di;
    const double sign = relative.w() < 0.0 ? -1.0 : 1.0;
    relative.coeffs() *= sign;
    Eigen::Matrix<double, 3, 4> d_log;
    Eigen::Vector3d unused;
    Quater::dlog(relative, unused, d_log);
    Eigen::Matrix4d conjugate = Eigen::Matrix4d::Identity();
    conjugate.bottomRightCorner<3, 3>() *= -1.0;
    Eigen::Matrix4d right_prediction, left_inverse, right_di, left_relative_base;
    Quater::Qright(old_pair * di, right_prediction);
    Quater::Qleft(new_pair.conjugate(), left_inverse);
    Quater::Qright(di, right_di);
    Quater::Qleft(new_pair.conjugate() * old_pair, left_relative_base);
    Eigen::Matrix<double, 3, 9> J;
    J.leftCols<6>() = sign * d_log *
        (right_prediction * conjugate * d_new + left_inverse * right_di * d_old);
    J.rightCols<3>() = sign * d_log * left_relative_base * ddi;
    return J;
}
}

Estimator::Estimator() {}

void Estimator::setState(int64_t dt_ns, int64_t start_t_ns, const Eigen::Vector3d& t0, const Eigen::Quaterniond& q0,
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& P, Parameters* param_)
{
    spl.init(dt_ns, 0, start_t_ns, 0, t0, q0);
    int64_t tk_i_ns = start_t_ns;
    for (int i = 0; i < 4; i++) {
        spl.addOneStateKnot(t0, Eigen::Vector3d::Zero(), tk_i_ns);
        tk_i_ns += dt_ns;
    }      
    cov_sys = Q;
    correction_missing = false;
    int_state_ready_ = false;
    first_imu = false;    
    cov_rcp = P;
    a_mat = Eigen::MatrixXd::Zero(P.rows(), P.cols());
    Eigen::Matrix<double, 6, 6> matblock = Eigen::Matrix<double, 6, 6>::Zero();
    matblock.topLeftCorner<3, 3>().setIdentity();
    matblock.bottomRightCorner<3, 3>().setIdentity();

    a_mat.block(0, 6, 6, 6) = matblock;
    a_mat.block(6, 12, 6, 6) = matblock;
    a_mat.block(12, 18, 6, 6) = matblock;
    a_mat.block(18, 18, 6, 6) = matblock;

    param = param_;
    a_mat.block<3,3>(BA_OFFSET, BA_OFFSET).setIdentity();
    a_mat.block<3,3>(BG_OFFSET, BG_OFFSET).setIdentity();
    int_state.tk = spl.getAllTK().back();
}

Eigen::VectorXd Estimator::getState()
{
    Eigen::Matrix<double, CP_SIZE, 1> cps_win = spl.getRCPs();
    Eigen::aligned_vector<Eigen::VectorXd> segments = {cps_win};
    segments.push_back(ba);
    segments.push_back(bg);
    Eigen::VectorXd stateXd = concat(segments);   
    return stateXd;
}  

void Estimator::updateIEKF(std::shared_ptr<VoxelMapManager> voxelmap, std::shared_ptr<VisualFrontend> visual_frontend,
    const CameraConfig& cam_config)
{
    constexpr double convergence_threshold = 0.001;
    int XSIZE = cov_rcp.rows();
    const Eigen::MatrixXd cov_prop = cov_rcp;
    bool prior_inverse_ready = false;
    Eigen::VectorXd rcp_prop = getState();
    num_tot_eff = 0;
    correction_missing = true;
    int num_iter = 0;
    correspondence_lookup_cache_.beginBatch(meas_data->pt_meas.size());
    for (int i = 0; i < max_iter; i++) {
        Eigen::VectorXd rcpi = getState();
        Association::findCorresp(num_tot_eff, &spl, voxelmap, meas_data->pt_meas,
            cov_rcp.template topLeftCorner<24, 24>(),
            &correspondence_lookup_cache_);
        if (num_tot_eff < 10) {
            updateState(rcp_prop);
            cov_rcp = cov_prop;
            cacheEndpointContinuity();
            return;
        }
        const bool information_form = runIekfIteration(rcp_prop, cov_prop, voxelmap, visual_frontend, cam_config, prior_inverse_ready);
        Eigen::VectorXd state_af = getState();
        num_iter++;
        if ((state_af - rcpi).norm() < convergence_threshold || (i == n_iter - 1) || (i == max_iter - 1)) {
            if (information_form) {
                KH.noalias() = llt_.solve(HT_R_inv_H_);
            }
            cov_rcp = ( Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;
            cov_rcp = (0.5*(cov_rcp + cov_rcp.transpose())).eval();
            break;
        }
    }
    correction_missing = (num_iter == 0);
    cacheEndpointContinuity();
    if (param->if_visual && visual_frontend && visual_frontend->has_frame_ && !meas_data->img_meas.empty()) {
        updateFramePose(meas_data->img_meas.back().time_ns, cam_config, *visual_frontend);
    }
}

void Estimator::addInitialProcessNoise()
{
    double dt_s = double(spl.getKnotTimeIntervalNs()) * 1e-9;
    cov_rcp += dt_s*dt_s*cov_sys;
    if (!param->bias_noise.continuous) {
        param->bias_noise.addProcessNoise(cov_rcp, dt_s);
    }
}

void Estimator::updateIntState()
{
    int64_t idx_new_cp = spl.numKnots();
    double tau_s = double(spl.getAllTK().back() - spl.getAllTK().at(spl.getAllTK().size() - 2))*1e-9;
    Eigen::Vector3d pk = spl.getKnotPos(idx_new_cp - 1);
    Eigen::Vector3d pkm1 = spl.getKnotPos(idx_new_cp - 2);
    const double velocity_factor = correction_missing ? 1.0 : 3.0;
    Eigen::Vector3d vk = velocity_factor/tau_s*(pk - pkm1);
    const double scale = vk.norm() > 10.0 ? 0.0 : 1.0;
    int64_t tk = spl.getAllTK().back();   
    int_state.init(tk, spl.getKnotOrt(idx_new_cp - 1), pk, vk*scale, Eigen::Vector3d::Zero());
    {
        Eigen::MatrixXd a_mat_new = a_mat;
        a_mat_new.middleRows(18, 6).setZero();  
        a_mat_new.block(18, 18, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();         
        cov_rcp = a_mat_new * cov_rcp * a_mat_new.transpose();   
        int XSIZE = cov_rcp.rows();
        Eigen::MatrixXd a_mat_aug = Eigen::MatrixXd::Zero(XSIZE+3, XSIZE);
        a_mat_aug.topLeftCorner(XSIZE, XSIZE).setIdentity();
        a_mat_aug.block(XSIZE, 6, 3, 3) = -scale*velocity_factor/tau_s*Eigen::Matrix3d::Identity();
        a_mat_aug.block(XSIZE, 12, 3, 3) = scale*velocity_factor/tau_s*Eigen::Matrix3d::Identity();
        cov_rcp_aug = a_mat_aug * cov_rcp * a_mat_aug.transpose();
    }            
    int_state_ready_ = true;
}

void Estimator::cacheEndpointContinuity()
{
    int64_t num_cp = spl.numKnots();

    double inv_dt_nu;
    Eigen::Matrix4d bom = spl.computeBlendingMatrixNU(num_cp - 2, inv_dt_nu);
    Eigen::Vector4d u(0, 0, 0, 1);
    Eigen::Vector4d du(0, 0, -3.0, 3.0);
    du *= inv_dt_nu;
    Eigen::Vector4d ddu = bom*Eigen::Vector4d(0, 0, 2, 6)*inv_dt_nu*inv_dt_nu;

    uu_pre.row(0) = u.tail<3>();
    uu_pre.row(1) = du.tail<3>();
    uu_pre.row(2) = ddu.tail<3>();
    if (correction_missing) {
        uu_pre.row(1) /= 3.0;
        uu_pre.row(2).setZero();
    }    

    dulam_pre = 3.0*inv_dt_nu;
    if_unclamp = true; 
}

Eigen::Matrix3d Estimator::resetPosRCP()
{
    if (!if_unclamp) {
        return Eigen::Matrix3d::Identity();
    }   
    int64_t num_cp = spl.numKnots();
    double inv_dt_nu;
    Eigen::Matrix4d bom_post = spl.computeBlendingMatrixNU(num_cp - 1, inv_dt_nu);

    Eigen::Matrix3d bm = bom_post.topLeftCorner<3, 3>().transpose();
    bm.row(1) *= inv_dt_nu;
    bm.row(2) *= 2*inv_dt_nu*inv_dt_nu;

    Eigen::Matrix3d coef = bm.inverse()*uu_pre;

    Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
    std::array<Eigen::Vector3d, 3> old_p;
    old_p[0] = cps_win.segment(6, 3);
    old_p[1] = cps_win.segment(12, 3);
    old_p[2] = cps_win.segment(18, 3);
    std::array<Eigen::Vector3d, 3> new_p;
    for (int i = 0; i < 3; i++) {
        new_p[i] = coef(i, 0)*old_p[0] + coef(i, 1)*old_p[1] + coef(i, 2)*old_p[2];
    }
    cps_win.segment(6, 3) = new_p[0];
    cps_win.segment(12, 3) = new_p[1];
    cps_win.segment(18, 3) = new_p[2];
    spl.updateRCPs(cps_win);         

    return coef;
}       

Eigen::Matrix<double, 6, 6> Estimator::resetOrtRCP()
{
    if (!if_unclamp) {
        return Eigen::Matrix<double, 6, 6>::Identity();
    }    
    Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
    std::array<Eigen::Vector3d, 2> old_del;
    old_del[0] = cps_win.segment(15, 3);
    old_del[1] = cps_win.segment(21, 3);

    Eigen::Matrix<double, 6, 1> del_i;
    del_i.head<3>() = old_del[0];
    del_i.tail<3>() = old_del[1];
     Eigen::Matrix<double, 6, 1> del_backup = del_i;

    std::array<Eigen::Quaterniond, 2> old_q;
    std::array<Eigen::Matrix<double, 4, 3>, 2> old_dexp;
    Quater::dexp(old_del[0], old_q[0], old_dexp[0]);
    Quater::dexp(old_del[1], old_q[1], old_dexp[1]);

    int64_t num_cp = spl.numKnots();
    double inv_dt_nu;
    Eigen::Matrix4d bph_post = spl.computeBlendingMatrixNU<true>(num_cp - 1, inv_dt_nu);    
    Eigen::Vector2d ula = bph_post.block<2, 1>(1, 0);
    Eigen::Vector2d dula = bph_post.block<2, 1>(1, 1)*inv_dt_nu;

    Eigen::Quaterniond rm = old_q[0]*old_q[1];
    Eigen::Matrix4d Q_m_l;
    Quater::Qleft(rm.inverse(), Q_m_l);

    Eigen::Matrix<double, 3, 4> V = Eigen::Matrix<double, 3, 4>::Zero();
    V.topRightCorner<3, 3>().setIdentity();       
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Quaterniond ri;
    int max_iters = 50;   
    double cost = 1e6; 
    for (int it = 0; it < max_iters; ++it) {
        Eigen::Matrix<double, 6, 1> z;

        std::array<Eigen::Quaterniond, 2> qi;
        std::array<Eigen::Matrix<double, 4, 3>, 2> dexp;
        Quater::dexp(ula[0]*del_i.head<3>(), qi[0], dexp[0]);
        Quater::dexp(ula[1]*del_i.tail<3>(), qi[1], dexp[1]);

        Eigen::Matrix4d Q_1_r;
        Quater::Qright(qi[1], Q_1_r);
        Eigen::Matrix4d Q_0_l;
        Quater::Qleft(qi[0], Q_0_l);

        ri = qi[0]*qi[1];
        z.head<3>() = (rm.inverse()*ri).vec();
        z.tail<3>() = qi[1].inverse().toRotationMatrix()*(2*dula[0]*del_i.head<3>()) + 2*dula[1]*del_i.tail<3>() - 2*dulam_pre*old_del[1];

        double cost_new = 0.5 * z.squaredNorm();
        if (cost_new > cost) {
            del_i = del_backup;
            break;
        }

        Eigen::Matrix<double, 3, 4> drot;
        Quater::drot((2*dula[0]*del_i.head<3>()), qi[1], drot);

        H.topLeftCorner<3, 3>() = V*Q_m_l*Q_1_r*dexp[0]*ula[0];
        H.topRightCorner<3, 3>() = V*Q_m_l*Q_0_l*dexp[1]*ula[1];
        H.bottomLeftCorner<3, 3>() = qi[1].inverse().toRotationMatrix()*(2*dula[0]);
        H.bottomRightCorner<3, 3>() = drot*dexp[1]*ula[1] + 2*dula[1]*Eigen::Matrix3d::Identity();

        del_backup = del_i;
        Eigen::Matrix<double, 6, 1> step = -H.colPivHouseholderQr().solve(z);        
        del_i += step;
        cost = cost_new;
    }        

    cps_win.segment<3>(15) = del_i.head<3>();
    cps_win.segment<3>(21) = del_i.tail<3>();
    spl.updateRCPs(cps_win);

    Eigen::Matrix<double, 6, 6> G;
    Eigen::Matrix4d Q_ri_r;
    Quater::Qright(ri, Q_ri_r);
    Eigen::Matrix4d D = Eigen::Matrix4d::Identity();
    D.bottomRightCorner<3, 3>() *= -1;
    Eigen::Matrix4d Q_1m_r;
    Quater::Qright(old_q[1], Q_1m_r);
    Eigen::Matrix4d Q_0m_l;
    Quater::Qleft(old_q[0], Q_0m_l);    

    G.topLeftCorner<3, 3>() = V*Q_ri_r*D*Q_1m_r*old_dexp[0];
    G.topRightCorner<3, 3>() = V*Q_ri_r*D*Q_0m_l*old_dexp[1];
    G.bottomLeftCorner<3, 3>().setZero();
    G.bottomRightCorner<3, 3>() = - 2*dulam_pre*Eigen::Matrix3d::Identity();    

    Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 6, 6>> solver(H);
    Eigen::Matrix<double, 6, 6> J = solver.solve(-G);
    return J;
}        

void Estimator::resetCov(const Eigen::Matrix3d& Js, const Eigen::Matrix<double, 6, 6>& Jr, int first_cp,
    const Eigen::Matrix<double, 3, 9>* J_new_delta)
{
    const int size = cov_rcp.rows();
    Eigen::MatrixXd J(size, size);
    J.setIdentity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            J.block<3, 3>((i + first_cp)*6, (j + first_cp)*6) = Js(i, j)*Eigen::Matrix3d::Identity();
        }
    }
    for (int i = 1; i < 3; i++) {
        for (int j = 1; j < 3; j++) {
            J.block<3, 3>((i + first_cp)*6+3, (j + first_cp)*6+3) = Jr.block<3, 3>((i-1)*3,(j-1)*3);
        }
    }    
    if (J_new_delta) {
        J.middleRows<3>(21).setZero();
        J.block<3, 3>(21, 9) = J_new_delta->leftCols<3>();
        J.block<3, 3>(21, 15) = J_new_delta->middleCols<3>(3);
        J.block<3, 3>(21, 21) = J_new_delta->rightCols<3>();
    }  
    cov_rcp = J*cov_rcp*J.transpose();
    cov_rcp = (0.5*(cov_rcp + cov_rcp.transpose())).eval();
}

void Estimator::propRCPNU(int64_t t)
{
    t += 1;        
    if (spl.getAllTK().back() >= t) {
        double dt_s = double(spl.getKnotTimeNs(spl.numKnots() - 1) - spl.getKnotTimeNs(spl.numKnots() - 2))*1e-9;
        cov_rcp += dt_s*dt_s*cov_sys;
        if (!param->bias_noise.continuous) {
            param->bias_noise.addProcessNoise(cov_rcp, dt_s);
        }
        spl.addOneTk(t, true);
    } else {
        if (!meas_data->imu_meas.empty() && int_state.tk != t) {
            throw std::logic_error("IMU prediction timestamp does not match the new spline knot");
        }        
        spl.addOneTk(t);
        spl.addOneTk(spl.getAllTK().back(), true);

        Eigen::Matrix<double, 6, 1> old_orientation_deltas;
        old_orientation_deltas.head<3>() = spl.getKnotOrtDel(spl.numKnots() - 2);
        old_orientation_deltas.tail<3>() = spl.getKnotOrtDel(spl.numKnots() - 1);
        const Eigen::Matrix<double, 6, 6> Jr = resetOrtRCP();
        const Eigen::Matrix3d Js = resetPosRCP();
        if (!meas_data->imu_meas.empty()) {
            const Eigen::Quaterniond predicted_orientation = int_state.qk;
            const Eigen::Vector3d predicted_position = int_state.pk;
            const Eigen::Vector3d predicted_increment = int_state.delk;
            const Eigen::Quaterniond q_last_adjusted =
                spl.getKnotOrt(spl.numKnots() - 1).normalized();
            const Eigen::Vector3d delta_new = endpointDelta(
                q_last_adjusted.conjugate() * predicted_orientation.normalized());
            Eigen::Matrix<double, 6, 1> adjusted_orientation_deltas;
            adjusted_orientation_deltas.head<3>() = spl.getKnotOrtDel(spl.numKnots() - 2);
            adjusted_orientation_deltas.tail<3>() = spl.getKnotOrtDel(spl.numKnots() - 1);
            const Eigen::Matrix<double, 3, 9> J_new_delta = endpointDeltaJacobian(
                old_orientation_deltas, adjusted_orientation_deltas, predicted_increment, Jr);
            resetCov(Js, Jr, 0, &J_new_delta);
            spl.addOneStateKnot(predicted_position, delta_new, t);
            double dt_s = double(spl.getKnotTimeNs(spl.numKnots() - 1) - spl.getKnotTimeNs(spl.numKnots() - 2))*1e-9;       
            cov_rcp += dt_s*dt_s*cov_sys;
            param->bias_noise.addProcessNoise(cov_rcp, dt_s, true);
        } else {
            if (if_unclamp) {
                resetCov(Js, Jr, 1);
            }            
            Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
            Eigen::Matrix<double, 24, 1> cps_win_new = a_mat.template topLeftCorner<24, 24>() * cps_win;
            Eigen::Matrix<double, 6, 1> cp_new = cps_win_new.tail<6>();
            spl.addOneStateKnot(cp_new.head<3>(), cp_new.tail<3>(), t);
            double dt_s = double(spl.getKnotTimeNs(spl.numKnots() - 1) - spl.getKnotTimeNs(spl.numKnots() - 2))*1e-9;        
            cov_rcp = a_mat * cov_rcp * a_mat.transpose() + dt_s*dt_s*cov_sys;
            param->bias_noise.addProcessNoise(cov_rcp, dt_s);
        }
    }
}       

ImuPredictionPreview Estimator::makePredictionPreview() const
{
    return ImuPredictionPreview({int_state.tk, int_state.qk, int_state.pk, int_state.vk},
        first_imu ? acc_0 : Eigen::Vector3d::Zero(),
        first_imu ? gyr_0 : Eigen::Vector3d::Zero(), first_imu, ba, bg, param->gravity);
}

void Estimator::propagateIntState(int64_t tkp1, const Eigen::aligned_deque<ImuData>& imu_meas, const Eigen::Vector3d& g)
{
    if (tkp1 < int_state.tk) {
        throw std::logic_error("IMU prediction target precedes the integrated state");
    }
    for (const auto& imu : imu_meas) {
        if (imu.time_ns > tkp1) break;
        if (imu.time_ns < int_state.tk) continue;
        if (imu.time_ns == int_state.tk) {
            acc_0 = imu.accel;
            gyr_0 = imu.gyro;
            first_imu = true;
            continue;
        }
        integrateStep(imu.time_ns - int_state.tk, imu, int_state.qk,
                      int_state.pk, int_state.vk, int_state.delk, g);
        int_state.tk = imu.time_ns;
        int_state_ready_ = true;
    }
    if (tkp1 > int_state.tk) {
        if (!first_imu) {
            return;
        }
        const ImuData held_imu(int_state.tk, gyr_0, acc_0);
        integrateStep(tkp1 - int_state.tk, held_imu, int_state.qk,
                      int_state.pk, int_state.vk, int_state.delk, g);
        int_state.tk = tkp1;
        int_state_ready_ = true;
    }
}            

void Estimator::integrateStep(int64_t dt_, const ImuData& imu, Eigen::Quaterniond& qs, Eigen::Vector3d& Ps, 
    Eigen::Vector3d& Vs, Eigen::Vector3d& dels, const Eigen::Vector3d& g)
{
    if (dt_ < 0) {
        throw std::invalid_argument("IMU integration interval must be nonnegative");
    }
    if (dt_ == 0) {
        return;
    }
    if (!first_imu) {
        first_imu = true;
        acc_0 = imu.accel;
        gyr_0 = imu.gyro;
    }
    Eigen::Quaterniond qs0 = qs;
    Eigen::Vector3d del0 = dels;

    double dt0 = dt_ * NS_TO_S;
    Eigen::Vector3d un_acc_0 = qs0 * (acc_0 - ba) - g;
    Eigen::Vector3d un_gyr_0 = gyr_0 - bg;
    Eigen::Matrix<double, 4, 3> dexp_del;
    Eigen::Matrix<double, 4, 3> dexp_del0;
    Eigen::Quaterniond delta_q;
    Eigen::Quaterniond exp_del0;
    Quater::dexp(un_gyr_0 * dt0/2.0, delta_q, dexp_del);
    Quater::dexp(del0, exp_del0, dexp_del0);
    qs *= delta_q;
    Ps += dt0 * Vs + 0.5 * dt0 * dt0 * un_acc_0;
    Vs += dt0 * un_acc_0;
    Eigen::Matrix<double, 3, 4> dlog_dels;
    Quater::dlog(exp_del0*delta_q, dels, dlog_dels);

    double dt2 = dt0*dt0;
    Eigen::Matrix<double, 3, 4> drot;
    Quater::drotInv((acc_0 - ba), qs0, drot);

    int64_t num_knots = spl.numKnots();
    Eigen::Quaterniond qkm3 = spl.getKnotOrt(num_knots - 4);

    std::array<Eigen::Quaterniond, 4> qdels;
    std::array<Eigen::Matrix<double, 4, 3>, 4> dexps;
    Eigen::Quaterniond ql = qkm3;
    Eigen::Matrix4d Q_l_all[4];
    for (int i = 0; i < 3; i++) {
        Eigen::Vector3d del = spl.getKnotOrtDel(num_knots - 3 + i);
        Quater::dexp(del, qdels[i], dexps[i]);
        Quater::Qleft(ql, Q_l_all[i]);
        ql *= qdels[i];
    }
    qdels[3] = exp_del0;
    dexps[3] = dexp_del0;
    Quater::Qleft(ql, Q_l_all[3]);

    Eigen::Matrix4d Q_r_all[4];
    Eigen::Quaterniond qr = Eigen::Quaterniond::Identity();
    for (int i = 3; i >= 0; i--) {
        Quater::Qright(qr, Q_r_all[i]);
        qr = qdels[i] * qr;
    }

    Eigen::Matrix4d Q_r_om;
    Quater::Qright(delta_q, Q_r_om);
    Eigen::Matrix4d Q_l_dekp1;
    Quater::Qleft(exp_del0, Q_l_dekp1);

    Eigen::Matrix3d Js[4];
    for (int i = 0; i < 4; i++) {
        Js[i] = drot*Q_l_all[i]*Q_r_all[i]*dexps[i];
    }
    Eigen::MatrixXd a_mat_new = Eigen::MatrixXd::Identity(cov_rcp_aug.rows(), cov_rcp_aug.cols());
    a_mat_new.middleRows(18, 6).setZero();
    a_mat_new.block(18, 3, 3, 3) = 0.5*dt2*Js[0];
    a_mat_new.block(18, 9, 3, 3) = 0.5*dt2*Js[1];
    a_mat_new.block(18, 15, 3, 3) = 0.5*dt2*Js[2];
    a_mat_new.block(18, 18, 3, 3).setIdentity();
    a_mat_new.block(18, 21, 3, 3) = 0.5*dt2*Js[3];
    a_mat_new.block(21, 21, 3, 3) = dlog_dels*Q_r_om*dexps[3];
    a_mat_new.block(18, 24, 3, 3) = -0.5*dt2*qs0.toRotationMatrix();
    Eigen::Matrix3d dbg = -dt0/2.0*dlog_dels*Q_l_dekp1*dexp_del;
    a_mat_new.block(21, 27, 3, 3) = dbg;
    int XSIZE = cov_rcp.rows();
    a_mat_new.block(18, XSIZE, 3, 3) = dt0*Eigen::Matrix3d::Identity();
    a_mat_new.block(XSIZE, 3, 3, 3) = dt0*Js[0];
    a_mat_new.block(XSIZE, 9, 3, 3) = dt0*Js[1];
    a_mat_new.block(XSIZE, 15, 3, 3) = dt0*Js[2];
    a_mat_new.block(XSIZE, 21, 3, 3) = dt0*Js[3];
    a_mat_new.block(XSIZE, 24, 3, 3) = -dt0*qs0.toRotationMatrix();
    a_mat_new.block(XSIZE, XSIZE, 3, 3).setIdentity();

    const Eigen::Matrix3d R_wb = qs0.toRotationMatrix();
    Eigen::Matrix3d Sigma_a = R_wb * param->pred_acc_cov.asDiagonal() * R_wb.transpose();
    Eigen::Matrix3d Q_ss = 0.25 * dt2 * dt2 * Sigma_a;
    Eigen::Matrix3d Q_sv = 0.5 * dt2 * dt0 * Sigma_a;
    Eigen::Matrix3d Q_vv = dt2 * Sigma_a;

    cov_rcp_aug = a_mat_new * cov_rcp_aug * a_mat_new.transpose();
    cov_rcp_aug.block(18, 18, 3, 3) += Q_ss;
    cov_rcp_aug.block<3, 3>(21, 21) += dbg * param->pred_gyro_cov.asDiagonal() * dbg.transpose();
    param->bias_noise.addProcessNoise(cov_rcp_aug, dt0);
    cov_rcp_aug.block(18, XSIZE, 3, 3) += Q_sv;
    cov_rcp_aug.block(XSIZE, 18, 3, 3) += Q_sv.transpose();
    cov_rcp_aug.block(XSIZE, XSIZE, 3, 3) += Q_vv;
    cov_rcp = cov_rcp_aug.topLeftCorner(XSIZE, XSIZE);

    acc_0 = imu.accel;
    gyr_0 = imu.gyro;
}

void Estimator::updateFramePose(int64_t t_ns, const CameraConfig& cam_data, VisualFrontend& visual_frontend)
{
    Eigen::Quaterniond orient_interp;
    Eigen::Vector3d t_interp = spl.itpPosition(t_ns);
    spl.itpQuaternionFromDelta(t_ns, &orient_interp);
    Eigen::Vector3d t_wc = orient_interp * cam_data.t_bc + t_interp;
    Eigen::Quaterniond q_wc = orient_interp * cam_data.q_bc;
    q_wc.normalize();
    visual_frontend.q_f_w_ = q_wc.inverse();
    visual_frontend.t_f_w_ = - visual_frontend.q_f_w_.toRotationMatrix() * t_wc;
}          

SplineState* Estimator::getSpline() {
    return &spl;
}            

Eigen::Matrix<double, 24, 24> Estimator::getCovRCP() {
    return cov_rcp.template topLeftCorner<24, 24>();
}

Eigen::Vector3d Estimator::getBg() { return bg;}

Eigen::Vector3d Estimator::getBa() { return ba;}

void Estimator::setBg(const Eigen::Vector3d& bg_) {
    bg = bg_;
}  

Eigen::VectorXd Estimator::concat(const Eigen::aligned_vector<Eigen::VectorXd> &vecs) const {
    int total = 0;
    for (const auto& v : vecs) {
        total += v.size();
    }
    Eigen::VectorXd out(total);
    int offset = 0;
    for (const auto& v : vecs) {
        out.segment(offset, v.size()) = v;
        offset += v.size();
    }
    return out;
}

double Estimator::prepCamera(ImgData& img_data, std::shared_ptr<VisualFrontend> visual_frontend, const CameraConfig& cam_config, int& n_vld, int XSIZE)
{
    double error = 0.0;
    n_vld = 0;
    img_data.if_valid.assign(img_data.m_rgb_pts_ptr_vec_in_last_frame.size(), false);
    if (img_data.m_last_tracked_pts.size() != img_data.m_rgb_pts_ptr_vec_in_last_frame.size()) {
        return -1.0;
    }
    const double pixel_variance = param->img_point_cov;
    if (!std::isfinite(pixel_variance) || pixel_variance <= 0.0) return -1.0;
    const Eigen::Matrix2d cov_meas_inv = Eigen::Matrix2d::Identity() / pixel_variance;
    if (!cov_meas_inv.allFinite()) return -1.0;
    int err_size = 2;
    int total_res_size = err_size*img_data.m_rgb_pts_ptr_vec_in_last_frame.size();
    img_data.H.resize(total_res_size, XSIZE);
    img_data.innv.resize(total_res_size, 1);
    img_data.mat_cov_inv.resize(total_res_size, 1);
    img_data.H.setZero();
    img_data.innv.setZero();
    img_data.mat_cov_inv.setConstant(cov_meas_inv(0, 0));
    Eigen::Quaterniond orient_interp;
    Eigen::Vector3d omg_itp;
    Jacobian43 J_ortdel;
    Jacobian J_pos;        
    Eigen::Vector3d t_interp = spl.itpPosition(img_data.time_ns, &J_pos);
    spl.itpQuaternionFromDelta(img_data.time_ns, &orient_interp, nullptr, &J_ortdel);

    Eigen::Vector3d t_wc = orient_interp * cam_config.t_bc + t_interp;
    Eigen::Quaterniond q_wc = orient_interp * cam_config.q_bc;
    q_wc.normalize();
    Eigen::Matrix3d Rcw = q_wc.inverse().toRotationMatrix();
    Eigen::Vector3d Pcw = - Rcw * t_wc;  

    int RCP_st_id = spl.numKnots() - 4;
    Eigen::Matrix<double, 3, 12> J_knot_p = Eigen::Matrix<double, 3, 12>::Zero();
    Eigen::Matrix<double, 4, 12> J_knot_q = Eigen::Matrix<double, 4, 12>::Zero();   
    for (int i = 0; i < (int) J_pos.d_val_d_knot.size(); i++) {
        int j = (int) J_pos.start_idx + i - RCP_st_id;
        if (j >= 0) {
            J_knot_p.block(0, j*3, 3, 3) = J_pos.d_val_d_knot[i] * Eigen::Matrix3d::Identity();
            J_knot_q.block(0, j*3, 4, 3) = J_ortdel.d_val_d_knot[i];          
        }
    }                    
    Eigen::Matrix<double, 3, 12> RwcT_J_knot_p = - Rcw * J_knot_p;
    Eigen::Matrix<double, 3, 4> drot2;   
    Quater::drotInv(cam_config.t_bc, orient_interp, drot2);        
    Eigen::Matrix4d Qbc_r; 
    Quater::Qright(cam_config.q_bc, Qbc_r);

    for (size_t i = 0; i < img_data.m_rgb_pts_ptr_vec_in_last_frame.size(); i++) {
        const auto& pt = img_data.m_rgb_pts_ptr_vec_in_last_frame[i];
        if (!pt) continue;
        Eigen::Vector3d pt_w = pt->pos_;
        cv::Point2f proj_pt = img_data.m_last_tracked_pts[i];
        Eigen::Vector2d uv(proj_pt.x, proj_pt.y);    

        Eigen::Vector3d pf = Rcw * pt_w + Pcw;
        if (!pf.allFinite() || pf.z() <= 1e-8 || !uv.allFinite()) {
            pt->if_valid = false;
            continue;
        }
        Eigen::Matrix<double, 2, 3> Jdpi;
        Eigen::Vector2d pc = visual_frontend->cam->world2cam(pf, &Jdpi);
        if (!pc.allFinite() || !Jdpi.allFinite()) continue;
        Eigen::Vector2d res = uv - pc;

        Eigen::Matrix<double, 3, 4> drot1;
        Quater::drot(pt_w - t_wc, q_wc, drot1);

        Eigen::Matrix<double, 3, 4> tmp = drot1 * Qbc_r - Rcw * drot2;   
        Eigen::Matrix<double, 3, 12> tmp_J_knot_q;
        tmp_J_knot_q.noalias() = tmp * J_knot_q;            

        Eigen::Matrix<double, 2, 12> J_knot_p_i;
        Eigen::Matrix<double, 2, 12> J_knot_q_i; 
        J_knot_p_i.noalias() = Jdpi * RwcT_J_knot_p;
        J_knot_q_i.noalias() = Jdpi * tmp_J_knot_q; 

        for (int j = 0; j < 4; j++) {
            img_data.H.block(i*err_size, j*6, err_size, 3) = J_knot_p_i.block(0, j*3, err_size, 3);
            img_data.H.block(i*err_size, j*6 + 3, err_size, 3) = J_knot_q_i.block(0, j*3, err_size, 3);
        }
        img_data.innv.segment(i*err_size, err_size) = res;
        const double normalized_error = std::sqrt(std::max(0.0, res.dot(cov_meas_inv * res)));
        const double delta = param->img_huber_loss;
        const double weight = normalized_error <= delta ? 1.0 : delta / normalized_error;
        const Eigen::Matrix2d cov_img_inv = weight * cov_meas_inv;
        img_data.mat_cov_inv.segment(i*err_size, err_size) = cov_img_inv.diagonal();
        img_data.if_valid[i] = true;
        n_vld++;
        error += normalized_error <= delta ? normalized_error * normalized_error
                                           : 2.0 * delta * normalized_error - delta * delta;
    }
    if (n_vld > 10) {
        return error;
    } else {
        img_data.if_valid.assign(img_data.if_valid.size(), false);
        n_vld = 0;
        return -1;
    }        
}     

void Estimator::prepIMU(ImuData& imu_data, const Eigen::Vector3d& g) const
{
    Eigen::Quaterniond q_itp;
    Eigen::Vector3d rot_vel;
    Jacobian43 J_ortdel;
    Jacobian J_line_acc;
    Jacobian33 J_gyro;
    spl.itpQuaternionFromDelta(imu_data.time_ns, &q_itp, &rot_vel, &J_ortdel, &J_gyro);
    Eigen::Vector3d a_w_no_g = spl.itpPosition<2>(imu_data.time_ns, &J_line_acc);
    Eigen::Vector3d a_w = a_w_no_g + g;
    Eigen::Matrix3d RT = q_itp.inverse().toRotationMatrix();   
    Eigen::Matrix<double, 3, 4> drot;
    Quater::drot(a_w, q_itp, drot);
    imu_data.H.setZero();
    int recur_st_id = spl.numKnots() - 4;
    for (int i = 0; i < (int) J_line_acc.d_val_d_knot.size(); i++) {
        int j = J_line_acc.start_idx + i - recur_st_id;
        if (j >= 0) {
            imu_data.H.block(0, j*6, 3, 3) = RT * J_line_acc.d_val_d_knot[i];
            imu_data.H.block(0, j*6 + 3, 3, 3) = drot * J_ortdel.d_val_d_knot[i];
            imu_data.H.block(3, j*6 + 3, 3, 3) = J_gyro.d_val_d_knot[i];                
        }
    }       
    imu_data.H.block(0, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
    imu_data.H.block(3, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();  
    imu_data.imu_itp.head<3>() = RT * a_w + ba;
    imu_data.imu_itp.tail<3>() = rot_vel + bg;
}    

void Estimator::prepKinVel(KinImuData& kin_data, const Eigen::Quaterniond& q_br, const Eigen::Vector3d& t_br) const
{
    kin_data.H = Eigen::Matrix<double, 12, 24>::Zero();
    kin_data.kin_meas.setZero();
    kin_data.kin_itp.setZero();
    if (std::none_of(std::begin(kin_data.contact_), std::end(kin_data.contact_),
                     [](bool contact) { return contact; })) return;
    Eigen::Quaterniond q_itp;
    Eigen::Vector3d omg_itp;
    Jacobian43 J_ortdel;
    Jacobian33 J_gyro;
    int64_t t_ns =  kin_data.time_ns;

    spl.itpQuaternionFromDelta(t_ns, &q_itp, &omg_itp, &J_ortdel, &J_gyro);
    Eigen::Matrix3d w_skew = CommonUtils::ssm(omg_itp);
    Jacobian J_vel;
    Eigen::Vector3d v_wi = spl.itpPosition<1>(t_ns, &J_vel);
    Eigen::Matrix3d R_wb = q_itp.toRotationMatrix();

    int RCP_st_id = spl.numKnots() - 4;
    Eigen::Matrix<double, 3, 12> J_knot_p = Eigen::Matrix<double, 3, 12>::Zero();
    Eigen::Matrix<double, 4, 12> J_knot_q = Eigen::Matrix<double, 4, 12>::Zero();
    Eigen::Matrix<double, 3, 12> J_knot_omg = Eigen::Matrix<double, 3, 12>::Zero();
    for (int i = 0; i < (int) J_vel.d_val_d_knot.size(); i++) {
        int j = (int) J_vel.start_idx + i - RCP_st_id;
        if (j >= 0) {
            J_knot_p.block(0, j*3, 3, 3) = J_vel.d_val_d_knot[i] * Eigen::Matrix3d::Identity();
            J_knot_q.block(0, j*3, 4, 3) = J_ortdel.d_val_d_knot[i];    
            J_knot_omg.block(0, j*3, 3, 3) = J_gyro.d_val_d_knot[i];          
        }
    }                   

    const Eigen::Matrix3d R_br = q_br.toRotationMatrix();
    for (int k = 0; k < 4; ++k) {
        if (!kin_data.contact_[k]) continue;
        const Eigen::Vector3d foot_pos(kin_data.foot_pos_[k][0], kin_data.foot_pos_[k][1], kin_data.foot_pos_[k][2]);
        const Eigen::Vector3d foot_vel(kin_data.foot_vel_[k][0], kin_data.foot_vel_[k][1], kin_data.foot_vel_[k][2]);
        const Eigen::Vector3d lever_b = t_br + R_br * foot_pos;
        const Eigen::Vector3d velocity_b = w_skew * lever_b + R_br * foot_vel;
        kin_data.kin_itp.segment<3>(3*k) = v_wi + R_wb * velocity_b;

        Eigen::Matrix<double, 3, 4> d_velocity_d_q;
        Quater::drotInv(velocity_b, q_itp, d_velocity_d_q);
        const Eigen::Matrix<double, 3, 12> J_knot_q_i =
            d_velocity_d_q * J_knot_q - R_wb * CommonUtils::ssm(lever_b) * J_knot_omg;
        for (int j = 0; j < 4; j++) {
            kin_data.H.block<3, 3>(3*k, 6*j) = J_knot_p.block<3, 3>(0, 3*j);
            kin_data.H.block<3, 3>(3*k, 6*j+3) = J_knot_q_i.block<3, 3>(0, 3*j);
        }
    }
}         

void Estimator::prepKinPt2Pl(KinImuData& kin_data, std::shared_ptr<VoxelMapManager> voxelmap, const Eigen::Quaterniond& q_br, const Eigen::Vector3d& t_br) const
{
    kin_data.H_pt2pl = Eigen::Matrix<double, 4, 24>::Zero();
    kin_data.kin_meas_pt2pl.setZero();
    kin_data.kin_itp_pt2pl.setZero();
    if (std::none_of(std::begin(kin_data.contact_), std::end(kin_data.contact_),
                     [](bool contact) { return contact; })) return;
    Eigen::Quaterniond q_itp;
    Jacobian43 J_ortdel;
    int64_t t_ns =  kin_data.time_ns;

    spl.itpQuaternionFromDelta(t_ns, &q_itp, nullptr, &J_ortdel);
    Jacobian J_pos;
    Eigen::Vector3d p_wi = spl.itpPosition(t_ns, &J_pos);    
    Eigen::Quaterniond q_wr = q_itp * q_br;
    Eigen::Vector3d p_wr = q_itp * t_br + p_wi;
    int RCP_st_id = spl.numKnots() - 4;
    Eigen::Vector4d vld = Eigen::Vector4d::Zero();
    for (int k = 0; k < 4; ++k) {
        if (kin_data.contact_[k]) { 
            Eigen::Vector3d foot_pos(kin_data.foot_pos_[k][0], kin_data.foot_pos_[k][1], kin_data.foot_pos_[k][2]);
            Eigen::Vector3d p_wf = q_wr * foot_pos + p_wr;
            PointData pt_data;
            Association::findCorresp(voxelmap, pt_data, p_wf);
            if (pt_data.if_valid) {
                Eigen::Matrix<double, 3, 4> drot;
                Quater::drotInv((q_br * foot_pos + t_br), q_itp, drot);
                Eigen::Matrix<double, 1, 4> tmp = pt_data.normvec.transpose() * drot;   
                kin_data.kin_itp_pt2pl[k] = pt_data.normvec.dot(p_wf) + pt_data.dist;
                    for (int i = 0; i < (int)J_pos.d_val_d_knot.size(); i++) {
                        int j = (int)J_pos.start_idx + i - RCP_st_id;
                        if (j >= 0) {
                            kin_data.H_pt2pl.block(k, j*6,     1, 3) = pt_data.normvec.transpose() * J_pos.d_val_d_knot[i];
                            kin_data.H_pt2pl.block(k, j*6 + 3, 1, 3) = tmp * J_ortdel.d_val_d_knot[i];
                        }
                    }  
                    vld[k] = 1;
            }

        }
    }
}          

void Estimator::prepLiDAR(PointData& pt_data, const Association::SplinePoseCache& c) const
{
    if (pt_data.if_valid) {
        const Eigen::Quaterniond& q_itp = c.q;
        const Eigen::Vector3d&    p_itp = c.pos;
        const Jacobian43&      J_ortdel = c.J_ortdel;
        const Jacobian&           J_pos = c.J_pos;
        Eigen::Matrix3d R_IL = pt_data.input().q_bl.toRotationMatrix();
        Eigen::Vector3d pt_w = q_itp * (R_IL * pt_data.input().pt_b + pt_data.input().t_bl) + p_itp;
        pt_data.zp = pt_data.normvec.dot(pt_w) + pt_data.dist;

        Eigen::Matrix<double, 3, 4> drot;
        Quater::drotInv((R_IL * pt_data.input().pt_b + pt_data.input().t_bl), q_itp, drot);
        Eigen::Matrix<double, 1, 4> tmp = pt_data.normvec.transpose() * drot;
        pt_data.H.setZero();
        for (int i = 0; i < (int)J_pos.d_val_d_knot.size(); i++) {
            int j = (int)J_pos.start_idx + i - c.RCP_st_id;
            if (j >= 0) {
                pt_data.H.block(0, j*6,     1, 3) = pt_data.normvec.transpose() * J_pos.d_val_d_knot[i];
                pt_data.H.block(0, j*6 + 3, 1, 3) = tmp * J_ortdel.d_val_d_knot[i];
            }
        }
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = pt_w - pt_data.center_;
        J_nq.block<1, 3>(0, 3) = -pt_data.normvec;
        double sigma_l = J_nq * pt_data.plane_var_ * J_nq.transpose();
        const Eigen::Matrix3d R_wl = (q_itp * pt_data.input().q_bl).toRotationMatrix();
        const Eigen::Vector3d n_l = R_wl.transpose() * pt_data.normvec;

        pt_data.var_pt = param->lidar_cov_floor + sigma_l + n_l.dot(pt_data.input().body_var * n_l);
    }
}

bool Estimator::runIekfIteration(const Eigen::VectorXd& x_prop, const Eigen::MatrixXd& P_prop, std::shared_ptr<VoxelMapManager> voxelmap, std::shared_ptr<VisualFrontend> visual_frontend, const CameraConfig& cam_config,
    bool& prior_inverse_ready)
{
    int XSIZE = P_prop.rows();
    int dim_meas = 0;
    if (!meas_data->pt_meas.empty()) {
        #pragma omp parallel num_threads(NUM_OF_THREAD)
        {
            int64_t last_t_ns = -1;
            Association::SplinePoseCache pose_cache;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < meas_data->pt_meas.size(); i++) {
                PointData& pt_data = meas_data->pt_meas.at(i);
                if (pt_data.if_valid && pt_data.input().time_ns != last_t_ns) {
                    pose_cache = Association::evalSpline(pt_data.input().time_ns, &spl);
                    last_t_ns  = pt_data.input().time_ns;
                }
                prepLiDAR(pt_data, pose_cache);
            }
        }
        dim_meas += num_tot_eff;
    }
    if (!meas_data->imu_meas.empty()) {
        #pragma omp parallel for num_threads(NUM_OF_THREAD)
        for (size_t i = 0; i < meas_data->imu_meas.size(); i++) {
            prepIMU(meas_data->imu_meas[i], param->gravity);
        }
        dim_meas += meas_data->imu_meas.size()*6;
    }

    if (!meas_data->kin_meas.empty()) {
        int num_contacts = 0;
        for (auto& kin_data : meas_data->kin_meas) {
            num_contacts += std::count(std::begin(kin_data.contact_), std::end(kin_data.contact_), true);
            prepKinVel(kin_data, param->q_br, param->t_br);
        }
        dim_meas += 3 * num_contacts;
        for (auto& kin_data : meas_data->kin_meas) {
            prepKinPt2Pl(kin_data, voxelmap, param->q_br, param->t_br);
        }
        dim_meas += num_contacts;
    }

    if (!meas_data->img_meas.empty() && param->img_point_cov > 0.0) {
        for (size_t i = 0; i < meas_data->img_meas.size(); i++) {
            {
                int n_vld = 0;
                prepCamera(meas_data->img_meas[i], visual_frontend, cam_config, n_vld, XSIZE);
                dim_meas += 2*n_vld;
            }
        }
    }

    measurement_H_.resize(dim_meas, XSIZE);
    measurement_innv_.resize(dim_meas);
    measurement_cov_inv_.resize(dim_meas);
    auto& H = measurement_H_;
    auto& innv = measurement_innv_;
    auto& mat_cov_inv = measurement_cov_inv_;
    H.setZero();
    innv.setZero();
    mat_cov_inv.setZero();

    int idx_offset = 0;
    for (size_t j = 0; j < meas_data->pt_meas.size(); j++) {
        PointData& pt_data = meas_data->pt_meas[j];
        if (pt_data.if_valid) {
            bool accepted = abs(pt_data.zp) < param->nn_thresh;
            if (!accepted) {
                const double state_var = pt_data.H*cov_rcp.topLeftCorner<24, 24>()*pt_data.H.transpose();
                const double lid_cov = state_var + pt_data.var_pt;
                accepted = lid_cov < pt_data.var_pt*param->coeff_cov;
            }
            if (accepted) {
                innv(idx_offset) = - pt_data.zp;
                H.block(idx_offset, 0, 1, 24) = pt_data.H;
            }
            mat_cov_inv(idx_offset) = 1.0/pt_data.var_pt;
            idx_offset++;
        }
    }
    Eigen::Matrix<double, 6, 1> cov_imu_inv =  Eigen::Matrix<double, 6, 1>(1/param->cov_acc[0], 1/param->cov_acc[1], 1/param->cov_acc[2], 1/param->cov_gyro[0], 1/param->cov_gyro[1], 1/param->cov_gyro[2]);
    size_t num_imu = meas_data->imu_meas.size();
    for (size_t j = 0; j < num_imu; j++) {
        ImuData& imu_data = meas_data->imu_meas[j];
        Eigen::Vector3d res = imu_data.accel - imu_data.imu_itp.head<3>();
        innv.segment<3>(idx_offset) = res;
        H.block(idx_offset, 0, 3, 30) = imu_data.H.topRows<3>();
        const double acc_weight = ImuRobustLoss::accelWeight(
            res, cov_imu_inv.head<3>(), param->acc_huber_delta);
        mat_cov_inv.segment<3>(idx_offset) = acc_weight * cov_imu_inv.head<3>();
        idx_offset += 3;
    }
    for (size_t j = 0; j < num_imu; j++) {
        ImuData& imu_data = meas_data->imu_meas[j];
        innv.segment<3>(idx_offset) = imu_data.gyro - imu_data.imu_itp.tail<3>();
        H.block(idx_offset, 0, 3, 30) = imu_data.H.bottomRows<3>();
        mat_cov_inv.segment<3>(idx_offset) = cov_imu_inv.tail<3>();
        idx_offset += 3;
    }

    for (const auto& kin_data : meas_data->kin_meas) {
        for (int foot = 0; foot < 4; ++foot) {
            if (!kin_data.contact_[foot]) continue;
            innv.segment<3>(idx_offset) = kin_data.kin_meas.segment<3>(3 * foot)
                                      - kin_data.kin_itp.segment<3>(3 * foot);
            H.block<3, 24>(idx_offset, 0) = kin_data.H.middleRows<3>(3 * foot);
            mat_cov_inv.segment<3>(idx_offset) = Eigen::Vector3d::Ones()/param->cov_kin_vel;
            idx_offset += 3;
        }
    }
    for (const auto& kin_data : meas_data->kin_meas) {
        for (int foot = 0; foot < 4; ++foot) {
            if (!kin_data.contact_[foot]) continue;
            innv(idx_offset) = kin_data.kin_meas_pt2pl(foot) - kin_data.kin_itp_pt2pl(foot);
            H.block<1, 24>(idx_offset, 0) = kin_data.H_pt2pl.row(foot);
            mat_cov_inv(idx_offset) = 1.0/param->cov_kin_pt2pl;
            ++idx_offset;
        }
    }

    if (!meas_data->img_meas.empty() && param->img_point_cov > 0.0) {
        for (size_t i = 0; i < meas_data->img_meas.size(); i++) {
            ImgData& img_data = meas_data->img_meas[i];
            int cnt_vld = 0;
            for(size_t i = 0; i < img_data.if_valid.size(); i++) {
                if (img_data.if_valid[i]) {
                    const int row = idx_offset + cnt_vld*2;
                    H.block(row, 0, 2, XSIZE) =  img_data.H.block(i*2, 0, 2, XSIZE);
                    innv.block(row, 0, 2, 1) =  img_data.innv.block(i * 2, 0, 2, 1);
                    mat_cov_inv.block(row, 0, 2, 1) = img_data.mat_cov_inv.block(i*2, 0, 2, 1);
                    cnt_vld++;
                }
            }
            idx_offset += cnt_vld*2;
        }
    }

    const bool information_form = update(innv, mat_cov_inv, H, x_prop, P_prop, prior_inverse_ready);

    return information_form;
}

void Estimator::updateState(const Eigen::VectorXd& xupd)
{
    Eigen::Matrix<double, CP_SIZE, 1> cp_win = xupd.segment(0, CP_SIZE);
    spl.updateRCPs(cp_win);
    ba = xupd.segment(BA_OFFSET, 3);
    bg = xupd.segment(BG_OFFSET, 3);
}    

template <int RSIZE>
bool Estimator::update(const Eigen::Matrix<double, RSIZE, 1>& innov, const Eigen::Matrix<double, RSIZE, 1>& R_inv, const Eigen::Matrix<double, RSIZE, Eigen::Dynamic>& H,
    const Eigen::VectorXd& x_prop, const Eigen::MatrixXd& cov_prop, bool& prior_inverse_ready)
{
    const int XSIZE = H.cols();
    const int num_pts = innov.rows();
    if (I_X_.rows() != XSIZE)
    {
        I_X_.resize(XSIZE, XSIZE);
        I_X_.setIdentity();

        cov_rcp_inv_.resize(XSIZE, XSIZE);
        HT_R_inv_H_.resize(XSIZE, XSIZE);
        S_.resize(XSIZE, XSIZE);
        KH.resize(XSIZE, XSIZE);

        delta_cur_.resize(XSIZE);
        information_rhs_.resize(XSIZE);
        deltax_.resize(XSIZE);
        RCPs_post_.resize(XSIZE);
    }

    if (HT_R_inv_.rows() != XSIZE || HT_R_inv_.cols() != num_pts)
        HT_R_inv_.resize(XSIZE, num_pts);

    const auto& state = getState();

    delta_cur_.noalias() = state - x_prop;

    if (num_pts > XSIZE)
    {
        if (!prior_inverse_ready) {
            llt_.compute(cov_prop);
            cov_rcp_inv_.noalias() = llt_.solve(I_X_);
            prior_inverse_ready = true;
        }

        HT_R_inv_.noalias() = H.transpose();
        HT_R_inv_.array().rowwise() *= R_inv.transpose().array();

        HT_R_inv_H_.noalias() = HT_R_inv_ * H;

        S_.noalias() = HT_R_inv_H_;
        S_.noalias() += cov_rcp_inv_;

        llt_.compute(S_);

        information_rhs_.noalias() = HT_R_inv_H_ * delta_cur_;
        information_rhs_.noalias() += HT_R_inv_ * innov;
        deltax_.noalias() = llt_.solve(information_rhs_);
        deltax_ -= delta_cur_;

        RCPs_post_.noalias() = state + deltax_;
    }
    else
    {
        Eigen::Matrix<double, RSIZE, RSIZE> R =
            R_inv.cwiseInverse().asDiagonal();

        Eigen::Matrix<double, RSIZE, RSIZE> S =
            H * cov_prop * H.transpose() + R;

        Eigen::LDLT<Eigen::Matrix<double, RSIZE, RSIZE>> ldlt(S);

        K_.noalias() =
            cov_prop * H.transpose() * ldlt.solve(
                Eigen::MatrixXd::Identity(num_pts, num_pts));

        KH.noalias() = K_ * H;

        deltax_.noalias() =
            KH * delta_cur_
          + K_ * innov
          - delta_cur_;

        RCPs_post_.noalias() = state + deltax_;
    }

    updateState(RCPs_post_);
    return num_pts > XSIZE;
}
