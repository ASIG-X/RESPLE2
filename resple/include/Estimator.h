#pragma once

#include "SplineState.h"
#include "Association.h"
#include "VisualFrontend.h"
#include <Eigen/Dense>
#include <fstream>
#include "utils/common_utils.h"

class Estimator
{
  public:
    static const int CP_SIZE = 24;
    static const int BA_OFFSET = 24;
    static const int BG_OFFSET = 27;

    int n_iter = 1;
    int num_tot_eff = 0;

    MeasData* meas_data;
    IntState int_state;

    Estimator();

    void setState(int64_t dt_ns, int64_t start_t_ns, const Eigen::Vector3d& t0, const Eigen::Quaterniond& q0,
        const Eigen::MatrixXd& Q, const Eigen::MatrixXd& P, Parameters* param_);

    void updateIEKF(std::shared_ptr<VoxelMapManager> voxelmap, std::shared_ptr<VisualFrontend> visual_frontend,
        const CameraConfig& cam_config);

    void addInitialProcessNoise();

    void propRCPNU(int64_t t);

    void updateIntState();

    void propagateIntState(int64_t tkp1, const Eigen::aligned_deque<ImuData>& imu_meas, const Eigen::Vector3d& g);

    ImuPredictionPreview makePredictionPreview() const;

    bool hasIntState() const { return int_state_ready_; }

    void updateFramePose(int64_t t_ns, const CameraConfig& cam_data, VisualFrontend& visual_frontend);

    SplineState* getSpline();

    Eigen::Matrix<double, 24, 24> getCovRCP();

    Eigen::Vector3d getBg();

    Eigen::Vector3d getBa();

    void setBg(const Eigen::Vector3d& bg_);

  private:
  static constexpr double NS_TO_S = 1e-9;

    SplineState spl;
    Eigen::MatrixXd cov_rcp;
    Eigen::MatrixXd cov_sys;
    Eigen::MatrixXd a_mat;
    Eigen::MatrixXd cov_rcp_aug;

    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Parameters* param;

    Eigen::MatrixXd measurement_H_;
    Eigen::VectorXd measurement_innv_;
    Eigen::VectorXd measurement_cov_inv_;
    Association::VoxelLookupCache correspondence_lookup_cache_;

    Eigen::MatrixXd I_X_;
    Eigen::MatrixXd cov_rcp_inv_;
    Eigen::MatrixXd HT_R_inv_;
    Eigen::MatrixXd HT_R_inv_H_;
    Eigen::MatrixXd S_;

    Eigen::MatrixXd K_;
    Eigen::MatrixXd KH;

    Eigen::VectorXd delta_cur_;
    Eigen::VectorXd information_rhs_;
    Eigen::VectorXd deltax_;
    Eigen::VectorXd RCPs_post_;

    Eigen::LLT<Eigen::MatrixXd> llt_;
    int max_iter = 5;

    Eigen::Matrix3d uu_pre;
    double dulam_pre;
    bool if_unclamp = false;
    bool correction_missing = false;

    bool int_state_ready_ = false;
    bool first_imu = false;
    Eigen::Vector3d acc_0;
    Eigen::Vector3d gyr_0;    

    Eigen::VectorXd getState();

    void cacheEndpointContinuity();

    Eigen::Matrix3d resetPosRCP();

    Eigen::Matrix<double, 6, 6> resetOrtRCP();

    void resetCov(const Eigen::Matrix3d& Js, const Eigen::Matrix<double, 6, 6>& Jr, int first_cp = 0,
                      const Eigen::Matrix<double, 3, 9>* J_new_delta = nullptr);

    void integrateStep(int64_t dt_, const ImuData& imu, Eigen::Quaterniond& qs, Eigen::Vector3d& Ps,
        Eigen::Vector3d& Vs, Eigen::Vector3d& dels, const Eigen::Vector3d& g);

    Eigen::VectorXd concat(const Eigen::aligned_vector<Eigen::VectorXd> &vecs) const;

    double prepCamera(ImgData& img_data, std::shared_ptr<VisualFrontend> visual_frontend, const CameraConfig& cam_config, int& n_vld, int XSIZE);

    void prepIMU(ImuData& imu_data, const Eigen::Vector3d& g) const;

    void prepKinVel(KinImuData& kin_data, const Eigen::Quaterniond& q_br, const Eigen::Vector3d& t_br) const;
    void prepKinPt2Pl(KinImuData& kin_data, std::shared_ptr<VoxelMapManager> voxelmap, const Eigen::Quaterniond& q_br, const Eigen::Vector3d& t_br) const;

    void prepLiDAR(PointData& pt_data, const Association::SplinePoseCache& cache) const;

    bool runIekfIteration(const Eigen::VectorXd& x_prop, const Eigen::MatrixXd& P_prop, std::shared_ptr<VoxelMapManager> voxelmap, std::shared_ptr<VisualFrontend> visual_frontend, const CameraConfig& cam_config,
                   bool& prior_inverse_ready);

    void updateState(const Eigen::VectorXd& xupd);

    template <int RSIZE>
    bool update(const Eigen::Matrix<double, RSIZE, 1>& innov, const Eigen::Matrix<double, RSIZE, 1>& R_inv, const Eigen::Matrix<double, RSIZE, Eigen::Dynamic>& H,
        const Eigen::VectorXd& x_prop, const Eigen::MatrixXd& cov_prop, bool& prior_inverse_ready);

};
