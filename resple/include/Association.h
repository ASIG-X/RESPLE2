#pragma once

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "utils/common_utils.h"
#include "SplineState.h"
#include "voxel_map.h"
#include <cassert>
#include <array>
#include <cstring>
#include <vector>

class Association
{
public:
    struct RootLookupCache {
        VOXEL_LOCATION key;
        VoxelOctoTree* root = nullptr;
        bool valid = false;
    };

    struct PointLookupCache {
        RootLookupCache primary;
        RootLookupCache neighbor;
        RootLookupCache* primaryFor(const VOXEL_LOCATION&) { return &primary; }
        RootLookupCache* neighborFor(const VOXEL_LOCATION&) { return &neighbor; }
    };

    class PreviewLookupCache {
    public:
        RootLookupCache* primaryFor(const VOXEL_LOCATION& key) {
            const uint64_t mixed = static_cast<uint64_t>(key.x) * UINT64_C(0x9e3779b185ebca87) ^
                static_cast<uint64_t>(key.y) * UINT64_C(0xc2b2ae3d27d4eb4f) ^
                static_cast<uint64_t>(key.z) * UINT64_C(0x165667b19e3779f9);
            return &entries_[(mixed ^ (mixed >> 32)) & 63];
        }
        RootLookupCache* neighborFor(const VOXEL_LOCATION& key) { return primaryFor(key); }
    private:
        std::array<RootLookupCache, 64> entries_{};
    };

    class VoxelLookupCache {
    public:
        void beginBatch(size_t points) {
            entries_.resize(points);
            for (auto& entry : entries_) {
                entry.primary.valid = false;
                entry.neighbor.valid = false;
            }
        }
        size_t size() const { return entries_.size(); }
        PointLookupCache& operator[](size_t i) { return entries_[i]; }

    private:
        std::vector<PointLookupCache> entries_;
    };

    struct SplinePoseCache {
        Eigen::Quaterniond q;
        Eigen::Vector3d pos;
        Jacobian   J_pos;
        Jacobian43 J_ortdel;
        int RCP_st_id;
    };

    struct RotationCache {
        bool valid = false;
        Eigen::Quaterniond q_bl;
        Eigen::Matrix3d R_wl;
        const Eigen::Matrix3d& get(const Eigen::Quaterniond& q, const Eigen::Quaterniond& extrinsic) {
            if (!valid || std::memcmp(q_bl.coeffs().data(), extrinsic.coeffs().data(),
                                      4 * sizeof(double)) != 0) {
                R_wl = (q * extrinsic).toRotationMatrix();
                q_bl = extrinsic;
                valid = true;
            }
            return R_wl;
        }
    };

    struct PoseCovarianceCache {
        Eigen::Matrix<double, 7, 7> covariance;
        bool valid = false;

        void prepare(const SplinePoseCache& c,
                     const Eigen::Matrix<double, 24, 24>& cov_rcp) {
            Eigen::Matrix<double, 7, 24> G = Eigen::Matrix<double, 7, 24>::Zero();
            for (int i = 0; i < (int)c.J_pos.d_val_d_knot.size(); ++i) {
                const int j = (int)c.J_pos.start_idx + i - c.RCP_st_id;
                if (j >= 0) {
                    G.block<3, 3>(0, j * 6) = c.J_pos.d_val_d_knot[i] * Eigen::Matrix3d::Identity();
                    G.block<4, 3>(3, j * 6 + 3) = c.J_ortdel.d_val_d_knot[i];
                }
            }
            covariance.noalias() = G * cov_rcp * G.transpose();
            valid = true;
        }

        Eigen::Matrix3d project(const Eigen::Matrix<double, 3, 4>& drot) const {
            Eigen::Matrix<double, 3, 7> D;
            D.leftCols<3>().setIdentity();
            D.rightCols<4>() = drot;
            return D * covariance * D.transpose();
        }
    };

    static SplinePoseCache evalSpline(int64_t t_ns, const SplineState* spline) {
        SplinePoseCache c;
        c.pos       = spline->itpPosition(t_ns, &c.J_pos);
        spline->itpQuaternionFromDelta(t_ns, &c.q, nullptr, &c.J_ortdel);
        c.RCP_st_id = static_cast<int>(spline->numKnots()) - 4;
        return c;
    }

    template<class PointType>
    static void pointBodyToWorld(const Eigen::Vector3d& pos, const Eigen::Quaterniond& q, const PointType& pi, PointType& po, const Eigen::Vector3d& t_bl, const Eigen::Quaterniond& q_bl)
    {    
        Eigen::Vector3f p_body(pi.x, pi.y, pi.z);
        Eigen::Vector3f p_global = q.cast<float>() * (q_bl.cast<float>() * p_body + t_bl.cast<float>()) + pos.cast<float>();
        po.x = p_global(0);
        po.y = p_global(1);
        po.z = p_global(2);
        po.curvature = pi.curvature;
    }     

    template<class PointType>
    static void pointBodyToWorld(const SplinePoseCache& c, const PointType& pi, PointType& po,
        const Eigen::Vector3d& t_bl, const Eigen::Quaterniond& q_bl,
        const Eigen::Matrix3d& var_body, Eigen::Matrix3d& var_w, Eigen::Matrix3d& cov_pose,
        const Eigen::Matrix<double, 24, 24>& cov_rcp,
        const Eigen::Matrix3d* cached_R_wl = nullptr,
        const PoseCovarianceCache* cached_pose_covariance = nullptr)
    {
        Eigen::Vector3f p_body(pi.x, pi.y, pi.z);
        Eigen::Vector3f p_i = (q_bl.cast<float>() * p_body + t_bl.cast<float>());
        Eigen::Vector3f p_global = c.q.cast<float>() * p_i + c.pos.cast<float>();
        po.x = p_global(0);
        po.y = p_global(1);
        po.z = p_global(2);
        po.curvature = pi.curvature;
        const Eigen::Matrix3d R_wl = cached_R_wl ? *cached_R_wl : (c.q * q_bl).toRotationMatrix();

        Eigen::Matrix<double, 3, 4> drot;
        Quater::drotInv(p_i.cast<double>(), c.q, drot);
        if (cached_pose_covariance && cached_pose_covariance->valid) {
            cov_pose = cached_pose_covariance->project(drot);
        } else {
            Eigen::Matrix<double, 3, 24> J = Eigen::Matrix<double, 3, 24>::Zero();
            for (int i = 0; i < (int)c.J_pos.d_val_d_knot.size(); i++) {
                int j = (int)c.J_pos.start_idx + i - c.RCP_st_id;
                if (j >= 0) {
                    J.block(0, j*6,     3, 3) = c.J_pos.d_val_d_knot[i] * Eigen::Matrix3d::Identity();
                    J.block(0, j*6 + 3, 3, 3) = drot * c.J_ortdel.d_val_d_knot[i];
                }
            }
            cov_pose = J * cov_rcp * J.transpose();
        }
        var_w    = R_wl * var_body * R_wl.transpose() + cov_pose;
    }

    static VoxelOctoTree* lookupRoot(const VoxelMapManager& voxelmap,
        const VOXEL_LOCATION& key, RootLookupCache* cache)
    {
        if (cache && cache->valid && cache->key == key) {
            return cache->root;
        }
        const auto iter = voxelmap.voxel_map_.find(key);
        VoxelOctoTree* root = iter == voxelmap.voxel_map_.end() ? nullptr : iter->second;
        if (cache) {
            cache->key = key;
            cache->root = root;
            cache->valid = true;
        }
        return root;
    }

    static void findCorrespAtWorld(const std::shared_ptr<VoxelMapManager>& voxelmap,
        PointData& pt_data, PointLookupCache* cache = nullptr)
    {
        const Eigen::Vector3d point(pt_data.pt_w.x, pt_data.pt_w.y, pt_data.pt_w.z);
        const VoxelPlane* plane = findPlaneAtWorld(voxelmap, point,
            pt_data.w_var, cache);
        pt_data.if_valid = plane != nullptr;
        if (plane) {
            pt_data.plane_var_ = plane->plane_var_;
            pt_data.normvec = plane->normal_;
            pt_data.center_ = plane->center_;
            pt_data.dist = plane->d_;
        }
    }

    template<class LookupCache = PointLookupCache>
    static const VoxelPlane* findPlaneAtWorld(const std::shared_ptr<VoxelMapManager>& voxelmap,
        const Eigen::Vector3d& point, const Eigen::Matrix3d& point_var,
        LookupCache* cache = nullptr)
    {
        const VOXEL_LOCATION position = VOXEL_LOCATION::fromPoint(
            point, voxelmap->config_setting_.max_voxel_size_);
        VoxelOctoTree* current_octo = lookupRoot(*voxelmap, position,
            cache ? cache->primaryFor(position) : nullptr);
        if (!current_octo) {
            return nullptr;
        }
        double prob = 0;
        const VoxelPlane* plane = voxelmap->findPlane(point, point_var, current_octo, 0,
            prob);
        if (!plane) {
            const auto& vc = current_octo->voxel_center_;
            const float ql = current_octo->quater_length_;
            VOXEL_LOCATION near_position = position;
            if      (point.x() > vc[0] + ql) near_position.x++;
            else if (point.x() < vc[0] - ql) near_position.x--;
            if      (point.y() > vc[1] + ql) near_position.y++;
            else if (point.y() < vc[1] - ql) near_position.y--;
            if      (point.z() > vc[2] + ql) near_position.z++;
            else if (point.z() < vc[2] - ql) near_position.z--;
            if (near_position == position) {
                return nullptr;
            }
            VoxelOctoTree* near_octo = lookupRoot(*voxelmap, near_position,
                cache ? cache->neighborFor(near_position) : nullptr);
            if (near_octo) {
                plane = voxelmap->findPlane(point, point_var, near_octo, 0, prob);
            }
        }
        return plane;
    }

    static void findCorresp(int& effect_num_k, const SplineState* spline,
        const std::shared_ptr<VoxelMapManager>& voxelmap, ActiveLidarBuffer& pt_meas,
        const Eigen::Matrix<double, 24, 24>& cov_rcp, VoxelLookupCache* cache = nullptr)
    {
        effect_num_k = 0;
        int num_pt = pt_meas.size();
        #pragma omp parallel num_threads(NUM_OF_THREAD) reduction(+:effect_num_k)
        {
            bool has_pose = false;
            int64_t last_t_ns = 0;
            SplinePoseCache pose_cache;
            RotationCache rotation;
            PoseCovarianceCache pose_covariance;
            unsigned int points_at_pose = 0;
            #pragma omp for schedule(static)
            for (int i = 0; i < num_pt; i++) {
                PointData &pt_data = pt_meas[i];
                if (!has_pose || pt_data.input().time_ns != last_t_ns) {
                    pose_cache = evalSpline(pt_data.input().time_ns, spline);
                    last_t_ns = pt_data.input().time_ns;
                    has_pose = true;
                    rotation.valid = false;
                    pose_covariance.valid = false;
                    points_at_pose = 0;
                }
                const Eigen::Matrix3d& R_wl = rotation.get(pose_cache.q, pt_data.input().q_bl);
                if (++points_at_pose == 3 && i + 1 < num_pt &&
                    pt_meas[i + 1].input().time_ns == last_t_ns) {
                    pose_covariance.prepare(pose_cache, cov_rcp);
                }
                Eigen::Matrix3d cov_pose;
                Association::pointBodyToWorld(pose_cache, pt_data.input().pt, pt_data.pt_w, pt_data.input().t_bl, pt_data.input().q_bl, pt_data.input().body_var, pt_data.w_var, cov_pose, cov_rcp, &R_wl, &pose_covariance);
                PointLookupCache* point_cache = cache ? &(*cache)[i] : nullptr;
                findCorrespAtWorld(voxelmap, pt_data, point_cache);
                if (pt_data.if_valid) effect_num_k++;
            }
        }
    }     

    static void findCorresp(std::shared_ptr<VoxelMapManager> voxelmap, PointData& pt_data, 
        const Eigen::Vector3d& point_w)
    {
        const VOXEL_LOCATION position = VOXEL_LOCATION::fromPoint(point_w, voxelmap->config_setting_.max_voxel_size_);
        auto iter = voxelmap->voxel_map_.find(position);
        if (iter == voxelmap->voxel_map_.end()) {
            pt_data.if_valid = false;
            return;
        }            
        VoxelOctoTree *current_octo = iter->second;
        bool is_sucess = false;
        voxelmap->build_single_residual_nocov(point_w, pt_data, current_octo, 0, is_sucess);
        if (!is_sucess) {
            const auto& vc = current_octo->voxel_center_;
            const float ql = current_octo->quater_length_;                
            VOXEL_LOCATION near_position = position;
            if      (point_w.x() > vc[0] + ql) near_position.x++;
            else if (point_w.x() < vc[0] - ql) near_position.x--;
            if      (point_w.y() > vc[1] + ql) near_position.y++;
            else if (point_w.y() < vc[1] - ql) near_position.y--;
            if      (point_w.z() > vc[2] + ql) near_position.z++;
            else if (point_w.z() < vc[2] - ql) near_position.z--;
            if (near_position == position) {
                pt_data.if_valid = false;
                return;
            }
            auto iter_near = voxelmap->voxel_map_.find(near_position);
            if (iter_near != voxelmap->voxel_map_.end()) {
                voxelmap->build_single_residual_nocov(point_w, pt_data, iter_near->second, 0, is_sucess);  
            }              
        }
        pt_data.if_valid = is_sucess;
    }         
};
