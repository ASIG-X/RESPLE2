#pragma once
#include "camera/abstract_camera.h"
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <stdexcept>

struct IbtcCameraProjection {
    const vk::AbstractCamera& camera;
    double downsample;
    Eigen::Matrix3d rectification;

    IbtcCameraProjection(const vk::AbstractCamera& model, double factor = 1.0,
                         const Eigen::Matrix3d& rotation = Eigen::Matrix3d::Identity())
        : camera(model), downsample(factor), rectification(rotation) {
        if (!std::isfinite(factor) || factor <= 0)
            throw std::invalid_argument("IBTC camera downsample factor must be positive");
    }
    Eigen::Vector2d project(const Eigen::Vector3d& point) const {
        const Eigen::Vector3d rectified = rectification * point;
        if (!point.allFinite() || point.z() <= 0 || rectified.z() <= 0)
            return Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
        return camera.world2cam(rectified) / downsample;
    }
    Eigen::Vector3d unproject(const Eigen::Vector2d& pixel, double depth_z) const {
        const Eigen::Vector3d ray = rectification.transpose() * camera.cam2world(pixel * downsample);
        if (!ray.allFinite() || ray.z() <= 1e-9 || !std::isfinite(depth_z) || depth_z <= 0)
            return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
        return ray * (depth_z / ray.z());
    }
};
