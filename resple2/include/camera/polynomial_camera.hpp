/*
Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "abstract_camera.h"

namespace mini_vikit {

using namespace Eigen;

class PolynomialCamera : public vk::AbstractCamera {
private:
    const double fx_, fy_;
    const double cx_, cy_;
    const double skew_;
    bool distortion_;
    double k2_, k3_, k4_, k5_, k6_, k7_;
    Matrix3d K_;
    Matrix3d K_inv_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PolynomialCamera(double width, double height,
                    double fx, double fy, double cx, double cy, double skew,
                    double k2=0.0, double k3=0.0, double k4=0.0,
                    double k5=0.0, double k6=0.0, double k7=0.0)
        : vk::AbstractCamera(static_cast<int>(width), static_cast<int>(height), 1.0),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy), skew_(skew),
          distortion_(true) {
        k2_ = k2; k3_ = k3; k4_ = k4; k5_ = k5; k6_ = k6; k7_ = k7;
        K_ << fx_, skew_, cx_,
               0.0, fy_,   cy_,
               0.0, 0.0,   1.0;
        K_inv_ = K_.inverse();
        model_ = "polynomial";
    }

    virtual Vector3d cam2world(const double& u, const double& v) const override {
        Vector3d xyz;
        if (!distortion_) {
            double y = (v - cy_) / fy_;
            double x = (u - cx_ - y * skew_) / fx_;
            xyz << x, y, 1.0;
        } else {
            double y = (v - cy_) / fy_;
            double x = (u - cx_ - y * skew_) / fx_;

            const double thetad = std::sqrt(x * x + y * y);
            if (thetad < 1e-12) return Vector3d(0, 0, 1);
            double theta = thetad;

            for (int i = 0; i < 7; ++i) {
                const double theta2 = theta * theta;
                const double theta3 = theta2 * theta;
                const double theta4 = theta3 * theta;
                const double theta5 = theta4 * theta;
                const double theta6 = theta5 * theta;
                theta = thetad / (1.0 + k2_ * theta + k3_ * theta2 + k4_ * theta3 +
                                 k5_ * theta4 + k6_ * theta5 + k7_ * theta6);
            }

            const double scaling = std::sin(theta) / thetad;
            x *= scaling;
            y *= scaling;
            xyz << x, y, std::cos(theta);
        }
        return xyz.normalized();
    }

    virtual Vector3d cam2world(const Vector2d& px) const override {
        return cam2world(px[0], px[1]);
    }

    virtual Vector2d world2cam(const Vector3d& xyz_c, Eigen::Matrix<double, 2, 3>* J = nullptr) const override {
        Vector2d px;
        if (!distortion_) {
            px[0] = fx_ * xyz_c[0] + cx_;
            px[1] = fy_ * xyz_c[1] + cy_;
            if (J) {
                *J << fx_,  0.0, 0.0,
                       0.0, fy_, 0.0;
            }
        } else {
            const double x = xyz_c[0], y = xyz_c[1], z = xyz_c[2];
            const double r = std::sqrt(x * x + y * y);
            if (r < 1e-12 && z > 0) {
                if (J) *J << fx_/z, skew_/z, 0, 0, fy_/z, 0;
                return Vector2d(cx_, cy_);
            }
            const double theta = std::atan2(r, z);
            const double thetad = thetad_from_theta(theta);
            const double scaling = thetad / r;
            const double xd = x * scaling;
            const double yd = y * scaling;
            px[0] = xd * fx_ + yd * skew_ + cx_;
            px[1] = yd * fy_ + cy_;
            if (J) {
                const double sp = dthetad_dtheta(theta);
                const double r2 = r * r;
                const double n2 = xyz_c.squaredNorm();
                const double A  = sp * z / (r2 * n2);
                const double B  = scaling / r2;
                const double AB = A - B;

                const double Jxd0 = scaling * y*y / r2 + A * x*x;
                const double Jxd1 = AB * x * y;
                const double Jxd2 = -sp * x / n2;
                const double Jyd0 = AB * x * y;
                const double Jyd1 = scaling * x*x / r2 + A * y*y;
                const double Jyd2 = -sp * y / n2;

                (*J)(0, 0) = fx_ * Jxd0 + skew_ * Jyd0;
                (*J)(0, 1) = fx_ * Jxd1 + skew_ * Jyd1;
                (*J)(0, 2) = fx_ * Jxd2 + skew_ * Jyd2;
                (*J)(1, 0) = fy_ * Jyd0;
                (*J)(1, 1) = fy_ * Jyd1;
                (*J)(1, 2) = fy_ * Jyd2;
            }
        }
        return px;
    }

    Vector2d world2cam(const Vector2d& uv) const {
        Vector2d px;
        if (!distortion_) {
            px[0] = fx_ * uv[0] + cx_;
            px[1] = fy_ * uv[1] + cy_;
        } else {
            double xd, yd;
            const double r = uv.norm();
            if (r < 1e-8) {
                return Vector2d(fx_*uv.x() + skew_*uv.y() + cx_, fy_*uv.y() + cy_);
            }
            const double theta = std::atan(r);
            const double thetad = thetad_from_theta(theta);
            const double scaling = thetad / r;
            xd = uv[0] * scaling;
            yd = uv[1] * scaling;
            px[0] = xd * fx_ + yd * skew_ + cx_;
            px[1] = yd * fy_ + cy_;
        }
        return px;
    }

    inline double thetad_from_theta(const double theta) const {
        const double theta2 = theta * theta;
        const double theta3 = theta2 * theta;
        const double theta4 = theta3 * theta;
        const double theta5 = theta4 * theta;
        const double theta6 = theta5 * theta;
        const double theta7 = theta6 * theta;
        return theta + k2_*theta2 + k3_*theta3 + k4_*theta4
                     + k5_*theta5 + k6_*theta6 + k7_*theta7;
    }

    inline double dthetad_dtheta(const double theta) const {
        const double theta2 = theta * theta;
        const double theta3 = theta2 * theta;
        const double theta4 = theta3 * theta;
        const double theta5 = theta4 * theta;
        const double theta6 = theta5 * theta;
        return 1.0 + 2.0*k2_*theta  + 3.0*k3_*theta2 + 4.0*k4_*theta3
                   + 5.0*k5_*theta4 + 6.0*k6_*theta5 + 7.0*k7_*theta6;
    }

    virtual double errorMultiplier2() const override { return std::fabs(fx_); }
    virtual double errorMultiplier() const override { return std::fabs(4.0 * fx_ * fy_); }

    inline const Matrix3d& K() const override { return K_; }
    inline const Matrix3d& K_inv() const override { return K_inv_; }

    virtual double fx() const override { return fx_; }
    virtual double fy() const override { return fy_; }
    virtual double cx() const override { return cx_; }
    virtual double cy() const override { return cy_; }
    double skew() const { return skew_; }
    bool has_distortion() const { return distortion_; }

    double k2() const { return k2_; }
    double k3() const { return k3_; }
    double k4() const { return k4_; }
    double k5() const { return k5_; }
    double k6() const { return k6_; }
    double k7() const { return k7_; }
};
}
