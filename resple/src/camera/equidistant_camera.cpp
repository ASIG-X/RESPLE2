/*
 * equidistant_camera.cpp
 *
 *  Created on: January 26, 2023
 *      Author: Chunran Zheng
 */

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "camera/equidistant_camera.h"

namespace vk {

EquidistantCamera::
EquidistantCamera(double width, double height, double scale,
              double fx, double fy,
              double cx, double cy,
              double k1, double k2, double k3, double k4) :
              AbstractCamera(width * scale , height * scale, scale),
              fx_(fx * scale), fy_(fy * scale), cx_(cx * scale), cy_(cy * scale),
              distortion_(true)
{
  k1_ = k1; k2_ = k2; k3_ = k3; k4_ = k4;
  K_ << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
  K_inv_ = K_.inverse();  
  model_ = "Equidistant";
}

EquidistantCamera::
~EquidistantCamera()
{}

Vector3d EquidistantCamera::
cam2world(const double& u, const double& v) const
{
  Vector3d xyz;
  if(!distortion_)
  {
    xyz[0] = (u - cx_)/fx_;
    xyz[1] = (v - cy_)/fy_;
    xyz[2] = 1.0;
  }
  else
  {
    double x = (u - cx_)/fx_;
    double y = (v - cy_)/fy_;
    const double thetad = std::sqrt(x * x + y * y);
    if (thetad < 1e-12) return Vector3d(0, 0, 1);
    double theta = thetad;
    for (int i = 0; i < 20; ++i)
    {
      const double theta2 = theta * theta;
      const double theta4 = theta2 * theta2;
      const double theta6 = theta4 * theta2;
      const double theta8 = theta4 * theta4;
      const double residual = theta * (1.0 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8) - thetad;
      const double derivative = 1.0 + 3*k1_*theta2 + 5*k2_*theta4 + 7*k3_*theta6 + 9*k4_*theta8;
      if (std::abs(derivative) < 1e-12) break;
      const double step = residual / derivative;
      theta -= step;
      if (std::abs(step) < 1e-12) break;
    }
    const double scaling = std::sin(theta) / thetad;
    x *= scaling;
    y *= scaling;
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = std::cos(theta);
  }
  return xyz.normalized();
}

Vector3d EquidistantCamera::
cam2world (const Vector2d& uv) const
{
  return cam2world(uv[0], uv[1]);
}

Vector2d EquidistantCamera::
world2cam(const Vector3d& xyz, Eigen::Matrix<double, 2, 3>* J) const
{
  Eigen::Vector2d uv = xyz.head<2>()/xyz[2];

  Vector2d px;
  if(!distortion_) {
    px[0] = fx_*uv[0] + cx_;
    px[1] = fy_*uv[1] + cy_;
    if (J != nullptr) {
      const double x = xyz[0];
      const double y = xyz[1];
      const double z_inv = 1. / xyz[2];
      const double z_inv_2 = z_inv * z_inv;
      *J << fx_ * z_inv, 0.0, -fx_ * x * z_inv_2,
            0.0, fy_ * z_inv, -fy_ * y * z_inv_2;
    }    
  } else {
    double xd, yd;
    const double r = uv.norm();
    if (r < 1e-8) {
      if (J != nullptr) {
        const double x = xyz[0];
        const double y = xyz[1];
        const double z_inv = 1. / xyz[2];
        const double z_inv_2 = z_inv * z_inv;
        *J << fx_ * z_inv, 0.0, -fx_ * x * z_inv_2,
              0.0, fy_ * z_inv, -fy_ * y * z_inv_2;
      }
      return Vector2d(fx_ * uv.x() + cx_, fy_ * uv.y() + cy_);
    }
    const double theta = std::atan(r);
    const double thetad = thetad_from_theta(theta);
    const double scaling = thetad / r;
    xd = uv[0] * scaling;
    yd = uv[1] * scaling;
    px[0] = xd*fx_ + cx_;
    px[1] = yd*fy_ + cy_;
    if (J != nullptr) {
      const double Z = xyz[2];
      const double invZ = 1.0 / Z;

      const double u = uv[0];
      const double v = uv[1];

      const double th2 = theta * theta;
      const double th4 = th2 * th2;
      const double th6 = th4 * th2;
      const double th8 = th4 * th4;

      const double dthetad_dtheta =
          1.0 +
          3.0 * k1_ * th2 +
          5.0 * k2_ * th4 +
          7.0 * k3_ * th6 +
          9.0 * k4_ * th8;

      const double s = thetad / r;

      const double dr_du = u / r;
      const double dr_dv = v / r;

      const double dtheta_du = dr_du / (1.0 + r * r);
      const double dtheta_dv = dr_dv / (1.0 + r * r);

      const double dthetad_du = dthetad_dtheta * dtheta_du;
      const double dthetad_dv = dthetad_dtheta * dtheta_dv;

      const double inv_r2 = 1.0 / (r * r);

      const double ds_du =
          (r * dthetad_du - thetad * dr_du) * inv_r2;

      const double ds_dv =
          (r * dthetad_dv - thetad * dr_dv) * inv_r2;

      const double du_dX = invZ;
      const double du_dY = 0.0;
      const double du_dZ = -u * invZ;

      const double dv_dX = 0.0;
      const double dv_dY = invZ;
      const double dv_dZ = -v * invZ;

      const double ds_dX =
          ds_du * du_dX +
          ds_dv * dv_dX;

      const double ds_dY =
          ds_du * du_dY +
          ds_dv * dv_dY;

      const double ds_dZ =
          ds_du * du_dZ +
          ds_dv * dv_dZ;

      const double dxd_dX =
          s * du_dX +
          u * ds_dX;

      const double dxd_dY =
          s * du_dY +
          u * ds_dY;

      const double dxd_dZ =
          s * du_dZ +
          u * ds_dZ;

      const double dyd_dX =
          s * dv_dX +
          v * ds_dX;

      const double dyd_dY =
          s * dv_dY +
          v * ds_dY;

      const double dyd_dZ =
          s * dv_dZ +
          v * ds_dZ;

      *J <<
          fx_ * dxd_dX, fx_ * dxd_dY, fx_ * dxd_dZ,
          fy_ * dyd_dX, fy_ * dyd_dY, fy_ * dyd_dZ;
    }    

  }
  return px;  
}

}
