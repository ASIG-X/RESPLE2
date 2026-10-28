/*
 * pinhole_camera.cpp
 *
 *  Created on: Jul 24, 2012
 *      Author: cforster
 */

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "camera/pinhole_camera.h"

namespace vk {

PinholeCamera::
PinholeCamera(double width, double height, double scale,
              double fx, double fy,
              double cx, double cy,
              double d0, double d1, double d2, double d3, double d4) :
              AbstractCamera(width * scale , height * scale, scale),
              fx_(fx * scale), fy_(fy * scale), cx_(cx * scale), cy_(cy * scale),
              distortion_(d0 != 0 || d1 != 0 || d2 != 0 || d3 != 0 || d4 != 0),
              undist_map1_(height_, width_, CV_16SC2),
              undist_map2_(height_, width_, CV_16SC2),
              use_optimization_(false)
{
  d_[0] = d0; d_[1] = d1; d_[2] = d2; d_[3] = d3; d_[4] = d4;
  cvK_ = (cv::Mat_<float>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
  cvD_ = (cv::Mat_<float>(1, 5) << d_[0], d_[1], d_[2], d_[3], d_[4]);
  cv::initUndistortRectifyMap(cvK_, cvD_, cv::Mat_<double>::eye(3,3), cvK_,
                              cv::Size(width_, height_), CV_16SC2, undist_map1_, undist_map2_);
  K_ << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
  K_inv_ = K_.inverse();
  model_ = "Pinhole";
}

PinholeCamera::
~PinholeCamera()
{}

Vector3d PinholeCamera::
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
    cv::Point2f uv(u,v), px;
    const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
    cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
    cv::undistortPoints(src_pt, dst_pt, cvK_, cvD_);
    xyz[0] = px.x;
    xyz[1] = px.y;
    xyz[2] = 1.0;
  }
  return xyz.normalized();
}

Vector3d PinholeCamera::
cam2world (const Vector2d& uv) const
{
  return cam2world(uv[0], uv[1]);
}

Vector2d PinholeCamera::
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
    double x, y, r2, r4, r6, a1, a2, a3, cdist, xd, yd;
    x = uv[0];
    y = uv[1];
    r2 = x*x + y*y;
    r4 = r2*r2;
    r6 = r4*r2;
    a1 = 2*x*y;
    a2 = r2 + 2*x*x;
    a3 = r2 + 2*y*y;
    cdist = 1 + d_[0]*r2 + d_[1]*r4 + d_[4]*r6;
    xd = x*cdist + d_[2]*a1 + d_[3]*a2;
    yd = y*cdist + d_[2]*a3 + d_[3]*a1;
    px[0] = xd*fx_ + cx_;
    px[1] = yd*fy_ + cy_;
    if (J != nullptr) {
      const double k1 = d_[0], k2 = d_[1], p1 = d_[2], p2 = d_[3], k3 = d_[4];
      const double A = k1 + 2.0*k2*r2 + 3.0*k3*r4;

      const double dcdx = 2.0*x*A;
      const double dcdy = 2.0*y*A;

      const double dxd_dx = cdist + x*dcdx + 2.0*p1*y + 6.0*p2*x;
      const double dxd_dy = x*dcdy + 2.0*p1*x + 2.0*p2*y;

      const double dyd_dx = y*dcdx + 2.0*p1*x + 2.0*p2*y;
      const double dyd_dy = cdist + y*dcdy + 6.0*p1*y + 2.0*p2*x;

      const double invZ = 1.0 / xyz.z();

      const double dxd_dX = invZ * dxd_dx;
      const double dxd_dY = invZ * dxd_dy;
      const double dxd_dZ = -invZ * (x*dxd_dx + y*dxd_dy);

      const double dyd_dX = invZ * dyd_dx;
      const double dyd_dY = invZ * dyd_dy;
      const double dyd_dZ = -invZ * (x*dyd_dx + y*dyd_dy);

      *J << fx_ * dxd_dX, fx_ * dxd_dY, fx_ * dxd_dZ,
          fy_ * dyd_dX, fy_ * dyd_dY, fy_ * dyd_dZ;      
    }    
  }
  return px;  
}

void PinholeCamera::
undistortImage(const cv::Mat& raw, cv::Mat& rectified)
{
  if(distortion_) {
    cv::remap(raw, rectified, undist_map1_, undist_map2_, cv::INTER_LINEAR);
  }
  else {
    rectified = raw.clone();
  }
}

}
