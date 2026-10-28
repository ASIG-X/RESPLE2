#include "camera/ImgPreprocessing.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

cv::Mat validatedIntrinsics(const cv::Mat& input)
{
  if (input.rows != 3 || input.cols != 3 || input.channels() != 1 ||
      (input.depth() != CV_32F && input.depth() != CV_64F))
    throw std::invalid_argument("Camera intrinsics must be a 3x3 floating-point matrix");
  cv::Mat result;
  input.convertTo(result, CV_64F);
  if (!cv::checkRange(result) || result.at<double>(0, 0) <= 0.0 || result.at<double>(1, 1) <= 0.0)
    throw std::invalid_argument("Camera intrinsics must be finite with positive focal lengths");
  return result;
}

}

ImgPreprocessing::ImgPreprocessing()
{
}

ImgPreprocessing::ImgPreprocessing(std::shared_ptr<rclcpp::Node>& nh, const std::string& ns)
{
  scale = CommonUtils::readParam<double>(nh, ns + "scale");  
  if_vignette_correct = CommonUtils::readParam<bool>(nh, ns + "if_vignette_correct");  
  if_undist = CommonUtils::readParam<bool>(nh, ns + "if_undist");  
  cam_model = CommonUtils::readParam<std::string>(nh, ns + "cam_model");  
  width = CommonUtils::readParam<int>(nh, ns + "cam_width");
  height = CommonUtils::readParam<int>(nh, ns + "cam_height");
  validateConfiguration();
}

ImgPreprocessing::~ImgPreprocessing()
{
}

void ImgPreprocessing::validateConfiguration() const
{
  if (cam_model != "Pinhole" && cam_model != "EquidistantCamera" && cam_model != "PolynomialCamera")
    throw std::invalid_argument("Unsupported camera model: " + cam_model);
  if (cam_model == "PolynomialCamera" && if_undist)
    throw std::invalid_argument("PolynomialCamera requires if_undist: false");
  if (width <= 0 || height <= 0 || !std::isfinite(scale) || scale <= 0.0 ||
      width * scale < 1.0 || height * scale < 1.0 ||
      width * scale > std::numeric_limits<int>::max() || height * scale > std::numeric_limits<int>::max())
    throw std::invalid_argument("Camera dimensions and scale must produce a positive, representable image size");
}

void ImgPreprocessing::init(const cv::Mat& K_, const cv::Mat& D_, vk::AbstractCamera*& cam)
{
  init(K_, D_, cv::Mat::eye(3, 3, CV_64F), cam);
}

void ImgPreprocessing::init(const cv::Mat& K_, const cv::Mat& D_, const cv::Mat& R, vk::AbstractCamera*& cam)
{
  validateConfiguration();
  if (cam_model == "PolynomialCamera")
    throw std::invalid_argument("PolynomialCamera requires YAML calibration (if_camera_info: false)");
  const cv::Mat new_K = validatedIntrinsics(K_);
  const int count = cam_model == "EquidistantCamera" ? 4 : 5;
  cv::Mat new_D = cv::Mat::zeros(count, 1, CV_64F);
  if (!D_.empty()) {
    if (D_.channels() != 1 || (D_.rows != 1 && D_.cols != 1) || !cv::checkRange(D_))
      throw std::invalid_argument("Camera distortion coefficients must be a finite vector");
    cv::Mat flat;
    D_.clone().reshape(1, static_cast<int>(D_.total())).convertTo(flat, CV_64F);
    for (int i = 0; i < flat.rows; ++i) {
      if (i < count) new_D.at<double>(i) = flat.at<double>(i);
      else if (flat.at<double>(i) != 0.0)
        throw std::invalid_argument("Unsupported extra CameraInfo distortion coefficients");
    }
  }

  K = new_K;
  D = new_D;
  buildCameraModel(R, cam);
}

void ImgPreprocessing::init(std::shared_ptr<rclcpp::Node>& nh, const std::string& ns, vk::AbstractCamera*& cam)
{
  validateConfiguration();
  double fx = CommonUtils::readParam<double>(nh, ns + "cam_fx");   
  double fy = CommonUtils::readParam<double>(nh, ns + "cam_fy");   
  double cx = CommonUtils::readParam<double>(nh, ns + "cam_cx");   
  double cy = CommonUtils::readParam<double>(nh, ns + "cam_cy");                 
  K = (cv::Mat_<double>(3,3) << fx, 0.0, cx,
                                  0.0, fy, cy,
                                  0.0, 0.0, 1.0);     
  K = validatedIntrinsics(K);
  if(cam_model != "PolynomialCamera") {
    double k1 = CommonUtils::readParam<double>(nh, ns + "k1");   
    double k2 = CommonUtils::readParam<double>(nh, ns + "k2");   
    double k3 = CommonUtils::readParam<double>(nh, ns + "k3");   
    double k4 = CommonUtils::readParam<double>(nh, ns + "k4");    
    double k5 = CommonUtils::readParam<double>(nh, ns + "k5");       
    D = (cv::Mat_<double>(5, 1) << k1, k2, k3, k4, k5);   
  } else {
    double skew = CommonUtils::readParam<double>(nh, ns + "skew"); 
    double k2 = CommonUtils::readParam<double>(nh, ns + "k2");   
    double k3 = CommonUtils::readParam<double>(nh, ns + "k3");   
    double k4 = CommonUtils::readParam<double>(nh, ns + "k4");    
    double k5 = CommonUtils::readParam<double>(nh, ns + "k5");      
    double k6 = CommonUtils::readParam<double>(nh, ns + "k6");  
    double k7 = CommonUtils::readParam<double>(nh, ns + "k7");  
    for (double value : {skew, k2, k3, k4, k5, k6, k7}) {
      if (!std::isfinite(value)) throw std::invalid_argument("Polynomial calibration must be finite");
    }
    cam = new mini_vikit::PolynomialCamera(width * scale, height * scale, scale * K.at<double>(0, 0), scale * K.at<double>(1, 1), scale * K.at<double>(0, 2),
    scale * K.at<double>(1, 2), scale * skew, k2, k3, k4, k5, k6, k7);
    undist_maps_initialized = true;
    return;
  }
  cv::Mat R = (cv::Mat_<double>(3,3) << 1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0);

  buildCameraModel(R, cam);
}

void ImgPreprocessing::buildCameraModel(const cv::Mat& R, vk::AbstractCamera*& cam)
{
  if (!cv::checkRange(D)) throw std::invalid_argument("Camera distortion coefficients must be finite");
  if(if_undist) {
    if (cam_model != "Pinhole" && cam_model != "EquidistantCamera")
      throw std::invalid_argument("Undistortion is unsupported for this camera model");
    getUndistMap(R);
    cam = new vk::PinholeCamera(width * scale, height * scale, 1.0,
        K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2));
    undist_maps_initialized = true;
    return;
  }
  if(cam_model == "Pinhole") {
      cam = new vk::PinholeCamera(width, height, scale, K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2),
      K.at<double>(1, 2), D.at<double>(0, 0), D.at<double>(1, 0), D.at<double>(2, 0), D.at<double>(3, 0),
      D.rows > 4 ? D.at<double>(4, 0) : 0.0);
  } else if(cam_model == "EquidistantCamera") {
      cam = new vk::EquidistantCamera(width, height, scale, K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2),
      K.at<double>(1, 2), D.at<double>(0, 0), D.at<double>(1, 0), D.at<double>(2, 0), D.at<double>(3, 0));
  } else if(cam_model == "PolynomialCamera") {
      cam = new mini_vikit::PolynomialCamera(width, height, K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2),
      K.at<double>(1, 2), D.at<double>(0, 0), D.at<double>(1, 0), D.at<double>(2, 0), D.at<double>(3, 0));      
  } else {
      throw std::invalid_argument("Unsupported camera model: " + cam_model);
  }
  undist_maps_initialized = true;
}

void ImgPreprocessing::preprocess(cv::Mat& img_get)
{
  if (if_vignette_correct) {
    precomputeVignettingMask(img_get.rows, img_get.cols);
    VignettingCorrect(img_get);
  }
  if (if_undist) {
    undistortImage(img_get);
  }
}

void ImgPreprocessing::precomputeVignettingMask(int height, int width) {
    if (height == vignetting_mask_f_.rows && width == vignetting_mask_f_.cols) return;

    vignetting_mask_f_.create(height, width, CV_32F);

    double cx = width / 2.0;
    double cy = height / 2.0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
        double r = std::sqrt(std::pow(y - cy, 2) + std::pow(x - cx, 2));
        double k = pow(r, 2) * vignetting_correction_a2_ + pow(r, 4) * vignetting_correction_a4_;
        vignetting_mask_f_.at<float>(y, x) = k;
        }
    }
    double max;
    cv::minMaxLoc(vignetting_mask_f_, NULL, &max, NULL, NULL);
    vignetting_mask_f_ = max > 0 ? vignetting_mask_f_ / max : vignetting_mask_f_;
    vignetting_mask_f_ = vignetting_mask_f_ * vignetting_correction_scale_;
    vignetting_mask_f_ += 1.0;
}    

void ImgPreprocessing::getUndistMap(const cv::Mat& R)
{
    const cv::Size input_size(width, height);
    const cv::Size image_size(width * scale, height * scale);
    const double rect_balance = 0.0;
    const double rect_fov_scale = 1.0;

    if (cam_model == "EquidistantCamera") {
      D = (cv::Mat_<double>(4, 1) << D.at<double>(0, 0), D.at<double>(1, 0), D.at<double>(2, 0), D.at<double>(3, 0));

      cv::Mat new_K;
      cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, input_size, R, new_K, rect_balance, image_size, rect_fov_scale);
      cv::fisheye::initUndistortRectifyMap(K, D, R, new_K, image_size, CV_32F, undistortion_map_x_, undistortion_map_y_);

      K = new_K;
    } else if (cam_model == "Pinhole") {
      cv::Mat new_K = cv::getOptimalNewCameraMatrix(K, D, input_size, 1, image_size, 0);
      cv::initUndistortRectifyMap(K, D, R, new_K, image_size, CV_16SC2, undistortion_map_x_, undistortion_map_y_);

      K = new_K;
    } else {
      return;
    }
    D.setTo(0);
}

void ImgPreprocessing::VignettingCorrect(cv::Mat& img_get)
{
  cv::cvtColor(img_get, lab_scratch_, cv::COLOR_BGR2Lab);
  cv::extractChannel(lab_scratch_, luminance_scratch_, 0);
  luminance_scratch_.convertTo(luminance_float_scratch_, CV_32FC1);
  cv::multiply(luminance_float_scratch_, vignetting_mask_f_,
      luminance_float_scratch_, 1.0, CV_32FC1);
  luminance_float_scratch_.convertTo(luminance_scratch_, CV_8UC1);
  cv::insertChannel(luminance_scratch_, lab_scratch_, 0);
  cv::cvtColor(lab_scratch_, img_get, cv::COLOR_Lab2BGR);
}

void ImgPreprocessing::undistortImage(cv::Mat& img_get)
{
  cv::remap(img_get, undistort_scratch_, undistortion_map_x_, undistortion_map_y_, cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT,
              0);
  undistort_scratch_.copyTo(img_get);
}
