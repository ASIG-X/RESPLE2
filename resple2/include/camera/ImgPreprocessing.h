#ifndef IMGPREPROCESSING_H_
#define IMGPREPROCESSING_H_

#include "utils/common_utils.h"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>

class ImgPreprocessing
{
public:

  ImgPreprocessing();
  ImgPreprocessing(std::shared_ptr<rclcpp::Node>& node, const std::string& ns);
  ~ImgPreprocessing();

  std::string cam_model;
  int width = 0;
  int height = 0;
  double scale = 1.0;
  cv::Mat K, D;
  bool undist_maps_initialized = false;
  bool if_vignette_correct = false;
  bool if_undist = false;

  double vignetting_correction_scale_ = 1.5;
  double vignetting_correction_a2_ = 0.001;
  double vignetting_correction_a4_ = 000001;
  cv::Mat vignetting_mask_f_;    
  cv::Mat undistortion_map_x_;
  cv::Mat undistortion_map_y_;
  cv::Mat undistort_scratch_;

  void init(const cv::Mat& K_, const cv::Mat& D_, vk::AbstractCamera*& cam);
  void init(const cv::Mat& K_, const cv::Mat& D_, const cv::Mat& R, vk::AbstractCamera*& cam);
  void init(std::shared_ptr<rclcpp::Node>& nh, const std::string& ns, vk::AbstractCamera*& cam);
  void preprocess(cv::Mat& img_get);
  void precomputeVignettingMask(int height, int width);
  void getUndistMap(const cv::Mat& R);
  void VignettingCorrect(cv::Mat& img_get);
  void undistortImage(cv::Mat& img_get);

private:
  cv::Mat lab_scratch_, luminance_scratch_, luminance_float_scratch_;
  void validateConfiguration() const;
  void buildCameraModel(const cv::Mat& R, vk::AbstractCamera*& cam);
};

#endif
