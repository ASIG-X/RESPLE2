#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/image_encodings.hpp>
#ifdef ROS_DISTRO_HUMBLE
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "SplineState.h"
#include "camera/ImgPreprocessing.h"

class GlobalMapPublisher
{
  public:
    using Cloud = pcl::PointCloud<pcl::PointXYZINormal>;
    using RgbCloud = pcl::PointCloud<pcl::PointXYZRGB>;

    GlobalMapPublisher() = default;
    ~GlobalMapPublisher() { stop(); }

    GlobalMapPublisher(const GlobalMapPublisher&) = delete;
    GlobalMapPublisher& operator=(const GlobalMapPublisher&) = delete;

    void start(int argc, char** argv)
    {
        if (worker_.joinable()) {
            return;
        }
        context_ = std::make_shared<rclcpp::Context>();
        context_->init(argc, argv);

        rclcpp::NodeOptions opt;
        opt.context(context_);
        opt.use_global_arguments(false);
        opt.start_parameter_services(false);
        opt.start_parameter_event_publisher(false);
        opt.enable_rosout(false);
        node_ = std::make_shared<rclcpp::Node>("global_map_publisher", opt);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(2)).reliable();
        pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("global_map", qos);
        pub_rgb_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
            "global_map_rgb_corrected", rclcpp::QoS(1).reliable().transient_local());

        running_ = true;
        worker_ = std::thread(&GlobalMapPublisher::run, this);
    }

    void submit(Cloud::Ptr cloud, bool priority = false)
    {
        if (!cloud) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!running_) {
                return;
            }
            (priority ? pending_hi_ : pending_lo_) = std::move(cloud);
        }
        cv_.notify_one();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!running_ && !worker_.joinable()) {
                return;
            }
            running_ = false;
        }
        cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
        pub_.reset();
        pub_rgb_.reset();
        node_.reset();
        if (context_ && context_->is_valid()) {
            context_->shutdown("global map publisher stop");
        }
        context_.reset();
    }

    void submitCorrectedRgb(RgbCloud::ConstPtr cloud)
    {
        if (!cloud) return;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!running_) return;
            pending_rgb_ = std::move(cloud);
        }
        cv_.notify_one();
    }

  private:
    template<typename PointT>
    void publishCloud(const pcl::PointCloud<PointT>& cloud,
                      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher)
    {
        pcl::PCLPointCloud2 pcl_msg;
        pcl::toPCLPointCloud2(cloud, pcl_msg);
        pcl_msg.header.stamp = cloud.header.stamp / 1000;
        sensor_msgs::msg::PointCloud2 msg;
        pcl_conversions::moveFromPCL(pcl_msg, msg);
        msg.header.stamp = rclcpp::Time(static_cast<int64_t>(cloud.header.stamp));
        msg.header.frame_id = "world";
        publisher->publish(std::move(msg));
    }

    void run()
    {
        while (true) {
            Cloud::Ptr cloud;
            RgbCloud::ConstPtr rgb;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return !running_ || pending_hi_ || pending_lo_ || pending_rgb_; });
                rgb.swap(pending_rgb_);
                if (pending_hi_) {
                    cloud.swap(pending_hi_);
                } else if (pending_lo_) {
                    cloud.swap(pending_lo_);
                } else if (!rgb) {
                    return;
                }
            }
            if (cloud && !cloud->empty()) {
                publishCloud(*cloud, pub_);
            }
            if (rgb) publishCloud(*rgb, pub_rgb_);
        }
    }

    rclcpp::Context::SharedPtr context_;
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_rgb_;

    std::thread worker_;
    std::mutex m_;
    std::condition_variable cv_;
    Cloud::Ptr pending_hi_;
    Cloud::Ptr pending_lo_;
    RgbCloud::ConstPtr pending_rgb_;
    bool running_ = false;
};

class RgbCameraColorizer
{
public:
    RgbCameraColorizer(rclcpp::Node::SharedPtr& nh, const std::string& name, bool subscribe_images)
    {
        config_.loadParameters(nh, name + ".");
        preprocessing_.reset(new ImgPreprocessing(nh, config_.pref));
        preprocessing_->scale = 1.0;
        preprocessing_->if_undist = false;
        if (!config_.if_camera_info) {
            vk::AbstractCamera* camera = nullptr;
            preprocessing_->init(nh, config_.pref, camera);
            camera_.reset(camera);
            if (!camera_) throw std::runtime_error("Unsupported RGB camera model");
        } else {
            info_sub_ = nh->create_subscription<sensor_msgs::msg::CameraInfo>(
                config_.topic_info, rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (camera_ || msg->k[0] <= 0 || msg->k[4] <= 0) return;
                    cv::Mat K(3, 3, CV_64F);
                    std::copy(msg->k.begin(), msg->k.end(), K.ptr<double>());
                    cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
                    std::copy_n(msg->d.begin(), std::min<size_t>(4, msg->d.size()), D.ptr<double>());
                    preprocessing_->width = msg->width;
                    preprocessing_->height = msg->height;
                    vk::AbstractCamera* camera = nullptr;
                    preprocessing_->init(K, D, cv::Mat::eye(3, 3, CV_64F), camera);
                    camera_.reset(camera);
                });
        }
        if (!subscribe_images) return;
        const auto qos = rclcpp::SensorDataQoS().keep_last(10);
        if (config_.if_compressed) {
            compressed_sub_ = nh->create_subscription<sensor_msgs::msg::CompressedImage>(
                config_.topic, qos, [this](sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) { bufferImage(msg); });
        } else {
            image_sub_ = nh->create_subscription<sensor_msgs::msg::Image>(
                config_.topic, qos, [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { bufferImage(msg); });
        }
    }

    template<typename MessagePtr>
    void addImage(const MessagePtr& msg) { bufferImage(msg); }

    template<typename PointType>
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorize(
        const pcl::PointCloud<PointType>& world, const SplineState* spline, int64_t boundary_ns,
        std::vector<bool>& assigned)
    {
        if (world.empty()) return nullptr;
        int64_t first = std::numeric_limits<int64_t>::max(), last = std::numeric_limits<int64_t>::min();
        for (const auto& point : world) {
            const int64_t time = static_cast<int64_t>(world.header.stamp) + static_cast<int64_t>(point.intensity * float(1e6));
            first = std::min(first, time);
            last = std::max(last, time);
        }
        const int64_t scan_time = first + (last - first) / 2;
        Image selected;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!camera_) {
                return nullptr;
            }
            while (!images_.empty() && images_.front().time_ns < spline->minTimeNs()) images_.pop_front();
            int64_t closest = std::numeric_limits<int64_t>::max();
            for (const auto& image : images_) {
                if (image.time_ns >= boundary_ns || image.time_ns > spline->maxTimeNs()) break;
                const int64_t distance = image.time_ns > scan_time ? image.time_ns - scan_time : scan_time - image.time_ns;
                if (distance < closest) {
                    closest = distance;
                    selected = image;
                }
            }
            if (!selected.raw && !selected.compressed) {
                return nullptr;
            }
        }
        if (selected.raw != decoded_source_.raw || selected.compressed != decoded_source_.compressed) {
            try {
                cv::Mat decoded = selected.raw
                    ? cv_bridge::toCvCopy(selected.raw, sensor_msgs::image_encodings::BGR8)->image
                    : cv_bridge::toCvCopy(selected.compressed, sensor_msgs::image_encodings::BGR8)->image;
                if (decoded.empty()) return nullptr;
                decoded_bgr_ = std::move(decoded);
                decoded_source_ = selected;
            } catch (const std::exception&) {
                return nullptr;
            }
        }
        const cv::Mat& bgr = decoded_bgr_;
        if (bgr.cols != camera_->width() || bgr.rows != camera_->height()) {
            return nullptr;
        }
        Eigen::Quaterniond q_wb;
        spline->itpQuaternionFromDelta(selected.time_ns, &q_wb);
        const Eigen::Vector3d t_wb = spline->itpPosition(selected.time_ns);
        auto colored = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        colored->header = world.header;
        colored->reserve(world.size());
        for (size_t i = 0; i < world.size(); ++i) {
            if (assigned[i]) continue;
            const auto& point = world[i];
            const Eigen::Vector3d p_world(point.x, point.y, point.z);
            const Eigen::Vector3d p_camera = config_.q_cb * (q_wb.conjugate() * (p_world - t_wb)) + config_.t_cb;
            if (!p_camera.allFinite() || p_camera.z() <= 0) continue;
            const Eigen::Vector2d pixel = camera_->world2cam(p_camera);
            if (!pixel.allFinite() || pixel.x() < 0 || pixel.y() < 0 ||
                pixel.x() >= bgr.cols || pixel.y() >= bgr.rows) continue;
            const auto color = bgr.at<cv::Vec3b>(static_cast<int>(pixel.y()), static_cast<int>(pixel.x()));
            pcl::PointXYZRGB out;
            out.x = point.x; out.y = point.y; out.z = point.z;
            out.r = color[2]; out.g = color[1]; out.b = color[0];
            colored->push_back(out);
            assigned[i] = true;
        }
        return colored;
    }

    void discardBefore(int64_t earliest_pending_ns)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (images_.size() > 1 && images_[1].time_ns < earliest_pending_ns) images_.pop_front();
    }

private:
    struct Image {
        int64_t time_ns = 0;
        sensor_msgs::msg::Image::ConstSharedPtr raw;
        sensor_msgs::msg::CompressedImage::ConstSharedPtr compressed;
        Image() = default;
        Image(int64_t time, sensor_msgs::msg::Image::ConstSharedPtr msg) : time_ns(time), raw(std::move(msg)) {}
        Image(int64_t time, sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) : time_ns(time), compressed(std::move(msg)) {}
    };

    template<typename MessagePtr>
    void bufferImage(const MessagePtr& msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t time_ns = config_.correctTimestampNs(rclcpp::Time(msg->header.stamp).nanoseconds());
        Image image(time_ns, msg);
        const auto position = std::lower_bound(images_.begin(), images_.end(), time_ns,
            [](const Image& buffered, int64_t time) { return buffered.time_ns < time; });
        if (position != images_.end() && position->time_ns == time_ns) *position = std::move(image);
        else images_.insert(position, std::move(image));
    }

    CameraConfig config_;
    std::unique_ptr<ImgPreprocessing> preprocessing_;
    std::unique_ptr<vk::AbstractCamera> camera_;
    std::mutex mutex_;
    std::deque<Image> images_;
    Image decoded_source_;
    cv::Mat decoded_bgr_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
};

class RgbMapColorizer
{
public:
    explicit RgbMapColorizer(rclcpp::Node::SharedPtr& nh, bool subscribe_images = true)
    {
        const auto names = CommonUtils::readParam<std::vector<std::string>>(nh, "cameras", {});
        if (names.empty()) throw std::runtime_error("RGB mapping requires a configured camera");
        for (const auto& name : names) {
            cameras_.emplace_back(new RgbCameraColorizer(nh, name, subscribe_images));
        }
        publisher_ = nh->create_publisher<sensor_msgs::msg::PointCloud2>("global_map_rgb", 2);
    }

    template<typename MessagePtr>
    void addImage(const MessagePtr& msg, size_t camera_index = 0)
    {
        cameras_.at(camera_index)->addImage(msg);
    }

    template<typename PointType>
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorize(
        const pcl::PointCloud<PointType>& world, const SplineState* spline, int64_t boundary_ns)
    {
        if (world.empty()) return nullptr;
        auto combined = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        combined->header = world.header;
        combined->reserve(world.size());
        std::vector<bool> assigned(world.size(), false);
        for (const auto& camera : cameras_) {
            auto colored = camera->colorize(world, spline, boundary_ns, assigned);
            if (colored) *combined += *colored;
            if (combined->size() == world.size()) break;
        }
        return combined;
    }

    void discardBefore(int64_t earliest_pending_ns)
    {
        for (const auto& camera : cameras_) camera->discardBefore(earliest_pending_ns);
    }

    void publish(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored)
    {
        if (!colored || colored->empty()) return;
        pcl::PCLPointCloud2 packed;
        pcl::toPCLPointCloud2(*colored, packed);
        packed.header.stamp = colored->header.stamp / 1000;
        sensor_msgs::msg::PointCloud2 msg;
        pcl_conversions::moveFromPCL(packed, msg);
        msg.header.frame_id = "world";
        msg.header.stamp = rclcpp::Time(static_cast<int64_t>(colored->header.stamp));
        publisher_->publish(msg);
    }

private:
    std::vector<std::unique_ptr<RgbCameraColorizer>> cameras_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};
