#pragma once

#ifdef __SSE2__
#include <emmintrin.h>
#endif
#include <rclcpp/rclcpp.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <omp.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include "utils/voxel_filter.h"
#include "utils/active_point_buffer.h"
#include <pcl/common/point_tests.h>
#include <pcl/common/common.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "utils/eigen_utils.hpp"
#include "camera/abstract_camera.h"
#include "camera/pinhole_camera.h"
#include "camera/equidistant_camera.h"
#include "camera/polynomial_camera.hpp"

extern int NUM_OF_THREAD;

#define RED         "\033[31m"
#define RESET       "\033[0m"

namespace resple2 {

class BufferTransferSchedule {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr auto period = std::chrono::milliseconds(5);

    bool due(Clock::time_point now) const { return now >= next_; }
    void attempted(Clock::time_point now) { next_ = now + period; }
    Clock::time_point deadline() const { return next_; }

private:
    Clock::time_point next_{};
};

class BufferHandoff {
public:
    template<class Append>
    void receive(std::mutex& mutex, Append&& append)
    {
        std::unique_lock<std::mutex> lock(mutex);
        std::forward<Append>(append)();
        lock.unlock();
    }

    template<class Container>
    bool take(std::mutex& mutex, Container& incoming, Container& local,
              bool blocking)
    {
        std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
        if (blocking) {
            lock.lock();
        } else if (!lock.try_lock()) {
            return false;
        }
        incoming.swap(local);
        lock.unlock();
        return true;
    }
};

inline void configureOpenCvThreads(const rclcpp::Node::SharedPtr& node)
{
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "OpenCV threads: -1 keeps the library default, 0 disables parallelism, positive values set the thread limit. Restart to change.";
    descriptor.read_only = true;
    const int64_t requested = node->declare_parameter<int64_t>(
        "opencv_num_threads", 5, descriptor);
    if (requested < -1 || requested > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("opencv_num_threads must be -1, 0, or a positive int");
    }
    if (requested >= 0) cv::setNumThreads(static_cast<int>(requested));
}

}

struct ImuData {
    int64_t time_ns;
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
    Eigen::Matrix<double, 6, 30> H;
    Eigen::Matrix<double, 6, 1> imu_itp;
    ImuData(){}
    ImuData(const int64_t s, const Eigen::Vector3d& w, const Eigen::Vector3d& a)
      : time_ns(s), gyro(w), accel(a) {}

    ImuData(const ImuData& other) : time_ns(other.time_ns), gyro(other.gyro), accel(other.accel),
    H(other.H), imu_itp(other.imu_itp) {
    }          
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PoseData {
    int64_t time_ns;
    Eigen::Quaterniond orient;
    Eigen::Vector3d pos;
    PoseData(){}
    PoseData(int64_t s, Eigen::Quaterniond& q, Eigen::Vector3d& t) : time_ns(s), orient(q), pos(t) {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class VisualPoint;
using VisualPointPtr = std::shared_ptr<VisualPoint>;

struct ImgData {
    int64_t time_ns;
    cv::Mat img;
    std::vector<cv::Point2f> m_last_tracked_pts;
    std::vector<VisualPointPtr> m_rgb_pts_ptr_vec_in_last_frame;
    Eigen::MatrixXd H;
    Eigen::VectorXd innv;
    Eigen::VectorXd mat_cov_inv;
    std::vector<bool> if_valid;

    ImgData(){}
    ImgData(const int64_t s, const cv::Mat& i) : time_ns(s), img(i) {}

    ImgData(const ImgData& other) : time_ns(other.time_ns), img(other.img),
        m_last_tracked_pts(other.m_last_tracked_pts),
        m_rgb_pts_ptr_vec_in_last_frame(other.m_rgb_pts_ptr_vec_in_last_frame) {
    }          
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct KinImuData {
    int64_t time_ns;
    double foot_pos_[4][3];
    double foot_vel_[4][3];
    bool contact_[4];
    Eigen::Matrix<double, 12, 24> H;
    Eigen::Matrix<double, 12, 1> kin_itp;    
    Eigen::Matrix<double, 12, 1> kin_meas;    

    Eigen::Matrix<double, 4, 24> H_pt2pl;
    Eigen::Matrix<double, 4, 1> kin_itp_pt2pl;    
    Eigen::Matrix<double, 4, 1> kin_meas_pt2pl;      
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct LoopData {
    int64_t time_i_ns;
    int64_t time_j_ns;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d translation;
    int idx_i;
    int idx_j;
    Eigen::Vector3d posi;
    Eigen::Vector3d posj;
    LoopData(){}
    LoopData(int64_t si, int64_t sj, Eigen::Quaterniond& q, Eigen::Vector3d& t, int idxi, int idxj) :
        time_i_ns(si), time_j_ns(sj), rotation(q), translation(t), idx_i(idxi), idx_j(idxj) {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<typename PointType>
struct LiDARKFs {
    Eigen::aligned_vector<PoseData> keyframePoses;
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> keyframeLaserClouds;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> keyframeRgbClouds;
    std::mutex m_frame_buff;    

    double thresh_kf_translation;
    double kf_distance_accumulated = 0;
    bool has_previous_odom = false;
    PoseData odom_pose_prev;
    Eigen::aligned_vector<PoseData> accumulated_odoms;    
    typename pcl::PointCloud<PointType>::Ptr accumulated_scans;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr accumulated_rgb;

    LiDARKFs() {
        accumulated_scans.reset(new pcl::PointCloud<PointType>());
        accumulated_rgb.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    }

    void reset(const Eigen::Quaterniond& rot_drift, const Eigen::Vector3d& trans_drift)
    {
        accumulated_odoms.clear();
        accumulated_scans->points.clear();
        accumulated_rgb->clear();
        kf_distance_accumulated = 0;
        odom_pose_prev.orient = rot_drift * odom_pose_prev.orient;
        odom_pose_prev.pos = rot_drift * odom_pose_prev.pos + trans_drift;
    }    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

namespace ouster_ros {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    }EIGEN_ALIGN16;
}  

POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint32_t, t, t)
)

namespace hesai_ros {
  struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (double, timestamp, timestamp)
)

namespace livox_mid360_boxi {
    struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint8_t tag;
      uint8_t line;
      double timestamp;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
  }
  POINT_CLOUD_REGISTER_POINT_STRUCT(livox_mid360_boxi::Point,
      (float, x, x)
      (float, y, y)
      (float, z, z)
      (float, intensity, intensity)
      (uint8_t, tag, tag)
      (uint8_t, line, line)
      (double, timestamp, timestamp)
  )

namespace rs_ros {
    struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(rs_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (double, timestamp, timestamp)
)  

namespace ls_ros {
    struct EIGEN_ALIGN16 Point {
        float x;
        float y;
        float z;
        uint8_t intensity;
        uint16_t confidence;
        float offset_time;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(ls_ros::Point,
      (float, x, x)
      (float, y, y)
      (float, z, z)
      (uint8_t, intensity, intensity)
      (uint16_t, confidence, confidence)
      (float, offset_time, offset_time)
)

typedef struct pointWithVar
{
  Eigen::Vector3d point_w;
  Eigen::Matrix3d var;
  float reflectivity;
  pointWithVar()
  {
    var = Eigen::Matrix3d::Zero();
    point_w = Eigen::Vector3d::Zero();
    reflectivity = 0;
  };

    pointWithVar(const pointWithVar& other) : point_w(other.point_w), var(other.var), 
        reflectivity(other.reflectivity){         
    }        

    pointWithVar& operator=(const pointWithVar& other) {
        if (this != &other) { 
            this->point_w = other.point_w;
            this->var = other.var;
            this->reflectivity = other.reflectivity;
        }
        return *this;
    }      
} pointWithVar;

class CommonUtils
{
public:
    template <typename T>
    static T readParam(rclcpp::Node::SharedPtr &n, std::string name)
    {
        T ans;
        if (!n->has_parameter(name)) {
            n->declare_parameter<T>(name);
        }        
        if (!n->get_parameter(name, ans)) {
            exit(1);
        }
        return ans;
    }

    template <typename T>
    static T readParam(rclcpp::Node::SharedPtr &n, std::string name, const T& alternative)
    {
        T ans;
        if (!n->has_parameter(name)) {
            n->declare_parameter<T>(name, alternative);
        }
        n->get_parameter_or(name, ans, alternative);
        return ans;
    }

    static Eigen::Vector3d readVector3d(rclcpp::Node::SharedPtr &n, const std::string& name)
    {
        std::vector<double> v = CommonUtils::readParam<std::vector<double>>(n, name);
        return Eigen::Vector3d(v[0], v[1], v[2]);
    }          

    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }      

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
                sin(y), cos(y), 0,
                0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
                0., 1., 0.,
                -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
                0., cos(r), -sin(r),
                0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
    {
        Eigen::Matrix3d R0;
        Eigen::Vector3d ng1 = g.normalized();
        Eigen::Vector3d ng2{0, 0, 1.0};
        R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        double yaw = CommonUtils::R2ypr(R0).x();
        R0 = CommonUtils::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        return R0;
    }    

    static int64_t ms2ns(const float t_ms) {
        return (t_ms * float(1e6));
    }    

    static bool time_list(pcl::PointXYZINormal &x, pcl::PointXYZINormal &y) {return (x.intensity < y.intensity);};

    static geometry_msgs::msg::PoseStamped pose2msg(const int64_t t, const Eigen::Vector3d& pos,
                                              const Eigen::Quaterniond& orient)
    {
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = rclcpp::Time(t);
        msg.pose.position.x = pos.x();
        msg.pose.position.y = pos.y();
        msg.pose.position.z = pos.z();
        msg.pose.orientation.w = orient.w();
        msg.pose.orientation.x = orient.x();
        msg.pose.orientation.y = orient.y();
        msg.pose.orientation.z = orient.z();
        return msg;
    }    

    struct BodyCovarianceNoise {
        float range_var;
        double angular_var;

        BodyCovarianceNoise(const float range_inc, const float degree_inc)
            : range_var(range_inc * range_inc), angular_var(pow(sin(DEG2RAD(degree_inc)), 2)) {}
    };

    static void calcBodyCov(Eigen::Vector3d &pb, const BodyCovarianceNoise& noise, Eigen::Matrix3d &cov) {
        if (pb[2] == 0) pb[2] = 0.0001;
        float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
        const float range_var = noise.range_var;
        Eigen::Matrix2d direction_var;
        direction_var << noise.angular_var, 0, 0, noise.angular_var;
        Eigen::Vector3d direction(pb);
        direction.normalize();
        Eigen::Matrix3d direction_hat;
        direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
        Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
        base_vector1.normalize();
        Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
        base_vector2.normalize();
        Eigen::Matrix<double, 3, 2> N;
        N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
        Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
        cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();        
    }

    static void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov) {
        calcBodyCov(pb, BodyCovarianceNoise(range_inc, degree_inc), cov);
    }

    static Eigen::Matrix3d ssm(const Eigen::Vector3d& v)
    {
        Eigen::Matrix3d mat;
        mat << 0.0, -v[2], v[1], 
                v[2], 0.0, -v[0], 
                -v[1], v[0], 0.0;
        return mat;
    }      

    static bool decideToAddByRange(const std::vector<pointWithVar>& pts,
        const pointWithVar& pt_to_add, float quater_length) {
        for (size_t i = 0; i < pts.size(); ++i) {
            if ((pts[i].point_w - pt_to_add.point_w).norm() < double(quater_length/3)) {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    static pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformCloud(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_in,
                                                                     const Eigen::Quaternion<T>& q, const Eigen::Matrix<T, 3, 1>& t)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_out(new pcl::PointCloud<pcl::PointXYZINormal>());

        size_t numPts = pc_in->points.size();
        pc_out->reserve(pc_in->size());
        for (size_t i = 0; i < numPts; ++i)
        {
            if (!pcl::isFinite(pc_in->points[i]))
                continue;            
            Eigen::Matrix<T, 3, 1> ptIn(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            Eigen::Matrix<T, 3, 1> ptOut = q * ptIn + t;

            pcl::PointXYZINormal pt = pc_in->points[i];
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = pc_in->points[i].intensity;
            pt.curvature = pc_in->points[i].curvature;
            pc_out->push_back(pt);
        }
        pc_out->width = pc_out->points.size();
        pc_out->height = 1;
        pc_out->is_dense = false;
        return pc_out;
    }  

};

struct LidarConfig {
    std::string topic;
    std::string type;
    int scan_line;
    float blind;
    Eigen::Quaterniond q_lb;
    Eigen::Vector3d t_lb;
    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;
    double w_pt;
    double beam_err;
    double dept_err;     
    resple2::VoxelDownsampler<pcl::PointXYZINormal> ds_filter_body;
    bool if_dense_map;

    LidarConfig() = default;

    LidarConfig(rclcpp::Node::SharedPtr& nh, const std::string& prefix) {
        topic = CommonUtils::readParam<std::string>(nh, prefix + "topic_lidar");
        type = CommonUtils::readParam<std::string>(nh, prefix + "lidar_type");
        scan_line = CommonUtils::readParam<int>(nh, prefix + "scan_line");
        blind = CommonUtils::readParam<float>(nh, prefix + "blind", 0.5f);
        std::vector<double> q_lb_v = CommonUtils::readParam<std::vector<double>>(nh, prefix + "q_lb");
        q_lb = Eigen::Quaterniond(q_lb_v.at(0), q_lb_v.at(1), q_lb_v.at(2), q_lb_v.at(3));
        t_lb = CommonUtils::readVector3d(nh, prefix + "t_lb");
        q_bl = q_lb.inverse();
        t_bl = q_lb.inverse() * (- t_lb);
        w_pt = CommonUtils::readParam<double>(nh, prefix + "w_pt");
        beam_err = CommonUtils::readParam<double>(nh, prefix + "beam_err");
        dept_err = CommonUtils::readParam<double>(nh, prefix + "dept_err");
        float ds_scan_voxel = CommonUtils::readParam<float>(nh, prefix + "ds_scan_voxel");
        ds_filter_body.setUseCustom(CommonUtils::readParam<bool>(nh, "if_custom_ds", false));
        ds_filter_body.setLeafSize(ds_scan_voxel, ds_scan_voxel, ds_scan_voxel);
        if_dense_map = CommonUtils::readParam<bool>(nh, prefix + "if_dense_map");  
    }
};

struct CameraConfig {
    std::string topic;
    std::string topic_info;
    std::string pref;
    Eigen::Quaterniond q_cb;
    Eigen::Vector3d t_cb;
    Eigen::Quaterniond q_bc;
    Eigen::Vector3d t_bc;
    bool if_compressed = false;
    bool if_vignette_correct = false;
    bool if_undist = false;
    double time_offset = 0;
    bool if_camera_info = false;
    int width;
    int height;
    double scale;

    CameraConfig() = default;

    void loadParameters(rclcpp::Node::SharedPtr& nh, const std::string& prefix) {
        topic = CommonUtils::readParam<std::string>(nh, prefix + "topic_camera");        
        topic_info = CommonUtils::readParam<std::string>(nh, prefix + "topic_info"); 
        pref = prefix;    
        if_vignette_correct = CommonUtils::readParam<bool>(nh, prefix + "if_vignette_correct");    
        if_undist =  CommonUtils::readParam<bool>(nh, prefix + "if_undist");    

        std::vector<double> q_cb_v = CommonUtils::readParam<std::vector<double>>(nh, prefix + "q_cb");
        q_cb = Eigen::Quaterniond(q_cb_v.at(0), q_cb_v.at(1), q_cb_v.at(2), q_cb_v.at(3)).normalized();
        t_cb = CommonUtils::readVector3d(nh, prefix + "t_cb");
        q_bc = q_cb.inverse().normalized();
        t_bc = q_cb.inverse() * (- t_cb);        

        if_compressed = CommonUtils::readParam<bool>(nh, prefix + "if_compressed", false);
        time_offset = CommonUtils::readParam<double>(nh, prefix + "time_offset");
        if_camera_info = CommonUtils::readParam<bool>(nh, prefix + "if_camera_info");
        scale = CommonUtils::readParam<double>(nh, prefix + "scale");    
        width = CommonUtils::readParam<int>(nh, prefix + "cam_width");
        width *= scale;
        height = CommonUtils::readParam<int>(nh, prefix + "cam_height");
        height *= scale;
    }

    int64_t correctTimestampNs(int64_t timestamp_ns) const {
        return timestamp_ns - static_cast<int64_t>(time_offset * 1e9);
    }
};

struct BiasNoise {
    Eigen::Vector3d initial_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d initial_gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d drift_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d drift_gyro = Eigen::Vector3d::Zero();
    bool continuous = false;

    static Eigen::Vector3d readDiagonal(const std::vector<double>& values,
                                      const std::string& name) {
        if (values.size() != 3) {
            throw std::invalid_argument(name + " requires exactly three values");
        }
        for (double value : values) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(name + " requires finite nonnegative variances");
            }
        }
        return Eigen::Vector3d(values[0], values[1], values[2]);
    }

    void addProcessNoise(Eigen::MatrixXd& covariance, double dt,
                         bool already_integrated = false) const {
        if (!std::isfinite(dt) || dt < 0.0) {
            throw std::invalid_argument("Bias noise interval must be finite and nonnegative");
        }
        if (continuous && already_integrated) return;
        const double scale = continuous ? dt : dt * dt;
        covariance.block<3, 3>(24, 24).diagonal() += scale * drift_acc;
        covariance.block<3, 3>(27, 27).diagonal() += scale * drift_gyro;
    }
};

struct Parameters {
    Eigen::Vector3d cov_acc;
    double acc_huber_delta = 0.0;
    Eigen::Vector3d cov_gyro;
    BiasNoise bias_noise;
    Eigen::Vector3d gravity;
    double nn_thresh;
    double coeff_cov;
    double lidar_cov_floor = 0.001;
    int num_imu_upd = 3;
    int64_t max_dt_ns = 1e7;
    Eigen::Vector3d pred_acc_cov;
    Eigen::Vector3d pred_gyro_cov;
    double img_point_cov = 1.0;
    double img_huber_loss = 1.0;
    double thresh_p;
    double thresh_del;
    double thresh_dv = 0.0;
    double dynamic_min_dt_ms = 10.0;
    bool if_kin;
    double cov_kin_vel;
    int num_points_upd;
    Eigen::Quaterniond q_br;
    Eigen::Vector3d t_br;
    bool if_visual;
    double cov_kin_pt2pl = 1.0;

    Parameters() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct OptParameters {
    double w_lc = 1.0;
    double w_odom = 1.0;
    bool if_loss_func = true;
    double loss_func_thresh = 3.0;

    OptParameters() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MapPointData {
private:
    struct ConstructWithCovariance {};
public:
    int64_t time_ns;
    pcl::PointXYZINormal pt;
    Eigen::Matrix3d body_var;
    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;

    MapPointData(const pcl::PointXYZINormal& point, int64_t frame_ns,
        const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation,
        const CommonUtils::BodyCovarianceNoise& noise)
        : MapPointData(ConstructWithCovariance{}, point, frame_ns, rotation, translation, [&noise](auto& p, auto& covariance) {
            CommonUtils::calcBodyCov(p, noise, covariance);
        }) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    template <class CovarianceFunction>
    MapPointData(ConstructWithCovariance, const pcl::PointXYZINormal& point, int64_t frame_ns,
        const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation,
        CovarianceFunction&& compute_covariance)
        : time_ns(frame_ns + CommonUtils::ms2ns(point.intensity)), pt(point),
          q_bl(rotation), t_bl(translation) {
        Eigen::Vector3d body_point(point.x, point.y, point.z);
        compute_covariance(body_point, body_var);
    }
};

struct BufferedLidarPoint {
private:
    struct ConstructWithCovariance {};
public:
    int64_t time_ns;
    pcl::PointXYZINormal pt;
    Eigen::Vector3d pt_b;
    Eigen::Matrix3d body_var;
    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;
    double initial_var;

    BufferedLidarPoint(const pcl::PointXYZINormal& point, int64_t frame_ns,
        const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation,
        double variance, const CommonUtils::BodyCovarianceNoise& noise)
        : BufferedLidarPoint(ConstructWithCovariance{}, point, frame_ns, rotation, translation, variance,
            [&noise](auto& p, auto& covariance) { CommonUtils::calcBodyCov(p, noise, covariance); }) {}

    BufferedLidarPoint(const pcl::PointXYZINormal& point, int64_t frame_ns,
        const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation,
        double variance, const Eigen::Matrix3d& covariance)
        : BufferedLidarPoint(ConstructWithCovariance{}, point, frame_ns, rotation, translation, variance,
            [&covariance](auto& p, auto& output) {
                if (p[2] == 0) p[2] = 0.0001;
                output = covariance;
            }) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    template <class CovarianceFunction>
    BufferedLidarPoint(ConstructWithCovariance, const pcl::PointXYZINormal& point, int64_t frame_ns,
        const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation,
        double variance, CovarianceFunction&& compute_covariance)
        : time_ns(frame_ns + CommonUtils::ms2ns(point.intensity)), pt(point),
          pt_b(point.x, point.y, point.z), q_bl(rotation), t_bl(translation), initial_var(variance) {
        compute_covariance(pt_b, body_var);
    }
};

struct PointData {
private:
    const BufferedLidarPoint* input_ = nullptr;
public:
    pcl::PointXYZINormal pt_w;
    Eigen::Matrix3d w_var;
    Eigen::Vector3d normvec;
    bool if_valid = false;
    Eigen::Vector3d center_;
    Eigen::Matrix<double, 6, 6> plane_var_;
    double dist = 0;
    double zp = 0;
    Eigen::Matrix<double, 1, 24> H;
    double var_pt = 0;

    PointData() = default;
    explicit PointData(const BufferedLidarPoint& source)
        : input_(&source), w_var(Eigen::Matrix3d::Zero()), var_pt(source.initial_var) {}
    const BufferedLidarPoint& input() const { return *input_; }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using PendingLidarBuffer = resple2::RetainedPointBuffer<BufferedLidarPoint>;
using ActiveLidarBuffer = resple2::ActivePointBuffer<BufferedLidarPoint, PointData>;

struct LidarData {
    Eigen::aligned_deque<Eigen::aligned_vector<pcl::PointXYZINormal>> pc_buff;
    std::deque<int64_t> t_buff;
    std::mutex mtx_pc;
    PendingLidarBuffer pt_buff;
};

struct MeasData {
    Eigen::aligned_deque<ImuData> imu_buff;
    Eigen::aligned_deque<ImuData> imu_meas;
    Eigen::aligned_deque<ImuData> imu_pred;
    Eigen::aligned_vector<sensor_msgs::msg::Imu::SharedPtr> imu_int_buff;   

    std::map<std::string, LidarData> lidars_data;  
    ActiveLidarBuffer pt_meas;

    Eigen::aligned_deque<ImgData> img_int_buff;  
    Eigen::aligned_deque<ImgData> img_buff; 
    Eigen::aligned_deque<ImgData> img_meas; 

    Eigen::aligned_deque<KinImuData> kin_imu_buff;
    Eigen::aligned_deque<KinImuData> kin_imu_int_buff;
    Eigen::aligned_deque<KinImuData> kin_meas;  

    std::mutex m_buff;
    std::mutex m_imu_buff;

    void removeOldMeas(int64_t t_thresh) {
        while (!imu_meas.empty() && imu_meas.front().time_ns < t_thresh) {
            imu_meas.pop_front();
        }
        while (!pt_meas.empty() && pt_meas.front().input().time_ns < t_thresh) {
            pt_meas.pop_front();
        }         
        while (!img_meas.empty() && img_meas.front().time_ns < t_thresh) {
            img_meas.pop_front();
        }         
        while (!kin_meas.empty() && kin_meas.front().time_ns < t_thresh) {
            kin_meas.pop_front();
        }                         
    }        
};

struct IntState {
    int64_t tk;
    Eigen::Quaterniond qk;
    Eigen::Vector3d pk;
    Eigen::Vector3d vk;   
    Eigen::Vector3d delk;  

    IntState() {};

    void init(const int64_t s, const Eigen::Quaterniond& q, const Eigen::Vector3d& p, 
        const Eigen::Vector3d& v, const Eigen::Vector3d& del) {
        tk = s;
        qk = q;
        pk = p;
        vk = v;
        delk = del;
    }
};

class ImuPredictionPreview {
public:
    struct Pose {
        int64_t time_ns;
        Eigen::Quaterniond q;
        Eigen::Vector3d p, v;
    };

    ImuPredictionPreview(const Pose& pose, const Eigen::Vector3d& acc,
        const Eigen::Vector3d& gyro, bool has_sample, const Eigen::Vector3d& ba,
        const Eigen::Vector3d& bg, const Eigen::Vector3d& gravity)
        : base_(pose), acc_(acc), gyro_(gyro), ba_(ba), bg_(bg), gravity_(gravity),
          has_sample_(has_sample) {
        if (has_sample_) refreshInterval();
    }

    bool hasSample() const { return has_sample_; }
    int64_t timeNs() const { return base_.time_ns; }
    void seed(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
        acc_ = acc; gyro_ = gyro; has_sample_ = true;
        refreshInterval();
    }

    Pose at(int64_t time_ns) const {
        if (time_ns < base_.time_ns) throw std::logic_error("Preview queried before its IMU boundary");
        if (time_ns == base_.time_ns) return base_;
        if (!has_sample_) throw std::logic_error("IMU preview has no held sample");
        const double dt = (time_ns - base_.time_ns) * 1e-9;
        Pose result = base_;
        result.time_ns = time_ns;
        result.p += dt * base_.v + 0.5 * dt * dt * acceleration_;
        result.v += dt * acceleration_;
        const double angle = angular_speed_ * dt;
        if (angle > 0.0) result.q = (base_.q * Eigen::Quaterniond(Eigen::AngleAxisd(angle, angular_axis_))).normalized();
        return result;
    }

    void pushSample(int64_t time_ns, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
        if (time_ns < base_.time_ns) return;
        if (!has_sample_) seed(acc, gyro);
        base_ = at(time_ns);
        seed(acc, gyro);
    }

private:
    void refreshInterval() {
        acceleration_ = base_.q * (acc_ - ba_) - gravity_;
        const Eigen::Vector3d w = gyro_ - bg_;
        angular_speed_ = w.norm();
        if (angular_speed_ > 0.0) angular_axis_ = w.normalized();
    }

    Pose base_;
    Eigen::Vector3d acc_, gyro_, ba_, bg_, gravity_;
    bool has_sample_;
    Eigen::Vector3d acceleration_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_axis_ = Eigen::Vector3d::Zero();
    double angular_speed_ = 0.0;
};

class ImgOpr
{
public:
    static float shiTomasiScore(const cv::Mat& img, int u, int v)
    {

        constexpr int halfbox_size = 4;
        constexpr int box_size     = 2 * halfbox_size;
        constexpr float inv        = 1.0f / (2.0f * box_size * box_size);

        const int x_min = u - halfbox_size, x_max = u + halfbox_size;
        const int y_min = v - halfbox_size, y_max = v + halfbox_size;

        if (x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
            return 0.0f;

        const int stride = img.step.p[0];
        int32_t iXX = 0, iYY = 0, iXY = 0;

#ifdef __SSE2__
        const __m128i zero = _mm_setzero_si128();
        __m128i acc_xx = _mm_setzero_si128();
        __m128i acc_yy = _mm_setzero_si128();
        __m128i acc_xy = _mm_setzero_si128();

        for (int y = y_min; y < y_max; ++y) {
            const uint8_t* row = img.data + stride * y + x_min;
            __m128i right = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(row + 1)), zero);
            __m128i left  = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(row - 1)), zero);
            __m128i top   = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(img.data + stride*(y-1) + x_min)), zero);
            __m128i bot   = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(img.data + stride*(y+1) + x_min)), zero);
            __m128i dx    = _mm_sub_epi16(right, left);
            __m128i dy    = _mm_sub_epi16(bot,   top);
            acc_xx = _mm_add_epi32(acc_xx, _mm_madd_epi16(dx, dx));
            acc_yy = _mm_add_epi32(acc_yy, _mm_madd_epi16(dy, dy));
            acc_xy = _mm_add_epi32(acc_xy, _mm_madd_epi16(dx, dy));
        }
        auto hsum = [](const __m128i v) -> int32_t {
            __m128i t = _mm_add_epi32(v, _mm_srli_si128(v, 8));
            t = _mm_add_epi32(t, _mm_srli_si128(t, 4));
            return _mm_cvtsi128_si32(t);
        };
        iXX = hsum(acc_xx);
        iYY = hsum(acc_yy);
        iXY = hsum(acc_xy);
#else        
        for (int y = y_min; y < y_max; ++y) {
            const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
            const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
            const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
            const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
            for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
                const int dx = *ptr_right - *ptr_left;
                const int dy = *ptr_bottom - *ptr_top;
                iXX += dx * dx;
                iYY += dy * dy;
                iXY += dx * dy;
            }
        }
#endif        

        const float dXX   = iXX * inv, dYY = iYY * inv, dXY = iXY * inv;
        const float trace = dXX + dYY;
        const float det   = dXX * dYY - dXY * dXY;
        return 0.5f * (trace - sqrtf(trace * trace - 4.0f * det));
    }

};
