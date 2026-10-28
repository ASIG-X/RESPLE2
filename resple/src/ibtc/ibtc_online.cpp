#include "ibtc/ibtc.h"
#include "utils/common_utils.h"
#include "camera/ImgPreprocessing.h"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <estimate_msgs/msg/loop_closure.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/empty.hpp>
#ifdef ROS_DISTRO_HUMBLE
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <unordered_set>
#include <opencv2/features2d.hpp>
#include <thread>
#include <chrono>
#include <stdexcept>

struct ImgEntry { int64_t stamp_ns; cv::Mat img; };

std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> lidar_buf;
std::deque<ImgEntry> img_buf;
std::map<std::string, nav_msgs::msg::Odometry::SharedPtr> odom_buf;
std::deque<nav_msgs::msg::Odometry::SharedPtr> odom_buf2;
std::mutex mtx_buffer;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom2_repub;
int scan_count = 0;
double last_lidar_msgtime = 0;
ImgPreprocessing* img_preprocess;
vk::AbstractCamera *cam = nullptr;
CameraConfig camera_data;
bool if_finished = false;

void finalizeService(const std::shared_ptr<rmw_request_id_t>, const std::shared_ptr<std_srvs::srv::Empty::Request>,
                    const std::shared_ptr<std_srvs::srv::Empty::Response>)
{
    if_finished = true;
}

void pointCloudCallBack(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    mtx_buffer.lock();
    scan_count++;
    if (rclcpp::Time(msg->header.stamp).seconds() < last_lidar_msgtime)
    {
        lidar_buf.clear();
    }
    lidar_buf.push_back(msg);
    last_lidar_msgtime = rclcpp::Time(msg->header.stamp).seconds();
    mtx_buffer.unlock();
}

void odomCallBack(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    std::string pose_time = std::to_string(rclcpp::Time(msg->header.stamp).seconds());
    std::string pose_t_str;
    if (pose_time.length() >= 15) pose_t_str = pose_time.erase(15);
    else  pose_t_str = pose_time;
    odom_buf[pose_t_str] = msg;
}

void odom2CallBack(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    odom_buf2.push_back(msg);
}

void imgCallBack(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    static int64_t last_img_ns = 0;
    int64_t t_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
    if (t_ns - last_img_ns > 3e7) {
        cv::Mat img_get = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image.clone();
        if (camera_data.if_vignette_correct || camera_data.if_undist)
            img_preprocess->preprocess(img_get);
        if (img_get.cols != cam->width() || img_get.rows != cam->height())
            cv::resize(img_get, img_get, cv::Size(cam->width(), cam->height()), 0, 0, cv::INTER_AREA);
        img_buf.push_back({t_ns, img_get});
        last_img_ns = t_ns;
    }
}

void imgCallBackRaw(const sensor_msgs::msg::Image::SharedPtr msg)
{
    static int64_t last_img_ns = 0;
    int64_t t_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
    if (t_ns - last_img_ns > 3e7) {
        cv::Mat img_get = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image.clone();
        if (camera_data.if_vignette_correct || camera_data.if_undist)
            img_preprocess->preprocess(img_get);
        if (img_get.cols != cam->width() || img_get.rows != cam->height())
            cv::resize(img_get, img_get, cv::Size(cam->width(), cam->height()), 0, 0, cv::INTER_AREA);
        img_buf.push_back({t_ns, img_get});
        last_img_ns = t_ns;
    }
}

void pub_keyframe_ids(std::vector<pcl::PointXYZI> &id_vec,
                      const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr &pub_handle,
                      const Eigen::Vector3d &pos, const int id, const std::string frame_id)
{
    visualization_msgs::msg::MarkerArray text_msg;
    visualization_msgs::msg::Marker a_text;
    a_text.header.frame_id = frame_id;
    a_text.ns = "pose_ids";
    a_text.action = visualization_msgs::msg::Marker::ADD;
    a_text.pose.orientation.w = 1.0;
    a_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    a_text.scale.z = 0.4;
    pcl::PointXYZI apt;
    apt.x = pos[0], apt.y = pos[1], apt.z = pos[2], apt.intensity = id;
    id_vec.push_back(apt);

    for (int i = 0; i < (int)id_vec.size(); i++){
        a_text.id = i;
        a_text.color.g = 1.0f;
        a_text.color.r = 1.0f;
        a_text.color.a = 1.0f;
        geometry_msgs::msg::Pose pose;
        pose.position.x = id_vec[i].x;
        pose.position.y = id_vec[i].y;
        pose.position.z = id_vec[i].z;
        a_text.pose = pose;
        a_text.text = std::to_string(int(id_vec[i].intensity));
        text_msg.markers.push_back(a_text);
    }

    pub_handle->publish(text_msg);
}

void pub_loop_links(const std::vector<Eigen::Vector3d> &posi_vec,
                    const std::vector<std::tuple<int,int,double>> &loop_pairs,
                    const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr &pub_handle,                    const std::string &frame_id)
{
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_all);

    auto make_line_marker = [&](int id, float r, float g, float b, const std::string &ns)
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame_id;
        m.ns = ns;
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.id = id;
        m.scale.x = 0.1;
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0f;
        m.pose.orientation.w = 1.0;
        return m;
    };
    auto line_blue    = make_line_marker(0, 0.0f, 0.0f, 1.0f, "transform");
    auto line_magenta = make_line_marker(1, 1.0f, 0.0f, 1.0f, "loop_links");

    for (const auto &lp : loop_pairs)
    {
        int cur_id      = std::get<0>(lp);
        int loop_id     = std::get<1>(lp);
        double t_norm   = std::get<2>(lp);
        if (cur_id  >= (int)posi_vec.size() ||
            loop_id >= (int)posi_vec.size()) continue;

        Eigen::Vector3d dir = posi_vec[loop_id] - posi_vec[cur_id];
        double dist = dir.norm();
        if (dist > 1e-6) dir /= dist;

        Eigen::Vector3d p_end_tnorm = posi_vec[cur_id] + dir * t_norm;
        Eigen::Vector3d p_end_loop  = posi_vec[loop_id];

        geometry_msgs::msg::Point p_cur, p_blue, p_magenta;
        p_cur.x     = posi_vec[cur_id][0]; p_cur.y     = posi_vec[cur_id][1]; p_cur.z     = posi_vec[cur_id][2];
        p_blue.x    = p_end_tnorm[0];      p_blue.y    = p_end_tnorm[1];      p_blue.z    = p_end_tnorm[2];
        p_magenta.x = p_end_loop[0];       p_magenta.y = p_end_loop[1];       p_magenta.z = p_end_loop[2];

        line_blue.points.push_back(p_cur);    line_blue.points.push_back(p_blue);
        line_magenta.points.push_back(p_cur); line_magenta.points.push_back(p_magenta);
    }

    marker_array.markers.push_back(line_blue);
    marker_array.markers.push_back(line_magenta);
    pub_handle->publish(marker_array);
}

void pyrDown_multiple(const cv::Mat& img_cur, cv::Mat& img_ds, int layer_num)
{
    assert(layer_num <= 8);
    cv::Mat img_tmp, img_tmp2, img_tmp3, img_tmp4;
    cv::Mat img_tmp5,img_tmp6, img_tmp7, img_tmp8;

    if (layer_num == 0) { img_ds = img_cur; }
    cv::pyrDown(img_cur, img_tmp);
    if (layer_num == 1) { img_ds = img_tmp; return; }
    cv::pyrDown(img_tmp, img_tmp2);
    if (layer_num == 2) { img_ds = img_tmp2; return; }
    cv::pyrDown(img_tmp2, img_tmp3);
    if (layer_num == 3) { img_ds = img_tmp3; return; }
    cv::pyrDown(img_tmp3, img_tmp4);
    if (layer_num == 4) { img_ds = img_tmp4; return; }
    cv::pyrDown(img_tmp4, img_tmp5);
    if (layer_num == 5) { img_ds = img_tmp5; return; }
    cv::pyrDown(img_tmp5, img_tmp6);
    if (layer_num == 6) { img_ds = img_tmp6; return; }
    cv::pyrDown(img_tmp6, img_tmp7);
    if (layer_num == 7) { img_ds = img_tmp7; return; }
    cv::pyrDown(img_tmp7, img_tmp8);
    if (layer_num == 8) { img_ds = img_tmp8; return; }
}

void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
     std::string::size_type pos1, pos2;
     pos2 = s.find(c);
     pos1 = 0;
     while(std::string::npos != pos2)
     {
         v.push_back(s.substr(pos1, pos2-pos1));
         pos1 = pos2 + c.size();
         pos2 = s.find(c, pos1);
     }
     if(pos1 != s.length())
         v.push_back(s.substr(pos1));
}

cv::Mat extract_image_from_msg(const sensor_msgs::msg::CompressedImage::SharedPtr &img_ptr, const int &pre_ds_image_level)
{
    cv::Mat temp_img = cv_bridge::toCvCopy(img_ptr, sensor_msgs::image_encodings::BGR8)->image.clone();
    cv::Mat img_tmp;
    pyrDown_multiple(temp_img, img_tmp, pre_ds_image_level);
    return img_tmp;
}

cv::Mat set_used_region_mask(cv::Mat &img_gray, const int &IMG_HEI, const int &IMG_WID)
{
    cv::Mat mask(IMG_HEI, IMG_WID, CV_8UC1);
    int check_range = 20;
    for (int r_cen = 0; r_cen < IMG_HEI; r_cen++)
    {
        for (int c_cen = 0; c_cen < IMG_WID; c_cen++)
        {
            bool ok_flag = true;
            for(int r = r_cen - check_range ; r < r_cen + check_range; r++)
            {
                for(int c = c_cen - check_range ; c < c_cen + check_range; c++)
                {
                    if (r >= 0 && r < IMG_HEI && c >= 0 && c < IMG_WID)
                    {
                        if (img_gray.at<uchar>(r,c) < 3)
                        {
                            ok_flag = false;
                            goto break_out_tmp;
                        }
                    }
                }
            }
            break_out_tmp:
            if (ok_flag)
                mask.at<uchar>(r_cen,c_cen) = 255;
            else
                mask.at<uchar>(r_cen,c_cen) = 0;
        }
    }
    return mask;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("ibtc_online");

    double icp_threshold = 0.5;

    int layer_num = 2;
    int pre_ds_image_level = 0;
    int set_used_region = 0;
    double thresh_tf_dist = 0;

    auto cam_names = node->declare_parameter<std::vector<std::string>>("cameras", std::vector<std::string>());
    if (cam_names.empty()) throw std::invalid_argument("IBTC requires at least one configured camera");
    camera_data.loadParameters(node, cam_names[0] + ".");  
    img_preprocess = new ImgPreprocessing(node, cam_names[0] + "."); 
    int IMG_WID = camera_data.width;
    int IMG_HEI = camera_data.height;
    Eigen::Matrix3d rectification = Eigen::Matrix3d::Identity();
    if (camera_data.if_camera_info) {
        sensor_msgs::msg::CameraInfo::SharedPtr cam_info_msg;
        auto sub_caminfo = node->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_data.topic_info, 1,
            [&](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                if (cam_info_msg) return;
                if (img_preprocess->cam_model == "PolynomialCamera")
                    throw std::invalid_argument("PolynomialCamera requires YAML calibration (if_camera_info: false)");
                if ((!msg->distortion_model.empty()) &&
                    ((img_preprocess->cam_model == "EquidistantCamera" && msg->distortion_model != "equidistant") ||
                     (img_preprocess->cam_model == "Pinhole" && msg->distortion_model != "plumb_bob" && msg->distortion_model != "rational_polynomial")))
                    throw std::invalid_argument("CameraInfo distortion_model does not match cam_model");
                img_preprocess->width = msg->width;
                img_preprocess->height = msg->height;
                cv::Mat K(3, 3, CV_64F, (void*)msg->k.data());
                cv::Mat D(msg->d.size(), 1, CV_64F, (void*)msg->d.data());
                cv::Mat R(3, 3, CV_64F, (void*)msg->r.data());
                if (camera_data.if_undist) {
                    for (int row = 0; row < 3; ++row)
                        for (int col = 0; col < 3; ++col) rectification(row, col) = msg->r[3*row+col];
                }
                img_preprocess->init(K, D, R, cam);
                cam_info_msg = msg;
            });
        while (rclcpp::ok() && !cam_info_msg) rclcpp::spin_some(node);
        if (!cam_info_msg) return 0;
    } else {
        img_preprocess->init(node, cam_names[0] + ".", cam);
    }
    if (!cam) throw std::invalid_argument("Unsupported IBTC camera model");
    IMG_WID = cam->width();
    IMG_HEI = cam->height();
    pre_ds_image_level = CommonUtils::readParam<int>        (node, "pre_ds_image_level", pre_ds_image_level);
    set_used_region    = CommonUtils::readParam<int>        (node, "set_used_region",    set_used_region);
    thresh_tf_dist     = CommonUtils::readParam<double>     (node, "thresh_tf_dist_ibtc",     (double)thresh_tf_dist);

    if (pre_ds_image_level < 0 || pre_ds_image_level > 8)
        throw std::invalid_argument("pre_ds_image_level must be between 0 and 8");
    const double pre_ds_scalar = std::ldexp(1.0, pre_ds_image_level);
    const IbtcCameraProjection projection(*cam, pre_ds_scalar, rectification);
    for (int level = 0; level < pre_ds_image_level; ++level) {
        IMG_WID = (IMG_WID + 1) / 2;
        IMG_HEI = (IMG_HEI + 1) / 2;
    }

    layer_num = floor(double(IMG_WID+IMG_HEI)*0.5/500.0f);

    auto pubOdomAftMapped  = node->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init_keypose", 10);
    auto pubRegisterCloud  = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_keycloud", 100);
    auto pubCurrentBinary  = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_key_points", 100);
    auto pubMatchedCloud   = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_matched", 100);
    auto pubMatchedBinary  = node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_matched_key_points", 100);
    auto pubSTD            = node->create_publisher<visualization_msgs::msg::MarkerArray>("descriptor_line", 10);
    auto pub_raw_img       = node->create_publisher<sensor_msgs::msg::Image>("/raw_in_img", 1000);
    auto pub_dep_img       = node->create_publisher<sensor_msgs::msg::Image>("/depth_img", 1000);
    auto pubCorrectionId   = node->create_publisher<visualization_msgs::msg::MarkerArray>("/ids_pose", 10);
    auto pubLoopLinks      = node->create_publisher<visualization_msgs::msg::MarkerArray>("/loop_links", 10);

    std::map<std::string, imu_lidar_camera> sorter;

    ConfigSetting config_setting;
    load_config_setting(node, config_setting);
    icp_threshold = config_setting.icp_threshold_;

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> key_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> history_plane_list;
    std::vector<std::vector<BinaryDescriptor>> history_binary_list;
    std::unordered_map<STD_LOC, std::vector<STD>> STD_map, STD_map_brf;
    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<pcl::PointXYZI> id_vec;

    int max_submap_img_num = CommonUtils::readParam<int>(node, "max_submap_img_num", 30);
    if (max_submap_img_num <= 0 || config_setting.sub_frame_num_ <= 0)
        throw std::invalid_argument("max_submap_img_num and sub_frame_num must be positive");
    const double camera_wait_timeout_s = CommonUtils::readParam<double>(node, "ibtc_camera_wait_timeout_s", 1.0);
    if (!std::isfinite(camera_wait_timeout_s) || camera_wait_timeout_s <= 0.0)
        throw std::invalid_argument("ibtc_camera_wait_timeout_s must be finite and positive");
    auto descriptor_wait_start = std::chrono::steady_clock::now();
    std::vector<cv::Mat> image_list(max_submap_img_num);
    std::vector<cv::Mat> image_gray_list(max_submap_img_num), image_gray_ds_list(max_submap_img_num);
    std::vector<double> image_times(max_submap_img_num);
    std::vector<std::vector<kpt_2d_info>> kpts_from_img_list(max_submap_img_num);
    std::vector<std::vector<std::vector<__m128i>>> brief_arrays_mm128_list(max_submap_img_num);
    std::vector<std::vector<cv::Point2f>> corners_tmp_list(max_submap_img_num);
    std::vector<std::vector<cv::Point2f>> corners_out_list(max_submap_img_num);
    std::vector<std::vector<float>> harrisR_list(max_submap_img_num);
    std::vector<cv::Mat> eigtmp_list(max_submap_img_num);
    std::vector<cv::Mat> eigmat_list, Dx_list, Dy_list, cov_list, descriptors_list, sum_list;
    cv::Mat mask_resize;

    std::vector<int*> depthf_mask_list(max_submap_img_num);
    std::vector<std::unique_ptr<int[]>> depth_arrs(max_submap_img_num);
    for (int si = 0; si < max_submap_img_num; si++)
    {
        depth_arrs[si] = std::make_unique<int[]>(IMG_WID * IMG_HEI);
        memset(depth_arrs[si].get(), -1, sizeof(int) * IMG_WID * IMG_HEI);
        depthf_mask_list[si] = depth_arrs[si].get();
    }

    for (int si = 0; si < max_submap_img_num; si++)
    {
        cv::Mat descriptors(10000, 64, CV_8U);
        descriptors_list.push_back(descriptors);
    }

    std::vector<std::vector<__m128i>> all_briefs;
    all_briefs.reserve(1e6);

    std::vector<std::tuple<int, int, float>> search_row_col;  int search_range = 10;
    for (int r = -search_range; r < search_range; r++)
        for (int c = -search_range; c < search_range; c++)
        {
            if (r == 0 && c == 0) continue;
            float pixeldis = sqrt((float)(r*r + c*c));
            search_row_col.push_back({r, c, pixeldis});
        }
    auto search_uv_sort = [](std::tuple<int, int, float> &a, std::tuple<int, int, float> &b) -> bool
    { return std::get<2>(a) < std::get<2>(b); };
    std::sort(search_row_col.begin(), search_row_col.end(), search_uv_sort);

    std::vector<std::vector<STD>> STD_list_vec;
    std::vector<STD> STD_list_last;

    bool is_build_descriptor = false;
    int key_frame_id = 0;

    Eigen::Vector3d init_translation;
    int count = 0;
    Eigen::Vector3d posi_now;
    Eigen::Vector3d pose_w2c_t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond pose_w2c_q = Eigen::Quaterniond::Identity();
    int64_t pose_t_ns = 0;
    std::vector<Eigen::Vector3d> posi_vec;
    std::vector<Eigen::Quaterniond> orti_vec;
    std::vector<int64_t> time_vec;
    std::vector<std::tuple<int,int,double>> loop_pairs;
    int is_init_starttime = false;
    int img_store_counter = 0;
    int64_t last_lidar_t_ns = 0;

#define sub_registred_cloud
#ifdef sub_registred_cloud
    auto sub_cloud = node->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/current_scan", 1000, pointCloudCallBack);
#else
    auto sub_cloud = node->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/cloud_undistort", 1000, pointCloudCallBack);
    auto sub_imuodom = node->create_subscription<nav_msgs::msg::Odometry>(
        "/aft_mapped_to_init", 1000, odomCallBack);
#endif
    rclcpp::SubscriptionBase::SharedPtr sub_img;
    if (camera_data.if_compressed)
        sub_img = node->create_subscription<sensor_msgs::msg::CompressedImage>(
            camera_data.topic, 1000, imgCallBack);
    else
        sub_img = node->create_subscription<sensor_msgs::msg::Image>(
            camera_data.topic, 1000, imgCallBackRaw);
    pub_odom2_repub = node->create_publisher<nav_msgs::msg::Odometry>("/camera/poses_inworld_camera_init", 1000);
    auto sub_camodom = node->create_subscription<nav_msgs::msg::Odometry>(
        "/camera/poses_inworld", 1000, odom2CallBack);
    auto lc_tranform_pub = node->create_publisher<estimate_msgs::msg::LoopClosure>(
        "/loop_closure_tranformation", 1);
    auto srv_finalize = node->create_service<std_srvs::srv::Empty>("/if_finish", finalizeService);

    rclcpp::Rate loop(1000);
    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);
        loop.sleep();
        if (if_finished && lidar_buf.empty() && !is_build_descriptor) {
            if (count == 0) break;
            count = 0;
            is_build_descriptor = true;
            descriptor_wait_start = std::chrono::steady_clock::now();
        }
        if (!lidar_buf.empty() && !is_build_descriptor)
        {
            sensor_msgs::msg::PointCloud2::SharedPtr cloud_ptr = lidar_buf.front();
            std::string pose_t_str;
            std::string pose_time = std::to_string(rclcpp::Time(cloud_ptr->header.stamp).seconds());
            if (pose_time.length() >= 15) pose_t_str = pose_time.erase(15);
            else  pose_t_str = pose_time;
            last_lidar_t_ns = rclcpp::Time(cloud_ptr->header.stamp).nanoseconds();
#ifndef sub_registred_cloud
            if (odom_buf.empty())   continue;
            auto odom_buf_iter = odom_buf.find(pose_t_str);
            if (odom_buf_iter == odom_buf.end())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                std::this_thread::yield();
                continue;
            }
            auto odom_msg = odom_buf_iter->second;
            auto w_q_l = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                                            odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
            auto w_t_l = Eigen::Vector3d(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y,
                                         odom_msg->pose.pose.position.z);
            sorter[pose_t_str].m_lidar_q = w_q_l;
            sorter[pose_t_str].m_lidar_t = w_t_l;

            Eigen::Vector3d translation = w_t_l;
            if (isnan(translation[0]) || isnan(translation[1]) || isnan(translation[2]))  continue;
            if (isinf(translation[0]) || isinf(translation[1]) || isinf(translation[2]))  continue;
            posi_now = translation;

            Eigen::Quaterniond q(w_q_l);
            nav_msgs::msg::Odometry odom;
            odom.header.frame_id = "world";
            odom.pose.pose.position.x = translation[0];
            odom.pose.pose.position.y = translation[1];
            odom.pose.pose.position.z = translation[2];
            odom.pose.pose.orientation.w = q.w();
            odom.pose.pose.orientation.x = q.x();
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z();
            pubOdomAftMapped->publish(odom);
            loop.sleep();

            if (!is_init_bag)
            {
                init_translation = translation;
                translation << 0, 0, 0;
                is_init_bag = true;
            }
            else
                translation = translation - init_translation;
#endif

            pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*cloud_ptr, *pcl_cloud);
            pcl::PointCloud<pcl::PointXYZI>::Ptr register_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            for (size_t i = 0; i < pcl_cloud->size(); i++)
            {
#ifndef sub_registred_cloud
                Eigen::Vector3d pv(pcl_cloud->points[i].x, pcl_cloud->points[i].y,
                                   pcl_cloud->points[i].z);
                pv = q * (R_lidar_to_imu * pv + t_lidar_to_imu) + translation;
                pcl::PointXYZI pi = pcl_cloud->points[i];
                pi.x = pv[0]; pi.y = pv[1]; pi.z = pv[2];
                if (isnan(pi.x)||isnan(pi.y)||isnan(pi.z))  continue;
                register_cloud->push_back(pi);
#else
                pcl::PointXYZI pi = pcl_cloud->points[i];
                pi.x = pcl_cloud->points[i].x;
                pi.y = pcl_cloud->points[i].y;
                pi.z = pcl_cloud->points[i].z;
                if (isnan(pi.x)||isnan(pi.y)||isnan(pi.z))  continue;
                register_cloud->push_back(pi);
#endif
            }

            down_sampling_voxel(*register_cloud, config_setting.ds_size_, config_setting.if_custom_ds_);

            if (count == 0)
            {
                pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                key_cloud_list.push_back(temp_cloud);
            }
            for (size_t i = 0; i < register_cloud->size(); i++)
                key_cloud_list.back()->points.push_back(register_cloud->points[i]);

            if (count < config_setting.sub_frame_num_ - 1)
                count++;
            else
            {
                count = 0;
                is_build_descriptor = true;
                descriptor_wait_start = std::chrono::steady_clock::now();
            }
            lidar_buf.pop_front();
        }
        bool if_build = false;
        while (!img_buf.empty() && !odom_buf2.empty()) {
            const auto image_ns = img_buf.front().stamp_ns;
            const auto odom_ns = rclcpp::Time(odom_buf2.front()->header.stamp).nanoseconds();
            if (image_ns == odom_ns) break;
            if (image_ns < odom_ns) img_buf.pop_front();
            else odom_buf2.pop_front();
        }
#ifdef disable_rgb
        if (0)
#else
        if (!img_buf.empty() && !odom_buf2.empty())
#endif
        {
            int64_t img_stamp = img_buf.front().stamp_ns;
            if(img_store_counter < max_submap_img_num && img_stamp <= last_lidar_t_ns) {
                const ImgEntry& img_entry = img_buf.front();
                std::string pose_time = std::to_string(img_entry.stamp_ns * 1e-9);
                std::string pose_t_str;
                if (pose_time.length() >= 15) pose_t_str = pose_time.erase(15);
                else  pose_t_str = pose_time;
                cv::Mat temp_img;
                pyrDown_multiple(img_entry.img, temp_img, pre_ds_image_level);
                image_list[img_store_counter] = temp_img;
                cv::Mat img_gray;
                cv::cvtColor(temp_img, img_gray, cv::COLOR_BGR2GRAY);

                image_gray_list[img_store_counter] = img_gray;
                cv::Mat img_resize;
                if (layer_num > 0)
                    pyrDown_multiple(img_gray, img_resize, layer_num);
                else
                    img_resize = img_gray;
                image_gray_ds_list[img_store_counter] = img_resize;

                image_times[img_store_counter] = img_entry.stamp_ns * 1e-9;
                img_store_counter++;
                if (!is_init_starttime)
                {
                    is_init_starttime = true;
                    for (int si = 0; si < max_submap_img_num; si++)
                    {
                        cv::Mat eig(img_resize.size(), CV_32FC1);
                        eigmat_list.push_back(eig);
                        cv::Mat cov(img_resize.size(), CV_32FC3);
                        cov_list.push_back(cov);
                        cv::Mat Dx(img_resize.size(), CV_32FC1);
                        Dx_list.push_back(Dx);
                        cv::Mat Dy(img_resize.size(), CV_32FC1);
                        Dy_list.push_back(Dy);
                        cv::Mat Sum(img_gray.rows+1, img_gray.cols+1, CV_32S);
                        sum_list.push_back(Sum);
                    }
                    if (set_used_region)
                        mask_resize = set_used_region_mask(img_resize, img_resize.rows, img_resize.cols);
                }

                auto odom_msg2 = odom_buf2.front();
                pose_w2c_q = Eigen::Quaterniond(odom_msg2->pose.pose.orientation.w, odom_msg2->pose.pose.orientation.x,
                                                     odom_msg2->pose.pose.orientation.y, odom_msg2->pose.pose.orientation.z);
                pose_w2c_t = Eigen::Vector3d(odom_msg2->pose.pose.position.x, odom_msg2->pose.pose.position.y,
                                             odom_msg2->pose.pose.position.z);
                pose_t_ns = rclcpp::Time(odom_msg2->header.stamp).nanoseconds();
                auto pose_c2w_q = pose_w2c_q.inverse();
                auto pose_c2w_t = -(pose_c2w_q * pose_w2c_t);
                sorter[pose_t_str].m_pose_c_2_w_q = pose_c2w_q;
                sorter[pose_t_str].m_pose_c_2_w_t = pose_c2w_t;
                sorter[pose_t_str].m_pose_w_2_c_q = pose_w2c_q;
                sorter[pose_t_str].m_pose_w_2_c_t = pose_w2c_t;
                sorter[pose_t_str].camera_poseset = true;

                img_buf.pop_front();
                odom_buf2.pop_front();
            } else if (img_stamp > last_lidar_t_ns) {
                if_build = true;
            } else {
                img_buf.pop_front();
                odom_buf2.pop_front();
            }
        }

        const bool camera_wait_expired = is_build_descriptor &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - descriptor_wait_start).count()
                >= camera_wait_timeout_s;
        if (is_build_descriptor && (camera_wait_expired || (if_finished &&
                (img_buf.empty() || odom_buf2.empty())))) {
            if_build = true;
        }
        if (is_build_descriptor && if_build && img_store_counter == 0) {
            key_cloud_list.pop_back();
            is_build_descriptor = false;
            continue;
        }

        if (is_build_descriptor && if_build)
        {

            int skip_ratio = std::rint(float(img_store_counter) / 10.0f);
            skip_ratio = std::max(1, skip_ratio);
            std::vector<size_t> index;
            for (size_t ii = 0; ii < (size_t)img_store_counter; )
            {
                index.push_back(ii);
                ii += skip_ratio;
            }

            history_plane_list.emplace_back(new pcl::PointCloud<pcl::PointXYZINormal>);
            history_binary_list.emplace_back();
            posi_vec.push_back(pose_w2c_t);
            orti_vec.push_back(pose_w2c_q);
            time_vec.push_back(pose_t_ns);
            std::vector<std::pair<int, float>> proposed_candidates;
            if (key_cloud_list[key_frame_id]->empty())
            {
                key_frame_id++;
                is_build_descriptor = false;
                img_store_counter = 0;
                continue;
            }

            std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
            init_voxel_map(config_setting, *key_cloud_list[key_frame_id], voxel_map);
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr frame_plane_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
            get_plane(voxel_map, frame_plane_cloud);
            history_plane_list[key_frame_id] = frame_plane_cloud;

            std::vector<Plane *> proj_plane_list;
            std::vector<Plane *> merge_plane_list;
            get_project_plane(config_setting, voxel_map, proj_plane_list);
            sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
            if (proj_plane_list.empty())
            {
                key_frame_id++;
                for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
                    delete (iter->second);
                if (!corner_list.empty()) std::vector<std::vector<cv::Point2f>>().swap(corner_list);
                is_build_descriptor = false;
                img_store_counter = 0;
                continue;
            }
            merge_plane(config_setting, proj_plane_list, merge_plane_list);
            sort(merge_plane_list.begin(), merge_plane_list.end(), plane_greater_sort);

            std::vector<BinaryDescriptor> binary_list;
            binary_extractor(config_setting, merge_plane_list, key_cloud_list[key_frame_id], binary_list);

#ifndef disable_rgb
            if (img_store_counter > 0)
#else
            if (0)
#endif
            {
                std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &img_id)
                {
                    cv::Mat &gray_ds_img  = image_gray_ds_list[img_id];
                    cv::Mat &eig          = eigmat_list[img_id];
                    cv::Mat &eigtmp       = eigtmp_list[img_id];
                    cv::Mat &cov          = cov_list[img_id];
                    cv::Mat &Dx           = Dx_list[img_id];
                    cv::Mat &Dy           = Dy_list[img_id];
                    cv::Mat &gray_img     = image_gray_list[img_id];
                    cv::Mat &sum          = sum_list[img_id];
                    std::string img_time_crop = std::to_string(image_times[img_id]).substr(0, 15);

                    if (set_used_region)
                        extract_2dkeypt_roughly(eig, eigtmp, gray_ds_img, cov, Dx, Dy, mask_resize,
                                                corners_tmp_list[img_id], corners_out_list[img_id], harrisR_list[img_id], layer_num);
                    else
                        extract_2dkeypt_roughly(eig, eigtmp, gray_ds_img, cov, Dx, Dy, cv::Mat(),
                                                corners_tmp_list[img_id], corners_out_list[img_id], harrisR_list[img_id], layer_num);

                    find_more_accurate_pixel(gray_img, corners_tmp_list[img_id], corners_out_list[img_id], layer_num);
                    extract_brief_fromimage(corners_out_list[img_id], gray_img, sum, descriptors_list[img_id],
                                            brief_arrays_mm128_list[img_id]);

                    std::vector<kpt_2d_info> kpts_from_img(corners_out_list[img_id].size());
                    for (int i = 0; i < (int)corners_out_list[img_id].size(); i++)
                    {
                        kpt_2d_info a_kpt2d;
                        a_kpt2d.pt_2d = corners_out_list[img_id][i];
                        a_kpt2d.brief_array = brief_arrays_mm128_list[img_id][i];
                        a_kpt2d.Rval = harrisR_list[img_id][i];
                        a_kpt2d.c_q_w = sorter.at(img_time_crop).m_pose_c_2_w_q;
                        a_kpt2d.c_t_w = sorter.at(img_time_crop).m_pose_c_2_w_t;
                        kpts_from_img[i] = a_kpt2d;
                    }
                    kpts_from_img_list[img_id] = kpts_from_img;
                });

                for (auto img_id : index)
                    memset(depthf_mask_list[img_id], -1, sizeof(int) * IMG_WID * IMG_HEI);

                find3dpts_for2dcorners(kpts_from_img_list, depthf_mask_list, key_cloud_list[key_frame_id],
                                       image_times, sorter, index, search_row_col,
                                       IMG_HEI, IMG_WID, projection);

                pcl::PointCloud<pcl::PointXYZINormal>::Ptr cor3d_cloud_ds(new pcl::PointCloud<pcl::PointXYZINormal>);
                std::vector<std::vector<__m128i>> out_brief_list;
                non_maximal_suppression(cor3d_cloud_ds, out_brief_list, kpts_from_img_list, index, config_setting);

                std::vector<Eigen::Vector3d> projected_cor3d;
                std::string img_time = std::to_string(image_times[0]).substr(0, 15);
                project_onimage(projected_cor3d, cor3d_cloud_ds, sorter[img_time].m_pose_c_2_w_q,
                                sorter[img_time].m_pose_c_2_w_t, sorter[img_time].m_pose_w_2_c_t,
                                IMG_HEI, IMG_WID, projection);
                for (int idx = 0; idx < (int)cor3d_cloud_ds->size(); idx++)
                {
                    auto a_cor3d = cor3d_cloud_ds->points[idx];
                    BinaryDescriptor abin;
                    abin.location_ = Eigen::Vector3d(a_cor3d.x, a_cor3d.y, a_cor3d.z);
                    abin.occupy_array_ = std::vector<bool>();
                    abin.summary_ = 0;
                    abin.brf_idx = all_briefs.size();
                    all_briefs.push_back(out_brief_list[idx]);
                    abin.harrisR_ = int(cor3d_cloud_ds->points[idx].intensity);
                    abin.breif_obs_t = image_times[int(cor3d_cloud_ds->points[idx].normal_x)];
                    binary_list.push_back(abin);
                }

                auto tmp_image = image_list[0];
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZINormal>());
                for (auto a_pcor3d : projected_cor3d)
                {
                    pcl::PointXYZINormal pt2d;
                    pt2d.x = a_pcor3d[1];
                    pt2d.y = a_pcor3d[0];
                    pt2d.z = 0;
                    pt2d.intensity = a_pcor3d[2];
                    cloud_2d->push_back(pt2d);
                    cv::circle(tmp_image, cv::Point2f(pt2d.x, pt2d.y), 6, cv::Scalar(0,255,0));
                    cv::putText(tmp_image, std::to_string(int(cor3d_cloud_ds->points[int(pt2d.intensity)].intensity)),
                                cv::Point((int)pt2d.x, (int)pt2d.y), cv::FONT_HERSHEY_DUPLEX, 1.5, CV_RGB(255, 0, 0), 1.5);
                }
                cv_bridge::CvImage tmp_msg;
                tmp_msg.encoding = sensor_msgs::image_encodings::BGR8;
                tmp_msg.image = tmp_image;
                pub_dep_img->publish(*tmp_msg.toImageMsg());
                loop.sleep();
            }

            std::vector<STD> STD_list;
            float max_triangle_side_len = generate_std(config_setting, binary_list, key_frame_id, STD_list);
#ifndef disable_rgb
            auto STD_list_bkq = STD_list;
            if (max_triangle_side_len < 9 || STD_list.size() < 500)
            {
                for (auto &var : STD_list_last)
                {
                    var.frame_number_ = key_frame_id;
                    STD_list.push_back(var);
                }
            }
#endif
            history_binary_list[key_frame_id] = binary_list;

            STD_list_vec.push_back(STD_list);
#ifndef disable_rgb
            STD_list_last = STD_list_bkq;
#endif

            pcl::PointCloud<pcl::PointXYZ> key_points_cloud;
            for (auto var : binary_list)
            {
                pcl::PointXYZ pi;
                pi.x = var.location_[0]; pi.y = var.location_[1]; pi.z = var.location_[2];
                key_points_cloud.push_back(pi);
            }
            sensor_msgs::msg::PointCloud2 pub_cloud;
            pcl::toROSMsg(key_points_cloud, pub_cloud);
            pub_cloud.header.frame_id = "world";
            pubCurrentBinary->publish(pub_cloud);
            loop.sleep();
            Eigen::Vector3d color1(1, 0, 0);
            publish_binary(binary_list, color1, "current", pubSTD);
            loop.sleep();

            sensor_msgs::msg::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*key_cloud_list[key_frame_id], laserCloudmsg);
            laserCloudmsg.header.stamp = node->now();
            laserCloudmsg.header.frame_id = "world";
            pubRegisterCloud->publish(laserCloudmsg);
            pub_keyframe_ids(id_vec, pubCorrectionId, pose_w2c_t, key_frame_id, "world");

            nav_msgs::msg::Odometry pose_msg;
            pose_msg.header.stamp = node->now();
            pose_msg.header.frame_id = "world";
            pose_msg.pose.pose.position.x = pose_w2c_t[0];
            pose_msg.pose.pose.position.y = pose_w2c_t[1];
            pose_msg.pose.pose.position.z = pose_w2c_t[2];
            pose_msg.pose.pose.orientation.w = pose_w2c_q.w();
            pose_msg.pose.pose.orientation.x = pose_w2c_q.x();
            pose_msg.pose.pose.orientation.y = pose_w2c_q.y();
            pose_msg.pose.pose.orientation.z = pose_w2c_q.z();
            pub_odom2_repub->publish(pose_msg);

            std::vector<STDMatchList> alternative_match;
#ifdef disable_rgb
            candidate_searcher(config_setting, STD_map, STD_list, alternative_match);
#else
            rgb_candidate_searcher(config_setting, STD_map, STD_map_brf, STD_list, alternative_match, all_briefs);
#endif

            bool triggle_loop = false;
            Eigen::Vector3d best_t;
            Eigen::Matrix3d best_rot;
            Eigen::Vector3d loop_translation;
            Eigen::Matrix3d loop_rotation;
            std::vector<std::pair<STD, STD>> sucess_match_list;
            std::vector<std::pair<STD, STD>> unsucess_match_list;
            std::vector<std::pair<STD, STD>> sucess_match_list_publish;
            int match_frame = 0;
            double best_score = 0;
            int best_frame = -1;
            bool is_rgb_case_final = false;
            bool btc_match_found = false;

            for (int i = 0; i < (int)alternative_match.size(); i++)
            {
                if (alternative_match[i].match_list_.size() > 0)
                {
                    bool fine_sucess = false;
                    Eigen::Matrix3d std_rot;
                    Eigen::Vector3d std_t;
                    sucess_match_list.clear();
                    bool is_rgb_case = false;
#ifdef disable_rgb
                    fine_loop_detection_tbb(
                        config_setting, alternative_match[i].match_list_, fine_sucess,
                        std_rot, std_t, sucess_match_list, unsucess_match_list);
#else
                    rgb_fine_loop_detection_tbb(
                        config_setting, alternative_match[i].match_list_, fine_sucess,
                        std_rot, std_t, sucess_match_list, unsucess_match_list, is_rgb_case, alternative_match[i].match_frame_);
#endif
                    if (fine_sucess)
                    {
                        double score = geometric_icp(config_setting,
                            frame_plane_cloud,
                            history_plane_list[alternative_match[i].match_frame_], std_rot, std_t);                        

                        proposed_candidates.push_back({alternative_match[i].match_frame_, score});
                        if (score > best_score)
                        {
                            sucess_match_list_publish = sucess_match_list;
                            best_frame = alternative_match[i].match_frame_;
                            best_score = score;
                            best_rot = std_rot;
                            best_t = std_t;
                            if (score > icp_threshold) is_rgb_case_final = is_rgb_case;
                            if (!is_rgb_case && score > icp_threshold) btc_match_found = true;
                        }
                    }
                }
            }

            if (best_score > icp_threshold)
            {
                loop_translation = best_t;
                loop_rotation    = best_rot;
                match_frame      = best_frame;
                triggle_loop     = true;
            }
            else
                triggle_loop = false;

            if (triggle_loop && (posi_vec[key_frame_id] - posi_vec[match_frame]).norm() > thresh_tf_dist) {
                triggle_loop = false;
            }

            is_build_descriptor = false;

            if (triggle_loop)
            {

                pcl::toROSMsg(*key_cloud_list[match_frame], pub_cloud);
                pub_cloud.header.frame_id = "world";
                pubMatchedCloud->publish(pub_cloud);
                loop.sleep();
                pcl::PointCloud<pcl::PointXYZ> matched_key_points_cloud;
                for (auto var : history_binary_list[match_frame])
                {
                    pcl::PointXYZ pi;
                    pi.x = var.location_[0]; pi.y = var.location_[1]; pi.z = var.location_[2];
                    matched_key_points_cloud.push_back(pi);
                }
                pcl::toROSMsg(matched_key_points_cloud, pub_cloud);
                pub_cloud.header.frame_id = "world";
                pubMatchedBinary->publish(pub_cloud);
                Eigen::Vector3d color2(0, 1, 0);
                if (is_rgb_case_final && !btc_match_found)
                    color2 = Eigen::Vector3d(0, 0, 1);
                publish_binary(history_binary_list[match_frame], color2, "history", pubSTD);
                loop.sleep();

                publish_std(sucess_match_list_publish, pubSTD, color2);
                loop.sleep();

                Eigen::Matrix4d T_lq = Eigen::Matrix4d::Identity();
                T_lq.topLeftCorner<3, 3>() = loop_rotation.cast<double>();
                T_lq.topRightCorner<3, 1>() = loop_translation;
                Eigen::Matrix4d T_qi = Eigen::Matrix4d::Identity();
                T_qi.topLeftCorner<3, 3>() = orti_vec[key_frame_id].toRotationMatrix();
                T_qi.topRightCorner<3, 1>() = posi_vec[key_frame_id];
                Eigen::Matrix4d T_li = Eigen::Matrix4d::Identity();
                T_li.topLeftCorner<3, 3>() = orti_vec[match_frame].toRotationMatrix();
                T_li.topRightCorner<3, 1>() = posi_vec[match_frame];           
                Eigen::Matrix4d T_lc_qc = T_li.inverse() * T_lq * T_qi;       

                Eigen::Quaterniond q_opt(T_lc_qc.topLeftCorner<3, 3>().cast<double>());
                estimate_msgs::msg::LoopClosure transform_msg;
                transform_msg.transform.position.x = T_lc_qc(0, 3);
                transform_msg.transform.position.y = T_lc_qc(1, 3);
                transform_msg.transform.position.z = T_lc_qc(2, 3);
                transform_msg.transform.orientation.x = q_opt.x();
                transform_msg.transform.orientation.y = q_opt.y();
                transform_msg.transform.orientation.z = q_opt.z();
                transform_msg.transform.orientation.w = q_opt.w();
                transform_msg.t_ref_ns = time_vec[match_frame];
                transform_msg.t_cur_ns = time_vec[key_frame_id];
                lc_tranform_pub->publish(transform_msg);

                loop_pairs.emplace_back(key_frame_id, match_frame, T_lc_qc.topRightCorner<3, 1>().norm());
                pub_loop_links(posi_vec, loop_pairs, pubLoopLinks, "world");
            }

            add_STD(STD_map, STD_list);
            add_STD_brf(STD_map_brf, STD_list);

            key_frame_id++;
            for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
                delete (iter->second);

            img_store_counter = 0;
            if (!corner_list.empty()) std::vector<std::vector<cv::Point2f>>().swap(corner_list);
        }
    }

    srv_finalize.reset();
    sub_cloud.reset();
    sub_img.reset();
    sub_camodom.reset();
    pub_odom2_repub.reset();
    lc_tranform_pub.reset();
    pubOdomAftMapped.reset();
    pubRegisterCloud.reset();
    pubCurrentBinary.reset();
    pubMatchedCloud.reset();
    pubMatchedBinary.reset();
    pubSTD.reset();
    pub_raw_img.reset();
    pub_dep_img.reset();
    pubCorrectionId.reset();
    pubLoopLinks.reset();
    node.reset();

    delete cam;
    delete img_preprocess;
    rclcpp::shutdown();
    return 0;
}
