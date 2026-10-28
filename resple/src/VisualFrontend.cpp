#include "VisualFrontend.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
bool isInImage(const cv::Point2f& pixel, int rows, int cols)
{
    return std::isfinite(pixel.x) && std::isfinite(pixel.y) &&
           pixel.x >= 0 && pixel.y >= 0 && pixel.x < cols && pixel.y < rows;
}
}

Eigen::Vector2d VisualFrontend::w2c(const Eigen::Vector3d &xyz_w) const {
    Eigen::Vector3d pc = q_f_w_ * xyz_w + t_f_w_;
    return cam->world2cam(pc);
}

Eigen::Vector3d VisualFrontend::w2f(const Eigen::Vector3d &xyz_w) const {
    return q_f_w_ * xyz_w + t_f_w_;
}

void VisualFrontend::updateLastTrackingVectorAndIds()
{
    int idx = 0;
    m_last_tracked_pts.clear();
    m_rgb_pts_ptr_vec_in_last_frame.clear();
    m_old_ids.clear();
    std::vector<std::pair<VisualPointPtr, cv::Point2f>> features(
        m_map_rgb_pts_in_last_frame_pos.begin(), m_map_rgb_pts_in_last_frame_pos.end());
    std::sort(features.begin(), features.end(), [](const auto& a, const auto& b) {
        return a.first->id < b.first->id;
    });
    for (const auto& feature : features)
    {
        m_rgb_pts_ptr_vec_in_last_frame.push_back(feature.first);
        m_last_tracked_pts.push_back(feature.second);
        m_old_ids.push_back( idx );
        idx++;
    }
}

void VisualFrontend::setMask()
{
    if (tracking_rows_ <= 0 || tracking_cols_ <= 0 || MIN_DIST < 0) {
        throw std::invalid_argument("Rgbmap_tracker: invalid mask dimensions or minimum distance");
    }
    cv::Mat mask = cv::Mat(tracking_rows_, tracking_cols_, CV_8UC1, cv::Scalar(255));

    std::vector<std::pair<VisualPointPtr, cv::Point2f>> features;
    for (const auto& it : m_map_rgb_pts_in_last_frame_pos) {
        if (it.first && it.first->if_valid && isInImage(it.second, tracking_rows_, tracking_cols_)) {
            features.push_back(it);
        }
    }
    std::sort(features.begin(), features.end(), [](const auto& a, const auto& b) {
        if (a.first->num_track != b.first->num_track)
            return a.first->num_track > b.first->num_track;
        return a.first->id < b.first->id;
    });
    m_map_rgb_pts_in_last_frame_pos.clear();
    for (const auto& it : features) {
        if ((int)m_map_rgb_pts_in_last_frame_pos.size() >= MAX_CNT) break;
        const cv::Point pixel = it.second;
        if (pixel.x < 0 || pixel.y < 0 || pixel.x >= tracking_cols_ || pixel.y >= tracking_rows_) continue;
        if (mask.at<uchar>(pixel) == 255) {
            m_map_rgb_pts_in_last_frame_pos[it.first] = it.second;
            cv::circle(mask, it.second, MIN_DIST, cv::Scalar(0), -1);
        }
    }
}
void VisualFrontend::applyTrackingResults()
{
    const size_t count = m_last_tracked_pts.size();
    if (m_current_tracked_pts.size() != count || m_old_ids.size() != count) {
        throw std::invalid_argument("Rgbmap_tracker: tracking coordinate and ID counts differ");
    }
    if (tracking_rows_ <= 0 || tracking_cols_ <= 0) {
        throw std::invalid_argument("Rgbmap_tracker: image dimensions must be positive");
    }
    for (const int id : m_old_ids) {
        if (id < 0 || static_cast<size_t>(id) >= m_rgb_pts_ptr_vec_in_last_frame.size()) {
            throw std::out_of_range("Rgbmap_tracker: tracking ID is outside the previous point vector");
        }
    }
    m_map_rgb_pts_in_last_frame_pos.clear();
    for (size_t i = 0; i < count; i++)
    {
        const auto& rgb_pts_ptr = m_rgb_pts_ptr_vec_in_last_frame[m_old_ids[i]];
        if (!rgb_pts_ptr || !rgb_pts_ptr->if_valid ||
            !isInImage(m_last_tracked_pts[i], tracking_rows_, tracking_cols_) ||
            !isInImage(m_current_tracked_pts[i], tracking_rows_, tracking_cols_)) {
            continue;
        }
        rgb_pts_ptr->num_track++;
        rgb_pts_ptr->type_track = 1;
        m_map_rgb_pts_in_last_frame_pos[ rgb_pts_ptr ] = m_current_tracked_pts[ i ];
        rgb_pts_ptr->m_img_pt_in_last_frame = Eigen::Vector2d( m_last_tracked_pts[ i ].x, m_last_tracked_pts[ i ].y );
    }
    updateLastTrackingVectorAndIds();

}

VisualFrontend::VisualFrontend()
{
}

VisualFrontend::~VisualFrontend()
{
}

void VisualFrontend::initialize()
{
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("VIO: image dimensions must be positive");
  }
  if (grid_size <= 10) {
    if (grid_n_height <= 0 || grid_n_height > height) {
      throw std::invalid_argument("VIO: grid_n_height must be between 1 and image height");
    }
    grid_size = height / grid_n_height;
  }
  grid_n_width = 1 + (width - 1) / grid_size;
  grid_n_height = 1 + (height - 1) / grid_size;
  if (grid_n_width > std::numeric_limits<int>::max() / grid_n_height) {
    throw std::overflow_error("VIO: grid dimensions are too large");
  }
  length = grid_n_width * grid_n_height;

  grid_num.resize(length);
  scan_value.resize(length);

  border = 0;
  append_voxel_points.reserve(length);
  append_img_points.reserve(length);
  MIN_DIST = grid_size/2;
}

int VisualFrontend::gridIndex(const Eigen::Vector2d& pixel) const
{
  if (!pixel.allFinite() || pixel.x() < border || pixel.y() < border ||
      pixel.x() >= width - border || pixel.y() >= height - border) {
    return -1;
  }
  const int col = static_cast<int>(pixel.x() / grid_size);
  const int row = static_cast<int>(pixel.y() / grid_size);
  if (col >= grid_n_width || row >= grid_n_height) return -1;
  return row * grid_n_width + col;
}

void VisualFrontend::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  append_img_points.clear();
  append_img_points.resize(length);  
}

void VisualFrontend::initNewFrame(cv::Mat& latest_img, double scale)
{
  if (width != latest_img.cols || height != latest_img.rows) {
      cv::resize(latest_img, latest_img, cv::Size(latest_img.cols * scale, latest_img.rows * scale), 0, 0, cv::INTER_LINEAR);
  }
  img_rgb = latest_img;

  if (latest_img.channels() == 3) {
    cv::cvtColor(img_rgb, gray_scratch_, cv::COLOR_BGR2GRAY);
    latest_img = gray_scratch_;
  }
  if (latest_img.empty()) { throw std::runtime_error("Frame: provided image is empty"); }

  if (latest_img.cols != cam->width() || latest_img.rows != cam->height())
  {
    throw std::runtime_error("Frame: provided image has not the same size as the camera model");
  }

  if (latest_img.type() != CV_8UC1) { throw std::runtime_error("Frame: provided image is not grayscale"); }

  has_frame_ = true;
  resetGrid();  
}

void VisualFrontend::updateAndAppendTrackPts()
{
    for (auto it = m_map_rgb_pts_in_last_frame_pos.begin();
      it != m_map_rgb_pts_in_last_frame_pos.end();) {
        const auto& rgb_pt = it->first;
        const Eigen::Vector2d tracked_pixel(it->second.x, it->second.y);
        bool if_remove = true;
        if (rgb_pt && rgb_pt->if_valid && gridIndex(tracked_pixel) >= 0) {
          const Eigen::Vector3d pf = w2f(rgb_pt->pos_);
          if (pf.allFinite() && pf.z() > 0) {
            const Eigen::Vector2d pc = cam->world2cam(pf);
            if_remove = gridIndex(pc) < 0 || (pc - tracked_pixel).norm() > max_allow_err;
          }
        }

        if (if_remove) {           
          it = m_map_rgb_pts_in_last_frame_pos.erase( it );
        } else {
          it++;
        }
    }    
}

void VisualFrontend::generateVisualPoints(cv::Mat img, const Eigen::aligned_vector<pcl::PointXYZINormal>& pg_w,
    bool render_debug)
{
  updateAndAppendTrackPts();
  tracking_rows_ = img_rgb.rows;
  tracking_cols_ = img_rgb.cols;
  setMask();
  for (auto it = m_map_rgb_pts_in_last_frame_pos.begin();
    it != m_map_rgb_pts_in_last_frame_pos.end();it++) {
      cv::Point2f pt = it->second;
      const int index = gridIndex(Eigen::Vector2d(pt.x, pt.y));
      if (index >= 0) grid_num[index] = TYPE_TRACK;
  }      
  const size_t num_candidates = pg_w.size() > 10 ? pg_w.size() : 0;
  for (size_t i = 0; i < num_candidates; i++) {
    Eigen::Vector3d pt_w(pg_w[i].x, pg_w[i].y, pg_w[i].z);
    Eigen::Vector3d pt_c(w2f(pt_w));
    if (!pt_c.allFinite() || pt_c.z() <= 0) {
      continue;
    }
    Eigen::Vector2d pc = cam->world2cam(pt_c);

    const int index = gridIndex(pc);
    if (index >= 0) {

      if (grid_num[index] != TYPE_MAP && grid_num[index] != TYPE_TRACK) {
        float cur_value = ImgOpr::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value < thresh_lidar_shiTomasi) continue;
        if (cur_value > scan_value[index]) {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pt_w;
          append_img_points[index] = pc;
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    } 
  }
  for (int i = 0; i < length; i++) {
    if (grid_num[i] == TYPE_POINTCLOUD) {
      auto pt_new = std::make_shared<VisualPoint>(append_voxel_points[i], cnt_id);

      pt_new->num_track++;
      cnt_id++;
      Eigen::Vector2d pc = append_img_points[i];
      m_map_rgb_pts_in_last_frame_pos[pt_new] = cv::Point2f(pc[0], pc[1]);
    }
  }
  updateLastTrackingVectorAndIds();
  if (!render_debug) {
    noted_img2.release();
    return;
  }
  img_rgb.copyTo(noted_img2);

  for (const auto &it : m_map_rgb_pts_in_last_frame_pos) {
      const auto& rgb_pts_ptr = it.first;
      cv::Scalar pt_color = color_map[rgb_pts_ptr->type_track];
      cv::circle(noted_img2, it.second, 7, pt_color, -1, 5);
      cv::putText(noted_img2, std::to_string(rgb_pts_ptr->num_track),it.second + cv::Point2f(5, 0), cv::FONT_HERSHEY_COMPLEX, 0.8, pt_color, 1, 30, 0);            
      if (rgb_pts_ptr->num_track > 1) {
        cv::arrowedLine(noted_img2, cv::Point2f(rgb_pts_ptr->m_img_pt_in_last_frame.x(), rgb_pts_ptr->m_img_pt_in_last_frame.y()), 
            it.second, pt_color, 3, 8, 0, 0.5);
      }
  }    
}

Eigen::Vector3f VisualFrontend::getInterpolatedPixel(cv::Mat img, Eigen::Vector2d pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  Eigen::Vector3f pixel(B, G, R);
  return pixel;
}

template <typename T>
void VisualFrontend::reduceVector(std::vector<T> &v, const std::vector<uchar>& status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

double VisualFrontend::distance(const cv::Point2f& pt1, const cv::Point2f& pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void VisualFrontend::trackImg()
{
  if (if_first_frame) {
    if_first_frame = false;
    img_prev = img_rgb;
    previous_pyramid_max_level_ = -1;
    return;
  } 
  m_current_tracked_pts.clear();
  if (m_last_tracked_pts.size() > 0) {
    auto& status = flow_status_;
    auto& err = flow_error_;
    const int max_level = 5;
    if (previous_pyramid_max_level_ != max_level) {
      cv::buildOpticalFlowPyramid(img_prev, previous_pyramid_, cv::Size(21, 21),
          max_level, true, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
    }
    cv::buildOpticalFlowPyramid(img_rgb, current_pyramid_, cv::Size(21, 21),
        max_level, true, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);

    cv::calcOpticalFlowPyrLK(previous_pyramid_, current_pyramid_, m_last_tracked_pts, m_current_tracked_pts, status, err, cv::Size(21, 21), 5);
    int succ_num = std::count(status.begin(), status.end(), (uchar)1);
    int cnt_thresh = 10;
    if (succ_num > cnt_thresh) {
      auto& reverse_status = reverse_status_;
      auto& reverse_pts = reverse_points_;
      reverse_pts = m_last_tracked_pts;
      cv::calcOpticalFlowPyrLK(current_pyramid_, previous_pyramid_, m_current_tracked_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3,
      cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 10, 0.05), cv::OPTFLOW_USE_INITIAL_FLOW);
      for(size_t i = 0; i < status.size(); i++) {
        status[i] = status[i] && reverse_status[i] &&
                    distance(m_last_tracked_pts[i], reverse_pts[i]) <= 0.5;
      }    
    } 

    reduceVector(m_last_tracked_pts, status);
    reduceVector(m_current_tracked_pts, status);
    reduceVector(m_old_ids, status);

    tracking_rows_ = img_rgb.rows;
    tracking_cols_ = img_rgb.cols;
    applyTrackingResults();
    previous_pyramid_.swap(current_pyramid_);
    previous_pyramid_max_level_ = max_level;
  } else {
    previous_pyramid_max_level_ = -1;
  }
  img_prev = img_rgb;
}
