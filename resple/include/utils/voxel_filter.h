#pragma once

#include <pcl/common/point_tests.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <boost/sort/spreadsort/integer_sort.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace resple2 {

struct VoxelFilterSelection {
    static constexpr uint8_t first = 1, second = 2;
    std::vector<uint8_t> mask;
    std::array<std::size_t, 2> counts{};

    void clear() { mask.clear(); counts = {}; }

};

template <typename PointT>
class VoxelFilter {
public:
    using Cloud = pcl::PointCloud<PointT>;
    using ConstPtr = typename Cloud::ConstPtr;

    void setLeafSize(float x, float y, float z) {
        const std::array<float, 3> sizes{x, y, z};
        for (float size : sizes) {
            if (!std::isfinite(size) || size <= 0.0f)
                throw std::invalid_argument("VoxelFilter leaf sizes must be finite and positive");
        }
        for (std::size_t axis = 0; axis < 3; ++axis)
            inverse_leaf_size_[axis] = 1.0 / static_cast<double>(sizes[axis]);
        configured_ = true;
    }

    void setInputCloud(const ConstPtr& cloud) {
        if (!cloud) throw std::invalid_argument("VoxelFilter input cloud must not be null");
        input_ = cloud;
    }

    void filter(Cloud& output) const {
        validate();
        if (input_.get() == &output) {
            Cloud temporary;
            filterInto(temporary);
            output.swap(temporary);
        } else {
            filterInto(output);
        }
    }

    void filterBoth(const VoxelFilter& other, Cloud& first, Cloud& second) const {
        validate();
        other.validate();
        if (input_ != other.input_)
            throw std::invalid_argument("VoxelFilter pair must share an input cloud");
        if (&first == &second)
            throw std::invalid_argument("VoxelFilter pair requires distinct output clouds");
        if (input_.get() == &first || input_.get() == &second) {
            Cloud a, b;
            filterBoth(other, a, b);
            first.swap(a);
            second.swap(b);
            return;
        }
        const bool same_leaf = inverse_leaf_size_ == other.inverse_leaf_size_;
        prepareOutput(first);
        other.prepareOutput(second);
        prepareGroups(other, same_leaf);
        finishFilter(first);
        if (same_leaf) {
            second = first;
        } else {
            other.finishFilter(second);
        }
    }

    bool selectBothInTimeOrder(const VoxelFilter& other, VoxelFilterSelection& selection) const {
        validate();
        other.validate();
        if (input_ != other.input_)
            throw std::invalid_argument("VoxelFilter pair must share an input cloud");
        selection.clear();
        float previous = -std::numeric_limits<float>::infinity();
        bool ordered = true;
        for (const auto& point : *input_) {
            if (!pcl::isFinite(point)) continue;
            if (!std::isfinite(point.intensity) || point.intensity < previous) {
                ordered = false;
                break;
            }
            previous = point.intensity;
        }
        if (!ordered) {
            return false;
        }
        const bool same_leaf = inverse_leaf_size_ == other.inverse_leaf_size_;
        prepareGroups(other, same_leaf);
        selection.mask.assign(input_->size(), 0);
        selection.counts = {groups_.size(), same_leaf ? groups_.size() : other.groups_.size()};
        for (const auto& group : groups_)
            selection.mask[group.selected] |= same_leaf ? VoxelFilterSelection::first | VoxelFilterSelection::second
                                                       : VoxelFilterSelection::first;
        if (!same_leaf) {
            for (const auto& group : other.groups_)
                selection.mask[group.selected] |= VoxelFilterSelection::second;
        }
        return true;
    }

private:
    void prepareGroups(const VoxelFilter& other, bool same_leaf) const {
        resetGroups();
        if (!same_leaf) other.resetGroups();
        for (std::size_t i = 0; i < input_->size(); ++i) {
            const auto& point = (*input_)[i];
            if (!pcl::isFinite(point)) {
                point_groups_[i] = empty_group;
                if (!same_leaf) other.point_groups_[i] = empty_group;
                continue;
            }
            const std::array<double, 3> xyz{point.x, point.y, point.z};
            accumulate(cellFor(xyz), point, i);
            if (!same_leaf) other.accumulate(other.cellFor(xyz), point, i);
        }
        computeCentroids();
        if (!same_leaf) other.computeCentroids();
        for (std::size_t i = 0; i < input_->size(); ++i) {
            if (point_groups_[i] == empty_group) continue;
            const auto& point = (*input_)[i];
            selectPoint(point, i);
            if (!same_leaf) other.selectPoint(point, i);
        }
    }

    using Cell = std::array<std::int64_t, 3>;
    static constexpr std::size_t empty_group = std::numeric_limits<std::size_t>::max();
    struct Group {
        Cell cell;
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        std::size_t count = 0;
        std::size_t selected = empty_group;
        double best_distance = std::numeric_limits<double>::infinity();
    };
    struct LinearIndex {
        std::uint64_t key;
        std::size_t group;
        bool operator<(const LinearIndex& other) const { return key < other.key; }
    };

    void validate() const {
        if (!input_) throw std::logic_error("VoxelFilter input cloud has not been set");
        if (!configured_) throw std::logic_error("VoxelFilter leaf size has not been set");
    }

    Cell cellFor(const std::array<double, 3>& xyz) const {
        Cell cell;
        constexpr double limit = 9223372036854775808.0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double coordinate = std::floor(xyz[axis] * inverse_leaf_size_[axis]);
            if (!std::isfinite(coordinate) || coordinate < -limit || coordinate >= limit)
                throw std::out_of_range("VoxelFilter coordinate exceeds the 64-bit voxel range");
            cell[axis] = static_cast<std::int64_t>(coordinate);
        }
        return cell;
    }

    static std::size_t hashCell(const Cell& cell) {
        std::uint64_t h = static_cast<std::uint64_t>(cell[0]) * 0x9e3779b185ebca87ULL ^
                          static_cast<std::uint64_t>(cell[1]) * 0xc2b2ae3d27d4eb4fULL ^
                          static_cast<std::uint64_t>(cell[2]) * 0x165667b19e3779f9ULL;
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebULL;
        return static_cast<std::size_t>(h ^ (h >> 31));
    }

    void prepareOutput(Cloud& output) const {
        output.clear();
        output.header = input_->header;
        output.sensor_origin_ = input_->sensor_origin_;
        output.sensor_orientation_ = input_->sensor_orientation_;
        output.height = 1;
        output.is_dense = input_->is_dense;
    }

    void resetGroups() const {
        groups_.clear();
        groups_.reserve(input_->size());
        point_groups_.resize(input_->size());
        if (input_->empty()) return;
        if (input_->size() > slots_.max_size() / 2)
            throw std::length_error("VoxelFilter hash table exceeds maximum size");
        const std::size_t required = input_->size() * 2;
        std::size_t capacity = 16;
        while (capacity < required) {
            if (capacity > slots_.max_size() / 2)
                throw std::length_error("VoxelFilter hash table exceeds maximum size");
            capacity *= 2;
        }
        slots_.assign(capacity, empty_group);
    }

    void accumulate(const Cell& cell, const PointT& point, std::size_t point_index) const {
        const std::size_t mask = slots_.size() - 1;
        std::size_t slot = hashCell(cell) & mask;
        while (slots_[slot] != empty_group && groups_[slots_[slot]].cell != cell)
            slot = (slot + 1) & mask;
        if (slots_[slot] == empty_group) {
            slots_[slot] = groups_.size();
            groups_.push_back(Group{cell});
        }
        const std::size_t group_index = slots_[slot];
        point_groups_[point_index] = group_index;
        auto& group = groups_[group_index];
        group.centroid += Eigen::Vector3d(point.x, point.y, point.z);
        ++group.count;
    }

    void computeCentroids() const {
        for (auto& group : groups_) group.centroid /= static_cast<double>(group.count);
    }

    void selectPoint(const PointT& point, std::size_t point_index) const {
        auto& group = groups_[point_groups_[point_index]];
        const double distance = (Eigen::Vector3d(point.x, point.y, point.z) - group.centroid).squaredNorm();
        if (distance < group.best_distance ||
            (distance == group.best_distance && point_index < group.selected)) {
            group.selected = point_index;
            group.best_distance = distance;
        }
    }

    void filterInto(Cloud& output) const {
        prepareOutput(output);
        resetGroups();
        for (std::size_t i = 0; i < input_->size(); ++i) {
            const auto& point = (*input_)[i];
            if (!pcl::isFinite(point)) {
                point_groups_[i] = empty_group;
                continue;
            }
            accumulate(cellFor({point.x, point.y, point.z}), point, i);
        }
        computeCentroids();
        for (std::size_t i = 0; i < input_->size(); ++i)
            if (point_groups_[i] != empty_group) selectPoint((*input_)[i], i);
        finishFilter(output);
    }

    void finishFilter(Cloud& output) const {
        if (groups_.empty()) return;
        Cell minimum = groups_.front().cell, maximum = minimum;
        for (const auto& group : groups_) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                minimum[axis] = std::min(minimum[axis], group.cell[axis]);
                maximum[axis] = std::max(maximum[axis], group.cell[axis]);
            }
        }
        std::array<std::uint64_t, 3> dimensions{};
        std::uint64_t volume = 1;
        bool linear_keys = true;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto span = static_cast<std::uint64_t>(maximum[axis]) -
                              static_cast<std::uint64_t>(minimum[axis]);
            if (span == std::numeric_limits<std::uint64_t>::max()) {
                linear_keys = false;
                break;
            }
            dimensions[axis] = span + 1;
            if (dimensions[axis] > std::numeric_limits<std::uint64_t>::max() / volume) {
                linear_keys = false;
                break;
            }
            volume *= dimensions[axis];
        }

        indices_.clear();
        indices_.reserve(groups_.size());
        if (linear_keys) {
            for (std::size_t i = 0; i < groups_.size(); ++i) {
                const auto& cell = groups_[i].cell;
                const auto x = static_cast<std::uint64_t>(cell[0]) - static_cast<std::uint64_t>(minimum[0]);
                const auto y = static_cast<std::uint64_t>(cell[1]) - static_cast<std::uint64_t>(minimum[1]);
                const auto z = static_cast<std::uint64_t>(cell[2]) - static_cast<std::uint64_t>(minimum[2]);
                indices_.push_back({x + dimensions[0] * (y + dimensions[1] * z), i});
            }
            boost::sort::spreadsort::integer_sort(indices_.begin(), indices_.end(),
                [](const LinearIndex& entry, unsigned shift) { return entry.key >> shift; });
        } else {
            for (std::size_t i = 0; i < groups_.size(); ++i) indices_.push_back({0, i});
            std::sort(indices_.begin(), indices_.end(), [&](const LinearIndex& a, const LinearIndex& b) {
                return groups_[a.group].cell < groups_[b.group].cell;
            });
        }
        output.reserve(groups_.size());
        for (const auto& entry : indices_) output.push_back((*input_)[groups_[entry.group].selected]);
    }

    ConstPtr input_;
    std::array<double, 3> inverse_leaf_size_{};
    bool configured_ = false;
    mutable std::vector<Group> groups_;
    mutable std::vector<std::size_t> slots_, point_groups_;
    mutable std::vector<LinearIndex> indices_;
};

template <typename PointT>
class VoxelDownsampler {
public:
    using Cloud = pcl::PointCloud<PointT>;
    void setUseCustom(bool enabled) { use_custom_ = enabled; }
    void setLeafSize(float x, float y, float z) {
        custom_.setLeafSize(x, y, z);
        pcl_.setLeafSize(x, y, z);
    }
    void setInputCloud(const typename Cloud::ConstPtr& cloud) {
        custom_.setInputCloud(cloud);
        pcl_.setInputCloud(cloud);
        input_ = cloud;
    }
    void filter(Cloud& output) {
        if (use_custom_) custom_.filter(output);
        else pcl_.filter(output);
    }

    bool selectBothInTimeOrder(VoxelDownsampler& other, VoxelFilterSelection& selection) const {
        selection.clear();
        if (!input_ || input_ != other.input_)
            throw std::invalid_argument("VoxelDownsampler pair must share an input cloud");
        if (!use_custom_ || !other.use_custom_) return false;
        return custom_.selectBothInTimeOrder(other.custom_, selection);
    }

    void filterBoth(VoxelDownsampler& other, Cloud& first, Cloud& second) {
        if (!input_ || input_ != other.input_)
            throw std::invalid_argument("VoxelDownsampler pair must share an input cloud");
        if (&first == &second)
            throw std::invalid_argument("VoxelDownsampler pair requires distinct output clouds");
        if (input_.get() == &first || input_.get() == &second) {
            Cloud a, b;
            filterBoth(other, a, b);
            first.swap(a);
            second.swap(b);
            return;
        }
        if (use_custom_ && other.use_custom_) {
            custom_.filterBoth(other.custom_, first, second);
        } else {
            filter(first);
            other.filter(second);
        }
    }

private:
    bool use_custom_ = false;
    typename Cloud::ConstPtr input_;
    VoxelFilter<PointT> custom_;
    pcl::VoxelGrid<PointT> pcl_;
};

}
