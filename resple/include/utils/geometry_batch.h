#pragma once

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace GeometryBatch {

struct Config {
    bool enabled = false;
    int first_points = 500;
    int check_stride = 64;
    int min_matches = 50;
    double min_information = 10000.0;
    double min_eigen_ratio = 0.01;
    double min_dt_ms = 10.0;
    double preview_position_std = 0.1;

    void validate(int64_t max_dt_ns) const {
        if (first_points < 1 || check_stride < 1 || min_matches < 1 ||
            !std::isfinite(min_information) || min_information <= 0.0 ||
            !std::isfinite(min_eigen_ratio) || min_eigen_ratio < 0.0 || min_eigen_ratio > 1.0/3.0 ||
            !std::isfinite(min_dt_ms) || min_dt_ms <= 0.0 || min_dt_ms * 1e6 > max_dt_ns ||
            !std::isfinite(preview_position_std) || preview_position_std <= 0.0)
            throw std::invalid_argument("Invalid geometry_batch thresholds or time bounds");
    }
};

struct Metrics {
    Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
    Eigen::Vector3d weak_direction = Eigen::Vector3d::Zero();
    double ratio = 0.0;
    std::array<bool, 3> passed{};
    bool valid = false;
    bool sufficient = false;
};

struct GeometryInformationAccumulator {
    size_t points = 0, matches = 0;
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();

    void addPoint() {
        ++points;
    }

    void addMatch(const Eigen::Vector3d& normal, double variance) {
        const double length = normal.norm();
        if (!normal.allFinite() || length < 1e-12 || !std::isfinite(variance) || variance <= 0.0) return;
        const Eigen::Vector3d n = normal / length;
        information.noalias() += (n * n.transpose()) / variance;
        ++matches;
    }

    Metrics metrics(const Config& config) const {
        Metrics result;
        if (matches && information.allFinite()) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(information);
            if (solver.info() == Eigen::Success) {
                result.valid = true;
                result.eigenvalues = solver.eigenvalues().cwiseMax(0.0);
                result.weak_direction = solver.eigenvectors().col(0);
                const double trace = result.eigenvalues.sum();
                if (trace > 0.0) result.ratio = result.eigenvalues[0] / trace;
            }
        }
        result.passed = {{matches >= static_cast<size_t>(config.min_matches),
            result.eigenvalues[0] >= config.min_information,
            result.ratio >= config.min_eigen_ratio}};
        result.sufficient = result.valid &&
            std::all_of(result.passed.begin(), result.passed.end(), [](bool pass) { return pass; });
        return result;
    }
};

struct Decision {
    int64_t cutoff_ns;
    std::string reason;
    GeometryInformationAccumulator batch;
    Metrics metrics;
};

template<class Point, class Evaluate, class Prepare>
std::optional<Decision> chooseCutoff(const std::vector<Point>& points,
    int64_t earliest_ns, int64_t limit_ns, int64_t available_ns,
    const std::string& limit_reason, const Config& config, Evaluate evaluate, Prepare prepare) {
    if (available_ns < limit_ns &&
        (available_ns < earliest_ns || points.empty() ||
         points.size() < static_cast<size_t>(config.first_points) ||
         points.back().time_ns < earliest_ns))
        return std::nullopt;
    GeometryInformationAccumulator batch;
    size_t checkpoint = static_cast<size_t>(config.first_points);
    size_t prepared_end = 0;
    const int64_t end = std::min(limit_ns, available_ns);
    for (size_t i = 0; i < points.size();) {
        const int64_t time_ns = points[i].time_ns;
        if (time_ns > end) break;
        if (i == prepared_end) {
            do {
                const int64_t prepared_time = points[prepared_end].time_ns;
                do { ++prepared_end; }
                while (prepared_end < points.size() && points[prepared_end].time_ns == prepared_time);
                if (batch.points + prepared_end - i >= checkpoint && prepared_time >= earliest_ns) break;
            } while (prepared_end < points.size() && points[prepared_end].time_ns <= end);
            prepare(i, prepared_end);
        }
        do { evaluate(points[i], batch); ++i; }
        while (i < points.size() && points[i].time_ns == time_ns);
        if (batch.points >= checkpoint && time_ns >= earliest_ns) {
            const Metrics metrics = batch.metrics(config);
            if (metrics.sufficient) return Decision{time_ns, "geometry", batch, metrics};
            checkpoint = batch.points + static_cast<size_t>(config.check_stride);
        }
    }
    if (available_ns >= limit_ns)
        return Decision{limit_ns, limit_reason, batch, batch.metrics(config)};
    return std::nullopt;
}

template<class Point, class Evaluate>
std::optional<Decision> chooseCutoff(const std::vector<Point>& points,
    int64_t earliest_ns, int64_t limit_ns, int64_t available_ns,
    const std::string& limit_reason, const Config& config, Evaluate evaluate) {
    return chooseCutoff(points, earliest_ns, limit_ns, available_ns, limit_reason,
        config, evaluate, [](size_t, size_t) {});
}

}
