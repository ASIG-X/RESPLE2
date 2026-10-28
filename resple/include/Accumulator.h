#pragma once

#include <array>
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/CholmodSupport>

template <class T>
using SparseLLT = Eigen::CholmodSupernodalLLT<T>;

class SparseHashAccumulator
{
 public:

    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorX;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Triplet<double> T;
    typedef Eigen::SparseMatrix<double> SparseMatrix;

    template <int ROWS, int COLS, typename Derived>
    inline void addH(int si, int sj, const Eigen::MatrixBase<Derived>& data)
    {
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, ROWS, COLS);
        KeyT id;
        id[0] = si;
        id[1] = sj;
        id[2] = ROWS;
        id[3] = COLS;
        auto it = hash_map.find(id);
        if (it == hash_map.end()) {
            hash_map.emplace(id, HessianBlock{data, {}});
            pattern_valid = false;
        } else {
            it->second.values += data;
        }
    }

    template <int ROWS, typename Derived>
    inline void addB(int i, const Eigen::MatrixBase<Derived>& data)
    {
        b.template segment<ROWS>(i) += data;
    }

    inline void setup_solver()
    {
        if (!pattern_valid) rebuildPattern();
        std::fill_n(smm.valuePtr(), smm.nonZeros(), 0.0);
        for (const auto& kv : hash_map) {
            const auto& block = kv.second;
            for (Eigen::Index i = 0; i < block.values.size(); ++i) {
                smm.valuePtr()[block.value_indices[i]] += block.values.data()[i];
            }
        }
        for (const auto index : diagonal_indices) {
            smm.valuePtr()[index] += std::numeric_limits<double>::min();
        }
    }

    inline VectorX Hdiagonal() const { return smm.diagonal(); }

    inline VectorX& getB() { return b; }

    inline const SparseMatrix& getH() const { return smm; }

    inline void scaleSystem(const VectorX& scaling)
    {
        for (int col = 0; col < smm.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(smm, col); it; ++it) {
                it.valueRef() *= scaling[it.row()];
                it.valueRef() *= scaling[col];
            }
        }
        b.array() *= scaling.array();
    }

    inline VectorX solve(const VectorX* diagonal)
    {
        std::copy_n(smm.valuePtr(), smm.nonZeros(), solve_matrix.valuePtr());
        if (diagonal) {
            for (Eigen::Index i = 0; i < b.size(); ++i) {
                solve_matrix.valuePtr()[diagonal_indices[i]] += (*diagonal)[i];
            }
        }
        VectorX res = VectorX::Constant(b.size(), std::numeric_limits<double>::quiet_NaN());
        if (b.size() == 0) return res;
        if (!solver) solver = std::make_unique<SparseLLT<SparseMatrix>>();
        if (!analysis_valid) {
            solver->analyzePattern(solve_matrix);
            if (solver->info() != Eigen::Success) return res;
            analysis_valid = true;
        }
        solver->factorize(solve_matrix);
        if (solver->info() != Eigen::Success) return res;
        res = solver->solve(b);
        if (solver->info() != Eigen::Success) {
            res.setConstant(std::numeric_limits<double>::quiet_NaN());
        }
        return res;
    }

    inline void reset(int opt_size)
    {
        hash_map.clear();
        b.setZero(opt_size);
        pattern_valid = false;
        analysis_valid = false;
    }

    inline void setZero()
    {
        for (auto& kv : hash_map) kv.second.values.setZero();
        b.setZero();
    }

    inline void join(const SparseHashAccumulator& other) {
        for (const auto& kv : other.hash_map) {
            auto it = hash_map.find(kv.first);

            if (it == hash_map.end()) {
                hash_map.emplace(kv.first, HessianBlock{kv.second.values, {}});
                pattern_valid = false;
            } else {
                it->second.values += kv.second.values;
            }
        }

        b += other.b;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:

    using KeyT = std::array<int, 4>;

    struct KeyHash {
        inline size_t operator()(const KeyT& c) const
        {
            size_t seed = 0;
            for (int i = 0; i < 4; i++) {
                seed ^= c[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    struct HessianBlock {
        MatrixX values;
        std::vector<SparseMatrix::StorageIndex> value_indices;
    };

    void rebuildPattern()
    {
        std::vector<T> triplets;
        size_t num_entries = b.size();
        for (const auto& kv : hash_map) num_entries += kv.second.values.size();
        triplets.reserve(num_entries);
        for (const auto& kv : hash_map) {
            const auto& values = kv.second.values;
            for (int col = 0; col < values.cols(); ++col) {
                for (int row = 0; row < values.rows(); ++row) {
                    triplets.emplace_back(kv.first[0] + row, kv.first[1] + col, 0.0);
                }
            }
        }
        for (int i = 0; i < b.size(); ++i) triplets.emplace_back(i, i, 0.0);
        smm.resize(b.size(), b.size());
        smm.setFromTriplets(triplets.begin(), triplets.end());
        smm.makeCompressed();

        auto valueIndex = [&](int row, int col) {
            const auto* begin = smm.innerIndexPtr() + smm.outerIndexPtr()[col];
            const auto* end = smm.innerIndexPtr() + smm.outerIndexPtr()[col + 1];
            const auto* entry = std::lower_bound(begin, end, row);
            return static_cast<SparseMatrix::StorageIndex>(entry - smm.innerIndexPtr());
        };
        for (auto& kv : hash_map) {
            auto& block = kv.second;
            block.value_indices.resize(block.values.size());
            for (int col = 0; col < block.values.cols(); ++col) {
                for (int row = 0; row < block.values.rows(); ++row) {
                    block.value_indices[col * block.values.rows() + row] =
                        valueIndex(kv.first[0] + row, kv.first[1] + col);
                }
            }
        }
        diagonal_indices.resize(b.size());
        for (int i = 0; i < b.size(); ++i) diagonal_indices[i] = valueIndex(i, i);
        solve_matrix = smm;
        pattern_valid = true;
        analysis_valid = false;
    }

    std::unordered_map<KeyT, HessianBlock, KeyHash> hash_map;
    VectorX b;
    SparseMatrix smm;
    SparseMatrix solve_matrix;
    std::vector<SparseMatrix::StorageIndex> diagonal_indices;
    std::unique_ptr<SparseLLT<SparseMatrix>> solver;
    bool pattern_valid = false;
    bool analysis_valid = false;
};
