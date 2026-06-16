/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    utils.h
 * @brief   Utility classes for data-matrix assembly, g2o parsing, and timing.
 *
 * @author  Zhexin Xu
 *
 * References:
 *   Zhexin Xu, Hanna Jiamei Zhang, Helena Calatrava, Pau Closas, David M. Rosen.
 *   "Implementing Robust M-Estimators with Certifiable Factor Graph
 *   Optimization." arXiv preprint arXiv:2603.20932, 2026.
 *
 *   Zhexin Xu, Nikolas R. Sanderson, Hanna Jiamei Zhang, David M. Rosen.
 *   "Certifiable Estimation with Factor Graphs." (Certi-fgo) arXiv:2603.01267, 2026.
 *
 * */

#ifndef STIEFELMANIFOLDEXAMPLE_UTILS_H
#define STIEFELMANIFOLDEXAMPLE_UTILS_H

#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/** 
 * @brief Floating point type used throughout the project. 
 */
typedef double Scalar;

/** 
 * @brief Dense vector type. 
 */
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

/** 
 * @brief Dense matrix type. 
 */
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

/** 
 * @brief Diagonal matrix type. 
 */
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;

/** 
 * @brief Sparse matrix type.
 * 
 * Uses row-major storage order to take advantage of fast sparse-matrix 
 * dense-vector multiplications when OpenMP is available.
 */
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;

#include "RelativePoseMeasurement.h"
#include <chrono>
#include <gtsam/base/types.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/Ordering.h>
#include <boost/pointer_cast.hpp>
namespace gtsam {

    /**
     * @brief Specifies how rotation and translation blocks are ordered in the data matrix.
     */
    enum class BlockOrderingType {
        RtRt, ///< Interleaved: [R1, t1, R2, t2, ...]
        tRtR,  ///< Blocked: [t1, t2, ..., R1, R2, ...]
        Generic ///< Based on ordering: [Var1, Var2, ...]
    };

    /**
     * @namespace DataParser
     * @brief Functions for parsing dataset files (.g2o, .pycfg).
     */
    namespace DataParser {

        /**
         * @brief Reads a g2o file and constructs measurement data.
         * 
         * @param[in] filename Path to the .g2o file.
         * @param[out] num_poses Total number of poses found in the file.
         * @return Measurement struct containing parsed data.
         */
        Measurement read_g2o_file(const std::string &filename, size_t &num_poses);

    } // namespace DataParser

    /**
     * @brief Structured block for Euclidean Hessian assembly.
     */
    struct EuclideanHessianBlock {
        std::vector<Key> keys;
        std::vector<int> dims; // Dimensions of each key in the Hessian
        Matrix Hessian; // Full joint Hessian w.r.t all keys
    };

    /**
     * @brief Interface for factors that can provide Euclidean Hessian blocks for SDP data matrix assembly.
     */
    class EuclideanFactor {
    public:
        virtual ~EuclideanFactor() = default;
        virtual EuclideanHessianBlock computeEuclideanHessian() const = 0;
    };

    /**
     * @brief Represents the global layout of variables in the data matrix.
     */
    class GlobalLayout {
    public:
        struct Component {
            int type; // 0 for Rotation/Stiefel (constrained), 1 for Translation/Landmark/Raw (unconstrained)
            int dim;
            size_t globalOffset;
            bool constrained;
        };

        struct Variable {
            Key key;
            std::vector<Component> components;
        };

    private:
        std::map<Key, Variable> variables_;
        size_t totalDim_;
        Ordering ordering_;
        BlockOrderingType type_;
        size_t d_;

    public:
        GlobalLayout() : totalDim_(0), d_(0) {}
        
        GlobalLayout(const Ordering& order, size_t d, BlockOrderingType type)
            : ordering_(order), type_(type), d_(d) {

            if (type == BlockOrderingType::Generic) {
                totalDim_ = 0;
                for (Key k : order) {
                    Variable var;
                    var.key = k;
                    if (symbolChr(k) == 'L') {
                        Component lmk = {1, 1, totalDim_, false};
                        var.components = {lmk};
                        totalDim_ += 1;
                    } else if (symbolChr(k) == 'R') {
                        Component sphere = {0, 1, totalDim_, true};
                        var.components = {sphere};
                        totalDim_ += 1;
                    } else {
                        Component rot = {0, (int)d, totalDim_, true};
                        Component trans = {1, 1, totalDim_ + d, false};
                        var.components = {rot, trans};
                        totalDim_ += (d + 1);
                    }
                    variables_[k] = var;
                }
                return;
            }

            size_t num_pose = 0;
            size_t num_landmark = 0;
            size_t num_range = 0;

            for (Key k : order) {
                if (symbolChr(k) == 'L') num_landmark++;
                else if (symbolChr(k) == 'R') num_range++;
                else num_pose++;
            }

            if (type == BlockOrderingType::RtRt) {
                // Interleaved: [R0, t0, R1, t1, ...] + [Landmarks] + [UnitSpheres]
                totalDim_ = num_pose * (d + 1) + num_landmark + num_range;
                size_t l_idx = 0;
                size_t r_idx = 0;
                for (size_t seq = 0; seq < order.size(); ++seq) {
                    Key k = order[seq];
                    Variable var;
                    var.key = k;
                    if (symbolChr(k) == 'L') {
                        Component lmk = {1, 1, num_pose * (d + 1) + l_idx++, false};
                        var.components = {lmk};
                    } else if (symbolChr(k) == 'R') {
                        Component sphere = {0, 1, num_pose * (d + 1) + num_landmark + r_idx++, true};
                        var.components = {sphere};
                    } else {
                        // For poses, we use the sequence in the ordering as the index
                        size_t pose_idx = 0;
                        for(size_t s=0; s<seq; ++s) {
                             Key prev_k = order[s];
                             if (symbolChr(prev_k) != 'L' && symbolChr(prev_k) != 'R') pose_idx++;
                        }
                        Component rot = {0, (int)d, pose_idx * (d + 1), true};
                        Component trans = {1, 1, pose_idx * (d + 1) + d, false};
                        var.components = {rot, trans};
                    }
                    variables_[k] = var;
                }
            } else {
                // Blocked: [t0, t1, ...] + [R0, R1, ...] + [Landmarks] + [UnitSpheres]
                totalDim_ = num_pose + num_pose * d + num_landmark + num_range;
                size_t l_idx = 0;
                size_t r_idx = 0;
                for (size_t seq = 0; seq < order.size(); ++seq) {
                    Key k = order[seq];
                    Variable var;
                    var.key = k;
                    if (symbolChr(k) == 'L') {
                        Component lmk = {1, 1, num_pose * (d + 1) + l_idx++, false};
                        var.components = {lmk};
                    } else if (symbolChr(k) == 'R') {
                        Component sphere = {0, 1, num_pose * (d + 1) + num_landmark + r_idx++, true};
                        var.components = {sphere};
                    } else {
                        size_t pose_idx = 0;
                        for(size_t s=0; s<seq; ++s) {
                             Key prev_k = order[s];
                             if (symbolChr(prev_k) != 'L' && symbolChr(prev_k) != 'R') pose_idx++;
                        }
                        Component trans = {1, 1, pose_idx, false};
                        Component rot = {0, (int)d, num_pose + pose_idx * d, true};
                        var.components = {rot, trans};
                    }
                    variables_[k] = var;
                }
            }
        }

        size_t getTotalDim() const { return totalDim_; }
        const Ordering& getOrdering() const { return ordering_; }
        
        size_t getGlobalIndex(Key k, int localRow, int claimedDim = -1) const {
            auto it = variables_.find(k);
            if (it == variables_.end()) throw std::runtime_error("Key not found in GlobalLayout");
            
            // Map localRow to component
            // For Pose: localRow < d is Rotation, localRow == d is Translation
            // For Landmark: localRow == 0 is Landmark
            // For UnitSphere: localRow == 0 is UnitSphere
            if (symbolChr(k) == 'L' || symbolChr(k) == 'R') {
                return it->second.components[0].globalOffset;
            } else {
                // If the factor claimed this key has dimension 1, it must be translation (since it's not 'L' or 'R)
                if (claimedDim == 1) {
                    return it->second.components[1].globalOffset;
                }

                if (localRow < (int)d_) {
                    return it->second.components[0].globalOffset + localRow; // Assuming Rotation is component 0 in internal def
                } else if (localRow == (int)d_) {
                    return it->second.components[1].globalOffset;
                } else {
                    throw std::runtime_error("Invalid localRow for Pose key");
                }
            }
        }

        const std::map<Key, Variable>& getVariables() const { return variables_; }
    };

    /**
     * @brief Utility for assembling global data matrices from factors.
     */
    class DataMatrixAssembler {
    public:
        /**
         * @brief Assemble the global data matrix using a modular, type-aware approach.
         */
        static SparseMatrix Assemble(const NonlinearFactorGraph& graph,
                                     const GlobalLayout& layout) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t totalDim = layout.getTotalDim();
            
            if (totalDim == 0) return SparseMatrix(0, 0);

            for (const auto& factor : graph) {
                if (auto eFactor = boost::dynamic_pointer_cast<EuclideanFactor>(factor)) {
                    EuclideanHessianBlock block = eFactor->computeEuclideanHessian();
                    ProcessBlock(block, layout, triplets);
                }
            }

            SparseMatrix L(totalDim, totalDim);
            L.setFromTriplets(triplets.begin(), triplets.end());
            L.makeCompressed();
            return L;
        }

    private:
        static void ProcessBlock(const EuclideanHessianBlock& block,
                                 const GlobalLayout& layout,
                                 std::vector<Eigen::Triplet<Scalar>>& triplets) {
            std::vector<int> keyLocalOffsets;
            int currentLocalOffset = 0;

            for (int dim : block.dims) {
                keyLocalOffsets.push_back(currentLocalOffset);
                currentLocalOffset += dim;
            }

            for (size_t i = 0; i < block.keys.size(); ++i) {
                for (size_t j = 0; j < block.keys.size(); ++j) {
                    FillSubBlock(block.Hessian, i, j, keyLocalOffsets, block.dims, 
                                block.keys[i], block.keys[j], layout, triplets);
                }
            }
        }

        static void FillSubBlock(const Matrix& H, size_t i, size_t j,
                                 const std::vector<int>& localOffsets,
                                 const std::vector<int>& dims,
                                 Key k1, Key k2,
                                 const GlobalLayout& layout,
                                 std::vector<Eigen::Triplet<Scalar>>& triplets) {
            int dim1 = dims[i];
            int dim2 = dims[j];
            int off1 = localOffsets[i];
            int off2 = localOffsets[j];

            for (int r = 0; r < dim1; ++r) {
                for (int c = 0; c < dim2; ++c) {
                    Scalar v = H(off1 + r, off2 + c);
                    if (v == 0) continue;

                    size_t gRow = layout.getGlobalIndex(k1, r, dim1);
                    size_t gCol = layout.getGlobalIndex(k2, c, dim2);
                    
                    triplets.emplace_back(gRow, gCol, v);
                }
            }
        }
    };

} // namespace gtsam



/** 
 * @namespace CFGStopwatch
 * @brief Convenience functions for measuring elapsed computation times.
 */
namespace CFGStopwatch {

    /** 
     * @brief Returns the current time point.
     * @return Current time point using high_resolution_clock.
     */
    inline std::chrono::time_point<std::chrono::high_resolution_clock> tick() {
        return std::chrono::high_resolution_clock::now();
    }

    /** 
     * @brief Returns the elapsed time in seconds since the given tick time.
     * @param tick_time The time point returned by a previous call to tick().
     * @return Elapsed time in seconds.
     */
    inline double
    tock(const std::chrono::time_point<std::chrono::high_resolution_clock>
         &tick_time) {
        auto counter = std::chrono::high_resolution_clock::now() - tick_time;
        return std::chrono::duration_cast<std::chrono::milliseconds>(counter)
                       .count() /
               1000.0;
    }
} // namespace CFGStopwatch


#endif //STIEFELMANIFOLDEXAMPLE_UTILS_H
