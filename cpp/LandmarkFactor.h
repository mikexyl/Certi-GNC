/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    LandmarkFactor.h
 * @brief   GTSAM-style lifted pose-to-landmark observation factor.
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
 */

#ifndef STIEFELMANIFOLDEXAMPLE_LANDMARKFACTOR_H
#define STIEFELMANIFOLDEXAMPLE_LANDMARKFACTOR_H

#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/inference/Symbol.h>
#include "LiftedPose.h"

#include <type_traits>
namespace gtsam {

    //******************************************************************************
    template <size_t d>
    class GTSAM_EXPORT LiftedLandmarkFactor : public NoiseModelFactorN<LiftedPoseDP, Vector>, public EuclideanFactor {
        Vector V_;    ///< measured translation difference between pose and landmark

        size_t p_, d_;               ///< dimensionality constants
        size_t pd_;

        using Trans = typename std::conditional<d == 2, Vector2 , Vector3>::type;

    public:

        // Provide access to the Matrix& version of evaluateError:
        using NoiseModelFactor2<LiftedPoseDP, Vector>::evaluateError;

        /// @name Constructor
        /// @{

        /// Constructor. Note we convert to d*p-dimensional noise model.
        LiftedLandmarkFactor(Key j1, Key j2, const Vector &T12 , size_t p,
                     const SharedNoiseModel &model = nullptr);

        /**
         * @brief Polymorphic copy used by GncOptimizer's cloneWithNewNoiseModel.
         */
        gtsam::NonlinearFactor::shared_ptr clone() const override {
            return std::static_pointer_cast<gtsam::NonlinearFactor>(
                std::make_shared<LiftedLandmarkFactor<d>>(*this));
        }

        /// @}
        /// @name Testable
        /// @{

        /// print with optional string
        void
        print(const std::string &s,
              const KeyFormatter &keyFormatter = DefaultKeyFormatter) const override;

        /// assert equality up to a tolerance
        bool equals(const NonlinearFactor &expected,
                    double tol = 1e-9) const override;

        /// @}
        /// @name NoiseModelFactorN methods
        /// @{


        Vector evaluateError(const LiftedPoseDP& P1, const Vector &L1, OptionalMatrixType H1, OptionalMatrixType H2) const override;


        /// Calculate Jacobians if asked, Only implemented for d=2 and 3 in .cpp
        void fillJacobians(const LiftedPoseDP &P1, const Vector &L1,
                           OptionalMatrixType H1,
                           OptionalMatrixType H2) const;

        /**
         * @brief Compute the Euclidean Hessian block in a structured format.
         */
        EuclideanHessianBlock computeEuclideanHessian() const override;

        // The functions below are used as the old had-crafted way to build data matrix, refer to SE-Sync, CPL-SLAM, and CORA paper
        // ******************************** Hand-crafted data matrix ****************************************
        // **************************************************************************************************
        // B4^T*B4
        std::vector<Eigen::Triplet<Scalar>> getSigmaTransBlock( ) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 1;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1()); // Pose idx,

            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /( 1 * sigmas(0) *  sigmas(0));
            triplets.emplace_back(i, i, precision);

            return triplets;
        }
        // B5^T*B5
        std::vector<Eigen::Triplet<Scalar>> getSigmaRotBlock(const size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = d_*d_;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1()); // Pose idx,
            size_t offset_trans = num_poses;
            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /(1 * sigmas(0) *  sigmas(0));

            // Add elements for Sigma^z
            for (size_t r = 0; r < d_; r++){
                for (size_t c = 0; c < d_; c++) {
                    triplets.emplace_back(offset_trans + i * d_ + r, offset_trans + i * d_ + c,
                                          precision * V_(r) *
                                          V_(c));
                }
            }
            return triplets;
        }

        // B6^T*B6
        std::vector<Eigen::Triplet<Scalar>> getSigmaLmkBlock(const size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 1;
            triplets.reserve(measurement_stride);
            size_t j = gtsam::symbolIndex(this->key2()); // Landmark idx, starting from 0

            size_t offset_pose = (d_ + 1) * num_poses;
            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /(1 * sigmas(0) *  sigmas(0));
            triplets.emplace_back(offset_pose + j, offset_pose + j, precision);
            return triplets;
        }

        // B4^T*B5
        std::vector<Eigen::Triplet<Scalar>> getTransToRotBlock(const size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = d_;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1()); // Pose idx,
            size_t offset_trans = num_poses;

            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /(1 * sigmas(0) *  sigmas(0));
            for (size_t k = 0; k < d_; k++) {
                triplets.emplace_back(i,
                                      offset_trans + i * d_ + k,
                                      precision * V_(k));
            }

            return triplets;
        }

        // B5^T*B4
        std::vector<Eigen::Triplet<Scalar>> getTransToRotBlockTranspose(const size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = d_;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1()); // Pose idx,
            size_t offset_trans = num_poses;

            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /(1 * sigmas(0) *  sigmas(0));

            for (size_t k = 0; k < d_; k++) {
                triplets.emplace_back(offset_trans + i * d_ + k,
                                      i,
                                      precision * V_(k));
            }
            return triplets;
        }

        //B4^T*B6
        std::vector<Eigen::Triplet<Scalar>> getTransToLmkBlock(const size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 1;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1()); // Pose idx,
            size_t j = gtsam::symbolIndex(this->key2()); // Landmark idx, starting from 0
            size_t offset_pose = (d_ + 1) * num_poses;

            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /(1 * sigmas(0) *  sigmas(0));
            triplets.emplace_back(i, j + offset_pose, -precision);

            return triplets;
        }

        // B6^T*B4
        std::vector<Eigen::Triplet<Scalar>> getTransToLmkBlockTranspose(const size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 1;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1()); // Pose idx,
            size_t j = gtsam::symbolIndex(this->key2()); // Landmark idx, starting from 0
            size_t offset_pose = (d_ + 1) * num_poses;

            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /(1 * sigmas(0) *  sigmas(0));
            triplets.emplace_back( j + offset_pose, i, -precision);

            return triplets;
        }

        // B5^T*B6
        std::vector<Eigen::Triplet<Scalar>> getRotToLmkBlock(const size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = d_;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1()); // Pose idx,
            size_t j = gtsam::symbolIndex(this->key2()); // Landmark idx, starting from 0
            size_t offset_trans = num_poses;
            size_t offset_pose = (d_ + 1) * num_poses;

            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /(1 * sigmas(0) *  sigmas(0));

            for (size_t k = 0; k < d_; k++)
            {
                triplets.emplace_back(offset_trans + i * d_ + k, j  + offset_pose,
                                      -precision * V_(k));
            }

            return triplets;
        }

        // B6^T*B5
        std::vector<Eigen::Triplet<Scalar>> getRotToLmkBlockTranspose(const size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = d_;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1()); // Pose idx,
            size_t j = gtsam::symbolIndex(this->key2()); // Landmark idx, starting from 0
            size_t offset_trans = num_poses;
            size_t offset_pose = (d_ + 1) * num_poses;

            auto diag = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            double precision = 2 /(1 * sigmas(0) *  sigmas(0));

            for (size_t k = 0; k < d_; k++)
            {
                triplets.emplace_back(j  + offset_pose, offset_trans + i * d_ + k,
                                                        -precision * V_(k));
            }

            return triplets;
        }

        size_t countTriplets() const {
            return 4 + 4 * d_ + d_*d_;
        }

        void appendBlocksFromFactor(const std::size_t                     num_poses,
                                    std::vector<Eigen::Triplet<Scalar>>&           triplets)
        {
            const auto& t1 = this->getSigmaTransBlock( );             //
            const auto& t2 = this->getSigmaRotBlock(num_poses);    //
            const auto& t3 = this->getSigmaLmkBlock(num_poses);          //
            const auto& t4 = this->getTransToRotBlock(num_poses);        //
            const auto& t5 = this->getTransToRotBlockTranspose(num_poses);        //
            const auto& t6 = this->getTransToLmkBlock(num_poses);        //
            const auto& t7 = this->getTransToLmkBlockTranspose(num_poses);        //
            const auto& t8 = this->getRotToLmkBlock(num_poses);        //
            const auto& t9 = this->getRotToLmkBlockTranspose(num_poses);        //

            for (const auto* blk : {&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9})
                triplets.insert(triplets.end(), blk->begin(), blk->end());
        }

        // ******************************** Hand-crafted data matrix ****************************************
        // **************************************************************************************************

    };

// Explicit instantiation for d=2 and d=3 in .cpp file:
    using LiftedLandmarkFactor2 = LiftedLandmarkFactor<2>;
    using LiftedLandmarkFactor3 = LiftedLandmarkFactor<3>;

} // namespace gtsam
#endif //STIEFELMANIFOLDEXAMPLE_LANDMARKFACTOR_H
