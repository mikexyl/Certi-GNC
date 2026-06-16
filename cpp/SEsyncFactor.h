/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    SEsyncFactor.h
 * @brief   GTSAM-style lifted relative-pose factor used by the certifiable
 *          PGO / Landmark-SLAM solvers.
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
 *   D. M. Rosen et al. "SE-Sync: A certifiably correct algorithm for
 *   synchronization over the special Euclidean group." IJRR 38(2-3):95-125, 2019.
 */

#ifndef STIEFELMANIFOLDEXAMPLE_SESYNCFACTOR_H
#define STIEFELMANIFOLDEXAMPLE_SESYNCFACTOR_H

#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/make_shared.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/inference/Symbol.h>
#include "LiftedPose.h"
#include <Eigen/Sparse>
#include <boost/optional/optional.hpp>
#include <boost/pointer_cast.hpp>
#include <type_traits>
namespace gtsam {

    template <size_t d>
    class GTSAM_EXPORT SEsyncFactor : public NoiseModelFactorN<LiftedPoseDP, LiftedPoseDP>, public EuclideanFactor {
        Matrix M_;                    ///< measured rotation
        Vector V_;                   ///< measured translation

        size_t p_, d_;               ///< dimensionality constants
        size_t pd_;
//        std::shared_ptr<Matrix> G_; ///< matrix of vectorized tangent space basis generators

        // Select Rot2 or Rot3 interface based template parameter d
        using Rot = typename std::conditional<d == 2, Rot2, Rot3>::type;
        using Trans = typename std::conditional<d == 2, Vector2 , Vector3>::type;

    public:

        // Provide access to the Matrix& version of evaluateError:
        using NoiseModelFactor2<LiftedPoseDP, LiftedPoseDP>::evaluateError;

        /// @name Constructor
        /// @{

        /**
         * @brief Construct a synchronization factor on SE(d) between two lifted poses.
         *
         * This factor encodes the measured relative rotation and translation
         * between two lifted pose variables in the factor graph. It inherits
         * from NoiseModelFactorN<LiftedPoseDP, LiftedPoseDP> to incorporate
         * a Gaussian noise model for the SE(d) measurement.
         *
         * @tparam d       The spatial dimension (2 for 2D, 3 for 3D).
         * @param j1       Key of the first lifted pose variable.
         * @param j2       Key of the second lifted pose variable.
         * @param R12      Measured rotation matrix between the two poses.
         * @param T12      Measured translation vector between the two poses.
         * @param p        Number of columns in the lifted-pose representation.
         * @param model    Shared pointer to the noise model encoding measurement uncertainty.
         *
         * @throws std::invalid_argument if the noise model dimension does not equal d * p + p.
         */
        SEsyncFactor(Key j1, Key j2, const Matrix &R12, const Vector &T12 , size_t p,
                     const SharedNoiseModel &model = nullptr);

        /**
         * @brief Polymorphic copy used by gtsam::NonlinearFactor::cloneWithNewNoiseModel.
         *
         * Required by GTSAM's GncOptimizer, which clones each factor with a
         * re-weighted noise model at every outer iteration. Without this
         * override, the base-class default throws ("Attempting to clone factor
         * with no clone() implemented").
         */
        gtsam::NonlinearFactor::shared_ptr clone() const override {
            return gtsam::make_shared<SEsyncFactor<d>>(*this);
        }

        /// @}
        /// @name Testable
        /// @{

        /**
         * @brief Print a human-readable representation of the SEsyncFactor.
         *
         * Outputs the factor’s identifier including its template parameters,
         * the associated variable keys, the measured rotation matrix, the
         * measured translation vector, and the noise model to standard output.
         *
         * @tparam d  Spatial dimension of the synchronization (2 or 3).
         * @param s             A prefix string printed before the factor information.
         * @param keyFormatter  A callable that formats Keys into strings for display.
         */
        void print(const std::string &s, const KeyFormatter &keyFormatter = DefaultKeyFormatter) const override;

        /**
         * @brief Check equality between this SEsyncFactor and another NonlinearFactor.
         *
         * Performs a type-safe comparison by dynamic casting the input factor to
         * SEsyncFactor<d>. Delegates to the base class’s equals method to compare
         * the noise model and then checks that the dimensionality `p_`, rotation
         * matrix `M_`, and translation vector `V_` match exactly.
         *
         * @tparam d     Spatial dimension of the synchronization (2 or 3).
         * @param expected  The other factor to compare against.
         * @param tol       Numerical tolerance for the base class comparison.
         * @return `true` if `expected` is an SEsyncFactor<d> with the same noise model,
         *         the same `p_`, identical `M_` and `V_`; `false` otherwise.
         */
        bool equals(const NonlinearFactor &expected,
                    double tol = 1e-9) const override;

        /// @}
        /// @name NoiseModelFactorN methods
        /// @{

        /**
         * @brief Compute the error vector for the SE synchronization factor.
         *
         * The error consists of two parts:
         *   1. Rotation residual: vec(Y2) − vec(Y1 * M_)
         *   2. Translation residual: t2 − t1 − Y1 * V_
         * The full error is the concatenation of these two residuals.
         *
         * If requested, the Jacobians H1 and H2 are computed via fillJacobians.
         *
         * @tparam d      Spatial dimension of the synchronization (2 or 3).
         * @param Q1      First lifted pose variable, providing Y1 (Stiefel manifold) and t1 (translation).
         * @param Q2      Second lifted pose variable, providing Y2 and t2.
         * @param H1      Optional pointer to the Jacobian w.r.t. Q1; if non-null, it will be resized
         *                and populated with the partial derivatives of the error w.r.t. Q1.
         * @param H2      Optional pointer to the Jacobian w.r.t. Q2; if non-null, it will be resized
         *                and populated with the partial derivatives of the error w.r.t. Q2.
         * @return        A Vector of length (p_*d + p_) containing the stacked rotation and translation errors.
         *
         * @throws std::invalid_argument if the row dimensions of Y1 or Y2 do not match p_.
         */
        Vector evaluateError(const LiftedPoseDP& P1, const LiftedPoseDP& P2,
                             boost::optional<Matrix&> H1,
                             boost::optional<Matrix&> H2) const override;

        /// @}

//    private:
        /**
         * @brief Compute the Jacobians of the SEsyncFactor error with respect to each lifted pose variable.
         *
         * This method computes and, if requested, populates the Jacobian matrices H1 and H2
         * for the SE synchronization factor. The error consists of both the Stiefel manifold
         * component (rotation) and the translation component. H1 and H2 are resized and filled
         * to represent the partial derivatives of the error with respect to Q1 and Q2, respectively.
         *
         * @tparam d      Spatial dimension of the synchronization (2 for 2D, 3 for 3D).
         * @param Q1      The first lifted pose variable, providing Y1 (Stiefel manifold) and t1 (translation).
         * @param Q2      The second lifted pose variable, providing Y2 and t2.
         * @param H1      Optional pointer to the output Jacobian w.r.t. Q1. If non-null, this matrix
         *                is resized to ((p_*d_ + p_) × (Dim(Y1) + p_)) and populated.
         * @param H2      Optional pointer to the output Jacobian w.r.t. Q2. If non-null, this matrix
         *                is resized to ((p_*d_ + p_) × (Dim(Y2) + p_)) and populated.
         */
        void fillJacobians(const LiftedPoseDP &P1, const LiftedPoseDP &P2,
                           boost::optional<Matrix&> H1,
                           boost::optional<Matrix&> H2) const;

        /**
         * @brief Compute the Euclidean Hessian block in a structured format.
         */
        EuclideanHessianBlock computeEuclideanHessian() const override;


        // The functions below are used as the old had-crafted way to build data matrix, refer to SE-Sync, CPL-SLAM, and CORA paper
        // ******************************** Hand-crafted data matrix ****************************************
        // **************************************************************************************************
        /**
         * @brief Build the sparse Laplacian block entries for the rotation connectivity measurement.
         *
         * Constructs the non-zero entries of the connectivity Laplacian matrix block corresponding
         * to the relative rotation measurement between two poses (key1() → key2()). The returned
         * triplets are offset by `num_poses` to position the block correctly in a larger system matrix.
         *
         * @tparam d            Dimensionality of the rotation component (size of the square matrix M_).
         * @param num_poses     Number of poses in the system; used to offset row and column indices.
         * @return              A vector of `Eigen::Triplet<Scalar>` containing the non-zero
         *                      entries of the Laplacian block in (row, column, value) form.
         *
         * @throws std::runtime_error if the factor’s noise model is not of type `noiseModel::Diagonal`.
         */
        std::vector<Eigen::Triplet<Scalar>> getRotationConnLaplacianBlock(size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 2 * (d + d * d);
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1());
            size_t j = gtsam::symbolIndex(this->key2());
            auto diag   = boost::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector& sigmas = diag->sigmas();
            double sigma_kappa = sigmas(0);
            double kappa  = 2.0 / (sigma_kappa*sigma_kappa);

            // Elements of ith block-diagonal
            for (size_t k = 0; k < d; k++) {
                triplets.emplace_back(num_poses + d * i + k, num_poses + d * i + k,
                                      kappa);
            }
            // Elements of jth block-diagonal
            for (size_t k = 0; k < d; k++) {
                triplets.emplace_back(num_poses + d * j + k, num_poses + d * j + k,
                                      kappa);
            }
            // Elements of ij block
            for (size_t r = 0; r < d; r++) {
                for (size_t c = 0; c < d; c++) {
                    triplets.emplace_back(num_poses + i * d + r, num_poses + j * d + c,
                                          -kappa *
                                          this->M_(r, c));
                }
            }
            // Elements of ji block
            for (size_t r = 0; r < d; r++) {
                for (size_t c = 0; c < d; c++) {
                    triplets.emplace_back(num_poses + j * d + r, num_poses + i * d + c,
                                          -kappa *
                                          this->M_(c, r));
                }
            }
            return triplets;
        }

        /**
         * @brief Build the sparse block entries connecting rotation variables to translation variables.
         *
         * Constructs the non-zero entries for the rotation-to-translation coupling block in the
         * connectivity Laplacian matrix for an SE(d) measurement between two poses. The block is
         * offset by `num_poses` to place it correctly within a larger system matrix.
         *
         * @tparam d           Dimensionality of the rotation component (rows/columns of V_).
         * @param num_poses    Number of poses in the system; used to offset translation indices.
         * @return             A vector of `Eigen::Triplet<Scalar>` containing the non-zero entries
         *                     for the rotation-to-translation block in (row, column, value) form.
         * @throws std::runtime_error if the factor’s noise model is not of type `noiseModel::Diagonal`.
         */
        std::vector<Eigen::Triplet<Scalar>> getRotToTransBlock(size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 2 * d;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1());
            size_t j = gtsam::symbolIndex(this->key2());

            auto diag = boost::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            size_t kappaBlock = static_cast<size_t>(p_ * d_);
            double sigma_tau = sigmas(kappaBlock);        // == √(1/(2 tau))
            double tau = 2.0 / (sigma_tau * sigma_tau);

            for (size_t k = 0; k < d; k++) {
                triplets.emplace_back(i, num_poses + i * d + k,
                                      tau * V_(k));
            }
            for (size_t k = 0; k < d; k++) {
                triplets.emplace_back(j, num_poses + i * d + k,
                                      -tau * V_(k));
            }
            return triplets;
        }

        /**
         * @brief Build the transpose of the rotation-to-translation coupling block.
         *
         * Constructs the non-zero entries for the transpose of the rotation-to-translation
         * coupling submatrix in the connectivity Laplacian for an SE(d) measurement
         * between two poses. The entries are offset by `num_poses` to locate them
         * correctly in a larger global matrix.
         *
         * @tparam d           Dimensionality of the rotation component (2 for 2D, 3 for 3D).
         * @param num_poses    Total number of poses; used to offset translation indices.
         * @return             A vector of `Eigen::Triplet<Scalar>` containing the non-zero
         *                     entries of the transposed rotation-to-translation block in
         *                     (row, column, value) form.
         *
         * @throws std::runtime_error if the factor’s noise model is not a Diagonal model.
         */
        std::vector<Eigen::Triplet<Scalar>> getRotToTransBlockTranspose(size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 2 * d;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1());
            size_t j = gtsam::symbolIndex(this->key2());

            auto diag = boost::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            size_t kappaBlock = static_cast<size_t>(p_ * d_);
            double sigma_tau = sigmas(kappaBlock);        // == √(1/(2 tau))
            double tau = 2.0 / (sigma_tau * sigma_tau);

            for (size_t k = 0; k < d; k++) {
                triplets.emplace_back(num_poses + i * d + k, i,
                                      tau * V_(k));
            }
            for (size_t k = 0; k < d; k++) {
                triplets.emplace_back(num_poses + i * d + k, j,
                                      -tau * V_(k));
            }
            return triplets;
        }

        /**
        * @brief Build the sparse Laplacian block for the translation connectivity measurement.
        *
        * Constructs the four non-zero entries of the translation connectivity Laplacian block
        * for the relative translation measurement between two poses identified by key1() and key2().
        * The precision τ is extracted from the Diagonal noise model’s sigmas.
        *
        * @return A std::vector of Eigen::Triplet<Scalar>;
        *
        * @throws std::runtime_error if the factor’s noise model is not of type noiseModel::Diagonal.
        *
        */
        std::vector<Eigen::Triplet<Scalar>> getTranslationConnLaplacianBlock() {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 4;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1());
            size_t j = gtsam::symbolIndex(this->key2());

            auto diag = boost::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            size_t kappaBlock = static_cast<size_t>(p_ * d_);
            double sigma_tau = sigmas(kappaBlock);        // == √(1/(2 tau))
            double tau = 2.0 / (sigma_tau * sigma_tau);

            triplets.emplace_back(i, i, tau);
            triplets.emplace_back(j, j, tau);
            triplets.emplace_back(i, j, -tau);
            triplets.emplace_back(j, i, -tau);

            return triplets;
        }

        /**
         * @brief Build the sparse Omega block for the rotation connectivity measurement.
         *
         * Constructs a d×d block of the connectivity Laplacian corresponding to the
         * rotation component of an SE(d) measurement between two poses. Each entry
         * at (r, c) is computed as τ * V_(r) * V_(c), where τ is the precision derived
         * from the translation noise model.
         *
         * @tparam d           Dimensionality of the rotation component (2 for Rot2, 3 for Rot3).
         * @param num_poses    Total number of poses in the system; used as an offset for global indexing.
         * @return             A vector of `Eigen::Triplet<Scalar>` containing the non-zero
         *                     entries of the rotation Omega block in (row, column, value) form.
         * @throws std::runtime_error if the factor’s noise model is not of type `noiseModel::Diagonal`.
         */
        std::vector<Eigen::Triplet<Scalar>> getRotationOmegaBlock(size_t num_poses) {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = d * d;
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1());

            auto diag = boost::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            if (!diag) {
                throw std::runtime_error("Expected a Diagonal noise model");
            }
            const Vector &sigmas = diag->sigmas();          // length = p*d + p
            size_t kappaBlock = static_cast<size_t>(p_ * d_);
            double sigma_tau = sigmas(kappaBlock);        // == √(1/(2 tau))
            double tau = 2.0 / (sigma_tau * sigma_tau);

            for (size_t r = 0; r < d; r++) {
                for (size_t c = 0; c < d; c++) {
                    triplets.emplace_back(num_poses + i * d + r, num_poses + i * d + c,
                                          tau * V_(r) *
                                          V_(c));
                }
            }
            return triplets;
        }

        size_t countTriplets() const {
            return 3 * d_ * d_ + 6 * d_ + 4;
        }

        /**
         * @brief Append all connectivity and coupling blocks for this SE synchronization factor.
         *
         * This method gathers the five constituent blocks that represent the factor’s contribution
         * to the global system matrix—namely:
         *   1. Rotation connectivity Laplacian (R-L)
         *   2. Rotation-to-translation coupling (R→T)
         *   3. Its transpose (T→R)
         *   4. Rotation Omega block (ΩR)
         *   5. Translation connectivity Laplacian (T-L)
         * and appends their non-zero entries to the supplied `triplets` vector.
         *
         * @tparam d           Spatial dimension of the synchronization (2 for 2D, 3 for 3D).
         * @param num_poses    Total number of poses in the system; used to offset row/column indices.
         * @param triplets     Reference to the vector of `Eigen::Triplet<Scalar>` where entries
         *                     will be appended.
         */
        void appendBlocksFromFactor(std::size_t                     num_poses,
                                    std::vector<Eigen::Triplet<Scalar>>&           triplets)
        {
            const auto& t1 = this->getRotationConnLaplacianBlock(num_poses);  // R-L
            const auto& t2 = this->getRotToTransBlock(num_poses);             // R→T
            const auto& t3 = this->getRotToTransBlockTranspose(num_poses);    // T→R
            const auto& t4 = this->getRotationOmegaBlock(num_poses);          // ΩR
            const auto& t5 = this->getTranslationConnLaplacianBlock();        // T-L

            for (const auto* blk : { &t1, &t2, &t3, &t4, &t5 })
                triplets.insert(triplets.end(), blk->begin(), blk->end());
        }

        // ******************************** Hand-crafted data matrix ****************************************
        // **************************************************************************************************

    };

// Explicit instantiation for d=2 and d=3 in .cpp file:
    using SEsyncFactor2 = SEsyncFactor<2>;
    using SEsyncFactor3 = SEsyncFactor<3>;

} // namespace gtsam




#endif //STIEFELMANIFOLDEXAMPLE_SESYNCFACTOR_H
