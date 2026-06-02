/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    Certifiable_problem.h
 * @brief   Base class of Riemannian-staircase implementation over factor graphs.
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
 *   David M. Rosen. "Scalable low-rank semidefinite programming for
 *   certifiably correct machine perception." Proc. Workshop on the
 *   Algorithmic Foundations of Robotics (WAFR), 2020.
 *
 * Background on the Riemannian staircase framework:
 *   N. Boumal, V. Voroninski, A. Bandeira. "The non-convex Burer-Monteiro
 *   approach works on smooth semidefinite programs." NeurIPS, 2016.
 *   F. Dellaert et al. "Shonan rotation averaging: Global optimality by
 *   surfing SO(p)^n." ECCV, 2020.
 *
 */

#ifndef STIEFELMANIFOLDEXAMPLE_CERTIFIABLE_PROBLEM_H
#define STIEFELMANIFOLDEXAMPLE_CERTIFIABLE_PROBLEM_H

#include "utils.h"
#include "CertifiableResults.h"
#include "CertifiableProblemOpts.h"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "Optimization/LinearAlgebra/LOBPCG.h"
#include "Optimization/Riemannian/TNT.h"
#include "ILDL/ILDL.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>

#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <map>
#include <vector>
#include <omp.h>
#include <cstddef>


namespace gtsam
{
    /**
     * @brief Abstract base for certifiable estimation problems.
     *
     * Defines the core interface for building factor graphs at different
     * relaxation levels, computing certificate matrices, and exporting results.
     */
    class CertifiableProblem {
    protected:
        /// Current factor graph at the active relaxation level (p).
        NonlinearFactorGraph currentGraph_;

        /// Current GTSAM Values: either the initial guess or the latest optimized result.
        Values currentValues_;

        /// Active relaxation rank p (dimension of the lifted space).
        size_t currentRank_ = 0;

        /// Number of poses in the SLAM/problem instance.
        size_t num_pose_ = 0;

        /// Ambient dimension of each pose (e.g., 2 for SE2, 3 for SE3).
        size_t d_ = 0;

        /// Global ordering of variables.
        Ordering ordering_;

        /// Global layout of variables in the data matrix.
        GlobalLayout layout_;

        /// Data or certificate matrix M (used for gradient, verification, etc.).
        SparseMatrix M_;

        /// Stacked variable matrix Y (size d × p·num_pose_), used in certificate computations.
        SparseMatrix Y_;

        /// All parsed measurements (e.g., relative poses, landmark observations).
        DataParser::Measurement measurements_;

        /// Container for storing certificate evaluation results (e.g., SDP values).
        CertificateResults certificateResults_;

        /// Indices (into currentGraph_) of pose-pose factors that are odometry
        /// (i.e., between consecutive poses). Populated by PGO/Landmark init().
        gtsam::FastVector<uint64_t> OdomEdgeList_;

        /// Optional per-factor weights applied when (re)building the graph.
        /// Empty vector means "no weights" (i.e., all weights = 1). Size, when
        /// non-empty, must equal the number of measurements in the order they
        /// are added to the graph.
        Vector currentWeights_;

        std::string dataPath;

    public:
        /// Options and parameters for certifiable‐problem routines (LM tolerances, max iterations).
        CertifiableProblemOpts opts_;

        /**
          * @brief Construct a certifiable problem instance.
          * @param d            Ambient dimension of the manifold (e.g., 2 or 3).
          * @param p            Initial rank / relaxation level.
          * @param measurements Parsed measurements (must contain num_poses, etc.).
          */
        CertifiableProblem(size_t d, size_t p, const DataParser::Measurement &measurements)
            : currentRank_(p),
              num_pose_(measurements.num_poses),
              d_(d),
              measurements_(measurements), certificateResults_() {
        }

        // Virtual Functions

        /**
          * @brief Virtual destructor.
          */
        virtual ~CertifiableProblem() = default;

        /**
           * @brief Build the nonlinear factor graph at relaxation level p.
           * @param p  Relaxation level (rank) at which to assemble the graph.
           * @return   A GTSAM NonlinearFactorGraph ready for optimization.
           */
        virtual NonlinearFactorGraph buildGraphAtLevel(size_t p) = 0;

        /**
          * @brief Compute the block‑diagonal certificate matrix Λ at level p.
          * @param Y  Stacked variable matrix
          * @return   Dense block matrix Λ_blocks, to be turned into a sparse Λ.
          */
        virtual Matrix computeLambdaBlocks(const Matrix &Y) = 0;

        /**
          * @brief Convert block‑diagonal Λ_blocks into a sparse Λ matrix.
          * @param LambdaBlocks  Dense block representation of Λ.
          * @return              Sparse certificate matrix Λ.
          */
        virtual SparseMatrix computeLambdaFromLambdaBlocks(const Matrix &LambdaBlocks) = 0;

        /**
         * @brief Recover the data matrix using the modular DataMatrixAssembler and global layout.
         */
        virtual SparseMatrix recoverDataMatrixFromAssembler() {
            return DataMatrixAssembler::Assemble(currentGraph_, layout_);
        }

        /**
          * @brief Create the element‑wise sparse matrix from variable Values.
          * @param values  Current GTSAM Values containing one or more variables.
          * @return        Sparse matrix S assembled from those variable blocks.
          */
        virtual SparseMatrix elementMatrix(const Values &values) = 0;

        /**
          * @brief Generate a random initialization of all variables at level Pmin.
          * @param Pmin  relaxation level required.
          * @return      GTSAM Values containing randomized starting points.
          */
        virtual Values randomInitAtLevelP(const size_t Pmin) = 0;

        /**
          * @brief Map a flat vector v into a GTSAM VectorValues on the tangent space at Y.
          * @param p       Relaxation level.
          * @param v       Flat vector
          * @param values  Current basepoint Values for linearization.
          * @return        VectorValues representing tangent vectors for each variable.
          */
        virtual VectorValues TangentVectorValues(size_t p, const Vector& v, const Values& values) = 0;

        /**
          * @brief Project ambient-space variations Ydot back to the tangent space at Y.
          * @param p     Relaxation level.
          * @param Y     Basepoint variable matrix.
          * @param Ydot  Ambient-space variation matrix.
          * @return      Tangent‑space projection of Ydot at Y.
          */
        virtual Matrix tangent_space_projection(const size_t p, const Matrix &Y,
                                        const Matrix &Ydot) = 0;

        /**
          * @brief Round solution from p back down to problem dimension (2D or 3D)
          * @return   Matrix containing the rounded solution.
          */
        virtual Matrix RoundSolutionS() = 0;

        /**
          * @brief Export results in either G2O or TUM format.
          * @param path  Base filename (without extension).
          * @param R     Element matrix to export.
          * @param g2o   If true, write G2O; otherwise, write TUM.
          */
        virtual void ExportData(const std::string &path, const Matrix &R, bool g2o) = 0;

        // Accessors

        /**
         * @brief Get the number of poses in the problem.
         * @return The number of poses (num_pose_).
         */
        size_t getNumPoses() const { return num_pose_; }

        /**
         * @brief Get the ambient dimension of the problem.
         * @return The dimension d_ (e.g., 2 or 3).
         */
        size_t getD() const { return d_; }

        /**
         * @brief Retrieve the results of the problem.
         * @return A CertificateResults struct summarizing metrics.
         */
        CertificateResults getCertificateResults() const { return certificateResults_; }

        /**
          * @brief Optimize the factor graph at a given relaxation level using Levenberg-Marquardt.
          *
          * Constructs the graph for level p, configures optimizer parameters from opts_,
          * runs the optimization starting from currentValues_, and records the final error.
          *
          * @param[in] p Relaxation level (rank) at which to assemble and optimize the graph.
          * @return GTSAM Values containing the optimized variable estimates.
          */
        Values tryOptimizingAtLevel(size_t p) {
            // Build the problem graph at this relaxation level.
            NonlinearFactorGraph graph = buildGraphAtLevel(p);

            // Copy and configure Levenberg–Marquardt parameters.
            auto lmParams = opts_.lmParams;
            lmParams.maxIterations    = opts_.maxIterations;
            lmParams.relativeErrorTol = opts_.relativeErrorTol;
            lmParams.absoluteErrorTol = opts_.absoluteErrorTol;
            lmParams.verbosityLM      = opts_.verbosityLM;
            lmParams.lambdaUpperBound = std::numeric_limits<double>::max();
            // Create and run the optimizer.
            auto lm = std::make_shared<LevenbergMarquardtOptimizer>(graph, currentValues_, lmParams);
            auto t0 = CFGStopwatch::tick();
            auto result = lm->optimize();
            auto t1 = CFGStopwatch::tock(t0);
            certificateResults_.elapsed_optimization_times.push_back(t1);
            certificateResults_.total_computation_time += t1;
            // Record the final LM error for certificate evaluation.
            certificateResults_.SDPval.push_back(lm->error());

            return result;
        }


        /**
          * @brief Compute the optimization cost (sum of squared errors) at a given relaxation level.
          *
          * This evaluates the current factor graph built for level \p p on the provided \p values,
          * without performing any optimization.
          *
          * @param p       Relaxation level (rank) at which to assemble the graph.
          * @param values  GTSAM Values containing variable assignments to evaluate.
          * @return        The total error (sum of squared residuals) of the graph at these values.
          */
        double costAtLevel(size_t p, const Values &values) {
            // Assemble the factor graph for level p
            const NonlinearFactorGraph graph = buildGraphAtLevel(p);

            // Evaluate and return the error of the graph on the given values
            return graph.error(values);
        }

        /**
         * @brief Initialize variable estimates at the next relaxation level via line search along the minimal-eigenvalue descent direction.
         *
         * Starting from the current estimates at level p-1, this routine "lifts" them to level p by moving
         * along the Riemannian descent direction given by the smallest eigenvector of the certificate matrix. 
         * This is known as saddle escape in certifiable estimation literature.
         *
         * @param[in] values Current GTSAM Values at level (currentRank_ - 1).
         * @param[in] minEigenVector Vector corresponding to the minimal eigenvalue (descent direction).
         * @param[in] minEigenValue The minimal eigenvalue.
         * @param[in] gradientTolerance Threshold on the norm of the Riemannian gradient for accepting a step.
         * @return New GTSAM Values at level currentRank_, lifted along the descent direction.
         */
        Values initializeWithDescentDirection(const Values &values,
                                              const Vector &minEigenVector,
                                              double minEigenValue,
                                              double gradientTolerance) {
            double funcVal = costAtLevel(currentRank_ - 1, values);
            double alphaMin = 1e-2;
            double alpha = std::max(1024 * alphaMin, 10 * gradientTolerance / fabs(minEigenValue));

            std::vector<double> alphas;
            std::vector<double> fvals;

            // Backtracking line search along descent direction
            while (alpha >= alphaMin) {
                Values Qplus = liftWithDescent(currentRank_, values, alpha * minEigenVector);
                double funcValTest = costAtLevel(currentRank_, Qplus);

                // Compute Riemannian gradient at Qplus
                Matrix Ytest          = elementMatrix(Qplus);
                Matrix Y_eucl_grad    = euclideanGradient(Ytest);
                Matrix gradTest       = riemannianGradient(currentRank_, Ytest, Y_eucl_grad);
                double gradTestNorm   = gradTest.norm();
                alphas.push_back(alpha);
                fvals.push_back(funcValTest);
                // Accept step if cost decreased and gradient norm is above tolerance
                if (funcVal > funcValTest && gradTestNorm > gradientTolerance) {
                    return Qplus;
                }
                alpha /= 2.0;
            }

            // If no acceptable step was found, pick the one with minimal cost
            auto fminIter = std::min_element(fvals.begin(), fvals.end());
            size_t minIdx = std::distance(fvals.begin(), fminIter);
            double aMin   = alphas[minIdx];

            if (fvals[minIdx] < funcVal) {
                return liftWithDescent(currentRank_, values, aMin * minEigenVector);
            }
            
            // Fallback: take the smallest step
            return liftWithDescent(currentRank_, values, alpha * minEigenVector);
        }

        /**
         * @brief Lift all variables in a GTSAM Values container to a higher relaxation level.
         *
         * @param p       Target relaxation level (rank) to lift variables to.
         * @param values  Input GTSAM Values containing variables at level (p−1).
         * @return        New Values container with all variables lifted to level \p p.
         */
        virtual Values LiftTo(size_t p, const Values& values) = 0;

        /**
         * @brief Lift variables to the next relaxation level and take a descent step along the minimal‐eigenvalue direction.
         *
         * First lifts the current \p values from level (p−1) to level p, then computes
         * the tangent‐space displacement corresponding to the descent direction minEigenVector,
         * and finally retracts back onto the manifold.
         *
         * @param p                 Target relaxation level to lift to.
         * @param values            Input GTSAM Values at level (p−1).
         * @param minEigenVector    Flattened minimal eigenvector of the certificate matrix, defining the descent direction.
         * @return                  New GTSAM Values at level \p p, after applying the descent retraction.
         */
        Values liftWithDescent(size_t p,
                               const Values &values,
                               const Vector &minEigenVector) {
            // Lift values from level (p−1) to level p
            Values lifted = LiftTo(p, values);

            // Convert the flat descent vector into per‐variable tangent displacements
            VectorValues delta = TangentVectorValues(p, minEigenVector, lifted);

            // Retract along the tangent directions back onto the manifold
            return lifted.retract(delta);
        }


        /**
         * @brief Compute the Riemannian gradient by projecting the Euclidean gradient onto the tangent space.
         *
         * @param[in] p Relaxation rank.
         * @param[in] Y Current point on the manifold.
         * @param[in] NablaF_Y Ambient-space (Euclidean) gradient of the objective at Y.
         * @return The Riemannian gradient.
         */
        Matrix riemannianGradient(const size_t p,
                                  const Matrix &Y,
                                  const Matrix &NablaF_Y) {
            return tangent_space_projection(p, Y, NablaF_Y);
        }


        /**
         * @brief Compute the ambient-space (Euclidean) gradient of the objective at Y.
         *
         * Using the current data/certificate matrix M_, this returns the Euclidean
         * gradient \f$ \nabla F(Y) = M Y \f$.
         *
         * @param[in] Y Current variable matrix of size (d × p·num_poses).
         * @return Matrix of the same dimensions representing the ambient-space gradient.
         */
        Matrix euclideanGradient(const Matrix &Y) {
            return M_ * Y;
        }

        /**
         * @brief Quickly verify the certificate matrix for negative curvature or positive semidefiniteness.
         *
         * Attempts a direct Cholesky factorization of the regularized certificate matrix M = S + η·I.
         * If M is not positive semidefinite, performs a (preconditioned) LOBPCG eigenpair search
         * to find a direction of sufficiently negative curvature.
         *
         * @param S              Input sparse certificate matrix.
         * @param eta            Regularization parameter (added to the diagonal).
         * @param nx             Number of eigenpairs to compute (typically 1).
         * @param[out] theta     On output, the Rayleigh quotient xᵀ·S·x of the found eigenvector.
         * @param[out] x         On output, the eigenvector corresponding to the minimal eigenvalue.
         * @param[out] num_iters Number of LOBPCG iterations performed.
         * @param max_iters      Maximum allowed LOBPCG iterations.
         * @param max_fill_factor Maximum fill factor for ILDL preconditioner.
         * @param drop_tol       Drop tolerance for ILDL preconditioner.
         * @return               True if M is positive semidefinite (no negative curvature found),
         *                       false otherwise (a negative curvature direction was found and returned).
         */
        static bool fast_verification(const SparseMatrix &S, Scalar eta, size_t nx,
                                      Scalar &theta, Vector &x, size_t &num_iters,
                                      size_t max_iters, Scalar max_fill_factor,
                                      Scalar drop_tol) {
            num_iters = 0;
            theta = 0;
            unsigned int n = S.rows();

            /// STEP 1:  Test positive-semidefiniteness of regularized certificate matrix
            /// M := S + eta * Id via direct factorization

            SparseMatrix Id(n, n);
            Id.setIdentity();
            SparseMatrix M = S + eta * Id;

            /// Test positive-semidefiniteness via direct Cholesky factorization
            Eigen::CholmodSupernodalLLT<SparseMatrix> MChol;
            /// Set various options for the factorization

            // Bail out early if non-positive-semidefiniteness is detected
            MChol.cholmod().quick_return_if_not_posdef = 1;

            // We know that we might be handling a non-PSD matrix, so suppress Cholmod's
            // printed error output
            MChol.cholmod().print = 0;

            // Calculate Cholesky decomposition!
            MChol.compute(M);

            // Test whether the Cholesky decomposition succeeded
            bool PSD = (MChol.info() == Eigen::Success);

            if (!PSD) {

                /// If control reaches here, then lambda_min(S) < -eta, so we must compute
                /// an approximate minimum eigenpair using LOBPCG

                Vector Theta; // Vector to hold Ritz values of S
                Matrix X;     // Matrix to hold eigenvector estimates for S
                size_t num_converged;

                /// Set up matrix-vector multiplication operator with regularized
                /// certificate matrix M

                // Matrix-vector multiplication with regularized certificate matrix M
                Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix> Mop =
                        [&M](const Matrix &X) -> Matrix { return M * X; };

                // Custom stopping criterion: terminate as soon as a direction of
                // sufficiently negative curvature is found:
                //
                // x'* S * x < - eta / 2
                //
                Optimization::LinearAlgebra::LOBPCGUserFunction<Vector, Matrix> stopfun =
                        [&S,
                                eta](size_t i,
                                     const Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>
                                     &M,
                                     const std::optional<
                                             Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>>
                                     &B,
                                     const std::optional<
                                             Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>>
                                     &T,
                                     size_t nev, const Vector &Theta, const Matrix &X, const Vector &r,
                                     size_t nc) {
                            // Calculate curvature along estimated minimum eigenvector X0
                            Scalar theta = X.col(0).dot(S * X.col(0));
                            return (theta < -eta / 2);
                        };

                // STEP 2: try unpreconditioned LOBPCG first (15% of the
                // iteration budget). Catches well-separated negative eigenpairs
                // without paying for the preconditioner build.
                double unprecon_iter_frac = .15;
                std::tie(Theta, X) = Optimization::LinearAlgebra::LOBPCG<Vector, Matrix>(
                        Mop,
                        std::optional<
                                Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>>(),
                        std::optional<
                                Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>>(),
                        n, nx, 1, static_cast<size_t>(unprecon_iter_frac * max_iters),
                        num_iters, num_converged, 0.0,
                        std::optional<
                                Optimization::LinearAlgebra::LOBPCGUserFunction<Vector, Matrix>>(
                                stopfun));

                // Extract eigenvector estimate
                x = X.col(0);

                // Calculate curvature along x
                theta = x.dot(S * x);

                if (!(theta < -eta / 2)) {
                    // STEP 3: preconditioned LOBPCG for the small-magnitude
                    // negative-eigenpair case. Build ILDL preconditioner of M.
                    Preconditioners::ILDLOpts ildl_opts;
                    ildl_opts.max_fill_factor = max_fill_factor;
                    ildl_opts.drop_tol = drop_tol;

                    Preconditioners::ILDL Mfact(M, ildl_opts);

                    Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix> T =
                            [&Mfact](const Matrix &X) -> Matrix {
                                // Preallocate output matrix TX
                                Matrix TX(X.rows(), X.cols());

                                for (unsigned int i = 0; i < X.cols(); ++i) {
                                    // Calculate TX by preconditioning the columns of X one-by-one
                                    TX.col(i) = Mfact.solve(X.col(i), true);
                                }

                                return TX;
                            };

                    /// Run preconditioned LOBPCG using the remaining alloted LOBPCG
                    /// iterations
                    std::tie(Theta, X) = Optimization::LinearAlgebra::LOBPCG<Vector, Matrix>(
                            Mop,
                            std::optional<
                                    Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>>(),
                            std::optional<
                                    Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>>(T),
                            n, nx, 1, static_cast<size_t>((1.0 - unprecon_iter_frac) * max_iters),
                            num_iters, num_converged, 0.0,
                            std::optional<
                                    Optimization::LinearAlgebra::LOBPCGUserFunction<Vector, Matrix>>(
                                    stopfun));

                    // Extract eigenvector estimate
                    x = X.col(0);

                    // Calculate curvature along x
                    theta = x.dot(S * x);

                    num_iters += static_cast<size_t>(unprecon_iter_frac * num_iters);
                } // if (!(theta < -eta / 2))

            } // if(!PSD)
            return PSD;
        }

        /**
         * @brief Project a square matrix onto the Special Orthogonal group SO(d)(SVD-based).
         *
         * Computes the singular value decomposition M = U·Σ·Vᵀ and returns R = U·Vᵀ.
         * If det(U·Vᵀ) is negative, flips the sign of the last column of U
         * so that det(R) = +1, ensuring R ∈ SO(d).
         *
         * @param M  Input square matrix to be projected.
         * @return   The closest rotation matrix R ∈ SO(d) to M under the Frobenius norm.
         */
        Matrix project_to_SOd(const Matrix &M) {
            // Compute the full SVD of M.
            Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

            // Determine the determinants of U and V to check overall orientation.
            Scalar detU = svd.matrixU().determinant();
            Scalar detV = svd.matrixV().determinant();

            // If U and V have the same orientation (det(U·Vᵀ) = +1), return U·Vᵀ directly.
            if (detU * detV > 0) {
                return svd.matrixU() * svd.matrixV().transpose();
            } else {
                // Otherwise, flip the sign of the last column of U to correct orientation.
                Matrix Uprime = svd.matrixU();
                Uprime.col(Uprime.cols() - 1) *= -1;
                return Uprime * svd.matrixV().transpose();
            }
        }

        /**
         * @brief Get the current relaxation rank.
         * @return The current rank p used for lifting and optimization.
         */
        inline size_t currentRank() const noexcept { return currentRank_; }

        /**
         * @brief Set the current relaxation rank.
         * @param rank  New rank p to use for subsequent operations.
         */
        inline void setCurrentRank(size_t rank) noexcept { currentRank_ = rank; }

        /**
         * @brief Increment the current relaxation rank by a given amount.
         * @param delta  Amount to increase the rank by (default is 1).
         */
        inline void increaseCurrentRank(size_t delta = 1) noexcept { currentRank_ += delta; }

        /**
         * @brief Update the stored currentValues_ for optimization initialization.
         * @param values  GTSAM Values to use as the current starting estimate.
         */
        inline void setCurrentValues(const Values &values) noexcept { currentValues_ = values; }

        /**
         * @brief Read-only access to the current values (last optimization result or initial guess).
         */
        inline const Values& getCurrentValues() const noexcept { return currentValues_; }

        /**
         * @brief Update the stored factor graph for the current rank.
         * @param graph  NonlinearFactorGraph to use for the next optimization.
         */
        inline void setCurrentGraph(const NonlinearFactorGraph &graph) noexcept { currentGraph_ = graph; }

        /**
         * @brief Read-only access to the current factor graph at the active rank.
         */
        inline const NonlinearFactorGraph& getCurrentGraph() const noexcept { return currentGraph_; }

        /**
         * @brief Indices of pose-pose factors that are odometry (consecutive poses).
         *
         * These factors are typically forced to be inliers by the GNC wrapper.
         */
        inline const gtsam::FastVector<uint64_t>& getOdometryPoseEdgeIds() const noexcept { return OdomEdgeList_; }

        /**
         * @brief Set per-factor weights to apply when (re)building the graph.
         * @param w  Weight per measurement, in the same order as they are emplaced
         *           into the graph (pose factors first, then landmark factors).
         *           Pass an empty vector to clear weights (i.e., use unit weights).
         */
        inline void setCurrentWeights(const Vector &w) noexcept { currentWeights_ = w; }

        /**
         * @brief Get the currently-applied per-factor weights (empty if none).
         */
        inline const Vector& getCurrentWeights() const noexcept { return currentWeights_; }

        /**
         * @brief Clear any per-factor weights (subsequent buildGraphAtLevel() uses unit weights).
         */
        inline void clearCurrentWeights() noexcept { currentWeights_.resize(0); }

        /**
         * @brief Update the data/certificate matrix M_.
         * @param M  SparseMatrix representing the current data or certificate matrix.
         */
        inline void setDataMatrix(const SparseMatrix &M) noexcept { M_ = M; }

        /**
         * @brief Retrieve the current data/certificate matrix.
         * @return The SparseMatrix M_ used in gradient and projection computations.
         */
        inline SparseMatrix getDataMatrix() const noexcept { return M_; }
    };

}



#endif //STIEFELMANIFOLDEXAMPLE_CERTIFIABLE_PROBLEM_H
