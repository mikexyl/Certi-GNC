/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    CertifiablePGO.h
 * @brief   Certifiable Pose Graph Optimization (PGO) via the Riemannian staircase.
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
 * For a related hand-crafted certifiable PGO solver, see:
 *   D. M. Rosen et al. "SE-Sync: A certifiably correct algorithm for
 *   synchronization over the special Euclidean group." IJRR 38(2-3):95-125, 2019.
 *
 */

#ifndef STIEFELMANIFOLDEXAMPLE_CERTIFIABLEPGO_H
#define STIEFELMANIFOLDEXAMPLE_CERTIFIABLEPGO_H

#include "Certifiable_problem.h"
#include "RandomInit.h"
#include "SEsyncFactor.h"

namespace gtsam {

/**
 * @brief Certifiable Pose Graph Optimization (PGO)
 *
 * Implements a hierarchy of relaxations of PGO via rank‐d Stiefel manifolds,
 * certifies solutions using eigenvalue tests, and refines by lifting along
 * descent directions when necessary.
 *
 * @tparam d  Ambient pose dimension (must be 2 or 3).
 */
    template<size_t d>
    class CertifiablePGO : public CertifiableProblem {
    BlockOrderingType orderingType_ = BlockOrderingType::Generic;

        static_assert(d == 2 || d == 3, "CertifiablePGO only supports d = 2 or 3.");
    public:
        Values LiftTo(size_t p, const Values& values) override {
            Values result;
            // Lift all Pose variables of type LiftedPoseDP
            for (const auto& it : values.extract<LiftedPoseDP>()) {
                result.insert(it.first, it.second.LiftTo(p));
            }
            return result;
        }

        /**
         * @brief Construct with initial rank and measurement data.
         * @param p             Initial relaxation rank.
         * @param measurements  Parsed measurement struct (num_poses, etc.).
         */
        CertifiablePGO(size_t p, const DataParser::Measurement &measurements)
                : CertifiableProblem(d, p, measurements) {
            certificateResults_.startingRank = p;
        }

        void setBlockOrderingType(BlockOrderingType type) { orderingType_ = type; }

        /**
         * @brief Initialize graph, data matrix, and random/odom values; record init time.
         *
         * Also populates OdomEdgeList_ with the indices of pose-pose factors that
         * correspond to odometry (consecutive-pose) measurements. These indices
         * are used by the GNC wrapper to mark known-inlier edges.
         */
        void init() {
            auto t0 = CFGStopwatch::tick();
            if (opts_.initType == CertifiableProblemOpts::InitType::Random) {
                currentValues_ = randomInitAtLevelP(currentRank_);
            } else if (opts_.initType == CertifiableProblemOpts::InitType::Odom) {
                currentValues_ = odomInitAtLevelP(currentRank_);
            } else {
                throw std::runtime_error("Unknown initialization type in CertifiableProblemOpts::initType");
            }
            // Build graph at initial level
            currentGraph_ = buildGraphAtLevel(currentRank_);

            // Mark odometry edges: edge (i, j) with j == i + 1 is an odometry edge.
            OdomEdgeList_.clear();
            OdomEdgeList_.reserve(measurements_.poseMeasurements.size());
            for (size_t k = 0; k < measurements_.poseMeasurements.size(); ++k) {
                const auto& m = measurements_.poseMeasurements[k];
                if (m.j == m.i + 1) {
                    OdomEdgeList_.push_back(k);
                }
            }

            // Initialize global ordering and layout
            ordering_ = Ordering::Colamd(currentGraph_);
            layout_ = GlobalLayout(ordering_, d, orderingType_);

            M_ = recoverDataMatrixFromAssembler();
            auto t1 = CFGStopwatch::tock(t0);
            certificateResults_.initialization_time.push_back(t1);
            certificateResults_.total_computation_time += t1;
        }

        /**
         * @brief Initialize all pose variables at level Pmin from g2o initial guesses.
         *
         * For each pose i, takes (R_i, t_i) from measurements_.initial_poses[{'A', i}]
         * if present, lifts the rotation to St(p, d) by zero-padding to a p x d matrix,
         * and lifts the translation to R^p by zero-padding. Poses without an initial
         * guess fall back to a small random perturbation around identity to keep the
         * Stiefel constraint well-defined.
         *
         * @param Pmin  Target relaxation rank.
         * @return      GTSAM Values container with LiftedPoseDP entries.
         */
        Values odomInitAtLevelP(const size_t Pmin) {
            Values initial;
            for (size_t j = 0; j < num_pose_; ++j) {
                auto it = measurements_.initial_poses.find({'A', j});
                if (it != measurements_.initial_poses.end()) {
                    const auto& iv = it->second;
                    // Lift the d x d rotation matrix into a p x d Stiefel element by zero-padding.
                    StiefelManifoldKP Y = StiefelManifoldKP::Lift(Pmin, iv.R);
                    Vector trans = LiftedPoseDP::LiftToRp(iv.t, Pmin);
                    initial.insert(j, LiftedPoseDP(Y, trans));
                } else {
                    // Fallback: random init for any pose missing an initial guess.
                    StiefelManifoldKP Y =
                            StiefelManifoldKP::Random(std::default_random_engine::default_seed, d, Pmin);
                    Vector trans = Vector::Random(Pmin);
                    initial.insert(j, LiftedPoseDP(Y, trans));
                }
            }
            return initial;
        }

        /**
         * @brief Assemble the factor graph at relaxation level p.
         *
         * If currentWeights_ is non-empty, scales each factor's precision
         * (kappa, tau) by the corresponding weight. weight = 0 effectively
         * removes the factor; weight = 1 leaves it unchanged.
         *
         * @param p  Relaxation rank.
         * @return   Factor graph containing only pose-to-pose factors.
         */
        NonlinearFactorGraph buildGraphAtLevel(size_t p) override {
            NonlinearFactorGraph inputGraph;
            const bool hasW = (currentWeights_.size() ==
                               static_cast<Eigen::Index>(measurements_.poseMeasurements.size()));
            for (size_t k = 0; k < measurements_.poseMeasurements.size(); ++k) {
                const auto &meas = measurements_.poseMeasurements[k];
                double w = hasW ? currentWeights_(k) : 1.0;
                // Numeric floor on weights so we never construct a zero-sigma noise model.
                if (w < 1e-12) w = 1e-12;

                // Construct diagonal noise sigmas from weighted kappa and tau.
                Vector sigmas = Vector::Zero(p * d + p);
                sigmas.head(p * d).setConstant(std::sqrt(1.0 / (2 * meas.kappa * w)));
                sigmas.tail(p).setConstant(std::sqrt(1.0 / (2 * meas.tau * w)));
                auto noise = noiseModel::Diagonal::Sigmas(sigmas);

                // Emplace the appropriate SE-sync factor
                if constexpr (d == 2) {
                    inputGraph.emplace_shared<SEsyncFactor2>(
                            meas.i, meas.j, meas.R, meas.t, p, noise
                    );
                } else {
                    inputGraph.emplace_shared<SEsyncFactor3>(
                            meas.i, meas.j, meas.R, meas.t, p, noise
                    );
                }
            }
            return inputGraph;
        }

        /**
         * @brief Compute the dense block‐diagonal certificate matrix.
         * @param Y  Variable matrix.
         * @return   Dense block matrix of size (d × d·num_poses).
         */
        Matrix computeLambdaBlocks(const Matrix &Y) override {
            Matrix SY = M_ * Y;
            Matrix Yt = Y.transpose();
            Matrix LambdaBlocks(d, num_pose_ * d);
            
            size_t pose_idx = 0;
            for (const auto& kv : layout_.getVariables()) {
                const auto& var = kv.second;
                for (const auto& comp : var.components) {
                    if (comp.constrained) {
                        size_t base = comp.globalOffset;
                        Matrix P = SY.block(base, 0, d, Y.cols())
                                   * Yt.block(0, base, Y.cols(), d);
                        LambdaBlocks.block(0, pose_idx * d, d, d) = 0.5 * (P + P.transpose());
                        pose_idx++;
                    }
                }
            }
            return LambdaBlocks;
        }

        /**
         * @brief Convert dense Λ_blocks into a sparse Λ matrix for certification.
         * @param LambdaBlocks  Dense block matrix.
         * @return              Sparse certificate matrix Λ.
         */
        SparseMatrix computeLambdaFromLambdaBlocks(
                const Matrix &LambdaBlocks) override {
            std::vector<Eigen::Triplet<Scalar>> elements;
            elements.reserve(d * d * num_pose_);
            
            size_t pose_idx = 0;
            for (const auto& kv : layout_.getVariables()) {
                const auto& var = kv.second;
                for (const auto& comp : var.components) {
                    if (comp.constrained) {
                        size_t base = comp.globalOffset;
                        for (size_t r = 0; r < d; ++r) {
                            for (size_t c = 0; c < d; ++c) {
                                elements.emplace_back(
                                        base + r,
                                        base + c,
                                        LambdaBlocks(r, pose_idx * d + c)
                                );
                            }
                        }
                        pose_idx++;
                    }
                }
            }
            SparseMatrix Lambda(layout_.getTotalDim(), layout_.getTotalDim());
            Lambda.setFromTriplets(elements.begin(), elements.end());
            return Lambda;
        }

        /**
         * @brief Build the element matrix S from current Values.
         * @param values  GTSAM Values containing LiftedPoseDP variables.
         * @return        Sparse matrix S of size.
         */
        SparseMatrix elementMatrix(const Values &values) override {
            const size_t p = currentRank_;
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(num_pose_ * p * (d + 1));

            for (const auto &kv: values.extract<LiftedPoseDP>()) {
                Key key = kv.first;
                const auto &pose = kv.second;
                const auto &t = pose.get_TranslationVector(); // p × 1
                const auto &mat = pose.matrix(); // p × d

                // Use layout to find offsets
                size_t r_base = layout_.getGlobalIndex(key, 0); // Rotation starts at 0
                size_t t_base = layout_.getGlobalIndex(key, d); // Translation at d

                // Rotation block
                for (size_t row = 0; row < p; ++row)
                    for (size_t col = 0; col < d; ++col)
                        triplets.emplace_back(r_base + col, row, mat(row, col));
                
                // Translation block
                for (size_t row = 0; row < p; ++row)
                    triplets.emplace_back(t_base, row, t(row));
            }
            
            SparseMatrix S(layout_.getTotalDim(), p);
            S.setFromTriplets(triplets.begin(), triplets.end());
            return S;
        }

        /**
         * @brief Randomly initialize all poses at level Pmin using uniform Stiefel and random translation.
         * @param Pmin  Target relaxation rank.
         * @return      GTSAM Values container with random LiftedPoseDP entries.
         */
        Values randomInitAtLevelP(const size_t Pmin) override {
            // Shared helper → same Values as the vanilla baseline for a given seed.
            return randomLiftedSE2dPoses(num_pose_, d, Pmin, opts_.randomSeed);
        }

        /**
         * @brief Convert a flat eigenvector into tangent VectorValues.
         * @param p  Relaxation rank.
         * @param v  Flattened direction of size.
         * @param values  Lifted Values at level p.
         * @return  VectorValues on each local tangent space.
         */
        VectorValues TangentVectorValues(
                size_t p, const Vector& v, const Values& values) override {
            VectorValues delta;
            Matrix Ydot = Matrix::Zero(v.size(), p);
            Ydot.rightCols<1>() = v;

            for (const auto &kv: values.extract<LiftedPoseDP>()) {
                Key key = kv.first;
                const auto &Y = kv.second.get_Y();
                
                size_t r_base = layout_.getGlobalIndex(key, 0);
                size_t t_base = layout_.getGlobalIndex(key, d);

                Matrix tangM = Ydot.block(r_base, 0, d, p).transpose();
                Vector transV = Ydot.block(t_base, 0, 1, p).transpose();

                Vector xi = StiefelManifoldKP::Vectorize(tangM);
                Vector tVec = Y.G_.transpose() * xi;
                Vector combined(tVec.size() + transV.size());
                combined << tVec, transV;
                delta.insert(gtsam::symbolIndex(key), combined);
            }
            return delta;
        }

        /**
         * @brief Project ambient‐space variation Ydot onto the tangent space at Y.
         * @param p     Relaxation rank.
         * @param Y     Basepoint matrix.
         * @param Ydot  Ambient variation.
         * @return      Tangent‐space projection.
         */
        Matrix tangent_space_projection(
                const size_t p, const Matrix &Y, const Matrix &Ydot) override {
            Matrix result = Ydot;
            const int P = static_cast<int>(p);
            
            for (const auto& kv : layout_.getVariables()) {
                Key key = kv.first;
                if (symbolChr(key) == 'L') continue; // Landmark handled in LandMark class

                size_t base = layout_.getGlobalIndex(key, 0);
                result.block(base, 0, d, P) =
                    StiefelManifoldKP::Proj(
                        Y.block(base, 0, d, P).transpose(),
                        result.block(base, 0, d, P).transpose()
                    ).transpose();
            }
            return result;
        }

        /**
         * @brief Clamp a value into [lower_bound, upper_bound].
         * @param val          Input value.
         * @param lower_bound  Minimum allowed.
         * @param upper_bound  Maximum allowed.
         * @return             Clamped value.
         */
        Scalar thresholdVal(Scalar val,
                            Scalar lower_bound,
                            Scalar upper_bound) {
            if (val < lower_bound) return lower_bound;
            if (val > upper_bound) return upper_bound;
            return val;
        }

        Matrix dataMatrixProduct(const Matrix &Y) {
            return M_ * Y;
        }

        // Without scale 0.5 in consistent with SE_Sync's objective
        Scalar evaluateObjective(const Matrix &Y) {
            return (Y.transpose() * dataMatrixProduct(Y)).trace();
        }

        /**
         * @brief Search for a certifiable solution by iteratively increasing the relaxation rank.
         *
         * Performs optimization, certificate computation, and verification at each level.
         * If verification fails, it lifts the solution along a descent direction to a higher rank.
         *
         * @param[in] pMin Minimum relaxation rank to start with.
         * @param[in] pMax Maximum relaxation rank allowed.
         * @return CertificateResults on success, or std::nullopt if no certifiable solution is found.
         */
        std::optional<CertificateResults> Solve(size_t pMin, size_t pMax) {
            Values Qstar;
            if (opts_.save_animation) {
                certificateResults_.visualizationTraj.push_back(RoundSolutionS());
            }
            auto t6 = CFGStopwatch::tick();
            for (size_t p = pMin; p <= pMax; ++p) {
                std::cout << "Starting optimization at rank = " << p << std::endl;
                setCurrentRank(p);
                Qstar = tryOptimizingAtLevel(p);
                setCurrentValues(Qstar);
                if (opts_.save_animation) {
                    certificateResults_.visualizationTraj.push_back(RoundSolutionS());
                }
                auto t4 = CFGStopwatch::tick();
                SparseMatrix S = elementMatrix(Qstar);
                Matrix lambdaBlocks = computeLambdaBlocks(S);
                SparseMatrix Lambda = computeLambdaFromLambdaBlocks(lambdaBlocks);
                Scalar obj = evaluateObjective(S);

                bool success = false;
                Scalar eta;
                if (opts_.useAbsoluteEta == true) {
                    eta = opts_.eta;
                } else {
                    eta = thresholdVal(obj * opts_.REL_CERT_ETA, opts_.MIN_CERT_ETA, opts_.MAX_CERT_ETA);
                }
                certificateResults_.etas.push_back(eta);
                size_t nx = opts_.nx;
                Vector v;
                Scalar theta;
                size_t num_lobpcg_iters;
                size_t max_iters = opts_.max_iters;
                Scalar max_fill_factor = opts_.max_fill_factor;
                Scalar drop_tol = opts_.drop_tol;

                success = fast_verification(M_ - Lambda, eta, nx, theta, v,
                                            num_lobpcg_iters, max_iters, max_fill_factor, drop_tol);
                auto t5 = CFGStopwatch::tock(t4);
                certificateResults_.verification_times.push_back(t5);
                certificateResults_.total_computation_time += t5;
                if (!success) {
                    std::cout << "Verification failed at level p = " << p << std::endl;
                    increaseCurrentRank();
                    currentValues_ = initializeWithDescentDirection(Qstar, v, theta, 1e-2);
                } else {
                    std::cout << "Solution verified at level p = " << p << std::endl;
                    certificateResults_.Yopt = S;
                    certificateResults_.Lambda = Lambda;
                    certificateResults_.xhat = RoundSolutionS();
                    certificateResults_.endingRank = p;
                    certificateResults_.Qstar = currentValues_;
                    certificateResults_.verification_status_ = VerificationStatus::VERIFIED;
                    certificateResults_.eta = eta;
                    // Keep currentGraph_ at the final rank for downstream error eval.
                    currentGraph_ = buildGraphAtLevel(p);
                    return certificateResults_;
                }
            }

            std::cout << "No certifiable solution found in p ∈ [" << pMin << ", " << pMax << "]" << std::endl;
            // Not certified: still expose the latest Values so GNC can keep iterating.
            certificateResults_.Qstar = currentValues_;
            certificateResults_.endingRank = currentRank_;
            certificateResults_.verification_status_ = VerificationStatus::FAILED;
            if (!certificateResults_.etas.empty()) {
                certificateResults_.eta = certificateResults_.etas.back();
            }
            currentGraph_ = buildGraphAtLevel(currentRank_);
            return certificateResults_;
        }

        /**
         * @brief Build a Values container at rank pMin from the rounded SE(d)
         *        solution stored in certificateResults_.xhat.
         *
         * Each pose's d x d rotation block is lifted to St(pMin, d) by
         * zero-padding (identity when pMin == d). Translations are lifted to
         * R^pMin. This is the warm-start path used between GNC outer
         * iterations: even if the previous Solve climbed the staircase to a
         * higher rank, the rounded SE(d) result still gives a high-quality
         * starting point at pMin.
         */
        Values valuesFromXhatAtLevel(size_t pMin) const {
            Values out;
            const Matrix& R = certificateResults_.xhat;
            if (R.size() == 0) {
                return out;  // caller should fall back to opts_.initType
            }
            for (size_t i = 0; i < num_pose_; ++i) {
                Key key = i;
                size_t r_base = layout_.getGlobalIndex(key, 0);
                size_t t_base = layout_.getGlobalIndex(key, d);
                Matrix Rd = R.block(0, r_base, d, d);
                Vector td = R.block(0, t_base, d, 1);
                StiefelManifoldKP Y = StiefelManifoldKP::Lift(pMin, Rd);
                Vector trans = LiftedPoseDP::LiftToRp(td, pMin);
                out.insert(i, LiftedPoseDP(Y, trans));
            }
            return out;
        }

        /**
         * @brief Run the certifiable solver with per-factor weights.
         *
         * Stores the supplied weights, rebuilds the current factor graph and
         * data matrix at level pMin from those weights, and invokes the
         * standard Solve() pipeline. For gncIter == 0 the values are reset to
         * a fresh initialization (random or odom, per opts_.initType); for
         * gncIter > 0 the previous certifier's rounded SE(d) solution is used
         * as a warm start at level pMin (avoiding rank-mismatch when the
         * previous call climbed the staircase).
         *
         * @param weights  Per-measurement weight (size == measurements_.poseMeasurements.size()).
         * @param pMin     Minimum relaxation rank.
         * @param pMax     Maximum relaxation rank allowed.
         * @param gncIter  GNC iteration index (0 means re-initialize values).
         * @return Optional CertificateResults with Qstar populated.
         */
        std::optional<CertificateResults> SolveWeighted(const Vector &weights,
                                                        size_t pMin,
                                                        size_t pMax,
                                                        size_t gncIter) {
            if (weights.size() !=
                static_cast<Eigen::Index>(measurements_.poseMeasurements.size())) {
                throw std::runtime_error(
                    "CertifiablePGO::SolveWeighted: weights size does not match poseMeasurements size.");
            }
            setCurrentWeights(weights);

            // Reset rank to pMin and rebuild the weighted graph + data matrix.
            setCurrentRank(pMin);
            if (gncIter == 0) {
                // Cold start at the first GNC iteration.
                if (opts_.initType == CertifiableProblemOpts::InitType::Random) {
                    currentValues_ = randomInitAtLevelP(pMin);
                } else if (opts_.initType == CertifiableProblemOpts::InitType::Odom) {
                    currentValues_ = odomInitAtLevelP(pMin);
                }
            } else {
                Values warm = valuesFromXhatAtLevel(pMin);
                if (warm.size() == num_pose_) {
                    currentValues_ = warm;
                } else {
                    if (opts_.initType == CertifiableProblemOpts::InitType::Odom) {
                        currentValues_ = odomInitAtLevelP(pMin);
                    } else {
                        currentValues_ = randomInitAtLevelP(pMin);
                    }
                }
            }
            currentGraph_ = buildGraphAtLevel(pMin);
            ordering_ = Ordering::Colamd(currentGraph_);
            layout_ = GlobalLayout(ordering_, d, orderingType_);
            M_ = recoverDataMatrixFromAssembler();

            return Solve(pMin, pMax);
        }


        /**
         * @brief Round the relaxed solution back to problem dimension.
         * @return Matrix R.
         */
        Matrix RoundSolutionS() override {
            const size_t p = currentRank_;
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(num_pose_ * p * (d + 1));
            for (const auto &kv: currentValues_.extract<LiftedPoseDP>()) {
                Key key = kv.first;
                const auto &pose = kv.second;
                const auto &t = pose.get_TranslationVector(); // p × 1
                const auto &mat = pose.matrix(); // p × d
                
                size_t r_base = layout_.getGlobalIndex(key, 0);
                size_t t_base = layout_.getGlobalIndex(key, d);

                // Translation entries
                for (size_t row = 0; row < p; ++row) {
                    triplets.emplace_back(t_base, row, t(row));
                }
                // Rotation entries
                for (size_t row = 0; row < p; ++row) {
                    for (size_t col = 0; col < d; ++col) {
                        triplets.emplace_back(r_base + col, row, mat(row, col));
                    }
                }
            }
            
            SparseMatrix P(layout_.getTotalDim(), p);
            P.setFromTriplets(triplets.begin(), triplets.end());
            Matrix S = P;
            Matrix St = S.transpose();
            Eigen::JacobiSVD<Matrix> svd(St, Eigen::ComputeFullV);

            Vector sigmas = svd.singularValues();

            // Construct a diagonal matrix comprised of the first d singular values
            DiagonalMatrix Sigma_d(d);
            DiagonalMatrix::DiagonalVectorType &diagonal = Sigma_d.diagonal();
            for (size_t i = 0; i < d; ++i) {
                diagonal(i) = sigmas(i);
            }

            // First, construct a rank-d truncated singular value decomposition for Y
            Matrix Vd = svd.matrixV().leftCols(d);
            Matrix R = Sigma_d * Vd.transpose();
            Vector determinants(num_pose_);

            size_t ng0 = 0; // This will count the number of blocks whose determinants have positive sign
            
            size_t pose_idx = 0;
            for (const auto& kv : layout_.getVariables()) {
                Key key = kv.first;
                if (symbolChr(key) == 'L') continue;
                
                size_t r_base = layout_.getGlobalIndex(key, 0);
                // Compute the determinant of the ith dxd block of R
                determinants(pose_idx) = R.block(0, r_base, d, d).determinant();
                if (determinants(pose_idx) > 0)
                    ++ng0;
                pose_idx++;
            }

            if (ng0 < num_pose_ / 2) {
                // Less than half of the total number of blocks have the correct sign, so reverse their orientations
                Matrix reflector = Matrix::Identity(d, d);
                reflector(d - 1, d - 1) = -1;
                R = reflector * R;
            }

            // Finally, project each dxd rotation block to SO(d)
            for (const auto& kv : layout_.getVariables()) {
                Key key = kv.first;
                if (symbolChr(key) == 'L') continue;
                size_t r_base = layout_.getGlobalIndex(key, 0);
                R.block(0, r_base, d, d) = project_to_SOd(R.block(0, r_base, d, d));
            }
            return R;
        }

        /**
         * @brief Export the solution in G2O or TUM format.
         * @param path  Base filename (without extension).
         * @param R     Rotation/translation solution matrix.
         * @param g2o   If true, write G2O; otherwise, write TUM.
         */
        void ExportData(const std::string &path, const Eigen::MatrixXd &R, bool g2o) override {
            if (g2o) {
                Values finalposes;
                if (d == 3) {
                    // Insert SE3 poses
                    for (std::size_t i = 0; i < num_pose_; ++i) {
                        Key key = i; // Assuming sequential keys for simple PGO
                        size_t r_base = layout_.getGlobalIndex(key, 0);
                        size_t t_base = layout_.getGlobalIndex(key, d);
                        finalposes.insert(
                                i,
                                Pose3(
                                        Rot3(R.block(0, r_base, d, d)),
                                        R.block(0, t_base, d, 1)
                                )
                        );
                    }
                } else {
                    // Insert SE2 poses
                    for (std::size_t i = 0; i < num_pose_; ++i) {
                        Key key = i;
                        size_t r_base = layout_.getGlobalIndex(key, 0);
                        size_t t_base = layout_.getGlobalIndex(key, d);
                        Rot2 rot = Rot2::fromCosSin(
                                R.block(0, r_base, d, d)(0, 0),
                                R.block(0, r_base, d, d)(1, 0)
                        );
                        finalposes.insert(i, Pose2(rot, R.block(0, t_base, d, 1)));
                    }
                }
                writeG2o(currentGraph_, finalposes, path + ".g2o");
            } else {
                std::ofstream file(path + ".txt");
                if (d == 2) {
                    for (std::size_t i = 0; i < num_pose_; ++i) {
                        Key key = i;
                        size_t r_base = layout_.getGlobalIndex(key, 0);
                        size_t t_base = layout_.getGlobalIndex(key, d);
                        Eigen::Matrix3d R3 = Eigen::Matrix3d::Identity();
                        R3.topLeftCorner<2, 2>() =
                                R.block(0, r_base, 2, 2);
                        auto q = Eigen::Quaterniond(R3);
                        Vector t = R.block(0, t_base, d, 1);
                        file << i << " " << t(0) << " " << t(1) << " " << 0
                             << " " << q.x() << " " << q.y()
                             << " " << q.z() << " " << q.w() << "\n";
                    }
                } else {
                    for (std::size_t i = 0; i < num_pose_; ++i) {
                        Key key = i;
                        size_t r_base = layout_.getGlobalIndex(key, 0);
                        size_t t_base = layout_.getGlobalIndex(key, d);
                        Quaternion q(R.block<3, 3>(0, r_base));
                        Vector t = R.block(0, t_base, d, 1);
                        file << i << " " << t(0) << " " << t(1) << " " << 0
                             << " " << t(2) << " " << q.x() << " "
                             << q.y() << " " << q.z() << " "
                             << q.w() << "\n";
                    }
                }
                file.close();
            }
        }

        /**
         * @brief Export all trajectories in CertificateResults::visualizationTraj to g2o files, for animation
         *
         * @param results  Container holding the trajectory matrices to export.
         * @param out_dir  Output directory; created if it does not exist.
         * @param stem     File name stem (default: "traj"), e.g., "traj_000.g2o".
         * @param g2o      If true, write .g2o files; otherwise write .txt files.
         * @return std::size_t  Number of files successfully written.
         */
        inline std::size_t ExportVisualizationTrajectories(
                const CertificateResults& results,
                const std::string& out_dir,
                const std::string& stem = "animation",
                bool g2o = true)
        {
            std::error_code ec;
            if (!out_dir.empty()) std::filesystem::create_directories(out_dir, ec);
            if (ec) {
                std::cerr << "[export] cannot create directory: " << out_dir
                          << " (" << ec.message() << ")\n";
                return 0;
            }

            const auto& trajs = results.visualizationTraj;
            if (trajs.empty()) {
                std::cerr << "[export] no trajectories to export.\n";
                return 0;
            }

            std::size_t written = 0;
            for (std::size_t i = 0; i < trajs.size(); ++i) {
                const Eigen::MatrixXd& R = trajs[i];
                std::ostringstream oss;
                oss << stem << "_" << std::setw(3) << std::setfill('0') << i;

                const std::string path = (std::filesystem::path(out_dir) / oss.str()).string();

                try {
                    ExportData(path, R, g2o);  // g2o=true => ".g2o" appended inside ExportData
                    ++written;
                } catch (const std::exception& e) {
                    std::cerr << "[export] failed to write " << path << ": " << e.what() << "\n";
                }
            }
            return written;
        }

    };


    // Convenience aliases
    using CertifiablePGO2 = CertifiablePGO<2>;
    using CertifiablePGO3 = CertifiablePGO<3>;

}
#endif // STIEFELMANIFOLDEXAMPLE_CERTIFIABLEPGO_H

