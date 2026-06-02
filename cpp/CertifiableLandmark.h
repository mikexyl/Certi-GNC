/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    CertifiableLandmark.h
 * @brief   Certifiable Landmark-based SLAM via the Riemannian staircase.
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
 * For related hand-crafted certifiable landmark-SLAM solvers, see:
 *   T. Fan et al. "CPL-SLAM: Efficient and certifiably correct planar
 *   graph-based SLAM using the complex-number representation." T-RO
 *   36(6):1719-1737, 2020.
 *   C. Holmes and T. D. Barfoot. "An efficient global optimality
 *   certificate for landmark-based SLAM." RA-L 8(3):1539-1546, 2023.
 *
 * */

#ifndef CERTIFIABLELANDMARK_H
#define CERTIFIABLELANDMARK_H

#include "Certifiable_problem.h"
#include "LandmarkFactor.h"
#include "RandomInit.h"
#include "SEsyncFactor.h"

namespace gtsam {
/**
 * @brief Certifiable landmark estimation problem.
 *
 * Implements certifiable estimation for simultaneous pose‐and‐landmark SLAM
 * problems via relaxation levels p.
 *
 * @tparam d  Ambient dimension (must be 2 or 3).
 */
    template<size_t d>
    class CertifiableLandmark : public CertifiableProblem {
        static_assert(d == 2 || d == 3, "CertifiableLandmark only supports d = 2 or 3.");

        /// Number of landmarks in the problem.
        size_t num_landmark_;

        /// Number of landmark measurements.
        size_t num_lmk_measurements_;

        /// Number of pose‐to‐pose measurements.
        size_t num_pose_measurements_;
     BlockOrderingType orderingType_ = BlockOrderingType::Generic;

    public:
        Values LiftTo(size_t p, const Values& values) override {
            Values result;
            // Lift all Pose variables of type LiftedPoseDP
            for (const auto& it : values.extract<LiftedPoseDP>()) {
                result.insert(it.first, it.second.LiftTo(p));
            }
            // Lift all raw Vector variables (landmarks)
            for (const auto& it : values.extract<Vector>()) {
                result.insert(it.first, LiftedPoseDP::LiftToRp(it.second, p));
            }
            return result;
        }

        void setBlockOrderingType(BlockOrderingType type) { orderingType_ = type; }

        /**
         * @brief Constructor.
         * @param[in] p Initial relaxation rank.
         * @param[in] measurements Parsed measurements struct (num_poses, landmarks, etc.).
         * @param[in] g2oPath Path to the g2o dataset file.
         */
        CertifiableLandmark(size_t p, const DataParser::Measurement &measurements, const std::string g2oPath)
                : CertifiableProblem(d, p, measurements) {
            num_landmark_ = measurements.num_landmarks;
            num_lmk_measurements_ = measurements.landmarkMeasurements.size();
            num_pose_measurements_ = measurements.poseMeasurements.size();
            dataPath = g2oPath;
            certificateResults_.startingRank = p;
        }

        /**
         * @brief Initialize the problem: build graph, random/odom init, and recover data matrix.
         *
         * Populates OdomEdgeList_ with the indices (within the assembled graph) of
         * pose-pose factors that correspond to odometry edges. Pose factors are
         * emplaced first (indices [0, num_pose_measurements_)), landmark factors
         * follow. Odometry edges are pose-pose factors with j == i + 1.
         */
        void init() {
            num_lmk_measurements_ = measurements_.landmarkMeasurements.size();
            num_pose_measurements_ = measurements_.poseMeasurements.size();
            auto t0 = CFGStopwatch::tick();
            if (opts_.initType == CertifiableProblemOpts::InitType::Random) {
                currentValues_ = randomInitAtLevelP(currentRank_);
            } else if (opts_.initType == CertifiableProblemOpts::InitType::Odom) {
                currentValues_ = odomInitAtLevelP(currentRank_);
            } else {
                throw std::runtime_error("LocalSearch initialization not supported for main branch, please find in paper_experiments branch");
            }
            // Build graph at initial level
            currentGraph_ = buildGraphAtLevel(currentRank_);

            // Mark odometry pose-pose factors (consecutive poses).
            OdomEdgeList_.clear();
            OdomEdgeList_.reserve(num_pose_measurements_);
            for (size_t k = 0; k < num_pose_measurements_; ++k) {
                const auto& m = measurements_.poseMeasurements[k];
                if (m.j == m.i + 1) {
                    OdomEdgeList_.push_back(k);
                }
            }

            // Initialize global ordering and layout
            ordering_ = Ordering::Natural(currentGraph_);
            layout_ = GlobalLayout(ordering_, d, orderingType_);

            M_ = recoverDataMatrixFromAssembler();
            auto t1 = CFGStopwatch::tock(t0);
            certificateResults_.initialization_time.push_back(t1);
            certificateResults_.total_computation_time += t1;
        }

        /**
         * @brief Initialize all pose and landmark variables at level Pmin from g2o initial guesses.
         *
         * For each pose touched by a measurement, takes (R, t) from
         * measurements_.initial_poses[{'A', i}] if present and lifts to rank Pmin.
         * For each landmark, takes t from measurements_.initial_landmarks[{'L', j}]
         * if present and lifts the translation to R^Pmin. Falls back to random
         * init for any variable lacking an initial guess.
         */
        Values odomInitAtLevelP(const size_t Pmin) {
            Values initial;
            auto liftPose = [&](size_t poseId) {
                if (initial.exists(poseId)) return;
                auto it = measurements_.initial_poses.find({'A', poseId});
                if (it != measurements_.initial_poses.end()) {
                    const auto& iv = it->second;
                    StiefelManifoldKP Y = StiefelManifoldKP::Lift(Pmin, iv.R);
                    Vector trans = LiftedPoseDP::LiftToRp(iv.t, Pmin);
                    initial.insert(poseId, LiftedPoseDP(Y, trans));
                } else {
                    StiefelManifoldKP Y =
                            StiefelManifoldKP::Random(std::default_random_engine::default_seed, d, Pmin);
                    Vector trans = Vector::Random(Pmin);
                    initial.insert(poseId, LiftedPoseDP(Y, trans));
                }
            };
            for (const auto &meas: measurements_.poseMeasurements) {
                liftPose(meas.i);
                liftPose(meas.j);
            }
            for (const auto &meas: measurements_.landmarkMeasurements) {
                liftPose(meas.i);
                Key key = Symbol('L', meas.j);
                if (!initial.exists(key)) {
                    auto it = measurements_.initial_landmarks.find({'L', meas.j});
                    if (it != measurements_.initial_landmarks.end()) {
                        Vector trans = LiftedPoseDP::LiftToRp(it->second.t, Pmin);
                        initial.insert(key, trans);
                    } else {
                        Vector trans = Vector::Random(Pmin);
                        initial.insert(key, trans);
                    }
                }
            }
            return initial;
        }

        /**
         * @brief Get the number of landmarks.
         * @return Number of landmarks.
         */
        inline size_t getNumLandmark() const { return num_landmark_; }

        /**
         * @brief Build the factor graph at relaxation level p.
         * @param p  Relaxation rank.
         * @return   A GTSAM NonlinearFactorGraph containing both pose and landmark factors.
         */
        NonlinearFactorGraph buildGraphAtLevel(size_t p) override {
            NonlinearFactorGraph inputGraph;

            // Weights vector is ordered [pose factors..., landmark factors...] matching
            // the order in which we emplace them below.
            const size_t expectedW =
                measurements_.poseMeasurements.size() +
                measurements_.landmarkMeasurements.size();
            const bool hasW = (currentWeights_.size() == static_cast<Eigen::Index>(expectedW));

            // Pose factors
            for (size_t k = 0; k < measurements_.poseMeasurements.size(); ++k) {
                const auto &meas = measurements_.poseMeasurements[k];
                double w = hasW ? currentWeights_(k) : 1.0;
                if (w < 1e-12) w = 1e-12;
                const double Kappa = meas.kappa * w;
                const double tau = meas.tau * w;
                Vector sigmas = Vector::Zero(p * d + p);
                sigmas.head(p * d).setConstant(std::sqrt(1.0 / (2 * Kappa)));
                sigmas.tail(p).setConstant(std::sqrt(1.0 / (2 * tau)));
                auto noise = noiseModel::Diagonal::Sigmas(sigmas);

                if constexpr (d == 2) {
                    inputGraph.emplace_shared<SEsyncFactor2>(
                            meas.i, meas.j, meas.R, meas.t, p, noise);
                } else {
                    inputGraph.emplace_shared<SEsyncFactor3>(
                            meas.i, meas.j, meas.R, meas.t, p, noise);
                }
            }

            // Landmark factors
            const size_t base = measurements_.poseMeasurements.size();
            for (size_t k = 0; k < measurements_.landmarkMeasurements.size(); ++k) {
                const auto &meas = measurements_.landmarkMeasurements[k];
                double w = hasW ? currentWeights_(base + k) : 1.0;
                if (w < 1e-12) w = 1e-12;
                const double nu = meas.nu * w;
                Vector sigmas = Vector::Zero(p);
                sigmas.tail(p).setConstant(std::sqrt(1.0 / (2 * nu)));
                auto noise = noiseModel::Diagonal::Sigmas(sigmas);

                if constexpr (d == 2) {
                    inputGraph.emplace_shared<LiftedLandmarkFactor2>(
                            meas.i, Symbol('L', meas.j), meas.l, p, noise);
                } else {
                    inputGraph.emplace_shared<LiftedLandmarkFactor3>(
                            meas.i, Symbol('L', meas.j), meas.l, p, noise);
                }
            }

            return inputGraph;
        }

        /**
         * @brief Compute block‐diagonal lambda matrix blocks from Y.
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
         * @brief Convert dense lambda blocks into a sparse lambda matrix.
         * @param LambdaBlocks  Dense block matrix.
         * @return              Sparse certificate matrix Λ.
         */
        SparseMatrix computeLambdaFromLambdaBlocks(const Matrix &LambdaBlocks) override {
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
            Lambda.setFromTriplets(elements.begin(), elements.end(), std::plus<Scalar>());
            return Lambda;
        }

        /**
         * @brief Build the sparse element matrix S from Values.
         * @param values  GTSAM Values containing LiftedPoseDP and raw vectors.
         * @return        Sparse matrix S of size.
         */
        SparseMatrix elementMatrix(const Values &values) override {
            const size_t p = currentRank_;
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(num_pose_ * p * (d + 1) + num_landmark_ * p);

            // Pose entries
            for (const auto &kv: values.extract<LiftedPoseDP>()) {
                Key key = kv.first;
                const auto &pose = kv.second;
                const auto &t = pose.get_TranslationVector(); // p × 1
                const auto &mat = pose.matrix();              // p × (d+1)

                size_t r_base = layout_.getGlobalIndex(key, 0);
                size_t t_base = layout_.getGlobalIndex(key, d);

                for (size_t row = 0; row < p; ++row)
                    triplets.emplace_back(t_base, row, t(row));
                for (size_t row = 0; row < p; ++row)
                    for (size_t col = 0; col < d; ++col)
                        triplets.emplace_back(r_base + col, row, mat(row, col));
            }

            // Landmark entries
            for (const auto &kv: values.extract<Vector>()) {
                Key key = kv.first;
                const auto &t = kv.second;
                size_t l_base = layout_.getGlobalIndex(key, 0);
                for (size_t row = 0; row < p; ++row)
                    triplets.emplace_back(l_base, row, t(row));
            }

            SparseMatrix S(layout_.getTotalDim(), p);
            S.setFromTriplets(triplets.begin(), triplets.end(), std::plus<Scalar>());
            return S;
        }

        /**
         * @brief Randomly initialize values at relaxation level Pmin.
         * @param Pmin  relaxation level.
         * @return      GTSAM Values with random LiftedPoseDP and landmark vectors.
         */
        Values randomInitAtLevelP(const size_t Pmin) override {
            // Shared helper → same Values as the vanilla baseline for a given seed.
            Values initial = randomLiftedSE2dPoses(
                measurements_.num_poses, d, Pmin, opts_.randomSeed);

            // Burn the pose draws so landmark sampling continues from the
            // same RNG state as the vanilla binary.
            std::mt19937 rng(opts_.randomSeed);
            std::uniform_real_distribution<double> ut(-10.0, 10.0);
            std::uniform_real_distribution<double> ur(-M_PI, M_PI);
            for (size_t i = 0; i < measurements_.num_poses; ++i) {
                if constexpr (d == 2) {
                    (void)ut(rng); (void)ut(rng); (void)ur(rng);
                } else {
                    (void)ut(rng); (void)ut(rng); (void)ut(rng);
                    (void)ut(rng); (void)ut(rng); (void)ut(rng); (void)ut(rng);
                }
            }
            addRandomLiftedLandmarks(measurements_.landmarkMeasurements,
                                     d, Pmin, rng, initial);
            return initial;
        }

        /**
         * @brief Convert a flat eigenvector into per-variable tangent VectorValues.
         * @param p      Relaxation level.
         * @param v      Flattened descent vector size.
         * @param values Lifted Values at level p.
         * @return       VectorValues for each variable on its tangent space.
         */
        VectorValues TangentVectorValues(size_t p,
                                         const Vector& v,
                                         const Values& values) override {
            VectorValues delta;
            Matrix Ydot = Matrix::Zero(v.size(), p);
            Ydot.rightCols<1>() = v;

            // Poses tangent blocks
            for (const auto &kv: values.extract<LiftedPoseDP>()) {
                Key key = kv.first;
                const auto &pose = kv.second;
                StiefelManifoldKP Y = pose.get_Y();
                
                size_t r_base = layout_.getGlobalIndex(key, 0);
                size_t t_base = layout_.getGlobalIndex(key, d);

                Matrix tangM = Ydot.block(r_base, 0, d, p).transpose();
                Vector transV = Ydot.block(t_base, 0, 1, p).transpose();

                Vector xi = StiefelManifoldKP::Vectorize(tangM);
                Vector tangentVec = Y.G_.transpose() * xi;
                Vector combined(tangentVec.size() + transV.size());
                combined << tangentVec, transV;
                delta.insert(gtsam::symbolIndex(key), combined);
            }

            // Landmarks tangent blocks
            for (const auto &kv: values.extract<Vector>()) {
                Key key = kv.first;
                size_t l_base = layout_.getGlobalIndex(key, 0);
                Vector lmkV = Ydot.block(l_base, 0, 1, p).transpose();
                delta.insert(gtsam::symbolIndex(key) + num_pose_, lmkV);
            }
            return delta;
        }

        /**
         * @brief Project ambient gradient Ydot onto tangent space at Y.
         * @param p     Relaxation level.
         * @param Y     Basepoint matrix.
         * @param Ydot  Ambient gradient matrix.
         * @return      Tangent‐space gradient at Y.
         */

        Matrix tangent_space_projection(
                const size_t p, const Matrix &Y, const Matrix &Ydot) override {
            Matrix result = Ydot;
            const int P = static_cast<int>(p);
            
            for (const auto& kv : layout_.getVariables()) {
                Key key = kv.first;
                if (symbolChr(key) == 'L') continue; // Landmark handled differently

                size_t r_base = layout_.getGlobalIndex(key, 0);
                result.block(r_base, 0, d, P) =
                    StiefelManifoldKP::Proj(
                        Y.block(r_base, 0, d, P).transpose(),
                        result.block(r_base, 0, d, P).transpose()
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

        Scalar evaluateObjective(const Matrix &Y) {
            return (Y.transpose() * dataMatrixProduct(Y)).trace();
        }

        /**
         * @brief Search for a certifiable solution by iteratively increasing the relaxation rank.
         *
         * Performs optimization, certificate computation, and verification at each level.
         *
         * @param[in] pMin Minimum relaxation rank to start with.
         * @param[in] pMax Maximum relaxation rank allowed.
         * @return CertificateResults on success, or std::nullopt if no certifiable solution is found.
         */
        std::optional<CertificateResults> Solve(size_t pMin, size_t pMax) {
            Values Qstar;
            if (opts_.save_animation) {
                certificateResults_.visualizationTraj.push_back(RoundSolutionS());
            }            auto t6 = std::chrono::high_resolution_clock::now();
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
                    std::cout << "Using eta = " << eta << ", Theta : " << theta << std::endl;
                    increaseCurrentRank();
                    currentValues_ = initializeWithDescentDirection(Qstar, v, theta, 1e-2);
                } else {
                    std::cout << "Solution verified at level p = " << p << std::endl;
                    std::cout << "Final eta = " << eta << ", Theta : " << theta << std::endl;
                    certificateResults_.Yopt = S;
                    certificateResults_.Lambda = Lambda;
                    certificateResults_.xhat = RoundSolutionS();
                    certificateResults_.endingRank = p;
                    certificateResults_.Qstar = currentValues_;
                    certificateResults_.verification_status_ = VerificationStatus::VERIFIED;
                    certificateResults_.eta = eta;
                    currentGraph_ = buildGraphAtLevel(p);
                    return certificateResults_;
                }
            }

            std::cout << "No certifiable solution found in p ∈ [" << pMin << ", " << pMax << "]" << std::endl;
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
         * @brief Build Values at rank pMin from the rounded SE(d) solution in xhat.
         */
        Values valuesFromXhatAtLevel(size_t pMin) const {
            Values out;
            const Matrix& R = certificateResults_.xhat;
            if (R.size() == 0) return out;
            const auto& vars = layout_.getVariables();
            // Poses
            for (size_t i = 0; i < num_pose_; ++i) {
                Key key = i;
                if (vars.find(key) == vars.end()) continue;
                size_t r_base = layout_.getGlobalIndex(key, 0);
                size_t t_base = layout_.getGlobalIndex(key, d);
                Matrix Rd = R.block(0, r_base, d, d);
                Vector td = R.block(0, t_base, d, 1);
                StiefelManifoldKP Y = StiefelManifoldKP::Lift(pMin, Rd);
                Vector trans = LiftedPoseDP::LiftToRp(td, pMin);
                out.insert(i, LiftedPoseDP(Y, trans));
            }
            // Landmarks
            for (size_t j = 0; j < num_landmark_; ++j) {
                Key key = Symbol('L', j);
                if (vars.find(key) == vars.end()) continue;
                size_t l_base = layout_.getGlobalIndex(key, 0);
                Vector td = R.block(0, l_base, d, 1);
                Vector trans = LiftedPoseDP::LiftToRp(td, pMin);
                out.insert(key, trans);
            }
            return out;
        }

        /**
         * @brief Run the certifiable landmark solver with per-factor weights.
         *
         * Weights are concatenated as [pose factors..., landmark factors...]
         * matching the order in buildGraphAtLevel(). gncIter == 0 cold starts;
         * subsequent calls warm-start from the previous SE(d) solution
         * lifted back to rank pMin (avoiding rank mismatches from staircase
         * climbs in earlier iterations).
         */
        std::optional<CertificateResults> SolveWeighted(const Vector &weights,
                                                        size_t pMin,
                                                        size_t pMax,
                                                        size_t gncIter) {
            const size_t expectedW =
                measurements_.poseMeasurements.size() +
                measurements_.landmarkMeasurements.size();
            if (weights.size() != static_cast<Eigen::Index>(expectedW)) {
                throw std::runtime_error(
                    "CertifiableLandmark::SolveWeighted: weights size does not match "
                    "(poseMeasurements + landmarkMeasurements) size.");
            }
            setCurrentWeights(weights);

            setCurrentRank(pMin);
            if (gncIter == 0) {
                if (opts_.initType == CertifiableProblemOpts::InitType::Random) {
                    currentValues_ = randomInitAtLevelP(pMin);
                } else if (opts_.initType == CertifiableProblemOpts::InitType::Odom) {
                    currentValues_ = odomInitAtLevelP(pMin);
                }
            } else {
                Values warm = valuesFromXhatAtLevel(pMin);
                if (warm.size() > 0) {
                    currentValues_ = warm;
                } else if (opts_.initType == CertifiableProblemOpts::InitType::Odom) {
                    currentValues_ = odomInitAtLevelP(pMin);
                } else {
                    currentValues_ = randomInitAtLevelP(pMin);
                }
            }
            currentGraph_ = buildGraphAtLevel(pMin);
            ordering_ = Ordering::Natural(currentGraph_);
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
            triplets.reserve(num_pose_ * p * (d + 1) + num_landmark_ * p);

            // Pose entries
            for (const auto &kv: currentValues_.extract<LiftedPoseDP>()) {
                Key key = kv.first;
                const auto &pose = kv.second;
                const auto &t = pose.get_TranslationVector(); // p × 1
                const auto &mat = pose.matrix();                // p × (d+1)

                size_t r_base = layout_.getGlobalIndex(key, 0);
                size_t t_base = layout_.getGlobalIndex(key, d);

                for (size_t row = 0; row < p; ++row)
                    triplets.emplace_back(t_base, row, t(row));

                for (size_t row = 0; row < p; ++row)
                    for (size_t col = 0; col < d; ++col)
                        triplets.emplace_back(r_base + col, row, mat(row, col));
            }

            // Landmark entries
            for (const auto &kv: currentValues_.extract<Vector>()) {
                Key key = kv.first;
                const auto &t = kv.second; // p × 1
                size_t l_base = layout_.getGlobalIndex(key, 0);
                for (size_t row = 0; row < p; ++row)
                    triplets.emplace_back(l_base, row, t(row));
            }

            SparseMatrix P(layout_.getTotalDim(), p);
            P.setFromTriplets(triplets.begin(), triplets.end(),
                              std::plus<Scalar>());
            Matrix S = P;
            Matrix St = S.transpose();

            // First, compute a thin SVD of Y
            Eigen::JacobiSVD<Matrix> svd(St, Eigen::ComputeFullV);

            Vector sigmas = svd.singularValues();

            // Construct a diagonal matrix comprised of the first d singular values
            DiagonalMatrix Sigma_d(d);
            DiagonalMatrix::DiagonalVectorType &diagonal = Sigma_d.diagonal();
            for (size_t i = 0; i < d; ++i)
                diagonal(i) = sigmas(i);

            // First, construct a rank-d truncated singular value decomposition for Y
            Matrix Vt = svd.matrixV().leftCols(d).transpose();
            Matrix R = Sigma_d * Vt;
            Vector determinants(num_pose_);

            size_t ng0 = 0; // This will count the number of blocks whose
            // determinants have positive sign
            
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
                // Less than half of the total number of blocks have the correct sign, so
                // reverse their orientations
                Matrix reflector = Matrix::Identity(d, d);
                reflector(d - 1, d - 1) = -1;
                R = Matrix(reflector * R);
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
                std::ofstream file(path + ".g2o");
                if (d == 2) {
                    for (std::size_t i = 0; i < num_pose_; ++i) {
                        Key key = i; // Assuming sequential keys
                        size_t r_base = layout_.getGlobalIndex(key, 0);
                        size_t t_base = layout_.getGlobalIndex(key, d);
                        Eigen::Matrix3d R3 = Eigen::Matrix3d::Identity();
                        R3.topLeftCorner<2, 2>() =
                                R.block(0, r_base, 2, 2);
                        auto theta = std::atan2(R3(1, 0), R3(0, 0));
                        Vector t = R.block(0, t_base, d, 1);
                        file << "VERTEX_SE2" << " " << i << " " << t(0) << " " << t(1)
                             << " " << theta << "\n";
                    }

                    for (std::size_t i = 0; i < num_landmark_; ++i) {
                        Key key = Symbol('L', i);
                        size_t l_base = layout_.getGlobalIndex(key, 0);
                        Vector t = R.block(0, l_base, d, 1);
                        file << "VERTEX_XY" << " " << i << " " << t(0) << " " << t(1)
                             << "\n";
                    }
                } else {
                    for (std::size_t i = 0; i < num_pose_; ++i) {
                        Key key = i;
                        size_t r_base = layout_.getGlobalIndex(key, 0);
                        size_t t_base = layout_.getGlobalIndex(key, d);
                        Quaternion q(R.block<3, 3>(0, r_base));
                        Vector t = R.block(0, t_base, d, 1);
                        file << "VERTEX_SE3:QUAT" << " " << i << " " << t(0) << " " << t(1)
                             << " " << t(2) << " " << q.x() << " "
                             << q.y() << " " << q.z() << " "
                             << q.w() << "\n";
                    }
                    for (std::size_t i = 0; i < num_landmark_; ++i) {
                        Key key = Symbol('L', i);
                        size_t l_base = layout_.getGlobalIndex(key, 0);
                        Vector t = R.block(0, l_base, d, 1);
                        file << "VERTEX_XYZ" << " " << i << " " << t(0) << " " << t(1) << " " << t(2)
                             << "\n";
                    }
                }
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
                        file << i << " " << t(0) << " " << t(1)
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

    using CertifiableLandmark2 = CertifiableLandmark<2>;
    using CertifiableLandmark3 = CertifiableLandmark<3>;
}

#endif //CERTIFIABLELANDMARK_H
