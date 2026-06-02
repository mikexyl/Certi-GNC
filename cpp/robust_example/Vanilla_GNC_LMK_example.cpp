/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file Vanilla_GNC_LMK_example.cpp
 * @brief Vanilla GTSAM GNC (TLS loss, LM inner solver) on the *same lifted*
 *        SE-sync + LiftedLandmark factor graph as the certifiable wrapper,
 *        but without the Riemannian staircase.
 *
 * The only algorithmic difference is the
 * inner solver: vanilla calls gtsam::LevenbergMarquardt once per GNC
 * iteration and stays at rank p = d; ours climbs the Riemannian staircase
 * until been verified.
 */

#include "../LandmarkFactor.h"
#include "../RandomInit.h"
#include "../SEsyncFactor.h"
#include "../utils.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

DEFINE_int32(d, 2, "Ambient dimension (only d=2 supported for landmarks).");
DEFINE_string(input_dir, "data/landmark/cityTrees_reduced_outl20.g2o", "Input g2o file.");
DEFINE_string(output_dir, "results/vanilla_lmk_demo", "Output file stem.");
DEFINE_int32(seed, 1, "Random seed for initial guess (only used if init_type=random).");
DEFINE_string(init_type, "random",
              "Initial guess source: 'random' or 'odom' (VERTEX_SE2 + VERTEX_XY from g2o).");
DEFINE_int32(inner_max_iterations, 200, "Max LM iterations per inner solve.");
DEFINE_int32(gnc_max_iterations, 30, "Max GNC outer iterations.");
DEFINE_double(mu_step, 1.4, "Multiplicative factor for mu update (TLS).");
DEFINE_double(rel_cost_tol, 1e-5, "GNC relative cost tolerance.");
DEFINE_double(weights_tol, 1e-4, "Weights binary-convergence tolerance.");
DEFINE_double(abs_error_tol, 1e-5, "LM absolute error tolerance.");
DEFINE_double(rel_error_tol, 1e-5, "LM relative error tolerance.");
DEFINE_int32(gnc_verbosity, 1, "GNC verbosity (0..4).");

using namespace gtsam;
using namespace std::chrono;

namespace {

template <size_t d>
NonlinearFactorGraph buildLiftedLandmarkGraph(
        const DataParser::Measurement& m, size_t p,
        FastVector<uint64_t>& odom_indices,
        size_t& num_pose_factors, size_t& num_lmk_factors) {
    NonlinearFactorGraph graph;
    odom_indices.clear();
    // Pose-pose factors
    for (size_t k = 0; k < m.poseMeasurements.size(); ++k) {
        const auto& pm = m.poseMeasurements[k];
        Vector sigmas = Vector::Zero(p * d + p);
        sigmas.head(p * d).setConstant(std::sqrt(1.0 / (2 * pm.kappa)));
        sigmas.tail(p).setConstant(std::sqrt(1.0 / (2 * pm.tau)));
        auto noise = noiseModel::Diagonal::Sigmas(sigmas);
        graph.emplace_shared<SEsyncFactor<d>>(pm.i, pm.j, pm.R, pm.t, p, noise);
        if (pm.j == pm.i + 1) odom_indices.push_back(k);
    }
    num_pose_factors = m.poseMeasurements.size();
    // Landmark-observation factors
    for (const auto& lm : m.landmarkMeasurements) {
        Vector sigmas = Vector::Zero(p);
        sigmas.tail(p).setConstant(std::sqrt(1.0 / (2 * lm.nu)));
        auto noise = noiseModel::Diagonal::Sigmas(sigmas);
        graph.emplace_shared<LiftedLandmarkFactor<d>>(
            static_cast<Key>(lm.i), Symbol('L', lm.j), lm.l, p, noise);
    }
    num_lmk_factors = m.landmarkMeasurements.size();
    return graph;
}

template <size_t d>
Values odomInitLifted(const DataParser::Measurement& m, size_t p) {
    Values out;
    for (size_t i = 0; i < m.num_poses; ++i) {
        auto it = m.initial_poses.find({'A', i});
        Matrix Rd;
        Vector td;
        if (it != m.initial_poses.end()) {
            Rd = it->second.R;
            td = it->second.t;
        } else {
            Rd = Matrix::Identity(d, d);
            td = Vector::Zero(d);
        }
        StiefelManifoldKP Y = StiefelManifoldKP::Lift(p, Rd);
        Vector tp = LiftedPoseDP::LiftToRp(td, p);
        out.insert(static_cast<Key>(i), LiftedPoseDP(Y, tp));
    }
    for (const auto& lm : m.landmarkMeasurements) {
        Key key = Symbol('L', lm.j);
        if (out.exists(key)) continue;
        auto it = m.initial_landmarks.find({'L', lm.j});
        Vector td = (it != m.initial_landmarks.end()) ? it->second.t : Vector::Zero(d);
        out.insert(key, LiftedPoseDP::LiftToRp(td, p));
    }
    return out;
}

template <size_t d>
Values randomInitLifted(const DataParser::Measurement& m, size_t p, unsigned seed) {
    Values out = randomLiftedSE2dPoses(m.num_poses, d, p, seed);
    // Rebuild a parallel generator and burn the pose draws so landmark
    // samples come from the same RNG state as ours' randomInitAtLevelP.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ut(-10.0, 10.0);
    std::uniform_real_distribution<double> ur(-M_PI, M_PI);
    for (size_t i = 0; i < m.num_poses; ++i) {
        if (d == 2) {
            (void)ut(rng); (void)ut(rng); (void)ur(rng);
        } else {
            (void)ut(rng); (void)ut(rng); (void)ut(rng);
            (void)ut(rng); (void)ut(rng); (void)ut(rng); (void)ut(rng);
        }
    }
    addRandomLiftedLandmarks(m.landmarkMeasurements, d, p, rng, out);
    return out;
}

template <size_t d>
void writeTumTrajectory(const Values& values, size_t num_poses,
                        const std::string& path) {
    std::ofstream f(path);
    f.precision(9);
    for (size_t i = 0; i < num_poses; ++i) {
        Key key = static_cast<Key>(i);
        if (!values.exists(key)) continue;
        const auto& lp = values.at<LiftedPoseDP>(key);
        const Matrix R = lp.get_StiefelElement().matrix().topRows(d);
        const Vector t = lp.get_TranslationVector().head(d);
        if (d == 2) {
            double th = std::atan2(R(1, 0), R(0, 0));
            double qz = std::sin(th * 0.5);
            double qw = std::cos(th * 0.5);
            f << i << " " << t(0) << " " << t(1) << " 0 0 0 " << qz << " " << qw << "\n";
        } else {
            Eigen::Matrix3d R3 = R;
            Eigen::Quaterniond q(R3);
            f << i << " " << t(0) << " " << t(1) << " " << t(2)
              << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
    }
}

void writeWeights(const Vector& w, const std::string& path) {
    std::ofstream f(path);
    f.precision(17);
    f << "# final_weights (size=" << w.size() << ")\n";
    for (Eigen::Index k = 0; k < w.size(); ++k) {
        f << k << " " << w(k) << "\n";
    }
}

template <size_t d>
int runMain(const DataParser::Measurement& m, const std::string& outputFile) {
    const size_t p = d;
    FastVector<uint64_t> odom_indices;
    size_t num_pose_factors = 0, num_lmk_factors = 0;
    NonlinearFactorGraph graph = buildLiftedLandmarkGraph<d>(
        m, p, odom_indices, num_pose_factors, num_lmk_factors);

    Values init;
    if (FLAGS_init_type == "odom" || FLAGS_init_type == "Odom") {
        init = odomInitLifted<d>(m, p);
    } else if (FLAGS_init_type == "random" || FLAGS_init_type == "Random") {
        init = randomInitLifted<d>(m, p, static_cast<unsigned>(FLAGS_seed));
    } else {
        LOG(FATAL) << "Unknown --init_type: " << FLAGS_init_type
                   << " (expected: random | odom)";
    }

    LevenbergMarquardtParams lmp = LevenbergMarquardtParams::CeresDefaults();
    lmp.maxIterations = FLAGS_inner_max_iterations;
    lmp.relativeErrorTol = FLAGS_rel_error_tol;
    lmp.absoluteErrorTol = FLAGS_abs_error_tol;
    lmp.lambdaUpperBound = std::numeric_limits<double>::max();
    lmp.verbosityLM = LevenbergMarquardtParams::SILENT;

    GncParams<LevenbergMarquardtParams> gncp(lmp);
    gncp.setLossType(GncLossType::TLS);
    gncp.setMaxIterations(FLAGS_gnc_max_iterations);
    gncp.setMuStep(FLAGS_mu_step);
    gncp.setRelativeCostTol(FLAGS_rel_cost_tol);
    gncp.setWeightsTol(FLAGS_weights_tol);
    gncp.setVerbosityGNC(
        static_cast<GncParams<LevenbergMarquardtParams>::Verbosity>(FLAGS_gnc_verbosity));
    gncp.setKnownInliers(odom_indices);

    GncOptimizer<GncParams<LevenbergMarquardtParams>> gnc(graph, init, gncp);

    auto t0 = steady_clock::now();
    Values result;
    bool ok = true;
    try {
        result = gnc.optimize();
    } catch (const std::exception& e) {
        ok = false;
        LOG(ERROR) << "[vanilla-lmk] GNC threw: " << e.what();
        result = init;
    }
    auto t1 = steady_clock::now();
    double total_sec = duration_cast<duration<double>>(t1 - t0).count();

    writeTumTrajectory<d>(result, m.num_poses, outputFile + ".txt");
    Vector w_meas = gnc.getWeights().head(
        static_cast<Eigen::Index>(num_pose_factors + num_lmk_factors));
    writeWeights(w_meas, outputFile + "_weights.txt");

    {
        std::ofstream summ(outputFile + "_summary.csv");
        summ.precision(17);
        summ << "field,value\n";
        summ << "method,vanilla\n";
        summ << "num_gnc_iters,\n";
        summ << "final_starting_rank," << p << "\n";
        summ << "final_ending_rank," << p << "\n";
        summ << "max_ending_rank," << p << "\n";
        summ << "levels_climbed_total,0\n";
        summ << "num_certifier_verified,\n";
        summ << "num_certifier_failed,\n";
        summ << "sum_lm_opt_time," << total_sec << "\n";
        summ << "sum_verification_time,0\n";
        summ << "sum_initialization_time,0\n";
        summ << "sum_certifier_time,0\n";
        summ << "final_objective,\n";
        summ << "final_eta,\n";
        summ << "final_total_time," << total_sec << "\n";
        summ << "status," << (ok ? "ok" : "error") << "\n";
    }

    LOG(INFO) << "[vanilla-lmk] done seed=" << FLAGS_seed
              << " time=" << total_sec << "s  ok=" << ok;
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;

    gflags::SetUsageMessage(
        "Vanilla GTSAM GNC (TLS, LM) on the lifted landmark factor graph at p=d. "
        "No Riemannian staircase.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const std::string inputFile = FLAGS_input_dir;
    const std::string outputFile = FLAGS_output_dir;
    std::filesystem::path outputParent =
        std::filesystem::path(outputFile).parent_path();
    if (!outputParent.empty())
        std::filesystem::create_directories(outputParent);

    size_t num_poses = 0;
    auto m = DataParser::read_g2o_file(inputFile, num_poses);
    LOG(INFO) << "[vanilla-lmk] " << m.poseMeasurements.size()
              << " pose factors + " << m.landmarkMeasurements.size()
              << " landmark factors over " << m.num_poses
              << " poses, " << m.num_landmarks << " landmarks (d=" << FLAGS_d << ")";

    if (FLAGS_d == 2) return runMain<2>(m, outputFile);
    LOG(FATAL) << "Landmark SLAM only supports d=2 in this build.";
    return 1;
}
