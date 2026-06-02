/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file Vanilla_GNC_PGO_example.cpp
 * @brief Vanilla GTSAM GNC (TLS loss, LM inner solver) on the *same lifted*
 *        SE-sync factor graph as the certifiable wrapper, but without the
 *        Riemannian staircase.
 *
 * The only algorithmic difference is the
 * inner solver: vanilla calls gtsam::LevenbergMarquardt once per GNC
 * iteration and stays at rank p = d; ours climbs the Riemannian staircase
 * until been verified.
 */

#include "../RandomInit.h"
#include "../SEsyncFactor.h"
#include "../utils.h"

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

DEFINE_int32(d, 2, "Ambient dimension (2 or 3).");
DEFINE_string(input_dir, "data/pgo/intel_outl20.g2o", "Input g2o file.");
DEFINE_string(output_dir, "results/vanilla_pgo_demo", "Output file stem.");
DEFINE_int32(seed, 1, "Random seed for initial guess (only used if init_type=random).");
DEFINE_string(init_type, "random",
              "Initial guess source: 'random' (lifted SE(d) i.i.d.) or 'odom' "
              "(VERTEX_SE2/SE3:QUAT values from the input g2o file).");

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

/**
 * Build the *lifted* SE-sync factor graph for PGO at rank p = d. Mirrors
 * the construction in CertifiablePGO::buildGraphAtLevel exactly so the
 * vanilla GNC sees the same factors as the certifiable wrapper.
 */
template <size_t d>
NonlinearFactorGraph buildLiftedPgoGraph(
        const DataParser::Measurement& measurements,
        size_t p,
        FastVector<uint64_t>& odom_indices) {
    NonlinearFactorGraph graph;
    odom_indices.clear();
    for (size_t k = 0; k < measurements.poseMeasurements.size(); ++k) {
        const auto& m = measurements.poseMeasurements[k];
        Vector sigmas = Vector::Zero(p * d + p);
        sigmas.head(p * d).setConstant(std::sqrt(1.0 / (2 * m.kappa)));
        sigmas.tail(p).setConstant(std::sqrt(1.0 / (2 * m.tau)));
        auto noise = noiseModel::Diagonal::Sigmas(sigmas);
        graph.emplace_shared<SEsyncFactor<d>>(m.i, m.j, m.R, m.t, p, noise);
        if (m.j == m.i + 1) odom_indices.push_back(k);
    }
    return graph;
}

/**
 * Lift the dataset's VERTEX_SE(d) values into LiftedPoseDP at rank p. At
 * p = d this is a no-op padding; for p > d, R is zero-row-padded into
 * St(p, d) and t is zero-padded to R^p.
 */
template <size_t d>
Values odomInitLifted(const DataParser::Measurement& measurements, size_t p) {
    Values out;
    for (size_t i = 0; i < measurements.num_poses; ++i) {
        auto it = measurements.initial_poses.find({'A', i});
        Matrix Rd;
        Vector td;
        if (it != measurements.initial_poses.end()) {
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
    return out;
}

/// Write a TUM trajectory by rounding each LiftedPoseDP at p = d back to SE(d).
template <size_t d>
void writeTumTrajectory(const Values& values, size_t num_poses,
                        const std::string& path) {
    std::ofstream f(path);
    f.precision(9);
    for (size_t i = 0; i < num_poses; ++i) {
        Key key = static_cast<Key>(i);
        if (!values.exists(key)) continue;
        const auto& lp = values.at<LiftedPoseDP>(key);
        const Matrix R = lp.get_StiefelElement().matrix().topRows(d);  // d x d
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
int runMain(const DataParser::Measurement& measurements,
            const std::string& outputFile) {
    const size_t p = d;  // vanilla stays at p = d; no staircase climbing.

    FastVector<uint64_t> odom_indices;
    NonlinearFactorGraph graph =
        buildLiftedPgoGraph<d>(measurements, p, odom_indices);

    Values init;
    if (FLAGS_init_type == "odom" || FLAGS_init_type == "Odom") {
        init = odomInitLifted<d>(measurements, p);
    } else if (FLAGS_init_type == "random" || FLAGS_init_type == "Random") {
        init = randomLiftedSE2dPoses(measurements.num_poses, d, p,
                                     static_cast<unsigned>(FLAGS_seed));
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

    GncParams<LevenbergMarquardtParams> gp(lmp);
    gp.setLossType(GncLossType::TLS);
    gp.setMaxIterations(FLAGS_gnc_max_iterations);
    gp.setMuStep(FLAGS_mu_step);
    gp.setRelativeCostTol(FLAGS_rel_cost_tol);
    gp.setWeightsTol(FLAGS_weights_tol);
    gp.setVerbosityGNC(
        static_cast<GncParams<LevenbergMarquardtParams>::Verbosity>(FLAGS_gnc_verbosity));
    gp.setKnownInliers(odom_indices);  // odometry edges always inliers

    GncOptimizer<GncParams<LevenbergMarquardtParams>> gnc(graph, init, gp);

    auto t0 = steady_clock::now();
    Values result;
    bool ok = true;
    try {
        result = gnc.optimize();
    } catch (const std::exception& e) {
        ok = false;
        LOG(ERROR) << "[vanilla-pgo] GNC threw: " << e.what();
        result = init;
    }
    auto t1 = steady_clock::now();
    double total_sec = duration_cast<duration<double>>(t1 - t0).count();

    writeTumTrajectory<d>(result, measurements.num_poses, outputFile + ".txt");
    Vector w_meas = gnc.getWeights().head(
        static_cast<Eigen::Index>(measurements.poseMeasurements.size()));
    writeWeights(w_meas, outputFile + "_weights.txt");

    {
        std::ofstream summ(outputFile + "_summary.csv");
        summ.precision(17);
        summ << "field,value\n";
        summ << "method,vanilla\n";
        summ << "num_gnc_iters,\n";  // GTSAM's GncOptimizer doesn't expose iter count
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

    LOG(INFO) << "[vanilla-pgo] done seed=" << FLAGS_seed
              << " time=" << total_sec << "s  ok=" << ok;
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;

    gflags::SetUsageMessage(
        "Vanilla GTSAM GNC (TLS, LM) on the lifted SE-sync factor graph at p=d. "
        "No Riemannian staircase.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const std::string inputFile = FLAGS_input_dir;
    const std::string outputFile = FLAGS_output_dir;
    std::filesystem::path outputParent =
        std::filesystem::path(outputFile).parent_path();
    if (!outputParent.empty())
        std::filesystem::create_directories(outputParent);

    size_t num_poses = 0;
    auto measurements = DataParser::read_g2o_file(inputFile, num_poses);
    LOG(INFO) << "[vanilla-pgo] " << measurements.poseMeasurements.size()
              << " measurements over " << measurements.num_poses << " poses (d="
              << FLAGS_d << ")";

    if (FLAGS_d == 2) return runMain<2>(measurements, outputFile);
    if (FLAGS_d == 3) return runMain<3>(measurements, outputFile);
    LOG(FATAL) << "Unsupported --d: " << FLAGS_d;
    return 1;
}
