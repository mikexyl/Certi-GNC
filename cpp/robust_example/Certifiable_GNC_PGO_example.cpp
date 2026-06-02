/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file Certifiable_GNC_PGO_example.cpp
 * @brief TLS-GNC wrapper around the certifiable PGO solver.
 *
 * Loads a g2o pose graph (which may contain outlier edges), constructs the
 * StiefelManifold CertifiablePGO problem, and runs the GNC outer loop with
 * truncated-least-squares loss. Odometry edges (j == i + 1) are treated as
 * known inliers.
 */

#include "../robust/CertifiableGncOptimizer.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_int32(d, 2, "Ambient dimension (2 or 3).");
DEFINE_int32(p, 2, "Initial relaxation rank.");
DEFINE_string(input_dir, "path_to_data/pgo/intel_outl20.g2o", "Input g2o file.");
DEFINE_string(output_dir, "results/pgo_demo", "Output file stem.");
DEFINE_string(init_type, "random", "Initialization type: random | odom.");

DEFINE_int32(inner_max_iterations, 200, "Max LM iterations per inner solve.");
DEFINE_int32(gnc_max_iterations, 30, "Max GNC outer iterations.");
DEFINE_double(mu_step, 1.4, "Multiplicative factor for mu update (TLS).");
DEFINE_double(rel_cost_tol, 1e-5, "Relative cost tolerance.");
DEFINE_double(abs_cost_tol, 0.0,
              "GNC absolute cost tolerance. Default 0 disables the abs check "
              "so only --rel_cost_tol controls convergence (matches "
              "gtsam::GncOptimizer which only exposes a relative criterion).");
DEFINE_double(weights_tol, 1e-4, "Weights binary-convergence tolerance.");
DEFINE_int32(gnc_verbosity, 1, "GNC verbosity (0=SILENT,1=SUMMARY,2=MU,3=WEIGHTS,4=VALUES).");
DEFINE_int32(seed, 1, "Random seed for initial guess (only used when init_type=random).");

DEFINE_double(abs_error_tol, 1e-5, "LM absolute error tolerance.");
DEFINE_double(rel_error_tol, 1e-5, "LM relative error tolerance.");
DEFINE_bool(use_abs_eta, true, "Use absolute-eta schedule.");
DEFINE_double(eta, 1e-3, "Verification eta (when use_abs_eta=true).");
DEFINE_double(MIN_CERT_ETA, 1e-3, "Lower bound for eta.");
DEFINE_double(MAX_CERT_ETA, 1e-2, "Upper bound for eta.");
DEFINE_double(REL_CERT_ETA, 1e-6, "Relative eta scaling.");

using namespace gtsam;

namespace {

CertifiableProblemOpts::InitType parseInitType(const std::string& s) {
    if (s == "random" || s == "Random") return CertifiableProblemOpts::InitType::Random;
    if (s == "odom" || s == "Odom") return CertifiableProblemOpts::InitType::Odom;
    LOG(FATAL) << "Unknown --init_type: " << s << " (expected: random | odom)";
    return CertifiableProblemOpts::InitType::Random;
}

template <size_t d>
void runGncPgo(const DataParser::Measurement& measurements,
               const std::string& outputFile) {
    using ProblemT = CertifiablePGO<d>;
    auto problem = std::make_shared<ProblemT>(FLAGS_p, measurements);
    problem->opts_.maxIterations = FLAGS_inner_max_iterations;
    problem->opts_.absoluteErrorTol = FLAGS_abs_error_tol;
    problem->opts_.relativeErrorTol = FLAGS_rel_error_tol;
    problem->opts_.useAbsoluteEta = FLAGS_use_abs_eta;
    problem->opts_.eta = FLAGS_eta;
    problem->opts_.MIN_CERT_ETA = FLAGS_MIN_CERT_ETA;
    problem->opts_.MAX_CERT_ETA = FLAGS_MAX_CERT_ETA;
    problem->opts_.REL_CERT_ETA = FLAGS_REL_CERT_ETA;
    problem->opts_.initType = parseInitType(FLAGS_init_type);
    problem->opts_.verbosityLM = LevenbergMarquardtParams::SILENT;
    problem->opts_.randomSeed = static_cast<unsigned>(FLAGS_seed);
    problem->init();

    CertifiableGncParams<LevenbergMarquardtParams> gncParams;
    gncParams.setLossType(Certifiable_TLS);
    gncParams.setMaxIterations(FLAGS_gnc_max_iterations);
    gncParams.setMuStep(FLAGS_mu_step);
    gncParams.setRelativeCostTol(FLAGS_rel_cost_tol);
    gncParams.setAbsoluteErrorTol(FLAGS_abs_cost_tol);
    gncParams.setWeightsTol(FLAGS_weights_tol);
    gncParams.setVerbosityGNC(
        static_cast<typename CertifiableGncParams<LevenbergMarquardtParams>::Verbosity>(
            FLAGS_gnc_verbosity));
    gncParams.export_path =
        std::filesystem::path(outputFile).parent_path().string();

    CertifiableGncOptimizer<CertifiableGncParams<LevenbergMarquardtParams>> gnc(problem, gncParams);
    gnc.setOutputPath(outputFile);

    const size_t pMin = FLAGS_p;
    const size_t pMax = d + 10;
    auto result = gnc.CertifiableOptimize(pMin, pMax);

    gnc.exportGncIterationLogTXTandCSVs(result, outputFile, /*dump_weights_csv=*/true);
    LOG(INFO) << "GNC finished: total time " << result.finalTotalTime_ << "s, "
              << result.iter_log_.size() << " iters.";
}

}  // namespace

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_logtostderr = true;

    gflags::SetUsageMessage("Run TLS-GNC on a certifiable PGO problem.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int d = FLAGS_d;
    const std::string inputFile = FLAGS_input_dir;
    const std::string outputFile = FLAGS_output_dir;

    CHECK(!inputFile.empty() && !outputFile.empty())
        << "--input_dir and --output_dir must be set";

    // Ensure output directory exists.
    std::filesystem::path outputParent = std::filesystem::path(outputFile).parent_path();
    if (!outputParent.empty()) std::filesystem::create_directories(outputParent);

    LOG(INFO) << "[START] d=" << d << " p=" << FLAGS_p
              << " init=" << FLAGS_init_type
              << " input=" << inputFile << " output=" << outputFile;

    size_t num_poses = 0;
    auto measurements = DataParser::read_g2o_file(inputFile, num_poses);
    LOG(INFO) << "Loaded " << measurements.poseMeasurements.size()
              << " pose measurements between " << measurements.num_poses << " poses.";

    try {
        if (d == 2) {
            runGncPgo<2>(measurements, outputFile);
        } else if (d == 3) {
            runGncPgo<3>(measurements, outputFile);
        } else {
            LOG(FATAL) << "Unsupported dimension d=" << d;
        }
    } catch (const std::exception& e) {
        LOG(FATAL) << "GNC threw exception: " << e.what();
    }

    LOG(INFO) << "[DONE]  Results written to " << outputFile;
    return 0;
}
