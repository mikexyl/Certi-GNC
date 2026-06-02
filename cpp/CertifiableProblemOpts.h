/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    CertifiableProblemOpts.h
 * @brief   Configuration options for the certifiable factor-graph solver
 *          (LM tolerances, staircase verification, init type).
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
 */

#ifndef CERTIFIABLEPROBLEMOPTS_H
#define CERTIFIABLEPROBLEMOPTS_H
#include "Certifiable_problem.h"

namespace gtsam
{
    /**
     * @brief Configuration options for certifiable problem routines.
     *
     * Contains parameters for the Levenberg–Marquardt optimizer as well as
     * settings for the fast verification (eigenvalue) step.
     */
    struct CertifiableProblemOpts
    {
        /// Levenberg–Marquardt optimizer parameters (default: Ceres defaults).
        LevenbergMarquardtParams lmParams = LevenbergMarquardtParams::CeresDefaults();

        /// Maximum number of LM iterations before termination.
        int maxIterations = 200;

        /// Relative error tolerance for LM convergence.
        double relativeErrorTol = 1e-15;

        /// Absolute error tolerance for LM convergence.
        double absoluteErrorTol = 1e-15;

        /// Level of verbosity for LM output (SUMMARY, SILENT, etc.).
        LevenbergMarquardtParams::VerbosityLM verbosityLM = LevenbergMarquardtParams::SUMMARY;

        /// Use absolute regularization term eat, or relative one
        /// Usually use relative eta in range-related example
        bool useAbsoluteEta = false;

        /// Regularization parameter eta for certificate matrix M = S + eta·I.
        Scalar eta = 1e-3;

        /// Lower bound for the eta
        Scalar MIN_CERT_ETA = 1e-4;

        /// Upper bound for the eta
        Scalar MAX_CERT_ETA = 1e-1;

        /// Relative ratio for the eta, i.e., eta = objective_value * REL_CERT_ETA
        Scalar REL_CERT_ETA =  4e-8;

        /// Block size (number of eigenvectors) for LOBPCG verification.
        size_t nx = 4;

        /// Maximum number of iterations for the LOBPCG solver.
        size_t max_iters = 100;

        /// Maximum fill factor for the ILDL preconditioner.
        Scalar max_fill_factor = 3;

        /// Drop tolerance for the ILDL preconditioner.
        Scalar drop_tol = 1;

        // Initialization type
        enum class InitType { Odom, Random, LocalSearch };
        InitType initType = InitType::Random;

        // Export the trajectories of each level, for generating animation
        bool save_animation = false;

        /// Base seed for random initialization. Each variable's per-variable
        /// seed is derived as randomSeed + variable_index, so changing this
        /// produces an independent random starting point for all variables.
        unsigned randomSeed = std::default_random_engine::default_seed;

    };
}  // namespace gtsam



#endif //CERTIFIABLEPROBLEMOPTS_H
