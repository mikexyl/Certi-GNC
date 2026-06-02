/* ----------------------------------------------------------------------------
 * This file is a derivative of gtsam::GncParams from GTSAM.
 *
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 *
 * Original source:
 *   gtsam/nonlinear/GncParams.h
 *   Authors: Jingnan Shi, Luca Carlone, Frank Dellaert
 *
 * Modifications copyright 2026, Northeastern University Robust Autonomy Lab.
 * Distributed under the same license as the original.
 * -------------------------------------------------------------------------- */

/**
 * @file    CertifiableGncParams.h
 * @brief   Parameter struct for the certifiable GNC wrapper. (Only TLS loss type for now)
 *
 * @author  Zhexin Xu
 *
 * References:
 *   Xu, Zhang, Calatrava, Closas, Rosen.
 *   "Implementing Robust M-Estimators with Certifiable Factor Graph
 *   Optimization." arXiv:2603.20932, 2026.
 *
 *   Xu, Sanderson, Zhang, Rosen. "Certifiable Estimation with Factor Graphs."
 *   (Certi-fgo) arXiv:2603.01267, 2026.
 *
 *   Yang, Antonante, Tzoumas, Carlone. "Graduated Non-Convexity for Robust
 *   Spatial Perception." ICRA/RAL, 2020.
 *
 * Derived from gtsam::GncParams (Shi, Carlone, Dellaert). TLS only;
 * Geman-McClure enum value and switch arms are removed. Adds `export_path`
 * used by the wrapper for per-run summary CSVs.
 */

#ifndef STIEFELMANIFOLDEXAMPLE_CERTIFIABLEGNCPARAMS_H
#define STIEFELMANIFOLDEXAMPLE_CERTIFIABLEGNCPARAMS_H

#include <gtsam/base/FastVector.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <algorithm>
#include <iostream>
#include <string>

namespace gtsam {

/// Robust loss function for GNC. Only TLS is supported in this build.
enum CertifiableGncLossType {
    Certifiable_TLS  ///< Truncated least squares.
};

template <class BaseOptimizerParameters>
class CertifiableGncParams {
public:
    typedef typename BaseOptimizerParameters::OptimizerType OptimizerType;

    enum Verbosity {
        SILENT = 0,
        SUMMARY,
        MU,
        WEIGHTS,
        VALUES
    };

    CertifiableGncParams() : baseOptimizerParams() {}
    explicit CertifiableGncParams(const BaseOptimizerParameters& bp)
        : baseOptimizerParams(bp) {}

    BaseOptimizerParameters baseOptimizerParams;

    CertifiableGncLossType lossType = Certifiable_TLS;
    size_t maxIterations = 100;
    double muStep = 1.4;
    double relativeCostTol = 1e-5;
    double absoluteErrorTol = 1e-5;
    double weightsTol = 1e-4;
    Verbosity verbosity = SILENT;

    using IndexVector = FastVector<uint64_t>;
    IndexVector knownInliers;
    IndexVector knownOutliers;

    /// Output directory for per-iteration logs and Monte-Carlo timing CSV.
    std::string export_path;

    void setLossType(const CertifiableGncLossType type) { lossType = type; }
    void setMaxIterations(const size_t maxIter) { maxIterations = maxIter; }
    void setMuStep(const double step) { muStep = step; }
    void setRelativeCostTol(double value) { relativeCostTol = value; }
    void setAbsoluteErrorTol(double value) { absoluteErrorTol = value; }
    void setWeightsTol(double value) { weightsTol = value; }
    void setVerbosityGNC(const Verbosity value) { verbosity = value; }

    void setKnownInliers(const IndexVector& knownIn) {
        for (size_t i = 0; i < knownIn.size(); i++) {
            knownInliers.push_back(knownIn[i]);
        }
        std::sort(knownInliers.begin(), knownInliers.end());
        // De-duplicate in case the user supplied overlapping sets.
        knownInliers.erase(std::unique(knownInliers.begin(), knownInliers.end()),
                           knownInliers.end());
    }

    void setKnownOutliers(const IndexVector& knownOut) {
        for (size_t i = 0; i < knownOut.size(); i++) {
            knownOutliers.push_back(knownOut[i]);
        }
        std::sort(knownOutliers.begin(), knownOutliers.end());
        knownOutliers.erase(std::unique(knownOutliers.begin(), knownOutliers.end()),
                            knownOutliers.end());
    }

    bool equals(const CertifiableGncParams& other, double tol = 1e-9) const {
        return baseOptimizerParams.equals(other.baseOptimizerParams)
            && lossType == other.lossType
            && maxIterations == other.maxIterations
            && std::fabs(muStep - other.muStep) <= tol
            && verbosity == other.verbosity
            && knownInliers == other.knownInliers
            && knownOutliers == other.knownOutliers;
    }

    void print(const std::string& str) const {
        std::cout << str << "\n";
        std::cout << "lossType: Truncated Least-squares\n";
        std::cout << "maxIterations: " << maxIterations << "\n";
        std::cout << "muStep: " << muStep << "\n";
        std::cout << "relativeCostTol: " << relativeCostTol << "\n";
        std::cout << "absoluteErrorTol: " << absoluteErrorTol << "\n";
        std::cout << "weightsTol: " << weightsTol << "\n";
        std::cout << "verbosity: " << verbosity << "\n";
        for (size_t i = 0; i < knownInliers.size(); i++)
            std::cout << "knownInliers: " << knownInliers[i] << "\n";
        for (size_t i = 0; i < knownOutliers.size(); i++)
            std::cout << "knownOutliers: " << knownOutliers[i] << "\n";
        baseOptimizerParams.print("Base optimizer params: ");
    }
};

}  // namespace gtsam

#endif  // STIEFELMANIFOLDEXAMPLE_CERTIFIABLEGNCPARAMS_H
