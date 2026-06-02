/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    RandomInit.h
 * @brief   Shared random-initialization helpers. Both the certifiable and
 *          vanilla GNC binaries call these so a given --seed produces
 *          identical lifted starting Values.
 *
 * @author  Zhexin Xu
 *
 * References:
 *   Zhexin Xu, Hanna Jiamei Zhang, Helena Calatrava, Pau Closas, David M. Rosen.
 *   "Implementing Robust M-Estimators with Certifiable Factor Graph
 *   Optimization." arXiv preprint arXiv:2603.20932, 2026.
 */

#ifndef STIEFELMANIFOLDEXAMPLE_RANDOMINIT_H
#define STIEFELMANIFOLDEXAMPLE_RANDOMINIT_H

#include "LiftedPose.h"
#include "RelativePoseMeasurement.h"
#include "StiefelManifold.h"

#include <gtsam/geometry/Rot2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Geometry>

#include <random>

namespace gtsam {

/**
 * @brief Uniform SE(d) sample per pose, lifted to St(p,d) + R^p.
 *
 * For each pose i = 0..num_poses-1, draws (x, y[, z], θ) uniformly and
 * lifts to LiftedPoseDP. At p = d the lift is a no-op; for p > d the
 * rotation is zero-row-padded and the translation is zero-padded.
 */
inline Values randomLiftedSE2dPoses(size_t num_poses, size_t d, size_t p,
                                    unsigned seed,
                                    double trans_range = 10.0) {
    Values out;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ut(-trans_range, trans_range);
    std::uniform_real_distribution<double> ur(-M_PI, M_PI);

    for (size_t i = 0; i < num_poses; ++i) {
        Matrix Rd;
        Vector td;
        if (d == 2) {
            const double x = ut(rng);
            const double y = ut(rng);
            const double th = ur(rng);
            Rd = Eigen::Rotation2D<double>(th).toRotationMatrix();
            td.resize(2);
            td << x, y;
        } else {
            const double x = ut(rng);
            const double y = ut(rng);
            const double z = ut(rng);
            const double qx = ut(rng), qy = ut(rng), qz = ut(rng), qw = ut(rng);
            Eigen::Quaterniond q(qw, qx, qy, qz);
            q.normalize();
            Rd = q.toRotationMatrix();
            td.resize(3);
            td << x, y, z;
        }
        StiefelManifoldKP Y = StiefelManifoldKP::Lift(p, Rd);
        Vector tp = LiftedPoseDP::LiftToRp(td, p);
        out.insert(static_cast<Key>(i), LiftedPoseDP(Y, tp));
    }
    return out;
}

/**
 * @brief Append random landmark Values, lifted to R^p.
 *
 * Iterates `landmarkMeasurements` in order and inserts a 2D/3D uniform
 * point on first sighting of each landmark id. Reuses the caller's
 * generator so the RNG sequence stays in sync with the pose draws.
 */
inline void addRandomLiftedLandmarks(
        const DataParser::landmarkMeasurements_t& obs,
        size_t d, size_t p,
        std::mt19937& rng,
        Values& out,
        double trans_range = 10.0) {
    std::uniform_real_distribution<double> ut(-trans_range, trans_range);
    for (const auto& m : obs) {
        Key key = Symbol('L', m.j);
        if (out.exists(key)) continue;
        Vector td;
        if (d == 2) {
            td.resize(2);
            td << ut(rng), ut(rng);
        } else {
            td.resize(3);
            td << ut(rng), ut(rng), ut(rng);
        }
        out.insert(key, LiftedPoseDP::LiftToRp(td, p));
    }
}

}  // namespace gtsam

#endif  // STIEFELMANIFOLDEXAMPLE_RANDOMINIT_H
