/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    CertifiableResults.h
 * @brief   Container for results / iteration information produced by the
 *          certifiable optimization routines.
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

#ifndef CERTIFIABLERESULTS_H
#define CERTIFIABLERESULTS_H

#include "utils.h"
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>

namespace gtsam
{

    /**
     * @brief Status of the global-optimality verification at the final rank.
     */
    enum class VerificationStatus : int {
        NOT_VERIFIED = 0,  ///< No verification has been attempted.
        VERIFIED     = 1,  ///< Solution passed PSD certificate test.
        FAILED       = 2   ///< Staircase reached pMax without certifying.
    };

    inline std::ostream& operator<<(std::ostream& os, VerificationStatus s) {
        switch (s) {
            case VerificationStatus::NOT_VERIFIED: return os << "NOT_VERIFIED";
            case VerificationStatus::VERIFIED:     return os << "VERIFIED";
            case VerificationStatus::FAILED:       return os << "FAILED";
        }
        return os << "UNKNOWN";
    }

    /**
     * @class CertificateResults
     * @brief Container for results produced by the certifiable optimization routines.
     *
     * Stores the relaxation ranks, optimized variables, certificate values, timing metrics,
     * and methods for exporting results.
     */
    class CertificateResults
    {
    public:
        /// @brief Relaxation rank at which optimization was initialized.
        size_t startingRank;

        /// @brief Relaxation rank at which solution was certified.
        size_t endingRank;

        /// @brief Variable matrix Y at the certified solution (stacked manifold representation).
        SparseMatrix Yopt;

        /**
         * @brief Objective values \f$ F(Z) = F(Y^T Y) \f$ recorded at each relaxation level.
         */
        std::vector<Scalar> SDPval;

        /**
         * @brief Norms of the Riemannian gradient \f$ \| \nabla F(Y) \| \f$ at each relaxation level.
         */
        std::vector<Scalar> gradnorm;

        /// @brief eta values recorded at each relaxation level.
        std::vector<Scalar> etas;

        /**
         * @brief Dual Lagrange multiplier matrix \f$ \Lambda \f$ corresponding to the certified Yopt.
         *
         * Computed per eq. (119) of the SE-Sync technical report. If \f$ Z = Y^T Y \f$
         * solves the dual SDP exactly, then \f$ \Lambda \f$ solves the corresponding primal
         * Lagrangian relaxation.
         */
        SparseMatrix Lambda;

        /**
         * @brief Rounded solution \f$ \hat{x} = [t | R] \in SE(d)^n \f$ after projecting onto the manifold.
         */
        Matrix xhat;

        /// @brief Total wall-clock time (ms) for the entire certifiable algorithm.
        double total_computation_time;

        /**
         * @brief Initialization times (ms) for each rank's random or descent-based start.
         */
        std::vector<double> initialization_time;

        /// @brief Optimization times (ms) taken by Levenberg-Marquardt at each level.
        std::vector<double> elapsed_optimization_times;

        /**
         * @brief Verification times (ms) spent on eigenvalue tests per relaxation level.
         */
        std::vector<double> verification_times;

        /**
         * @brief Trajectories for visualization at each relaxation level.
         */
        std::vector<Matrix> visualizationTraj;

        /**
         * @brief Lifted GTSAM Values at the final certified rank (Qstar).
         *
         * Contains LiftedPoseDP (and Vector for landmarks) at the rank at
         * which the solution was either verified or last optimized. Required by
         * the GNC wrapper to evaluate factor-graph errors during convergence
         * checks and weight updates.
         */
        Values Qstar;

        /**
         * @brief Global-optimality verification status at the returned solution.
         */
        VerificationStatus verification_status_ = VerificationStatus::NOT_VERIFIED;

        /**
         * @brief Convenience: smallest eta used during the (last) verification.
         *
         * The GNC wrapper records this per outer iteration for logging.
         */
        double eta = 0.0;

        /**
         * @brief Export all fields of CertificateResults to a single CSV file.
         *
         * Writes a three-column CSV with rows labeled by field name,
         * index within vector fields, and the corresponding value.
         *
         * @param[in] R CertificateResults instance to export.
         * @param[in] filename Path to the output CSV file.
         */
        static void exportCertificateResultsSingleCSV(
            const CertificateResults& R,
            const std::string& filename)
        {
            std::ofstream out(filename);
            out << "field,index,value\n";

            // 1) Export SDPval vector entries
            for (size_t k = 0; k < R.SDPval.size(); ++k) {
                out << "SDPval," << k << "," << R.SDPval[k] << "\n";
            }

            // 2) Export gradnorm vector entries
            for (size_t k = 0; k < R.gradnorm.size(); ++k) {
                out << "gradnorm," << k << "," << R.gradnorm[k]<< "\n";
            }

            // 3) Export scalar fields
            out << "startingRank,0," << R.startingRank << "\n";
            out << "endingRank,0,"   << R.endingRank   << "\n";

            // 4) Export initialization_time vector entries
            for (size_t k = 0; k < R.initialization_time.size(); ++k) {
                out << "initialization_time," << k << "," << R.initialization_time[k] << "\n";
            }

            // 5) Export elapsed_optimization_times vector entries
            double FinalOptTime = 0;
            for (size_t k = 0; k < R.elapsed_optimization_times.size(); ++k) {
                out << "elapsed_optimization_times," << k << ","
                    << R.elapsed_optimization_times[k] << "\n";
                FinalOptTime += R.elapsed_optimization_times[k];
            }
            // 6) Export verification_times vector entries
            for (size_t k = 0; k < R.verification_times.size(); ++k) {
                out << "verification_times," << k << ","
                    << R.verification_times[k] << "\n";
            }

            // 7) Export etas vector entries
            for (size_t k = 0; k < R.etas.size(); ++k) {
                out << "etas," << k << "," << R.etas[k] << "\n";
            }

            out << "FinalOptimizationTime,0," << FinalOptTime << "\n";
            out << "FinalTotalTime,0," << R.total_computation_time << "\n";

            std::cout << "All fields exported to \"" << filename << "\"\n";
        }
    };

}  // namespace gtsam




#endif //CERTIFIABLERESULTS_H
