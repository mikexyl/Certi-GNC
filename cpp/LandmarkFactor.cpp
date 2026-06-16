/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    LandmarkFactor.cpp
 * @brief   GTSAM-style lifted pose-to-landmark observation factor (implementation).
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
 */

#include <gtsam/base/timing.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include "LandmarkFactor.h"
#include <iostream>
#include <vector>

namespace gtsam
{

    //******************************************************************************
    /**
     * @brief Constructor for a lifted landmark factor.
     * @tparam d  Ambient dimension (2 or 3).
     * @param j1    Key of the observing pose variable.
     * @param j2    Key of the landmark variable.
     * @param T12   Measured translation vector V_ from pose to landmark.
     * @param p     Relaxation rank.
     * @param model Shared noise model; must have dimension == p.
     * @throws std::invalid_argument if noise model dimension ≠ p.
     */
    template <size_t d>
    LiftedLandmarkFactor<d>::LiftedLandmarkFactor(Key j1, Key j2,
                                                  const Vector &T12,
                                                  size_t p,
                                                  const SharedNoiseModel &model)
      : NoiseModelFactorN<LiftedPoseDP, Vector>(model, j1, j2),
        V_(T12),                     // store measured offset
        p_(p),                       // store relaxation rank
        d_(d),                       // ambient dimension
        pd_(p * d)                   // product p*d for convenience
    {
        // ensure noise model matches expected dimension
        if (noiseModel()->dim() != p_)
            throw std::invalid_argument(
                "LandmarkFactor: model with incorrect dimension.");
    }

    //******************************************************************************
    /**
     * @brief Print this factor’s contents for debugging.
     * @param s             Prefix string.
     * @param keyFormatter  Function to convert Keys → printable strings.
     */
    template <size_t d>
    void LiftedLandmarkFactor<d>::print(const std::string &s,
                                        const KeyFormatter &keyFormatter) const
    {
        std::cout << s
                  << "LandmarkFactor<" << p_ << " * " << d
                  << ">(" << keyFormatter(key<1>()) << ","
                  << keyFormatter(key<2>()) << ")\n";
        traits<Matrix>::Print(V_, "  V: ");         // print measured vector
        noiseModel_->print("  noise model: ");      // print noise details
    }

    //******************************************************************************
    /**
     * @brief Check equality with another factor.
     * @param expected  Other factor to compare.
     * @param tol       Numeric tolerance.
     * @return          True if same type, keys, rank, and measurement.
     */
    template <size_t d>
    bool LiftedLandmarkFactor<d>::equals(const NonlinearFactor &expected,
                                         double tol) const
    {
        auto e = dynamic_cast<const LiftedLandmarkFactor *>(&expected);
        return e != nullptr
            && NoiseModelFactorN<LiftedPoseDP, Vector>::equals(*e, tol)
            && p_ == e->p_
            && V_ == e->V_;
    }

    //******************************************************************************
    /**
     * @brief Compute Jacobians of the error w.r.t. pose and landmark.
     * @param Q1  Current lifted pose variable.
     * @param L1  Current landmark variable vector.
     * @param H1  Optional output Jacobian w.r.t. Q1.
     * @param H2  Optional output Jacobian w.r.t. L1.
     */
    template <size_t d>
    void LiftedLandmarkFactor<d>::fillJacobians(const LiftedPoseDP &Q1,
                                                const Vector &L1,
                                                boost::optional<Matrix&> H1,
                                                boost::optional<Matrix&> H2) const
    {
        const StiefelManifoldKP Y1 = Q1.get_Y();    // rotation part
        const Vector t1 = Q1.get_t();              // translation from pose
        const Vector t2 = L1;                      // landmark translation
        const size_t St_OutDim = p_ * d_;          // flattened rotation block size
        const size_t St_dim    = StiefelManifoldKP::Dimension(d_, p_);
        const size_t t_OutDim  = p_;               // translation output dim
        const size_t t_dim     = p_;               // translation input dim

        if (H1) {
            // Jacobian w.r.t. the lifted pose Q1: [derror/dY | derror/dt1]
            H1->resize(p_, St_dim + t_dim);
            H1->setZero();

            // build derivative of translation error w.r.t. Y1
            Matrix dt_dY = Matrix::Zero(t_OutDim, St_OutDim);
            for (std::size_t j = 0; j < d_; ++j) {
                dt_dY.block(0, j * p_, p_, p_) -= V_(j)
                    * Matrix::Identity(p_, p_);
            }
            H1->block(0, 0, t_OutDim, St_dim) = dt_dY * Y1.G_;
            H1->block(0, St_dim, t_OutDim, t_dim) =
                -Matrix::Identity(t_OutDim, t_dim);
        }

        if (H2) {
            // Jacobian w.r.t. landmark L1: identity in translation
            H2->resize(p_, t_dim);
            H2->setZero();
            H2->block(0, 0, t_OutDim, t_dim) =
                Matrix::Identity(t_OutDim, t_dim);
        }
    }

    //******************************************************************************
    /**
     * @brief Evaluate the error vector for this factor.
     * @param Q1   Current lifted pose variable.
     * @param L1   Current landmark variable vector.
     * @param H1   Optional Jacobian w.r.t. Q1.
     * @param H2   Optional Jacobian w.r.t. L1.
     * @return     Error = t2 - t1 - Y1.matrix() * V_.
     */
    template <size_t d>
    Vector LiftedLandmarkFactor<d>::evaluateError(const LiftedPoseDP &Q1,
                                                  const Vector &L1,
                                                  boost::optional<Matrix&> H1,
                                                  boost::optional<Matrix&> H2) const
    {
        const StiefelManifoldKP Y1 = Q1.get_Y();
        // translation error: measured minus predicted
        Vector error_trans = L1 - Q1.get_t() - Y1.matrix() * V_;

        // fill H1/H2 if requested
        this->fillJacobians(Q1, L1, H1, H2);

        return error_trans;
    }

    template <size_t d>
    EuclideanHessianBlock LiftedLandmarkFactor<d>::computeEuclideanHessian() const {
        using namespace Eigen;
        MatrixXd Id = MatrixXd::Identity(d, d);
        
        int D_pose = d + 1;
        int D_lmk = 1;
        int D = D_pose + D_lmk;
        Matrix H = Matrix::Zero(D, D);
        int R_idx = 0;
        int t_idx = d;
        int L_idx = d + 1;

        // H_RR = VV^T
        H.block(R_idx, R_idx, d, d) = 2 * V_ * V_.transpose();
        // H_Rt = V
        H.block(R_idx, t_idx, d, 1) = 2 * V_;
        H.block(t_idx, R_idx, 1, d) = 2 * V_.transpose();
        // H_RL = -V
        H.block(R_idx, L_idx, d, 1) = 2 * -V_;
        H.block(L_idx, R_idx, 1, d) = 2 * -V_.transpose();
        // H_tt = 1
        H(t_idx, t_idx) = 2 * 1.0;
        // H_tL = -1
        H(t_idx, L_idx) = 2 * -1.0;
        H(L_idx, t_idx) = 2 * -1.0;
        // H_LL = 1
        H(L_idx, L_idx) = 2 * 1.0;

        double tau_sigma = noiseModel_->sigmas()(0);
        double tau = 1.0 / (tau_sigma * tau_sigma);
        return {{key1(), key2()}, {(int)D_pose, (int)D_lmk}, tau * H};
    }

    // Explicit instantiations for d=2 and d=3
    template class LiftedLandmarkFactor<2>;
    template class LiftedLandmarkFactor<3>;

} // namespace gtsam

