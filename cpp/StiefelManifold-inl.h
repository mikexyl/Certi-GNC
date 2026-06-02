/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    StiefelManifold-inl.h
 * @brief   Template implementations for the Stiefel-manifold variable type.
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

#ifndef STIEFELMANIFOLD_STIEFELMANIFOLD_INL_H
#define STIEFELMANIFOLD_STIEFELMANIFOLD_INL_H

#include "StiefelManifold.h"
#include <gtsam/base/Matrix.h>

namespace gtsam {

template<int K_, int P_>
StiefelManifold<K_, P_> StiefelManifold<K_, P_>::projectToManifold(const gtsam::Matrix &A)
{
    Eigen::Index  p = A.rows();
    Eigen::Index  k = A.cols();
    gtsam::Matrix P(p,k);

    if (A.rows() != p || A.cols() != k)
    {
        throw std::runtime_error("Error in projectToManifold");
    }

    Eigen::JacobiSVD<gtsam::Matrix> SVD(A.block(0,0,p,k),Eigen::ComputeThinU | Eigen::ComputeThinV);
    P.block(0,0, p,k) = SVD.matrixU() * SVD.matrixV().transpose();

    return StiefelManifold<K_, P_>::FromMatrix(P);
}

template<int K_, int P_>
Matrix StiefelManifold<K_, P_>::SymBlockDiagProduct(const gtsam::Matrix &A, const gtsam::Matrix &BT, const gtsam::Matrix &C)
{
    Eigen::Index  p = A.rows();
    Eigen::Index  k = A.cols();
    gtsam::Matrix R(p, k);
    gtsam::Matrix P(k,k);
    gtsam::Matrix S(k,k);

    P = BT.block(0,0,k,p) * C.block(0,0,p,k);
    S = 0.5 * (P + P.transpose());
    R.block(0,0,p,k) =  A.block(0,0,p,k) * S;

    return R;
}
template<int K_, int P_>
StiefelManifold<K_, P_> StiefelManifold<K_, P_>::random_sample(const std::default_random_engine::result_type &seed) const
{
    std::default_random_engine generator(seed);
    std::normal_distribution<double> g;

    Matrix R(p_,k_);
    for (size_t r = 0; r<p_;++r)
        for (size_t c = 0; c< k_; ++c)
            R(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = g(generator);

    return projectToManifold(R);
}

    template<int K_, int P_>
    StiefelManifold<K_, P_> StiefelManifold<K_, P_>::Random(const std::default_random_engine::result_type &seed, size_t k, size_t p)
    {
        std::default_random_engine generator(seed);
        std::normal_distribution<double> g;

        Matrix R(p,k);
        for (size_t r = 0; r<p;++r)
            for (size_t c = 0; c< k; ++c)
                R(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = g(generator);

        return projectToManifold(R);
    }

template<int K_, int P_>
void StiefelManifold<K_, P_>::print(const std::string &s) const {
    std::cout << s << matrix_ << std::endl;
}


} // namespace gtsam





#endif //STIEFELMANIFOLD_STIEFELMANIFOLD_INL_H
