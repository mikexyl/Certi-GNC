/* ----------------------------------------------------------------------------
* Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/make_shared.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <boost/optional/optional.hpp>
#include "../utils.h"
#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

using namespace gtsam;
using namespace std;

/**
 * @brief A simple user-defined factor for testing the DataMatrixAssembler.
 * It represents a simple quadratic cost 0.5 * w * ||x1 - x2||^2.
 * The Euclidean Hessian for this is:
 * [  w  -w ]
 * [ -w   w ]
 */
class SimpleDistanceFactor : public NoiseModelFactor2<Vector, Vector>, public EuclideanFactor {
    double w_;
public:
    SimpleDistanceFactor(Key k1, Key k2, double w) 
        : NoiseModelFactor2<Vector, Vector>(noiseModel::Unit::Create(1), k1, k2), w_(w) {}

    Vector evaluateError(const Vector& x1, const Vector& x2,
                         boost::optional<Matrix&> H1,
                         boost::optional<Matrix&> H2) const override {
        if (H1) *H1 = Matrix::Identity(x1.size(), x1.size());
        if (H2) *H2 = -Matrix::Identity(x2.size(), x2.size());
        return x1 - x2;
    }

    EuclideanHessianBlock computeEuclideanHessian() const override {
        Matrix H(2, 2);
        H << w_, -w_,
            -w_,  w_;
        std::vector<Key> k_vec;
        for(Key k : keys()) k_vec.push_back(k);
        return {k_vec, {1, 1}, H};
    }
};

TEST(DataMatrixAssembler, SimpleManualVerification) {
    // 1. Create a simple graph with our new factor
    NonlinearFactorGraph graph;
    Key x1 = Symbol('L', 1);
    Key x2 = Symbol('L', 2);
    double w = 5.0;
    graph.add(gtsam::make_shared<SimpleDistanceFactor>(x1, x2, w));

    // 2. Define ordering and layout
    Ordering ordering;
    ordering.push_back(x1);
    ordering.push_back(x2);
    // Landmark variables have dim 1
    GlobalLayout layout(ordering, 1, BlockOrderingType::RtRt);

    // 3. Assemble the data matrix
    SparseMatrix Q = DataMatrixAssembler::Assemble(graph, layout);

    // 4. Verify its properties against a manually derived matrix
    // For this simple case, the global Q should be:
    // [  5  -5 ]
    // [ -5   5 ]
    Matrix Q_dense = Matrix(Q);
    Matrix Q_expected(2, 2);
    Q_expected << 5, -5,
                 -5,  5;

    EXPECT_TRUE(Q_dense.isApprox(Q_expected));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
