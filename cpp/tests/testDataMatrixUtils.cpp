/* ----------------------------------------------------------------------------
* Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/Ordering.h>
#include <gtest/gtest.h>
#include "../utils.h"

using namespace gtsam;
using namespace std;

TEST(DataMatrixUtils, GlobalLayoutRtRt) {
    Ordering ordering;
    Key x0 = Symbol('x', 0);
    Key x1 = Symbol('x', 1);
    Key l0 = Symbol('L', 0);
    ordering.push_back(x0);
    ordering.push_back(x1);
    ordering.push_back(l0);
    
    size_t d = 3;
    // Interleaved: [R0(3), t0(1), R1(3), t1(1), L0(1)]
    GlobalLayout layout(ordering, d, BlockOrderingType::RtRt);
    
    EXPECT_EQ(layout.getTotalDim(), 9); // 2*(3+1) + 1 = 9
    
    // Check offsets
    // x0
    EXPECT_EQ(layout.getGlobalIndex(x0, 0), 0); // R0 row 0
    EXPECT_EQ(layout.getGlobalIndex(x0, 2), 2); // R0 row 2
    EXPECT_EQ(layout.getGlobalIndex(x0, 3), 3); // t0 (localRow = d = 3)
    
    // Check claimedDim mapping for Pose
    EXPECT_EQ(layout.getGlobalIndex(x0, 0, 1), 3); // Claimed dim 1 -> must be translation
    
    // x1
    EXPECT_EQ(layout.getGlobalIndex(x1, 0), 4); // R1 row 0
    EXPECT_EQ(layout.getGlobalIndex(x1, 3), 7); // t1
    EXPECT_EQ(layout.getGlobalIndex(x1, 0, 1), 7); // Claimed dim 1 -> must be translation
    
    // l0
    EXPECT_EQ(layout.getGlobalIndex(l0, 0), 8); // L0
    EXPECT_EQ(layout.getGlobalIndex(l0, 0, 1), 8); // Landmark always mapped same way
}

TEST(DataMatrixUtils, GlobalLayouttRtR) {
    Ordering ordering;
    Key x0 = Symbol('x', 0);
    Key x1 = Symbol('x', 1);
    Key l0 = Symbol('L', 0);
    ordering.push_back(x0);
    ordering.push_back(x1);
    ordering.push_back(l0);
    
    size_t d = 3;
    // Blocked: [t0(1), t1(1), R0(3), R1(3), L0(1)]
    GlobalLayout layout(ordering, d, BlockOrderingType::tRtR);
    
    EXPECT_EQ(layout.getTotalDim(), 9);
    
    // Check offsets
    // x0
    EXPECT_EQ(layout.getGlobalIndex(x0, 3), 0); // t0
    EXPECT_EQ(layout.getGlobalIndex(x0, 0), 2); // R0 row 0
    EXPECT_EQ(layout.getGlobalIndex(x0, 0, 1), 0); // Claimed dim 1 -> translation
    
    // x1
    EXPECT_EQ(layout.getGlobalIndex(x1, 3), 1); // t1
    EXPECT_EQ(layout.getGlobalIndex(x1, 0), 5); // R1 row 0
    EXPECT_EQ(layout.getGlobalIndex(x1, 0, 1), 1); // Claimed dim 1 -> translation
    
    // l0
    EXPECT_EQ(layout.getGlobalIndex(l0, 0), 8); // L0
}

TEST(DataMatrixUtils, GlobalLayoutNonSequential) {
    Ordering ordering;
    Key x10 = Symbol('x', 10);
    Key l5 = Symbol('L', 5);
    ordering.push_back(x10);
    ordering.push_back(l5);
    
    size_t d = 2;
    // RtRt: [R10(2), t10(1), L5(1)] -> Total 4
    GlobalLayout layout(ordering, d, BlockOrderingType::RtRt);
    
    EXPECT_EQ(layout.getTotalDim(), 4);
    EXPECT_EQ(layout.getGlobalIndex(x10, 0), 0);
    EXPECT_EQ(layout.getGlobalIndex(x10, 2), 2);
    EXPECT_EQ(layout.getGlobalIndex(l5, 0), 3);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
