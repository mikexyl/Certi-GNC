# Tests Overview

This folder contains unit tests for the `cpp` module.

## Why there are two test styles

We use two test paths on purpose:

1. `gtsamAddTestsGlob(...)` + GTSAM/CppUnitLite macros
2. Standard GoogleTest (`GTest::gtest`, `GTest::gtest_main`)

### 1) GTSAM test macros (mainly for custom variables and custom factors)

Use this path when testing GTSAM-specific behavior, especially:
- custom variables on manifolds (for example `StiefelManifold`, `LiftedPose`)
- custom factors and Jacobian correctness checks with GTSAM helpers such as `EXPECT_CORRECT_FACTOR_JACOBIANS`

Current files in this group:
- `testLiftedPose.cpp`
- `testLandmarkFactor.cpp`
- `testPriorFactor.cpp`
- `testRaFactor.cpp`
- `testSEsyncFactor.cpp`
- `testStiefelManifold.cpp`
- `testLiftedRangeFactor.cpp`

### 2) General GoogleTest files

Use standard GTest for more general unit tests that do not need the GTSAM factor-testing macros.

Current files in this group:
- `testAssemblerComparison.cpp`
- `testAssemblerSimple.cpp`
- `testDataMatrixUtils.cpp`

## Note

The grouping above is configured in `cpp/tests/CMakeLists.txt`. If you add a new test file, place it in the matching list there.
