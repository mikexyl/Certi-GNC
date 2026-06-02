file(GLOB test_files "*.cpp")

# CppUnitLite tests (factor-graph unit tests via GTSAM's macro).
set(CPPUNITLITE_TEST_FILES
    testLiftedPose.cpp
    testLandmarkFactor.cpp
    testSEsyncFactor.cpp
    testStiefelManifold.cpp
)

# GTest tests (assembler / utility tests via standard GoogleTest).
set(GTEST_TEST_FILES
    testAssemblerSimple.cpp
    testDataMatrixUtils.cpp
)

# Build the CppUnitLite tests via GTSAM's test macro.
set(CPPUNITLITE_GLOB "")
foreach(test_file ${CPPUNITLITE_TEST_FILES})
    set(CPPUNITLITE_GLOB "${CPPUNITLITE_GLOB}${test_file};")
endforeach()

set(GTSAM_BUILD_TESTS ON)
gtsamAddTestsGlob(StiefelManifold "${CPPUNITLITE_GLOB}" "" "${PROJECT_NAME};gtsam")

# Build the GTest tests directly.
foreach(test_file ${GTEST_TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name}
        PRIVATE
            ${PROJECT_NAME}
            GTest::gtest
            GTest::gtest_main
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
