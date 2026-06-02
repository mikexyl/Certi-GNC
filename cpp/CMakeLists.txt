# --- Headers and Sources ---
file(GLOB EX_HEADERS "*.h")
file(GLOB ROBUST_HEADERS "robust/*.h")
file(GLOB EX_SRCS "*.cpp")

# --- Install Headers ---
install(FILES ${EX_HEADERS} DESTINATION include/StiefelManifoldExample)
install(FILES ${ROBUST_HEADERS} DESTINATION include/StiefelManifoldExample/robust)

# --- Library Definition ---
add_library(${PROJECT_NAME} SHARED ${EX_SRCS} ${EX_HEADERS} ${ROBUST_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/robust>
)

# --- Linking ---
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        gtsam
        Optimization
        ILDL
        gflags
        glog
        ${BLAS_LIBRARIES}
        ${SPQR_LIBRARIES}
        ${LAPACK}
        cholmod
)

# --- Unit Tests ---
add_subdirectory(tests)

# --- Robust (GNC) Examples ---
add_subdirectory(robust_example)