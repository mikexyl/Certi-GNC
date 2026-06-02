/* ----------------------------------------------------------------------------
 * Copyright 2026, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu (xu.zhex@northeastern.edu), et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    utils.cpp
 * @brief   Data-matrix builder, g2o parser, and timing utilities (implementation).
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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Geometry>
#include "utils.h"
#include <unordered_set>
#include <set>

namespace gtsam {
    namespace DataParser {
        Matrix fromAngle(double angle_rad) {
            Matrix rotation_matrix_2d(2, 2);
            rotation_matrix_2d << cos(angle_rad), -sin(angle_rad), sin(angle_rad),
                    cos(angle_rad);
            return rotation_matrix_2d;
        }

        Matrix fromQuat(double qx, double qy, double qz, double qw) {
            Eigen::Quaterniond q(qw, qx, qy, qz);
            auto rot_mat = q.toRotationMatrix();
            // Not sure why we can't cast it directly?
            Matrix result(3, 3);
            result << rot_mat(0, 0), rot_mat(0, 1), rot_mat(0, 2), rot_mat(1, 0),
                    rot_mat(1, 1), rot_mat(1, 2), rot_mat(2, 0), rot_mat(2, 1), rot_mat(2, 2);
            return result;
        }

        Measurement read_g2o_file(const std::string &filename, size_t &num_poses_) {
            std::unordered_set<size_t> pose_ids, landmark_ids;

            // Preallocate output vector
            Measurement measurements;
            RelativePoseMeasurement posemeasurement;
            RelativeLandmarkMeasurement landmarkmeasurement;


            // A string used to contain the contents of a single line
            std::string line;

            // A string used to extract tokens from each line one-by-one
            std::string token;

            // Preallocate various useful quantities
            Scalar dx, dy, dz, dtheta, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16,
                    I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66;

            size_t i, j;

            // Open the file for reading
            std::ifstream infile(filename);
            size_t &num_poses = measurements.num_poses;
            size_t &num_landmarks = measurements.num_landmarks;

            std::unordered_map<size_t, size_t> poses;
            std::unordered_map<size_t, size_t> landmarks;

            while (std::getline(infile, line)) {
                // Construct a stream from the string
                std::stringstream strstrm(line);

                // Extract the first token from the string
                strstrm >> token;

                if (token == "EDGE_SE2") {
                    // This is a 2D pose measurement

                    /** The g2o format specifies a 2D relative pose measurement in the
                     * following form:
                     *
                     * EDGE_SE2 id1 id2 dx dy dtheta, I11, I12, I13, I22, I23, I33
                     *
                     */

                    // Extract formatted output
                    strstrm >> i >> j >> dx >> dy >> dtheta >> I11 >> I12 >> I13 >> I22 >>
                            I23 >> I33;
                    if (poses.insert({i, num_poses}).second) num_poses++;

                    if (poses.insert({j, num_poses}).second) num_poses++;

                    // Pose ids
                    posemeasurement.i = poses[i];
                    posemeasurement.j = poses[j];
                    pose_ids.insert(i);
                    pose_ids.insert(j);

                    // Raw measurements
                    posemeasurement.t = Eigen::Matrix<Scalar, 2, 1>(dx, dy);
                    posemeasurement.R = Eigen::Rotation2D<Scalar>(dtheta).toRotationMatrix();

                    Eigen::Matrix<Scalar, 2, 2> TranInfo;
                    TranInfo << I11, I12, I12, I22;
                    posemeasurement.tau = 2 / TranInfo.inverse().trace();

                    posemeasurement.kappa = I33;



                    measurements.poseMeasurements.push_back(posemeasurement);

                } else if (token == "EDGE_SE3:QUAT") {

                    // This is a 3D pose measurement

                    /** The g2o format specifies a 3D relative pose measurement in the
                     * following form:
                     *
                     * EDGE_SE3:QUAT id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw
                     *
                     * I11 I12 I13 I14 I15 I16
                     *     I22 I23 I24 I25 I26
                     *         I33 I34 I35 I36
                     *             I44 I45 I46
                     *                 I55 I56
                     *                     I66
                     */

                    // Extract formatted output
                    strstrm >> i >> j >> dx >> dy >> dz >> dqx >> dqy >> dqz >> dqw >> I11 >>
                            I12 >> I13 >> I14 >> I15 >> I16 >> I22 >> I23 >> I24 >> I25 >> I26 >>
                            I33 >> I34 >> I35 >> I36 >> I44 >> I45 >> I46 >> I55 >> I56 >> I66;

                    // Fill in elements of the measurement

                    // Pose ids
                    if (poses.insert({i, num_poses}).second) num_poses++;

                    if (poses.insert({j, num_poses}).second) num_poses++;
                    posemeasurement.i = poses[i];
                    posemeasurement.j = poses[j];
                    pose_ids.insert(i);
                    pose_ids.insert(j);

                    // Raw measurements
                    posemeasurement.t = Eigen::Matrix<Scalar, 3, 1>(dx, dy, dz);
                    posemeasurement.R =
                            Eigen::Quaternion<Scalar>(dqw, dqx, dqy, dqz).toRotationMatrix();

                    // Compute precisions

                    // Compute and store the optimal (information-divergence-minimizing) value
                    // of the parameter tau
                    Eigen::Matrix<Scalar, 3, 3> TranInfo;
                    TranInfo << I11, I12, I13, I12, I22, I23, I13, I23, I33;
                    posemeasurement.tau = 3 / TranInfo.inverse().trace();
                    posemeasurement.trans_precision = 3 / TranInfo.inverse().trace();

                    // Information-divergence-minimizing isotropic precision for the
                    // rotation block (kappa) and the corresponding trans precision.
                    Eigen::Matrix<Scalar, 3, 3> RotInfo;
                    RotInfo << I44, I45, I46, I45, I55, I56, I46, I56, I66;
                    posemeasurement.kappa = 3 / (2 * RotInfo.inverse().trace());
                    posemeasurement.rot_precision = 3 / (2* RotInfo.inverse().trace());



                    measurements.poseMeasurements.push_back(posemeasurement);

                } else if (token == "LANDMARK2" || token == "EDGE_SE2_XY")
                {
                    strstrm >> i >> j >> dx >> dy >> I11 >> I12 >> I22;

                    if (poses.insert({i, num_poses}).second) num_poses++;

                    if (landmarks.insert({j, num_landmarks}).second) num_landmarks++;

                    landmarkmeasurement.i = poses[i];     // pose index (if you need a mapping there too, do the same)
                    landmarkmeasurement.j = landmarks[j];
                    pose_ids.insert(i);
                    landmark_ids.insert(j);    // the “j” is a landmark

                    landmarkmeasurement.l = Eigen::Matrix<Scalar,2,1>(dx,dy);

                    Eigen::Matrix<Scalar,2,2> TranCov;
                    TranCov << I11, I12,
                               I12, I22;
                    // Check if inverse() is possible to avoid potential issues if trace is 0
                    auto inv = TranCov.inverse();
                    landmarkmeasurement.nu = 2.0 / inv.trace();

                    measurements.landmarkMeasurements.push_back(landmarkmeasurement);
                } else if (token == "VERTEX_SE2") {
                    // g2o: VERTEX_SE2 id x y theta
                    Scalar x, y, theta;
                    strstrm >> i >> x >> y >> theta;
                    if (poses.find(i) == poses.end()) {
                        poses.insert({i, num_poses++});
                    }
                    Measurement::InitialValue initial;
                    initial.t = Eigen::Vector2d(x, y);
                    initial.R = Eigen::Rotation2D<Scalar>(theta).toRotationMatrix();
                    measurements.initial_poses[{'A', poses[i]}] = initial;
                } else if (token == "VERTEX_SE3:QUAT") {
                    // g2o: VERTEX_SE3:QUAT id x y z qx qy qz qw
                    Scalar x, y, z, qx, qy, qz, qw;
                    strstrm >> i >> x >> y >> z >> qx >> qy >> qz >> qw;
                    if (poses.find(i) == poses.end()) {
                        poses.insert({i, num_poses++});
                    }
                    Measurement::InitialValue initial;
                    initial.t = Eigen::Vector3d(x, y, z);
                    initial.R = Eigen::Quaternion<Scalar>(qw, qx, qy, qz).toRotationMatrix();
                    measurements.initial_poses[{'A', poses[i]}] = initial;
                } else if (token == "VERTEX_XY") {
                    // g2o: VERTEX_XY id x y
                    Scalar x, y;
                    strstrm >> i >> x >> y;
                    if (landmarks.find(i) == landmarks.end()) {
                        landmarks.insert({i, num_landmarks++});
                    }
                    Measurement::InitialValue initial;
                    initial.t = Eigen::Vector2d(x, y);
                    measurements.initial_landmarks[{'L', landmarks[i]}] = initial;
                } else if (token == "FIX") {
                    // g2o: FIX id
                    // Ignore this token as it's used for fixing nodes in optimization, which is not needed here
                } else {
                    std::cout << "Error: unrecognized type: " << token << "!" << std::endl;
                    assert(false);
                }


            } // while

            infile.close();
            measurements.num_poses = pose_ids.size();
            measurements.num_landmarks = landmark_ids.size();
            return measurements;
        }



    }// namespace DataParser
} // namespace gtsam
