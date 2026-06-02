import numpy as np

# -----------------------------------------------------------------------------
# 1) Change these to your input/output files:
# -----------------------------------------------------------------------------
tum_path = "results/CIN/MH3500_incincremental_result_Final.tum"
g2o_path = "trajectory.g2o"

# -----------------------------------------------------------------------------
# 2) Helper functions for quaternion math
# -----------------------------------------------------------------------------
def quat_conj(q):
    """Conjugate of quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_mul(q1, q2):
    """Hamilton product of quaternions q1 * q2."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def rotate(q, v):
    """Rotate vector v by quaternion q."""
    vq = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, vq), quat_conj(q))[:3]

# -----------------------------------------------------------------------------
# 3) Read TUM file
# -----------------------------------------------------------------------------
def read_tum(path):
    poses = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            t, x, y, z, qx, qy, qz, qw = map(float, line.split())
            poses.append((t, np.array([x, y, z]), np.array([qx, qy, qz, qw])))
    return poses

poses = read_tum(tum_path)

# -----------------------------------------------------------------------------
# 4) Write g2o file
# -----------------------------------------------------------------------------
with open(g2o_path, 'w') as out:
    # -- write vertices
    for idx, (_, p, q) in enumerate(poses):
        out.write(f"VERTEX_SE3:QUAT {idx} "
                  f"{p[0]} {p[1]} {p[2]} "
                  f"{q[0]} {q[1]} {q[2]} {q[3]}\n")

    # -- prepare an identity information matrix (upper‐triangle, 21 entries)
    info = [1.0 if i == j else 0.0
            for i in range(6)
            for j in range(i, 6)]

    # -- write odometry edges between consecutive poses
    for i in range(len(poses) - 1):
        _, p_i, q_i = poses[i]
        _, p_j, q_j = poses[i+1]

        # relative rotation & translation
        q_rel = quat_mul(quat_conj(q_i), q_j)
        t_rel = rotate(quat_conj(q_i), p_j - p_i)

        out.write(f"EDGE_SE3:QUAT {i} {i+1} "
                  f"{t_rel[0]} {t_rel[1]} {t_rel[2]} "
                  f"{q_rel[0]} {q_rel[1]} {q_rel[2]} {q_rel[3]} ")
        out.write(" ".join(f"{v}" for v in info))
        out.write("\n")

print("TUM → g2o conversion complete!")
