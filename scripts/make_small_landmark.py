#!/usr/bin/env python3
"""Subset a landmark-SLAM g2o file (with LANDMARK2 observations) to its first
N poses, keeping only landmarks observed from that pose range.

Also synthesizes initial VERTEX_SE2 poses by integrating odometry from
(0, 0, 0), so the resulting file can be used with --init_type=odom.

Inputs use these tokens:
  VERTEX_XY id x y               (landmark)
  EDGE_SE2 i j dx dy dtheta ...  (pose-pose odometry)
  LANDMARK2 i j lx ly ...        (pose -> landmark observation)

Usage:
    python make_small_landmark.py \\
        --input  /path/to/your/landmark.g2o \\
        --output data/landmark_small/my_tiny.g2o \\
        --num_poses 80

Note: this project ships only the pre-extracted small dataset
data/landmark_small/cityTrees_tiny.g2o. To regenerate it or generate one
from a different source, point --input at an external copy.
"""

import argparse
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_poses", type=int, required=True)
    args = ap.parse_args()

    landmarks_in = {}        # id -> (x, y)
    pose_edges = []          # list of (i, j, dx, dy, dtheta, info...)
    lmk_obs = []             # list of (i, j, lx, ly, info...)

    with open(args.input) as f:
        for line in f:
            t = line.strip().split()
            if not t:
                continue
            if t[0] == "VERTEX_XY":
                landmarks_in[int(t[1])] = (float(t[2]), float(t[3]))
            elif t[0] == "EDGE_SE2":
                pose_edges.append(t)
            elif t[0] in ("LANDMARK2", "EDGE_SE2_XY"):
                lmk_obs.append(t)

    # Keep odometry edges whose endpoints are within [0, num_poses).
    kept_pose_edges = []
    for e in pose_edges:
        i, j = int(e[1]), int(e[2])
        if i < args.num_poses and j < args.num_poses:
            kept_pose_edges.append(e)

    # Keep landmark observations from poses in range; collect landmark ids.
    raw_obs = []
    for e in lmk_obs:
        i, j = int(e[1]), int(e[2])
        if i < args.num_poses:
            raw_obs.append(e)

    # Drop landmarks observed only once: they are unobservable in GNC because
    # the landmark snaps onto the (potentially outlier) measurement, giving zero
    # residual and weight=1 even when the measurement is bogus.
    obs_count = {}
    for e in raw_obs:
        obs_count[int(e[2])] = obs_count.get(int(e[2]), 0) + 1
    kept_obs = [e for e in raw_obs if obs_count[int(e[2])] >= 2]
    kept_lmk_ids = {int(e[2]) for e in kept_obs}

    # Integrate odometry from (0,0,0) to get pose initial values.
    poses_init = [(0.0, 0.0, 0.0)] * args.num_poses
    # Build adjacency: outgoing edges sorted by i, j == i+1 are the most natural.
    succ = {}
    for e in kept_pose_edges:
        i, j = int(e[1]), int(e[2])
        if j == i + 1:
            dx, dy, dtheta = float(e[3]), float(e[4]), float(e[5])
            succ[i] = (dx, dy, dtheta)
    for i in range(args.num_poses - 1):
        if i in succ:
            x, y, th = poses_init[i]
            dx, dy, dth = succ[i]
            c, s = math.cos(th), math.sin(th)
            nx = x + c * dx - s * dy
            ny = y + s * dx + c * dy
            nth = th + dth
            # wrap to [-pi, pi]
            nth = (nth + math.pi) % (2 * math.pi) - math.pi
            poses_init[i + 1] = (nx, ny, nth)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        # Initial poses
        for i, (x, y, th) in enumerate(poses_init):
            f.write(f"VERTEX_SE2 {i} {x:.9g} {y:.9g} {th:.9g}\n")
        # Landmark vertices (keep their input positions as initial guesses)
        for lid in sorted(kept_lmk_ids):
            if lid in landmarks_in:
                lx, ly = landmarks_in[lid]
                f.write(f"VERTEX_XY {lid} {lx:.9g} {ly:.9g}\n")
        # Odometry edges
        for e in kept_pose_edges:
            f.write(" ".join(e) + "\n")
        # Landmark observations
        for e in kept_obs:
            f.write(" ".join(e) + "\n")

    # Save GT trajectory (the integrated odometry IS the truth for this subset).
    gt_path = os.path.splitext(args.output)[0] + "_GT.txt"
    with open(gt_path, "w") as f:
        for i, (x, y, th) in enumerate(poses_init):
            qx, qy, qz, qw = 0.0, 0.0, math.sin(th * 0.5), math.cos(th * 0.5)
            f.write(f"{i} {x:.9g} {y:.9g} 0 {qx:.9g} {qy:.9g} {qz:.9g} {qw:.9g}\n")

    print(f"Wrote {args.num_poses} poses, {len(kept_lmk_ids)} landmarks, "
          f"{len(kept_pose_edges)} odometry edges, {len(kept_obs)} landmark observations.")
    print(f"Output: {args.output}")
    print(f"GT:     {gt_path}")


if __name__ == "__main__":
    main()
