#!/usr/bin/env python3
"""Subset a 2D PGO g2o file to its first N poses.

Keeps all VERTEX_SE2 with id < N and all EDGE_SE2 whose both endpoints have
id < N. Writes the resulting g2o to --output. Also writes a TUM-style GT file
(<output stem>_GT.txt) holding the same pose coordinates as ground truth.

Usage:
    python make_small_pgo.py \\
        --input  /path/to/your/clean_pgo.g2o \\
        --output data/pgo_small/my_tiny.g2o \\
        --num_poses 500

Note: this project ships only the pre-extracted small datasets under
data/pgo_small/. To use this script on full benchmark g2o files (e.g.
intel.g2o, MIT.g2o) you must point --input at an external copy.
"""

import argparse
import math
import os


def quat_from_yaw(yaw):
    h = yaw * 0.5
    return (0.0, 0.0, math.sin(h), math.cos(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_poses", type=int, required=True)
    args = ap.parse_args()

    vertices = []          # list of (id, x, y, theta)
    edges = []             # list of edge lines (raw, after id-checks)

    with open(args.input, "r") as f:
        for line in f:
            t = line.strip().split()
            if not t:
                continue
            if t[0] == "VERTEX_SE2":
                vid = int(t[1])
                if vid < args.num_poses:
                    x, y, theta = float(t[2]), float(t[3]), float(t[4])
                    vertices.append((vid, x, y, theta))
            elif t[0] == "EDGE_SE2":
                i, j = int(t[1]), int(t[2])
                if i < args.num_poses and j < args.num_poses:
                    edges.append(line.rstrip("\n"))
            else:
                # Pass through any other tokens (FIX, etc.) if vertex 0 is in range
                pass

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for v in vertices:
            f.write(f"VERTEX_SE2 {v[0]} {v[1]:.9g} {v[2]:.9g} {v[3]:.9g}\n")
        for e in edges:
            f.write(e + "\n")

    # GT file in TUM-like format: id x y z qx qy qz qw (z=0, quat from yaw)
    gt_path = os.path.splitext(args.output)[0] + "_GT.txt"
    with open(gt_path, "w") as f:
        for v in vertices:
            qx, qy, qz, qw = quat_from_yaw(v[3])
            f.write(f"{v[0]} {v[1]:.9g} {v[2]:.9g} 0 {qx:.9g} {qy:.9g} {qz:.9g} {qw:.9g}\n")

    print(f"Wrote {len(vertices)} vertices, {len(edges)} edges to {args.output}")
    print(f"Wrote GT to {gt_path}")


if __name__ == "__main__":
    main()
