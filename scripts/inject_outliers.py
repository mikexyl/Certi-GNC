#!/usr/bin/env python3
"""Inject outliers into a 2D PGO or landmark-SLAM g2o file.

Outliers are injected by replacing the *measurement* of randomly selected
non-odometry edges (and/or all landmark observations) with random values
drawn far from the truth:
  - SE2 outlier: dx,dy ~ N(0, sigma_trans^2); dtheta ~ Uniform[-pi, pi]
  - XY  outlier: lx,ly ~ N(0, sigma_landmark^2)

The information matrix (last 6 / 3 fields of the edge) is left unchanged so
the GNC weight update sees the same precision as for inliers.

Writes:
  <output>.g2o            same as input but with selected edges perturbed
  <output>_mask.txt       one line "edge_index is_outlier" for every edge
                          (in the order they appear in the output file).

Edge classification:
  EDGE_SE2 with |j - i| == 1 are odometry; never perturbed.
  EDGE_SE2 with |j - i|  > 1 are loop closures; candidates for outlier.
  EDGE_SE2_XY / LANDMARK     are landmark observations; candidates for outlier.

Usage:
    python inject_outliers.py --input clean.g2o --pct 10 --seed 0 \\
                              --output out.g2o
"""

import argparse
import math
import os
import random


def is_odometry_se2(i, j):
    return abs(i - j) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pct", type=float, required=True,
                    help="Outlier percentage in [0, 100] of candidate edges.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma_trans", type=float, default=10.0,
                    help="StdDev of injected translation outliers (meters).")
    ap.add_argument("--sigma_landmark", type=float, default=10.0,
                    help="StdDev of injected landmark outliers (meters).")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # First pass: read all lines, classify edges, count candidates.
    with open(args.input, "r") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    edge_indices = []     # global index in 'lines' list for each edge
    candidates = []       # subset of edge_indices that are outlier candidates
    edge_kind = []        # "odom" | "loop" | "landmark" for each edge

    for idx, line in enumerate(lines):
        t = line.split()
        if not t:
            continue
        if t[0] == "EDGE_SE2":
            i, j = int(t[1]), int(t[2])
            edge_indices.append(idx)
            if is_odometry_se2(i, j):
                edge_kind.append("odom")
            else:
                edge_kind.append("loop")
                candidates.append(len(edge_indices) - 1)  # relative edge index
        elif t[0] in ("EDGE_SE2_XY", "LANDMARK", "LANDMARK2"):
            edge_indices.append(idx)
            edge_kind.append("landmark")
            candidates.append(len(edge_indices) - 1)

    n_to_perturb = int(round(args.pct / 100.0 * len(candidates)))
    chosen = set(rng.sample(candidates, n_to_perturb)) if n_to_perturb > 0 else set()

    # Second pass: rewrite chosen edges with random measurements.
    out_lines = list(lines)
    for rel_idx in chosen:
        idx = edge_indices[rel_idx]
        t = out_lines[idx].split()
        if t[0] == "EDGE_SE2":
            # EDGE_SE2 i j dx dy dtheta I11 I12 I13 I22 I23 I33
            t[3] = f"{rng.gauss(0, args.sigma_trans):.6f}"
            t[4] = f"{rng.gauss(0, args.sigma_trans):.6f}"
            t[5] = f"{rng.uniform(-math.pi, math.pi):.6f}"
            out_lines[idx] = " ".join(t)
        elif t[0] in ("EDGE_SE2_XY", "LANDMARK", "LANDMARK2"):
            # EDGE_SE2_XY i j lx ly I11 I12 I22
            t[3] = f"{rng.gauss(0, args.sigma_landmark):.6f}"
            t[4] = f"{rng.gauss(0, args.sigma_landmark):.6f}"
            out_lines[idx] = " ".join(t)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for line in out_lines:
            f.write(line + "\n")

    # Write mask: one row per edge, columns: edge_index_in_graph kind is_outlier
    mask_path = os.path.splitext(args.output)[0] + "_mask.txt"
    with open(mask_path, "w") as f:
        f.write("# edge_index kind is_outlier\n")
        for rel_idx, kind in enumerate(edge_kind):
            f.write(f"{rel_idx} {kind} {1 if rel_idx in chosen else 0}\n")

    n_odom = sum(1 for k in edge_kind if k == "odom")
    n_loop = sum(1 for k in edge_kind if k == "loop")
    n_lmk  = sum(1 for k in edge_kind if k == "landmark")
    print(f"Edges: total={len(edge_kind)} odom={n_odom} loop={n_loop} landmark={n_lmk}")
    print(f"Candidates for outlier: {len(candidates)}; perturbed: {len(chosen)} ({args.pct}%)")
    print(f"Wrote {args.output}")
    print(f"Wrote {mask_path}")


if __name__ == "__main__":
    main()
