#!/usr/bin/env python3
"""Evaluate a GNC run against ground truth.

Reads a TUM-format trajectory at <result>.txt and compares against the GT
TUM trajectory at <gt>, reporting absolute trajectory error (translation
RMSE) after rigid 2D alignment to cancel the global gauge freedom.

Usage:
    python eval_gnc_pgo.py --gt data/pgo/intel_GT.txt \\
                           --result results/pgo_demo
"""

import argparse
import math
import os
import sys


def read_tum(path):
    """Read a TUM-style file: id x y z qx qy qz qw."""
    poses = {}
    with open(path) as f:
        for line in f:
            t = line.strip().split()
            if not t or t[0].startswith("#"):
                continue
            poses[int(t[0])] = (float(t[1]), float(t[2]), float(t[3]))
    return poses


def umeyama_2d(src, dst):
    """Find rigid 2D transform (R, t) aligning src -> dst (no scale)."""
    n = len(src)
    if n == 0:
        return [[1, 0], [0, 1]], [0, 0]
    cx_s = sum(p[0] for p in src) / n
    cy_s = sum(p[1] for p in src) / n
    cx_d = sum(p[0] for p in dst) / n
    cy_d = sum(p[1] for p in dst) / n
    sxx = sxy = syx = syy = 0.0
    for s, d in zip(src, dst):
        dx_s, dy_s = s[0] - cx_s, s[1] - cy_s
        dx_d, dy_d = d[0] - cx_d, d[1] - cy_d
        sxx += dx_s * dx_d
        sxy += dx_s * dy_d
        syx += dy_s * dx_d
        syy += dy_s * dy_d
    theta = math.atan2(sxy - syx, sxx + syy)
    c, s = math.cos(theta), math.sin(theta)
    tx = cx_d - (c * cx_s - s * cy_s)
    ty = cy_d - (s * cx_s + c * cy_s)
    return [[c, -s], [s, c]], [tx, ty]


def apply_2d(R, t, p):
    return (R[0][0] * p[0] + R[0][1] * p[1] + t[0],
            R[1][0] * p[0] + R[1][1] * p[1] + t[1])


def ate(est, gt):
    common = sorted(set(est) & set(gt))
    if not common:
        return None
    src = [(est[i][0], est[i][1]) for i in common]
    dst = [(gt[i][0], gt[i][1]) for i in common]
    R, t = umeyama_2d(src, dst)
    se = 0.0
    for s, d in zip(src, dst):
        a = apply_2d(R, t, s)
        se += (a[0] - d[0]) ** 2 + (a[1] - d[1]) ** 2
    return math.sqrt(se / len(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--result", required=True,
                    help="Output stem of GNC run (without .txt).")
    args = ap.parse_args()

    traj_path = args.result + ".txt"
    if not os.path.exists(traj_path):
        print(f"[ERROR] missing trajectory: {traj_path}", file=sys.stderr)
        sys.exit(1)

    est = read_tum(traj_path)
    gt = read_tum(args.gt)
    e = ate(est, gt)
    n = len(set(est) & set(gt))
    print(f"ATE (translation RMSE) = {e:.4f} m (n={n})")


if __name__ == "__main__":
    main()
