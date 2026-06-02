#!/usr/bin/env python3
"""Full-scale Landmark SLAM benchmark: certifiable GNC vs. vanilla GTSAM GNC.

Same grid as scripts/run_benchmark_pgo.py but for landmark binaries on
data/landmark/cityTrees_reduced.g2o (1600 poses, ~100 landmarks).
Outlier injection scales LANDMARK2 observation perturbations by
sigma_landmark (default 100 m for cityTrees).

Outputs:
  <out_dir>/data/             outlier-injected datasets, one per trial
  <out_dir>/runs/             per-run artifacts
  <out_dir>/per_run.csv       one row per individual run
  <out_dir>/summary.csv       one row per (method, init, rate) aggregate

Usage:
    python scripts/run_benchmark_lmk.py \\
        --input data/landmark/cityTrees_reduced.g2o \\
        --gt    data/landmark/cityTrees_reduced_GT.txt \\
        --num_trials 10 --rates 10 20 30 \\
        --out_dir results/benchmark_cityTrees
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
BIN = os.path.join(ROOT, "build", "bin")


def sh(cmd, label, log_path=None):
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    out = proc.stdout + proc.stderr
    if log_path:
        with open(log_path, "w") as f:
            f.write(out)
    if proc.returncode != 0:
        sys.stderr.write(f"[{label}] rc={proc.returncode}\n{out[-500:]}\n")
    return proc.returncode, t1 - t0, out


def parse_ate(text):
    m = re.search(r"ATE \(translation RMSE\)\s*=\s*([\d\.eE\+\-]+)", text)
    return float(m.group(1)) if m else None


def evaluate(gt, stem):
    cmd = ["python3", os.path.join(HERE, "eval_gnc_pgo.py"),
           "--gt", gt, "--result", stem]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return parse_ate(proc.stdout + proc.stderr)


def read_summary(stem):
    path = stem + "_summary.csv"
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            field, value = row[0], row[1]
            if value == "":
                out[field] = None
            else:
                try:
                    out[field] = float(value) if any(c in value for c in ".eE") else int(value)
                except ValueError:
                    out[field] = value
    return out


def mean_std(xs):
    xs = [x for x in xs if x is not None
          and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan"), 0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var), len(xs)


def inject(input_g2o, output_g2o, pct, seed, sigma_landmark):
    cmd = ["python3", os.path.join(HERE, "inject_outliers.py"),
           "--input", input_g2o, "--output", output_g2o,
           "--pct", str(pct), "--seed", str(seed),
           "--sigma_landmark", str(sigma_landmark)]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   required=True)
    ap.add_argument("--gt",      required=True)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "benchmark_cityTrees"))
    ap.add_argument("--num_trials", type=int, default=10)
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="Trial s uses seed (seed_offset + s) for both outlier "
                         "injection and random initial guess. Default 0 = seeds "
                         "0..num_trials-1.")
    ap.add_argument("--rates", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--methods", nargs="+", default=["ours", "vanilla"])
    ap.add_argument("--inits",   nargs="+", default=["random", "odom"])
    ap.add_argument("--sigma_landmark", type=float, default=100.0,
                    help="StdDev of injected landmark-outlier offsets (m).")
    # Both methods get the same outer-loop cap so the comparison is fair.
    ap.add_argument("--gnc_max_iterations_ours",    type=int, default=30)
    ap.add_argument("--gnc_max_iterations_vanilla", type=int, default=30)
    args = ap.parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    runs_dir = os.path.join(args.out_dir, "runs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    per_run_csv = os.path.join(args.out_dir, "per_run.csv")
    summary_csv = os.path.join(args.out_dir, "summary.csv")

    per_run_rows = []
    aggregated = {}

    total_runs = len(args.rates) * args.num_trials * len(args.methods) * len(args.inits)
    done = 0
    overall_t0 = time.time()

    for rate in args.rates:
        for trial in range(args.num_trials):
            seed = args.seed_offset + trial
            data_file = os.path.join(data_dir, f"cityTrees_outl{rate}_seed{seed}.g2o")
            if not os.path.exists(data_file):
                inject(args.input, data_file, rate, seed, args.sigma_landmark)
            for method in args.methods:
                for init in args.inits:
                    stem = os.path.join(runs_dir,
                                        f"{method}_init-{init}_rate{rate}_seed{seed}")
                    if method == "ours":
                        cmd = [
                            os.path.join(BIN, "Certifiable_GNC_LMK_example"),
                            "--d=2", "--p=2",
                            "--input_dir=" + data_file,
                            "--output_dir=" + stem,
                            "--init_type=" + init,
                            f"--seed={seed}",
                            "--gnc_verbosity=0",
                            f"--gnc_max_iterations={args.gnc_max_iterations_ours}",
                        ]
                    else:
                        cmd = [
                            os.path.join(BIN, "Vanilla_GNC_LMK_example"),
                            "--input_dir=" + data_file,
                            "--output_dir=" + stem,
                            "--init_type=" + init,
                            f"--seed={seed}",
                            "--gnc_verbosity=0",
                            f"--gnc_max_iterations={args.gnc_max_iterations_vanilla}",
                        ]
                    rc, secs, _ = sh(cmd, f"{method}/{init} r={rate} s={seed}",
                                     log_path=stem + ".log")
                    ate = evaluate(args.gt, stem)
                    summ = read_summary(stem)
                    row = {
                        "method": method,
                        "init": init,
                        "rate": rate,
                        "seed": seed,
                        "ate": ate,
                        "time": summ.get("final_total_time", secs),
                        "ok": rc == 0,
                        "num_gnc_iters":         summ.get("num_gnc_iters"),
                        "final_ending_rank":     summ.get("final_ending_rank"),
                        "max_ending_rank":       summ.get("max_ending_rank"),
                        "levels_climbed_total":  summ.get("levels_climbed_total"),
                        "num_certifier_verified": summ.get("num_certifier_verified"),
                        "num_certifier_failed":   summ.get("num_certifier_failed"),
                        "sum_lm_opt_time":        summ.get("sum_lm_opt_time"),
                        "sum_verification_time":  summ.get("sum_verification_time"),
                    }
                    per_run_rows.append(row)
                    aggregated.setdefault((method, init, rate), []).append(row)
                    done += 1
                    elapsed = time.time() - overall_t0
                    remaining = elapsed / done * (total_runs - done)
                    print(f"[{done}/{total_runs}] {method:8s} init={init:6s} "
                          f"rate={rate}% seed={seed}  ATE={ate}  "
                          f"time={row['time']:.2f}s  "
                          f"(elapsed={elapsed:.0f}s, eta={remaining:.0f}s)")

    cols = ["method", "init", "rate", "seed",
            "ate", "time", "ok",
            "num_gnc_iters", "final_ending_rank", "max_ending_rank",
            "levels_climbed_total", "num_certifier_verified",
            "num_certifier_failed", "sum_lm_opt_time", "sum_verification_time"]
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_run_rows:
            w.writerow(r)

    agg_cols = ["method", "init", "rate", "n",
                "ate_mean", "ate_std", "ate_median",
                "time_mean", "time_std",
                "iters_mean", "max_rank_mean",
                "levels_total_mean", "verified_mean",
                "lm_opt_time_mean", "verify_time_mean",
                "n_failures"]
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(agg_cols)
        for (method, init, rate) in sorted(aggregated.keys()):
            entries = aggregated[(method, init, rate)]
            ates_sorted = sorted([e["ate"] for e in entries if e["ate"] is not None])
            ate_m, ate_s, n = mean_std([e["ate"] for e in entries])
            ate_med = (ates_sorted[len(ates_sorted) // 2]
                       if ates_sorted else float("nan"))
            tm_m, tm_s, _ = mean_std([e["time"] for e in entries])
            iters_m, _, _ = mean_std([e["num_gnc_iters"] for e in entries])
            rank_m, _, _ = mean_std([e["max_ending_rank"] for e in entries])
            lvl_m, _, _ = mean_std([e["levels_climbed_total"] for e in entries])
            ver_m, _, _ = mean_std([e["num_certifier_verified"] for e in entries])
            lm_m, _, _ = mean_std([e["sum_lm_opt_time"] for e in entries])
            vt_m, _, _ = mean_std([e["sum_verification_time"] for e in entries])
            n_fail = sum(1 for e in entries if not e["ok"])
            w.writerow([method, init, rate, n,
                        ate_m, ate_s, ate_med,
                        tm_m, tm_s,
                        iters_m, rank_m, lvl_m, ver_m,
                        lm_m, vt_m, n_fail])

    print("\n=== Benchmark summary ===")
    print(f"{'method':<9} {'init':<7} {'rate':>5} "
          f"{'ATE [m] mean±sd':>22} {'median':>9} "
          f"{'time [s]':>10} {'iters':>7} "
          f"{'max rank':>9} {'lvls':>5} {'n':>3} {'fail':>4}")
    for (method, init, rate) in sorted(aggregated.keys()):
        entries = aggregated[(method, init, rate)]
        ate_m, ate_s, n = mean_std([e["ate"] for e in entries])
        ates_sorted = sorted([e["ate"] for e in entries if e["ate"] is not None])
        ate_med = (ates_sorted[len(ates_sorted) // 2]
                   if ates_sorted else float("nan"))
        tm_m, _, _ = mean_std([e["time"] for e in entries])
        iters_m, _, _ = mean_std([e["num_gnc_iters"] for e in entries])
        rank_m, _, _ = mean_std([e["max_ending_rank"] for e in entries])
        lvl_m, _, _ = mean_std([e["levels_climbed_total"] for e in entries])
        n_fail = sum(1 for e in entries if not e["ok"])
        iters_s = f"{iters_m:>7.1f}" if not math.isnan(iters_m) else f"{'-':>7}"
        rank_s = f"{rank_m:>9.1f}" if not math.isnan(rank_m) else f"{'-':>9}"
        lvl_s = f"{lvl_m:>5.1f}" if not math.isnan(lvl_m) else f"{'-':>5}"
        print(f"{method:<9} {init:<7} {rate:>4}% "
              f"{ate_m:>10.3f} ±{ate_s:>8.3f} {ate_med:>9.3f} "
              f"{tm_m:>10.2f} {iters_s} {rank_s} {lvl_s} {n:>3} {n_fail:>4}")

    print(f"\nWrote: {per_run_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Per-run artifacts under: {runs_dir}")


if __name__ == "__main__":
    main()
