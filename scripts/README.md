# Scripts — runbook

All scripts live under `scripts/`. They are Python 3. Only the
visualization helpers need extra packages (`matplotlib`, `numpy`); the
experiment scripts are pure stdlib.

### Experiment scripts

| Script                       | Purpose                                                                |
|------------------------------|------------------------------------------------------------------------|
| `inject_outliers.py`         | Random-replacement outlier injection into a g2o file (+ mask).         |
| `make_small_pgo.py`          | Subset a PGO g2o to its first N poses; emit g2o + TUM GT.              |
| `make_small_landmark.py`     | Subset a landmark g2o to first N poses; drop single-obs lmks; GT.      |
| `eval_gnc_pgo.py`            | ATE (translation RMSE) vs. a TUM-format ground truth.                  |
| `run_benchmark_pgo.py`       | Monte Carlo PGO benchmark: ours vs. vanilla GTSAM GNC.                 |
| `run_benchmark_lmk.py`       | Monte Carlo landmark benchmark: ours vs. vanilla GTSAM GNC.            |

### Visualization / format-conversion helpers

| Script              | Purpose                                                                |
|---------------------|------------------------------------------------------------------------|
| `plotg2o.py`        | Visualize a g2o trajectory (poses + landmarks) with matplotlib.        |
| `plottum.py`        | Plot a TUM-format trajectory.                                          |
| `g2o_to_tum.py`     | Convert a g2o file's vertices to TUM format.                           |
| `tum_to_g2o.py`     | Convert a TUM trajectory back to g2o vertices.                         |
| `rmse.py`           | Standalone RMSE / ATE between two TUM trajectories (numpy-based).      |

All commands below assume the project root (`StiefelManifold/`) is the
working directory and the binaries have been built (`build/bin/`).

---

## Monte Carlo runners

Each runner sweeps **methods × inits × outlier-rates × trials** on a
single dataset and writes per-run artifacts plus an aggregated summary.

### Common flags

| Flag | Default | Notes |
|---|---|---|
| `--input` | *(required)* | Clean source g2o file. Outliers are injected per trial. |
| `--gt` | *(required)* | TUM-format ground-truth trajectory (for ATE). |
| `--out_dir` | `results/benchmark_intel` (PGO) / `results/benchmark_cityTrees` (LMK) | Output folder. |
| `--num_trials` | `10` | Trials per outlier rate. Trial `s` uses seed `(seed_offset + s)` for both outlier injection and random init. |
| `--seed_offset` | `0` | Shifts the seed sequence. `--num_trials 10 --seed_offset 100` → seeds 100…109. Use to (re)run an independent MC realization. |
| `--rates` | `10 20 30` | Outlier percentages to sweep, space-separated. |
| `--methods` | `ours vanilla` | Subset of {`ours`, `vanilla`}. |
| `--inits` | `random odom` | Subset of {`random`, `odom`}. |
| `--gnc_max_iterations_ours` | `25` | Outer-loop cap for the certifiable wrapper. |
| `--gnc_max_iterations_vanilla` | `30` | Outer-loop cap for `gtsam::GncOptimizer`. |
| `--sigma_landmark` *(LMK only)* | `100.0` | StdDev of injected landmark-outlier offsets (m). |

### PGO benchmark

> **Note.** Both runners take a *clean* g2o source via `--input` and
> inject outliers per trial. The demo files shipped in `data/` already
> have outliers, so they are not suitable here — point `--input` at an
> external clean dataset (e.g. an unmodified intel.g2o / cityTrees.g2o).

PGO Monte Carlo (10 trials × 3 rates × 2 methods × 2 inits = 120 runs, ~20 min):

```bash
python3 scripts/run_benchmark_pgo.py \
    --input /path/to/clean_intel.g2o \
    --gt    /path/to/intel_GT.txt \
    --num_trials 10 --rates 10 20 30 \
    --out_dir results/benchmark_intel
```

Quick sweep, one rate, 3 trials:

```bash
python3 scripts/run_benchmark_pgo.py \
    --input /path/to/clean_intel.g2o --gt /path/to/intel_GT.txt \
    --rates 20 --num_trials 3 \
    --out_dir results/quick_pgo_20
```

Only our method, only random init:

```bash
python3 scripts/run_benchmark_pgo.py \
    --input /path/to/clean_intel.g2o --gt /path/to/intel_GT.txt \
    --num_trials 5 \
    --methods ours --inits random \
    --out_dir results/ours_random
```

Vanilla only, both init modes, single rate:

```bash
python3 scripts/run_benchmark_pgo.py \
    --input /path/to/clean_intel.g2o --gt /path/to/intel_GT.txt \
    --rates 20 --num_trials 10 \
    --methods vanilla \
    --out_dir results/vanilla_pgo
```

### Landmark benchmark

```bash
python3 scripts/run_benchmark_lmk.py \
    --input /path/to/clean_cityTrees.g2o \
    --gt    /path/to/cityTrees_GT.txt \
    --num_trials 10 --rates 10 20 30 \
    --out_dir results/benchmark_cityTrees
```

Lower outlier-sigma (milder outliers):

```bash
python3 scripts/run_benchmark_lmk.py \
    --input /path/to/clean_cityTrees.g2o \
    --gt    /path/to/cityTrees_GT.txt \
    --rates 10 20 30 --num_trials 5 \
    --sigma_landmark 20 \
    --out_dir results/lmk_sigma20
```

---

## What the runners produce

Inside `--out_dir`:

```
<out_dir>/
├── data/                                          per-trial outlier-injected
│   └── <dataset>_outl{R}_seed{S}.g2o              g2o files (+ _mask.txt)
├── runs/                                          per-run artifacts
│   ├── {method}_init-{init}_rate{R}_seed{S}.txt        TUM trajectory
│   ├── ..._iter_log.txt                                GNC iteration log
│   ├── ..._iters.csv                                   per-iter scalars
│   ├── ..._summary.csv                                 15-field run summary
│   ├── ..._weights.txt  ..._weights.csv                final / per-iter weights
│   └── ....log                                         stdout/stderr
├── per_run.csv                                    one row per individual run
└── summary.csv                                    one row per (method,init,rate) cell
```

`per_run.csv` columns:
```
method, init, rate, seed,
ate, time, ok,
num_gnc_iters, final_ending_rank, max_ending_rank,
levels_climbed_total, num_certifier_verified, num_certifier_failed,
sum_lm_opt_time, sum_verification_time
```

`summary.csv` columns:
```
method, init, rate, n,
ate_mean, ate_std, ate_median,
time_mean, time_std,
iters_mean, max_rank_mean, levels_total_mean, verified_mean,
lm_opt_time_mean, verify_time_mean,
n_failures
```

The runner also prints a formatted table to stdout at the end of the run.

---

## Re-running / partial re-runs

Outlier-injected datasets in `<out_dir>/data/` are only generated if they
don't already exist. If you've already run an MC at one rate and want to
add another rate, just point at the same `--out_dir` and the previously
injected files are reused. **Per-run artifacts in `<out_dir>/runs/` are
always overwritten** when their `(method, init, rate, seed)` cell is
re-executed — so re-running just regenerates them.

To do a clean rerun:

```bash
rm -rf results/benchmark_intel
python3 scripts/run_benchmark_pgo.py --input ... --gt ... ...
```

---

## Single-run usage (without the MC harness)

Inject + run + eval as three separate commands:

```bash
# 1. Inject (point --input at your own clean g2o)
python3 scripts/inject_outliers.py \
    --input  /path/to/clean.g2o \
    --output /tmp/clean_outl15.g2o \
    --pct 15 --seed 7

# 2. Run (ours)
./build/bin/Certifiable_GNC_PGO_example \
    --input_dir=/tmp/clean_outl15.g2o \
    --output_dir=/tmp/run \
    --init_type=random --seed=7

# 3. Score
python3 scripts/eval_gnc_pgo.py \
    --gt data/pgo/intel_GT.txt --result /tmp/run
```

For all program-argument options on the four binaries, see the top-level
`README.md` "Use" section or run `<binary> --help`.
