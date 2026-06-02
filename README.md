# Implementing Robust M-Estimators with Certifiable Factor Graph Optimization

Implementation of robustified certifiable factor graph optimization

<p align="center">
  <img src="media/comparision.png" alt="Comparison" width="520"><br>
  <img src="media/pgo_trajectories.gif" alt="PGO trajectories" width="260">
  <img src="media/landmark_slam_trajectories.gif" alt="Landmark SLAM trajectories" width="260">
</p>

## Abstract

Adaptive reweighting converts robust M-estimation into a sequence of weighted least-squares (WLS) subproblems. We solve each subproblem using a Riemannian-staircase optimizer with certifiable optimality checks, making the overall algorithm more robust to outliers and poor initial guesses. The solver integrates naturally into the standard GTSAM factor-graph workflow and builds on the *Certifiable Factor Graph Optimization* framework of Xu, Sanderson, Zhang, and Rosen (2026).

## Citation

If you use this work, please cite:

```bibtex
@article{xu2026implementing,
  title   = {Implementing Robust M-Estimators with Certifiable Factor Graph Optimization},
  author  = {Xu, Zhexin and Zhang, Hanna Jiamei and Calatrava, Helena and Closas, Pau and Rosen, David M.},
  journal = {arXiv preprint arXiv:2603.20932},
  year    = {2026}
}
```

The underlying certifiable factor-graph framework that this work is built upon:

```bibtex
@article{xu2026certifiable,
  title   = {Certifiable Estimation with Factor Graphs},
  author  = {Xu, Zhexin and Sanderson, Nikolas R. and Zhang, Hanna Jiamei and Rosen, David M.},
  journal = {arXiv preprint arXiv:2603.01267},
  year    = {2026}
}
```

## Build

### Dependencies

Tested on Ubuntu 22.04 / 24.04. C++17 required.

- CMake
- GCC / Clang
- Eigen
- BLAS / LAPACK
- SuiteSparse (CHOLMOD + SPQR)
- gflags
- glog
- GoogleTest
- Boost
- GTSAM 4.2+

System packages (everything except GTSAM):

```bash
sudo apt install \
    build-essential cmake \
    libeigen3-dev libopenblas-dev liblapack-dev libsuitesparse-dev \
    libgflags-dev libgoogle-glog-dev libgtest-dev libboost-all-dev
```

Two inner dependencies are pulled in as git submodules:
`Optimization/` (LOBPCG + Riemannian TNT) and `Preconditioners/` (ILDL).
The top-level `CMakeLists.txt` builds them automatically.

### Build this project

```bash
git clone --recurse-submodules git@github.com:NEU-RAL/Certi-GNC.git
cd Certi-GNC
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

If you cloned without `--recurse-submodules`, run
`git submodule update --init --recursive` before the cmake step.

If your GTSAM lives in a non-standard location, pass its build directory:
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DGTSAM_DIR=/path/to/gtsam/build \
      -DGTSAMCMakeTools_DIR=/path/to/gtsam/cmake ..
```

Produces four GNC driver binaries plus two unit tests under `build/bin/`:

```text
Certifiable_GNC_PGO_example   # ours, PGO
Certifiable_GNC_LMK_example   # ours, landmark SLAM
Vanilla_GNC_PGO_example       # gtsam::GncOptimizer on the same lifted graph, no staircase
Vanilla_GNC_LMK_example       # gtsam::GncOptimizer on the same lifted graph, no staircase
testAssemblerSimple  testDataMatrixUtils
```

The vanilla baselines use **identical** factor types (`SEsyncFactor` /
`LiftedLandmarkFactor`), variable types (`LiftedPoseDP` / `Vector`
landmarks), random initial guess (via `cpp/RandomInit.h`, same seed),
LM solver, and GNC parameters as ours. The only algorithmic difference
is the inner solver: vanilla runs a single LM call per GNC iteration at
rank `p = d`, while ours climbs the Riemannian staircase until the
global-optimality certificate passes.

## Use

### Single-run example

One demo dataset per problem is shipped under `data/`:

```text
data/pgo/intel_outl20.g2o                       # 1728 poses, 20 % outliers
data/pgo/intel_GT.txt                           # TUM-format ground truth
data/landmark/cityTrees_reduced_outl20.g2o      # 1600 poses, 20 % landmark-obs outliers
data/landmark/cityTrees_reduced_GT.txt          # TUM-format ground truth
```

#### Common flags (all four binaries)

| Flag                       | Default     | Description                                                |
|----------------------------|-------------|------------------------------------------------------------|
| `--d`                      | `2`         | Ambient dimension (2 = SE(2), 3 = SE(3); LMK is d=2 only)  |
| `--p`                      | `2`         | Initial relaxation rank (`Certifiable_GNC_*` only)         |
| `--input_dir`              | demo file   | Input g2o file                                             |
| `--output_dir`             | demo stem   | Output file stem (extension is added per artifact)         |
| `--init_type`              | `random`    | `random` or `odom` (= initial values from the dataset's `VERTEX_*` lines; see below) |
| `--seed`                   | `1`         | RNG seed for random init                                   |
| `--inner_max_iterations`   | `200`       | Max LM iterations per inner solve                          |
| `--gnc_max_iterations`     | `30`        | Max GNC outer iterations                                   |
| `--mu_step`                | `1.4`       | TLS mu update factor                                       |
| `--rel_cost_tol`           | `1e-5`      | GNC relative cost tolerance                                |
| `--weights_tol`            | `1e-4`      | "Weights converged to binary" tolerance                    |
| `--abs_error_tol`          | `1e-5`      | LM absolute error tolerance                                |
| `--rel_error_tol`          | `1e-5`      | LM relative error tolerance                                |
| `--gnc_verbosity`          | `1`         | `0` SILENT, `1` SUMMARY, `2` MU, `3` WEIGHTS, `4` VALUES   |

`Certifiable_GNC_*` adds certifier-specific flags: `--use_abs_eta`,
`--eta`, `--MIN_CERT_ETA`, `--MAX_CERT_ETA`, `--REL_CERT_ETA`,
`--abs_cost_tol` (default `0`, disabled; matches vanilla which has no
absolute-cost criterion). Run `<binary> --help` for the full list with
defaults.

#### PGO

```bash
# init_type=odom: initial guess loaded from the g2o file's VERTEX_SE2 lines
# (not actual odometry integration — see "Note on --init_type=odom" above)
./build/bin/Certifiable_GNC_PGO_example \
    --input_dir=data/pgo/intel_outl20.g2o \
    --output_dir=results/pgo_demo \
    --init_type=odom

# Random init, seed 0
./build/bin/Certifiable_GNC_PGO_example \
    --input_dir=data/pgo/intel_outl20.g2o \
    --output_dir=results/pgo_demo_rand \
    --init_type=random --seed=0

# Vanilla baseline on the same data
./build/bin/Vanilla_GNC_PGO_example \
    --input_dir=data/pgo/intel_outl20.g2o \
    --output_dir=results/vanilla_pgo_demo \
    --init_type=random --seed=0

# Score against ground truth
python3 scripts/eval_gnc_pgo.py \
    --gt     data/pgo/intel_GT.txt \
    --result results/pgo_demo
```

#### Landmark SLAM

```bash
./build/bin/Certifiable_GNC_LMK_example \
    --input_dir=data/landmark/cityTrees_reduced_outl20.g2o \
    --output_dir=results/lmk_demo \
    --init_type=odom

./build/bin/Vanilla_GNC_LMK_example \
    --input_dir=data/landmark/cityTrees_reduced_outl20.g2o \
    --output_dir=results/vanilla_lmk_demo \
    --init_type=random --seed=0

python3 scripts/eval_gnc_pgo.py \
    --gt     data/landmark/cityTrees_reduced_GT.txt \
    --result results/lmk_demo
```

#### Inject your own outliers

```bash
python3 scripts/inject_outliers.py \
    --input  /path/to/your/clean.g2o \
    --output /tmp/your_outl15.g2o \
    --pct 15 --seed 7
```

#### Output files

Each run writes (under `--output_dir`):

| File                  | Contents                                                |
|-----------------------|---------------------------------------------------------|
| `<stem>.txt`          | TUM-format trajectory (id x y z qx qy qz qw)            |
| `<stem>_iter_log.txt` | Human-readable per-iteration GNC log                    |
| `<stem>_iters.csv`    | Per-iteration scalars (objective, ranks, timing)        |
| `<stem>_weights.txt`  | Final per-factor weight, one row per factor             |
| `<stem>_weights.csv`  | Long-form per-iter weights (ours only)                  |
| `<stem>_summary.csv`  | One-row run summary with staircase diagnostics          |

## Known Issues

## Contact

For questions or suggestions, please contact [xu.zhex@northeastern.edu](mailto:xu.zhex@northeastern.edu).

## License

Copyright (c) 2026 Northeastern University Robust Autonomy Lab.
Licensed under the [MIT License](LICENSE).
