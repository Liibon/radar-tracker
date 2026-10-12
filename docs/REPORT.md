# Radar Multi-Object Tracking Capstone Report

## 1. Objective and scope

This project is an end-to-end radar tracking study: Python produces repeatable
target trajectories, noisy measurements, clutter, plots, and evaluation data;
C++ performs filtering, measurement-to-track association, and track lifecycle
management. The standard target is a multi-object Extended Kalman Filter (EKF)
tracker with two-sensor fusion, deterministic tests, and measured CPU/GPU
performance.

The project is educational rather than safety critical. It intentionally begins
with a transparent nearest-neighbor baseline. More sophisticated association
(global nearest neighbor, JPDA, or MHT) is future work, not implied by baseline
results.

## 2. Mathematical model

### 2.1 State and motion

Each track uses a planar constant-velocity state

```text
x = [p_x, p_y, v_x, v_y]^T
```

For sample interval `dt`, the linear transition is

```text
F = [[1, 0, dt,  0],
     [0, 1,  0, dt],
     [0, 0,  1,  0],
     [0, 0,  0,  1]]
```

The prediction equations are `x^- = F x` and `P^- = F P F^T + Q`. A common
white-acceleration model uses

```text
Q = sigma_a^2 * [[dt^4/4, 0,      dt^3/2, 0],
                 [0,      dt^4/4, 0,      dt^3/2],
                 [dt^3/2, 0,      dt^2,   0],
                 [0,      dt^3/2, 0,      dt^2]]
```

`Q` represents unmodeled acceleration; it is not measurement noise. Larger
values adapt faster to maneuvers but increase estimate variance.

### 2.2 Cartesian Kalman update

For Cartesian position measurements `z = [p_x, p_y]^T`, `h(x) = Hx`, where
`H` selects the first two state components. Innovation `y`, innovation
covariance `S`, gain `K`, and update are

```text
y = z - H x^-
S = H P^- H^T + R
K = P^- H^T S^-1
x = x^- + K y
P = (I - K H) P^- (I - K H)^T + K R K^T
```

The final covariance expression is the Joseph form. It costs slightly more
than `P = (I-KH)P^-` but better preserves symmetry and positive
semidefiniteness under floating-point error. Implementations should solve
linear systems involving `S` rather than form an explicit inverse.

### 2.3 Polar radar EKF update

For a radar at `(s_x, s_y)`, define `d_x = p_x-s_x`, `d_y = p_y-s_y`, and
`r = sqrt(d_x^2+d_y^2)`. The measurement function is

```text
h(x) = [r, atan2(d_y, d_x)]^T
```

The relevant Jacobian rows are

```text
[d_x/r,       d_y/r,       0, 0]
[-d_y/r^2,    d_x/r^2,     0, 0]
```

The EKF substitutes `H = dh/dx` evaluated at the predicted state. Angle
innovations must be normalized to `[-pi, pi)`. Near-zero range makes the
Jacobian singular, so the implementation must reject or regularize such an
update. Range and angle variances have different units and belong explicitly
in `R`.

### 2.4 Association and gating

For measurement `j` and track `i`, the squared normalized innovation distance

```text
d^2(i,j) = y(i,j)^T S(i,j)^-1 y(i,j)
```

is a covariance-aware association cost. A chi-square gate rejects implausible
pairs before nearest-neighbor selection. With two measurement dimensions, a
95% gate is approximately `5.991`; the chosen probability must be recorded
with results. Greedy nearest neighbor is order dependent and can assign at
most one measurement to each track. Unassigned measurements may start
tentative tracks; unassigned tracks receive misses.

### 2.5 Track lifecycle

A new detection creates a tentative track. Confirmation requires a configured
number of hits in a recent window, preventing single clutter points from
becoming reported tracks. Tracks are coasting while temporarily unobserved and
are deleted after a configured miss count or covariance limit. Track IDs are
monotonic and never silently reused during one run. Each lifecycle transition
is testable independently of filtering accuracy.

### 2.6 Multi-sensor fusion

Measurements from sensors with known extrinsics are transformed into a common
world frame. Asynchronous measurements are processed in timestamp order:
predict to the measurement timestamp, update using that sensor's model and
covariance, then continue. Treating correlated measurements as independent
causes overconfidence; the baseline assumes independent sensor noise and must
state that assumption in experiments.

## 3. Architecture

The design separates concerns so they can be tested and timed:

```text
Python scenario generator -> timestamped measurement file/stream
                                      |
                                      v
C++ input -> prediction -> gating/association -> filter update
                                      |                  |
                                      v                  v
                               track manager       track estimates
                                                          |
                                                          v
Python evaluator/plots <- truth and estimate files <- C++ output
```

Core filter objects own state and covariance. Sensor models own `h(x)`,
Jacobians, and noise. The association layer consumes predicted tracks and
measurements without changing lifecycle state. The manager applies matches,
misses, births, confirmations, and deletions. This boundary permits a CPU or
CUDA association backend without changing tracking semantics.

Reproducibility requires the scenario seed, configuration, build type, commit,
compiler, hardware, and backend to be stored with every result.

The measurement CSV is self-contained. Every row includes the required finite
`sensor_x` and `sensor_y` world-frame coordinates; even an origin sensor writes
both as zero. Readers must reject a missing pose rather than silently assume an
origin, because that would change polar-to-world geometry.

## 4. Implementation milestones

1. **Single Cartesian target:** implement prediction/update, generate seeded
   noise, and plot truth, measurements, estimates, position error, and
   covariance bounds.
2. **Polar radar EKF:** implement range/bearing measurement, Jacobian,
   angle wrapping, and near-origin guard.
3. **Multiple targets and clutter:** add gating, one-to-one nearest-neighbor
   association, track birth/confirmation/deletion, and ID metrics.
4. **Second sensor:** add coordinate transforms, per-sensor covariance, and
   timestamp-ordered fusion.
5. **Parallel CPU:** parallelize independent prediction/cost calculations,
   measure determinism, scaling, latency, and throughput.
6. **GPU requirement:** implement and validate an actual CUDA backend for a
   batch operation large enough to benefit (normally association-cost matrix
   and/or batched predict/update), then benchmark it against the same CPU
   workload.
7. **Verification and write-up:** run unit, integration, statistical, and
   performance tests; archive raw measurements and disclose limitations.

Each milestone is complete only when its relevant tests pass and a seeded
example can be reproduced from a clean build.

## 5. Evaluation

### 5.1 Accuracy and consistency

Report position and velocity RMSE against truth after matching estimates to
truth. Also report median and 95th-percentile absolute position error. For
multi-object runs, report confirmed-track recall, false-track count, ID
switches, and track fragmentation; RMSE alone can hide identity failures.

Normalized Estimation Error Squared (NEES) evaluates state/covariance
consistency when truth is known. Normalized Innovation Squared (NIS) evaluates
measurement consistency without using truth in the update. Both should be
examined across many seeded Monte Carlo runs against chi-square intervals.
Passing one random trajectory is not statistical validation.

### 5.2 GPU speedup requirement

GPU acceleration is considered delivered only when a CUDA-capable build:

- compiles a real CUDA kernel or calls a documented CUDA library path;
- reports that CUDA path at runtime, including device name and relevant
  versions;
- produces results within declared tolerances of the CPU reference; and
- is timed using CUDA events with synchronization at measurement boundaries.

A threaded, SIMD, mock, stub, or CPU fallback is useful functionality but is
**not GPU acceleration**. The original Apple M2 run therefore records CUDA as
unavailable rather than treating its CPU fallback as GPU work. A later Runpod
run used an actual NVIDIA device and is reported separately below.

Speedup is `T_cpu / T_gpu` for identical inputs and semantics. Record both
kernel-only timing and end-to-end timing including allocation and host/device
copies. The latter is the operational result. Use a release build, warm up both
paths, pre-generate inputs, run enough iterations for stable samples, and
report median plus dispersion (for example p5/p95 or interquartile range), not
only the fastest run. Test multiple batch sizes because transfer and launch
overheads can make a GPU slower for small scenes. CPU and GPU runs must use the
same gating, precision, and output requirements.

A defensible benchmark record includes:

```text
commit, build flags, compiler/CUDA versions, CPU, GPU, OS
seed, target count, measurement count, clutter rate, duration
backend, warm-up count, measured repetitions
kernel median/p95, end-to-end median/p95, peak memory
CPU/GPU agreement tolerance and validation result
```

### 5.3 Measured CUDA result

The CUDA requirement was executed on an NVIDIA RTX 2000 Ada Generation with
CUDA 12.4.131 and driver 580.159.04. All three C++ tests passed, runtime output
identified the active CUDA device, and the greatest absolute difference from
the serial CPU output was `7.45058059692e-09`.

For a 4,096 by 4,096 distance matrix over 20 measured repetitions, serial CPU
median latency was 105.875 ms, CUDA kernel median latency was 0.843 ms, and
CUDA end-to-end median latency was 79.111 ms. This is a **125.60x kernel-only
speedup** and a **1.34x end-to-end speedup**. At 512 by 512, end-to-end CUDA
was slower than serial CPU (0.63x), demonstrating that launch, allocation, and
transfer overhead matters for small workloads. The operational crossover in
this run occurred between 512 and 1,024 objects.

These are association distance-matrix microbenchmark results, not a claim that
the full multi-object tracker is 125.60x faster. The complete environment,
p95 values, and 320 raw timing samples are preserved in
`docs/BENCHMARK_RECORD.md` and `benchmarks/results/runpod_rtx2000_*.csv`.

## 6. Expected limitations

Constant velocity degrades during sharp maneuvers. EKF linearization is weak
for large uncertainty or geometry near the radar origin. Greedy association
can swap identities at crossings and performs poorly in dense clutter. Simple
hit/miss lifecycle thresholds trade delayed confirmation against false tracks.
The initial fusion model assumes known sensor registration, synchronized clocks,
and independent noise. These constraints define useful follow-on work rather
than invalidating the baseline.

## 7. Reproducibility checklist

- Build and test from a clean checkout in release and sanitizer/debug modes.
- Pin or record Python and C++ dependencies.
- Make all random seeds explicit and store scenario configuration.
- Store raw benchmark samples, not only plots or aggregates.
- Never mix CUDA, CPU-parallel, and CPU-fallback samples under one label.
- Confirm plots are derived from archived CSV output, not hand-entered values.
- Record known failures and skipped tests with reasons.
