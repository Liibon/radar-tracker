# Performance Record

Do not enter estimates in this file. Preserve raw samples beside every completed
record and link to them from `raw data`.

## Environment

| Field | Value |
|---|---|
| Date/time and timezone | 2026-10-11 20:55 CDT |
| Git commit and dirty state | Initial project, before first commit |
| OS/kernel | macOS 15.3.2 / Darwin 24.3.0 arm64 |
| CPU and logical/physical cores | Apple M2; 8 logical / 8 physical |
| Thread count and affinity policy | `std::thread::hardware_concurrency()` = 8; OS default affinity |
| RAM | 8 GiB |
| Compiler and flags | Apple clang 17.0.0; CMake Release build |
| CUDA toolkit/driver | `nvcc` and NVIDIA driver tools not found |
| GPU | CUDA not available; GPU benchmark not run |
| Tracker backend reported at runtime | CPU threaded available; CUDA unavailable |
| Precision | IEEE-754 `double` |

Attach the output of:

```bash
python3 tests/probe_cuda.py
```

The probe only checks the environment. A device/toolkit result is not evidence
that tracker work ran on the GPU. Preserve tracker backend logs and a profiler
trace or other implementation-level evidence with CUDA results.

## Workload

| Field | Value |
|---|---|
| Scenario/seed | Synthetic association matrix; seed 20261011 |
| Targets | 1024 track positions |
| Measurements per frame | 1024 positions |
| Clutter per frame | Not applicable to distance-kernel microbenchmark |
| Frames/sample interval | One 1,048,576-element distance matrix per repetition |
| Association gate | Not applicable; distance computation only |
| Warm-up/measured repetitions | 3 / 10 per backend |
| Raw data | `docs/benchmark_cpu_m2.csv`; probe: `benchmarks/results/apple_m2_cuda_probe.json` |

## Correctness before timing

| Comparison | Tolerance | Maximum observed difference | Result |
|---|---:|---:|---|
| CPU serial vs CPU parallel | Exact equality | 0 | Pass |
| CPU serial vs CUDA | Relative 1e-12 | Not measured | CUDA unavailable |

Compare not only floating-point states but also association decisions, lifecycle
events, and output row counts. A fast backend with different tracking semantics
is not an equivalent speedup result.

## Timing results

| Backend | Scope | Median | p95 | Repetitions | Speedup vs serial CPU |
|---|---|---:|---:|---:|---:|
| Serial CPU | distance core | 1.620 ms | 2.430 ms | 10 | 1.0 baseline |
| Threaded CPU | distance core | 0.707 ms | 1.459 ms | 10 | 2.29x |
| CUDA | kernel only | Not run | Not run | 0 | Not measured |
| CUDA | end to end | Not run | Not run | 0 | Not measured |

End-to-end CUDA time includes allocations (unless explicitly reused in both
real operation and benchmark), host/device transfers, kernel execution, output
transfer, and synchronization. State explicitly when CUDA is unavailable; do
not put CPU fallback measurements in a row labeled CUDA.

The CPU result above was measured with
`./build/association_benchmark --tracks 1024 --measurements 1024 --warmup 3
--repetitions 10`. Speedup is the serial median divided by the threaded median.
This is a distance-matrix microbenchmark, not end-to-end tracker speedup.

The CUDA rows are populated only when runtime output identifies an active CUDA
backend and the result agrees with serial CPU. `cuda-kernel` isolates device
execution; `cuda-end-to-end` includes transfers and synchronization. If CUDA is
unavailable, leave those rows as `Not run` and attach the probe output. A CPU
fallback must remain labeled CPU.

## Interpretation

- Crossover batch size: Not measured.
- Throughput: 646.99 million distances/s serial and 1.48 billion distances/s threaded, based on medians.
- Tail-latency change: p95 fell from 2.430 ms to 1.459 ms in this 10-sample run.
- Peak memory: Not measured.
- Main bottleneck and profiler evidence: Not measured.
- Threats to validity: short microbenchmark, one machine, OS scheduling noise,
  no affinity pinning, and no full association sort or end-to-end tracker work.
