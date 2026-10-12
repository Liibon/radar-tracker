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

## CUDA record: Runpod RTX 2000 Ada

This is a separate measurement from the Apple M2 CPU record above. It was
executed on real CUDA hardware on 2026-10-12 at git commit `6766088` using a
Release build configured with `-DTRACKER_ENABLE_CUDA=ON`.

### Environment

| Field | Value |
|---|---|
| Provider / pod | Runpod Secure Cloud; ephemeral pod, deleted after collection |
| Allocated CPU / RAM | 6 vCPU / 31 GiB; AMD EPYC 7443P host |
| GPU | NVIDIA RTX 2000 Ada Generation, 16,380 MiB |
| Driver / CUDA compiler | 580.159.04 / CUDA 12.4.131 |
| Compiler / build | GCC 11.4.0 / CMake 3.22.1 / Release |
| Runtime backend | `CUDA backend active; device=NVIDIA RTX 2000 Ada Generation` |
| Warm-up / measured runs | 3 / 20 per backend and matrix size |
| Correctness | Maximum CPU/CUDA absolute difference `7.45058059692e-09` |
| Unit tests | 3 of 3 CTest tests passed |

The workload computes the complete pairwise squared-distance matrix for equal
numbers of tracks and measurements. `cuda-kernel` is measured with CUDA events.
`cuda-end-to-end` additionally includes allocation, host/device transfers,
result transfer, and synchronization.

### Timing results

| Matrix | Backend | Median (ms) | p95 (ms) | Speedup vs serial CPU |
|---:|---|---:|---:|---:|
| 512 x 512 | Serial CPU | 0.521544 | 0.587780 | 1.00x |
| 512 x 512 | CUDA end to end | 0.826970 | 0.848852 | 0.63x |
| 512 x 512 | CUDA kernel | 0.020064 | 0.020323 | 25.99x |
| 1,024 x 1,024 | Serial CPU | 2.052989 | 2.264666 | 1.00x |
| 1,024 x 1,024 | CUDA end to end | 1.543601 | 1.566384 | 1.33x |
| 1,024 x 1,024 | CUDA kernel | 0.059360 | 0.060006 | 34.59x |
| 2,048 x 2,048 | Serial CPU | 23.413266 | 23.869210 | 1.00x |
| 2,048 x 2,048 | CUDA end to end | 18.881013 | 23.097942 | 1.24x |
| 2,048 x 2,048 | CUDA kernel | 0.218512 | 0.219499 | 107.15x |
| 4,096 x 4,096 | Serial CPU | 105.875467 | 107.428888 | 1.00x |
| 4,096 x 4,096 | CUDA end to end | 79.111374 | 82.787774 | 1.34x |
| 4,096 x 4,096 | CUDA kernel | 0.842928 | 0.844453 | 125.60x |

The operational crossover occurred between 512 and 1,024 objects. At 4,096,
the kernel itself was 125.60x faster than serial CPU, while transfer and
allocation overhead reduced the honest end-to-end speedup to 1.34x. The small
maximum difference is consistent with GPU/CPU floating-point evaluation order
and passed the benchmark's declared correctness check.

Raw samples and environment capture:

- `benchmarks/results/runpod_rtx2000_512.csv`
- `benchmarks/results/runpod_rtx2000_1024.csv`
- `benchmarks/results/runpod_rtx2000_2048.csv`
- `benchmarks/results/runpod_rtx2000_4096.csv`
- `benchmarks/results/runpod_rtx2000_environment.txt`
