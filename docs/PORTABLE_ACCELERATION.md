# Portable acceleration

The association distance matrix has three interchangeable implementations:

| Backend | Precision | Target |
|---|---|---|
| Serial/threaded C++ | float64 | Any C++17 CPU |
| CUDA | float64 | NVIDIA GPUs |
| OpenCL 1.2 | float32 | Vendor-neutral GPUs, CPUs, and embedded accelerators |

OpenCL is found automatically. To require it during configuration and testing:

```bash
cmake -S . -B build -DTRACKER_ENABLE_OPENCL=ON
cmake --build build --parallel
./build/association_benchmark --tracks 1024 --measurements 1024 \
  --require-opencl --output benchmarks/results/opencl.csv
```

Use it in the tracker with `--accelerator opencl`. The old `--gpu` flag remains
as a compatibility alias for `--accelerator cuda`.

## Measured Apple M2 result

For a 1,024 x 1,024 distance matrix over 20 measured runs:

| Backend | Median | p95 | Speedup vs serial CPU |
|---|---:|---:|---:|
| Serial CPU | 1.816 ms | 2.160 ms | 1.00x |
| Threaded CPU | 0.810 ms | 0.912 ms | 2.24x |
| OpenCL kernel | 0.030 ms | 0.087 ms | 59.65x |
| OpenCL end to end | 3.100 ms | 5.369 ms | 0.59x |

The portable kernel is fast, but allocating buffers and moving/transforming data
for every call costs more than the computation at this size. A production path
should keep buffers resident across frames and batch multiple association jobs.
The maximum observed difference from the float64 CPU reference was 7.18 squared
meters on values in the millions, within the benchmark's float32 tolerance.

Raw samples are in `benchmarks/results/opencl_apple_m2_1024.csv`.
