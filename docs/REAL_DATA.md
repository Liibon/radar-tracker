# Real-World Radar Evaluation

## Dataset and provenance

The real-data path uses **RadarScenes sequence 158**, downloaded from the
official Zenodo record: <https://doi.org/10.5281/zenodo.4559821>. RadarScenes
contains recordings from four automotive radar sensors mounted on a moving
vehicle, ego odometry, compensated radial velocity, global detection
coordinates, semantic labels, and point-wise object track IDs.

The complete archive is about 11 GB. `python/download_radarscenes_sample.py`
uses HTTP byte ranges to read the official ZIP central directory and extract
only these files:

- `data/sequence_158/radar_data.h5`
- `data/sequence_158/scenes.json`
- `data/sensors.json`
- the dataset README

The script verifies the ZIP CRC for every extracted member and records SHA-256
hashes in `provenance.json`. Downloaded data is excluded from Git and remains
under **CC BY-NC-SA 4.0**; the project's source-code license does not replace
the dataset license.

## Processing without label leakage

Each real radar scan is processed as follows:

1. Discard non-finite returns and returns whose ego-compensated absolute radial
   speed is below 0.5 m/s. This suppresses most static infrastructure.
2. Transformations supplied by RadarScenes (`x_seq`, `y_seq`, ego odometry,
   and sensor calibration) place returns in the global sequence frame.
3. Form connected components within 1.8 m. Components require at least two
   returns in the default high-precision configuration.
4. Convert each component centroid to world-frame range/bearing around the
   moving sensor origin and pass it to the C++ EKF.

No semantic label or track ID is consulted in these four steps. The resulting
`measurements.csv` contains `true_target_id` only as evaluation metadata. The
C++ filter and association cost do not read that field when making decisions.

Separately, point-wise RadarScenes track IDs are grouped into annotation
centroids in `truth.csv`. These centroids are used to score and color output
after tracking.

## Reproduce the included run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

.venv/bin/python -m python.download_radarscenes_sample
.venv/bin/python -m python.run_radarscenes \
  real_data/radarscenes_sequence_158
```

Default real-data tracker parameters are a 3 m association gate, 12 tolerated
missed aggregate sensor scans, and 3 hits for confirmation. They are explicit
CLI options rather than changes to the synthetic defaults.

## Measured sequence-158 result

The downloaded sequence contains 74,616 raw returns over 14.526 seconds. The
label-free front end retained 1,393 moving returns and formed 313 multi-return
components. The reference run produced:

| Metric | Result |
|---|---:|
| Annotated targets in sequence | 13 |
| Targets surviving the conservative front end | 5 |
| Targets with confirmed tracks | 4 |
| Confirmed track IDs | 5 |
| False confirmed tracks | 1 |
| Track fragmentations | 0 |
| Matched estimate samples | 315 |
| Position RMSE against annotation centroids | 0.366 m |
| Median position error | 0.257 m |
| 95th-percentile position error | 0.696 m |

These values are generated in `real_output/metrics.json` and are not substituted
for the synthetic benchmark results.

## Interpretation and limitations

RadarScenes provides point-wise object annotations rather than independent,
survey-grade target-center ground truth. The evaluation position is therefore
the centroid of annotated radar returns, and the reported sub-meter error tests
temporal smoothing and track continuity relative to that centroid—not absolute
radar accuracy. Finite-difference centroid velocity is too noisy for a credible
velocity score, so the real-data plot intentionally reports position only.

The two-return component requirement strongly reduces false alarms but misses
sparsely observed pedestrians and bicycles. Lowering `--min-cluster-points` to
one increases recall and false tracks. This precision/recall tradeoff is a real
front-end detection limitation; labels are not used to hide it. A production
extension should add Doppler-aware clustering and covariance-aware global
assignment before drawing broader conclusions.
