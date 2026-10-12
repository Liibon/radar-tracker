# Python simulator and analysis

Install the only runtime dependencies:

```bash
python3 -m pip install -r python/requirements.txt
```

Generate a deterministic two-radar range/bearing scenario and plots:

```bash
python3 python/runner.py --output-dir output --seed 7
```

Use `--cartesian`, `--one-sensor`, `--targets N`, `--duration S`, and `--dt S`
to vary the scenario. To run the C++ executable in the same workflow:

```bash
python3 python/runner.py --output-dir output \
  --tracker ./build/radar_tracker
```

The runner appends `--input PATH --output PATH`. For a different executable,
`{measurements}` and `{estimates}` placeholders can be embedded in the tracker
command instead. An existing estimate
file can instead be supplied with `--estimates output/estimates.csv`.

## CSV contract

- `truth.csv`: `timestamp,target_id,x,y,vx,vy`
- `measurements.csv`: `timestamp,sensor_id,measurement_id,type,value1,value2,true_target_id`
- C++ output: `timestamp,track_id,x,y,vx,vy[,truth_target_id]`

For a `polar` measurement, `value1` is range in metres and `value2` is bearing
in radians counter-clockwise from global +x. For `cartesian`, they are global
x/y metres. `true_target_id=-1` denotes clutter. `scenario.json` records units,
sensor locations, noise settings, target definitions, and the random seed.

Plot benchmark results (`method,objects,elapsed_ms[,frames]`) with:

```bash
python3 python/plot_performance.py benchmark.csv --output output/performance.png
```

The performance chart shows latency, throughput, and speedup relative to the
serial/CPU method (or the first method if neither name is present).
