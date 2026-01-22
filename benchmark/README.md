# Benchmark

End-to-end inference benchmarking for VoxCPM.

## Run

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --concurrency 4 --iters 5 --warmup 1
```

Key flags:

- `--concurrency`: number of concurrent `generate()` requests
- `--max-generate-length`: maximum number of generation steps per request
- `--devices`: CUDA devices, e.g. `0` or `0,1`
- `--json-out`: write machine-readable results

## Notes

- Metrics are measured from the parent process wall time and include IPC overhead.
- If the model directory is local, the script reads `config.json` to infer `sample_rate` for RTF; otherwise provide `--sample-rate`.
- `RTF_per_req_mean` is computed as the average over requests of `(request_wall_time / request_audio_duration)`.
