# Benchmark baseline

Real-time factor (×RT = audio seconds ÷ wall seconds; higher is faster), transcribing
a 55.8 s English clip as a single file. ×RT is scrybe's own reported figure (inference
wall time only). Reproduce with [`run.sh`](run.sh).

## CPU — Apple M1, 8 cores, 16 GB

| Model | ×RT | Wall |
| --- | ---: | ---: |
| `tiny` | 30.9× | 1.8 s |
| `base` | 20.6× | 2.7 s |
| `large-v3-turbo` | 3.1× | 17.8 s |

`small` and `large-v3` need a model download first; `run.sh` adds them when asked.

## Metal — Apple Silicon

The `metal` build offloads inference to the GPU; the speedup depends on the chip, so
measure it on the target rather than quote a number here:

```sh
cargo build --release --features metal
bench/run.sh large-v3-turbo large-v3
```

## Method

- Sample: 55.8 s, 16 kHz mono — the committed test clip looped (`run.sh` regenerates it).
- Release build (thin LTO, stripped), default thread count (all cores).
- Numbers above are a single M1 run; expect a few percent of variance between runs.
