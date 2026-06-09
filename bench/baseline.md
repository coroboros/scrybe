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

## Metal — Apple M1, 8 cores, 16 GB

The `metal` build (`cargo build --release --features metal`) offloads inference to the GPU:

| Model | ×RT | Wall |
| --- | ---: | ---: |
| `tiny` | 33.6× | 1.7 s |
| `base` | 31.3× | 1.8 s |
| `large-v3-turbo` | 8.1× | 6.9 s |

The GPU win grows with model size — `large-v3-turbo` is ~2.6× faster than CPU here,
while the small models are already overhead-bound and barely move. Larger chips (M-series
Pro/Max/Ultra) widen the gap. Run `bench/run.sh` on a `--features metal` build to measure
a given chip.

## Method

- Sample: 55.8 s, 16 kHz mono — the committed test clip looped (`run.sh` regenerates it).
- Release build (thin LTO, stripped), default thread count (all cores).
- Numbers above are a single M1 run; expect a few percent of variance between runs.
