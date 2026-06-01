# scrybe

> Collapse sound into signal.

A fast, offline Whisper transcription CLI. Point it at a file or a folder; it
transcribes audio to text from the terminal — Metal on Apple Silicon, CPU
everywhere, no Python and no system `ffmpeg`.

## Status

In progress. Audio decode, the model cache, and the whisper.cpp engine (VAD-free
quality gating, language auto-detect, timestamped segments) work on the CPU
backend. Parallel batch orchestration, output writers, and distribution are
landing. Metal is a build feature (`--features metal`) for Apple Silicon; the
default build is CPU so it compiles anywhere.

## Install

From source, until the release channels (Homebrew, `npx`, `cargo binstall`,
curl) are wired:

```sh
cargo install --path .
```

## Usage

```sh
scrybe ./recordings          # transcribe a folder
scrybe talk.mp3 --format srt # one file, SubRip output
scrybe --dry-run ./in        # resolve the plan without transcribing
scrybe models list           # show the model family, sizes, cache status
scrybe models pull large-v3  # pre-fetch a model
scrybe --offline ./in        # cached models only, no network
```

Run `scrybe --help` for the full flag list.

## Codecs

Decoded natively, no system `ffmpeg` required:

| Extension | Codec |
|-----------|-------|
| `wav` | PCM |
| `mp3` | MP3 |
| `flac` | FLAC |
| `ogg` / `oga` | Vorbis |
| `m4a` / `mp4` / `m4b` | AAC-LC, ALAC |

HE-AAC/SBR is not handled by the built-in decoder — it fails with exit code `10`
rather than emit garbled audio. Re-encode with `ffmpeg`, or pass `--decoder ffmpeg`
to decode through a system `ffmpeg` when one is on `PATH`.

## Models

| Model | Notes |
|-------|-------|
| `tiny` / `base` / `small` | small and fast, lower accuracy |
| `large-v3` | most accurate, translation-capable |
| `large-v3-turbo` | default — near-`large-v3` accuracy, much faster |
| `distil-large-v3.5` | distilled, fast, English-leaning |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | success |
| `1` | unexpected error (e.g. failed to write output) |
| `2` | usage error (bad flag or value) |
| `10` | unsupported codec |
| `11` | model download failed |
| `12` | out of memory |
| `13` | GPU init failed |
| `14` | file not found |
| `20` | partial batch failure |

## License

MIT © Coroboros
