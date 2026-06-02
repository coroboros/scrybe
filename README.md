<div align="center">

<img src="assets/logo.png" width="288" height="288" alt="scrybe"/>

<!-- omit in toc -->
# scrybe

**Collapse sound into signal — a fast, offline Whisper transcription CLI.**

Pure-Rust audio decode, whisper.cpp via whisper-rs, Metal on Apple Silicon and CPU everywhere. Point it at one file or a whole folder and get text back from the terminal. No Python, no system `ffmpeg`.

[![crates.io](https://img.shields.io/crates/v/scrybe?style=flat-square&color=000000)](https://crates.io/crates/scrybe)
[![ci](https://img.shields.io/github/actions/workflow/status/coroboros/scrybe/ci.yml?branch=main&style=flat-square&label=ci&color=000000)](https://github.com/coroboros/scrybe/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-000000?style=flat-square)](https://opensource.org/licenses/MIT)
[![stars](https://img.shields.io/github/stars/coroboros/scrybe?style=flat-square&label=stars&color=000000)](https://github.com/coroboros/scrybe)
[![coroboros.com](https://img.shields.io/badge/coroboros.com-000000?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48cGF0aCBkPSJNMiAxMmgyME0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==)](https://coroboros.com)

</div>

<!-- omit in toc -->
## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Usage](#usage)
- [Why this exists](#why-this-exists)
- [Models](#models)
- [Codecs](#codecs)
- [Options](#options)
- [Output formats](#output-formats)
- [Exit codes](#exit-codes)
- [Limitations](#limitations)
- [Compared to alternatives](#compared-to-alternatives)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- macOS (Apple Silicon or Intel), Linux, or Windows.
- A few hundred MB to ~8 GB of free RAM, depending on the model. scrybe auto-selects the largest model that fits detected RAM, so it runs on small machines and scales up on large ones.
- **From a prebuilt binary** — nothing else. The binary embeds whisper.cpp and the voice-activity model.
- **From source** — a C/C++ toolchain and CMake (whisper.cpp is built by `whisper-rs-sys`). Apple Silicon adds the Metal backend with `--features metal`.

## Install

> scrybe is pre-release. Build from source today; the binary distributions below go live on the first published release.

```sh
brew install coroboros/tap/scrybe   # macOS — the blessed path
npx @coroboros/scrybe               # Node toolchains
cargo binstall scrybe               # prebuilt binary via cargo
```

From source (any platform; Apple Silicon adds the Metal backend):

```sh
cargo install --path .                   # CPU
cargo install --path . --features metal  # Apple Silicon (Metal)
```

A raw GitHub-release download on macOS may be quarantined — clear it with `xattr -d com.apple.quarantine ./scrybe`. The Homebrew, npx, and cargo paths are not quarantined.

## Usage

```sh
scrybe ./recordings           # transcribe a folder
scrybe talk.mp3 --format srt  # one file, SubRip output
scrybe talk.mp3 --json        # stream a JSON transcript to stdout
scrybe --dry-run ./in         # resolve the plan without transcribing
scrybe models list            # show the model family, sizes, cache status
scrybe models pull large-v3   # pre-fetch a model
scrybe --offline ./in         # cached models only, no network
```

Run `scrybe --help` for the full flag list.

## Why this exists

Running Whisper from the terminal usually means a Python environment, a system `ffmpeg`, and a separate step to convert audio to 16 kHz WAV before the model ever sees it. scrybe collapses that into one binary.

- **Offline and private.** Models download once, verified against a pinned SHA-256, into the standard Hugging Face cache. After that, nothing leaves the machine — `--offline` enforces it.
- **No Python, no system `ffmpeg`.** A single Rust binary decodes mp3, wav, flac, ogg, and m4a natively via [symphonia](https://github.com/pdeljanov/Symphonia), resamples to 16 kHz mono, and runs [whisper.cpp](https://github.com/ggml-org/whisper.cpp) through [whisper-rs](https://github.com/tazz4843/whisper-rs).
- **Metal-accelerated on Apple Silicon**, CPU everywhere else. The default build needs no GPU toolchain.
- **Zero-config.** Omit `--model` and `--jobs` and scrybe picks the largest model and the concurrency that fit detected RAM, so a flag-free run is never refused by its own memory guard.
- **Fails loud, never silent.** Voice-activity segmentation is always on as a correctness floor. Unsupported codecs, out-of-memory runs, and output collisions stop with an actionable message and a stable exit code rather than emitting garbled or overwritten files.

## Models

| Model | Notes |
|-------|-------|
| `tiny` / `base` / `small` | small and fast, lower accuracy |
| `large-v3` | most accurate, translation-capable |
| `large-v3-turbo` | default — near-`large-v3` accuracy, much faster |
| `distil-large-v3.5` | distilled, fast, English-leaning |

Weights are ggml builds from the whisper.cpp Hugging Face repos. Only `large-v3` translates to English (`--task translate`); the gate rejects the others before any download. With `--model` omitted, scrybe resolves the largest model that fits detected RAM at the chosen job count.

On an Apple M1, `tiny` transcribes at ~31×RT on CPU and `large-v3-turbo` at ~3×RT, rising to ~8×RT on Metal. Full numbers and method: [`bench/baseline.md`](bench/baseline.md).

## Codecs

Decoded natively, no system `ffmpeg` required:

| Extension | Codec |
|-----------|-------|
| `wav` | PCM |
| `mp3` | MP3 |
| `flac` | FLAC |
| `ogg` / `oga` | Vorbis |
| `m4a` / `mp4` / `m4b` | AAC-LC, ALAC |

HE-AAC/SBR is not handled by the built-in decoder — it fails with exit code `10` rather than emit garbled audio. Re-encode with `ffmpeg`, or pass `--decoder ffmpeg` to decode through a system `ffmpeg` when one is on `PATH`.

Both decoders stream straight to 16 kHz mono, so the full-resolution source is never held in memory — a long, high-bitrate file (an hour of 48 kHz stereo) decodes fine. Only the 16 kHz output is resident, bounded at ~4.6 hours per file; a longer single clip fails loud with exit code `10`, so split marathon recordings first.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `<paths>...` | *(required)* | Audio files or directories to transcribe. |
| `--recursive` | `false` | Recurse into subdirectories. |
| `--model <MODEL>` | largest that fits RAM | Whisper model. See [Models](#models). |
| `--lang <LANG>` | auto-detect | Source language code (`en`, `fr`, …). |
| `--task <TASK>` | `transcribe` | `transcribe` or `translate` (to English). |
| `--format <FMT,…>` | `txt` | Output formats, comma-separated. See [Output formats](#output-formats). |
| `--out-dir <DIR>` | beside input | Write outputs here instead of next to each input. |
| `--jobs <N>` | device-aware | Files decoded concurrently ahead of inference. |
| `--threads <N>` | device-aware | CPU threads per inference job. |
| `--force` | `false` | Reprocess inputs even when an up-to-date output exists. |
| `--dry-run` | `false` | Print the resolved plan without transcribing. |
| `--decoder <BACKEND>` | `symphonia` | `symphonia` (built-in) or `ffmpeg` (system). |
| `--json` | `false` | Force JSON; stream to stdout for one file, `.json` sidecars for many. |
| `--offline` | `false` | Use only cached models; never access the network. |
| `--no-color` | `false` | Disable colored output. |

<details>
<summary><code>scrybe models</code> — manage models on disk</summary>

<br>

| Command | Description |
| --- | --- |
| `models list` | List the model family, sizes, and which are cached. |
| `models pull <MODEL>` | Download a model into the cache. |
| `models remove <MODEL>` | Remove a cached model. |
| `models path` | Print the model cache directory. |

```sh
scrybe models list
scrybe models pull large-v3-turbo
scrybe models path     # → ~/.cache/huggingface/hub
```

The cache honors `HF_HOME`. Downloads are resumable and verified against a pinned SHA-256; a corrupt cache entry is re-fetched once, or rejected under `--offline`.

</details>

## Output formats

`--format` accepts any comma-separated combination; `--json` overrides it. Outputs land beside each input (`talk.mp3` → `talk.srt`) or in `--out-dir`. An up-to-date output is skipped unless `--force`, and two inputs that would write the same file stop the run rather than overwrite.

| Format | Extension | Contents |
| --- | --- | --- |
| `txt` | `.txt` | One segment per line. |
| `srt` | `.srt` | SubRip cues with `HH:MM:SS,mmm` timing. |
| `vtt` | `.vtt` | WebVTT cues with `HH:MM:SS.mmm` timing. |
| `tsv` | `.tsv` | `start`, `end` (milliseconds), `text` columns. |
| `csv` | `.csv` | `start`, `end` (milliseconds), `text` — RFC 4180 quoted. |
| `json` | `.json` | Stable versioned schema — model, language, duration, segments. |

Subtitle timestamps are sanitized: never negative, never overlapping. JSON carries a `schema_version` so downstream tooling can pin it.

<details>
<summary>JSON schema</summary>

<br>

```json
{
  "schema_version": 1,
  "model": "large-v3-turbo",
  "language": "en",
  "duration": 12.84,
  "segments": [
    {
      "start": 0.0,
      "end": 2.4,
      "text": "the quick brown fox",
      "words": [
        { "start": 0.0, "end": 0.5, "text": "the" },
        { "start": 0.5, "end": 1.1, "text": "quick" }
      ]
    }
  ]
}
```

Timestamps are in seconds. Each segment carries a `words` array of per-word timing, emitted only with JSON output (the other formats carry segment-level timing only). It is additive and optional — absent on a word-less segment — so `schema_version` stays `1`.

</details>

## Exit codes

Stable across releases — only ever added, never renumbered.

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
| `15` | model load failed (corrupt or incompatible ggml) |
| `16` | transcription failed (compute or decode failure) |
| `20` | partial batch failure, or interrupted before completion |

## Limitations

- **Per-file length ceiling** — decode streams to 16 kHz mono, so the source size is unbounded, but the resident output caps at ~4.6 hours per file (exit `10` beyond that). Split marathon recordings.
- **HE-AAC/SBR** — the built-in decoder rejects it rather than mis-decode. Use `--decoder ffmpeg` or re-encode.
- **No speaker diarization** — scrybe transcribes; it does not label who spoke. Planned for v2, alongside an alternative inference engine.
- **GPU backends build from source** — Metal ships in Apple Silicon prebuilts; `cuda` and `vulkan` are opt-in cargo features built locally.

## Compared to alternatives

| Feature | `openai-whisper` | `whisper.cpp` (cli) | `faster-whisper` | `WhisperX` | **`scrybe`** |
| --- | :---: | :---: | :---: | :---: | :---: |
| Runtime | Python + PyTorch | C/C++ | Python | Python | Rust |
| No Python required | no | yes | no | no | yes |
| Single self-contained binary | no | yes (after build) | no | no | yes |
| Native multi-codec decode, no system `ffmpeg` | no | no (16 kHz WAV / ffmpeg) | no | no | yes |
| Apple Silicon GPU (Metal) | no (CPU/CUDA) | yes | no (CPU/CUDA) | no (CUDA) | yes |
| Folder/batch with progress + summary | no | no | no | no | yes |
| Output txt/srt/vtt/json/tsv/csv | yes (no csv) | yes (no tsv) | via wrapper | yes (no tsv/csv) | yes |
| Zero-config model + concurrency | no | no | no | no | yes |
| Stable exit-code contract | no | no | no | no | yes |
| Word-level timestamps | yes | yes | yes | yes (alignment) | yes (JSON) |
| Speaker diarization | no | no | no | yes | not yet (v2) |

scrybe's niche is a single self-contained binary that decodes any common codec and batch-transcribes offline, with Metal acceleration, no Python environment and no system `ffmpeg`. The Python tools — [`openai-whisper`](https://github.com/openai/whisper), [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), and [`WhisperX`](https://github.com/m-bain/whisperX) — are richer (word-level timestamps, and diarization in WhisperX) but require a Python environment, a system `ffmpeg`, and usually CUDA. [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) is the engine scrybe embeds; its own CLI expects pre-converted 16 kHz WAV (or an ffmpeg-enabled build), and leaves model selection, codec decode, batch UX, and output formatting to the caller. scrybe adds those on top, including word-level timestamps in its JSON. For speaker diarization today, reach for WhisperX; scrybe plans it for v2.

## Contributing

Bug reports and PRs welcome.

- Open an issue before submitting non-trivial PRs.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/).
- Run `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test` before pushing.
- Target the `main` branch.

## License

[MIT](LICENSE.md)
