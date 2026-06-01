# Changelog

## v0.1.0 - 01/06/2026

### Features
- CLI surface — `scrybe <paths>` plus `models list/pull/remove/path`, with `--model --lang --task --format --out-dir --jobs --threads --recursive --force --dry-run --decoder --no-color --json --offline`
- Audio ingest — discover files/folders, decode mp3/wav/flac/ogg/m4a (AAC-LC, ALAC) with symphonia, resample to 16 kHz mono via rubato; HE-AAC/SBR fails loud (exit `10`) with the `--decoder ffmpeg` escape
- Model cache — registry of six whisper.cpp ggml models with pinned SHA-256, resumable hf-hub downloads, `--offline`, and an 8 GB-aware memory guard
- Transcription engine — whisper.cpp via whisper-rs, CPU by default with `metal`/`cuda`/`vulkan` cargo features; `condition_on_previous_text` off, quality-gating thresholds, no-speech filtering, language auto-detect, timestamped segments
- Parallel batch — bounded-pipeline decode feeding serial inference, live progress, colored run summary (×RT, language, outputs), skip up-to-date outputs unless `--force`, graceful Ctrl-C, partial-batch resilience (exit `20`)
- Output writers — `txt`/`srt`/`vtt`/`json`/`tsv`, sidecar or `--out-dir`, a stable versioned JSON schema, and single-file `--json` to stdout
- Void-tinted color layer — honors `NO_COLOR`, `CLICOLOR_FORCE`, `--no-color`, and non-TTY auto-strip
- Structured error taxonomy — one actionable line per failure with stable exit codes (`1`, `10`–`14`, `20`)

### Configuration
- Rust crate (lib + bin), `rust-toolchain.toml` pinned to `1.96`, `rustfmt.toml`, clippy lints denying `unwrap`/`expect`/`panic`, MIT `LICENSE.md`
- CI — fmt, clippy (`-D warnings`), and CPU-backend tests on macOS + Linux with the golden-transcript model cached
- Release automation config — `cargo-dist` targets/installers (Homebrew, npm, shell, PowerShell) and `release-plz`
