# Changelog

## v0.1.4 - 10/06/2026

### Configuration
- clean up comments (#6)
- drop the redundant CARGO_REGISTRY_TOKEN secret [skip ci]


## v0.1.3 - 09/06/2026

### Refactor
- production-grade cleanup pass (comments, I/O exit code, dead type)
- single-source the VAD filename, drop WHAT-narration comments

### Documentation
- `changelog` — consolidate the 0.1.x notes under 0.1.2 (#4)


## v0.1.2 - 09/06/2026

### Features
- CLI surface — `scrybe <paths>` plus `models list/pull/remove/path`, with `--model --lang --task --format --out-dir --jobs --threads --recursive --force --dry-run --decoder --no-color --json --offline`
- Audio ingest — discover files/folders, decode mp3/wav/flac/ogg/m4a (AAC-LC, ALAC) with symphonia, resample to 16 kHz mono via rubato; HE-AAC/SBR fails loud (exit `10`) with the `--decoder ffmpeg` escape
- Model cache — registry of six whisper.cpp ggml models with pinned SHA-256, resumable hf-hub downloads, `--offline`, and an 8 GB-aware memory guard
- Transcription engine — whisper.cpp via whisper-rs, CPU by default with `metal`/`cuda`/`vulkan` cargo features; `condition_on_previous_text` off, quality-gating thresholds, no-speech filtering, language auto-detect, timestamped segments
- Parallel batch — bounded-pipeline decode feeding serial inference, live progress, colored run summary (×RT, language, outputs), skip up-to-date outputs unless `--force`, graceful Ctrl-C, partial-batch resilience (exit `20`)
- Output writers — `txt`/`srt`/`vtt`/`json`/`tsv`/`csv`, sidecar or `--out-dir`, a stable versioned JSON schema, and single-file `--json` to stdout
- Void-tinted color layer — honors `NO_COLOR`, `CLICOLOR_FORCE`, `--no-color`, and non-TTY auto-strip
- Structured error taxonomy — one actionable line per failure with stable exit codes (`1`, `10`–`16`, `20`)

### Configuration
- Rust crate (lib + bin), `rust-toolchain.toml` pinned to `1.96`, `rustfmt.toml`, clippy lints denying `unwrap`/`expect`/`panic`, MIT `LICENSE.md`
- CI — fmt, clippy (`-D warnings`), and CPU-backend tests on Linux, macOS, and Windows with the golden-transcript model cached, a Metal compile-only smoke, and a `cargo-deny` supply-chain gate
- Release automation config — `cargo-dist` targets/installers (Homebrew, npm, shell, PowerShell), released through the shared `coroboros/ci` pipeline
