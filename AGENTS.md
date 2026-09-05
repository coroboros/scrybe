# scrybe

Offline transcription and speaker diarization CLI in Rust, with whisper.cpp and native audio decoding.

## Project constraints

- `README.md` owns the CLI contract; `src/cli.rs` and `src/error.rs` implement it. Preserve exit codes, argument behavior and output formats; add codes without renumbering existing ones.
- Route user-facing failures through `ScrybeError`. Keep the deny-level panic lints; validate external inputs at their boundaries.
- Render colors through `color::paint` and `anstream`, preserving `--no-color` and `NO_COLOR`. `cli::DEFAULT_MODEL` remains the shared default.
- `skills/scrybe/SKILL.md` is the single skill source, embedded by `src/skills/mod.rs`; edit it once for installation and `skills get`.
- Native build and test setup lives in `ci/setup.sh`, `ci/test.env` and `ci/test-setup.sh`. Keep public artifacts free of private paths and infrastructure references.

## Validation

The toolchain is pinned in `rust-toolchain.toml`. Rust or dependency changes require `cargo fmt --check`, `cargo clippy --all-targets` and `cargo test`; documentation-only edits need Markdown and reference checks.

Tests cover transcription, decoder fallback and diarization only when their fixtures are available. For those changes and release verification, use the setup hooks and all three `SCRYBE_REQUIRE_*` flags in `ci/test.env`; report missing fixtures or skipped paths. Model setup can download files. Reuse passing results while the tested inputs remain unchanged.

## Release

Target `main` through a PR and squash-merge the reviewed head. After release approval, tag the merge commit with the next SemVer. `.github/workflows/ci.yml` delegates version updates, changelog, policy checks and publication to the shared Rust pipeline; keep version artifacts and cargo-deny policy centrally owned. Supplemental Metal and ONNX checks cover the native substrates. Distribution targets and hooks are declared in `Cargo.toml`.
