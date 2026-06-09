# scrybe

A fast, offline Whisper transcription CLI — one file or a whole folder, from the
terminal. Rust, whisper.cpp via whisper-rs (Metal on Apple Silicon, CPU
everywhere), pure-Rust audio decode.

## Canonical rules

Follows the Coroboros engineering global rules. Repo-specific divergences are
stated inline in `## Rules` below.

> **Public-repo hygiene:** this ships into a public community repo. Never
> reference private rule paths, local machine paths, or internal tooling here —
> keep it generic.

## Tech Stack
- Rust, edition 2024, toolchain pinned in `rust-toolchain.toml`
- `clap` (derive) for the command surface; `anstream` + `anstyle` for color
- `cargo fmt` (rustfmt) and `cargo clippy` for format/lint — the Biome analog
- `assert_cmd` + `predicates` for CLI tests
- `whisper-rs` (engine), `symphonia` + `rubato` (decode), `hf-hub` (model cache), `rayon` + `indicatif` (batch UX), `sysinfo` (memory guard)

## Commands
- `cargo build` — debug build
- `cargo build --release` — optimized binary (`strip`, thin LTO)
- `cargo test` — unit + CLI acceptance tests
- `cargo clippy --all-targets` — lint; the no-panic lints are deny-level
- `cargo fmt` / `cargo fmt --check` — format / verify

## Important Files
- `src/main.rs` — entry point; parse, init color, dispatch, map errors to exit codes
- `src/cli.rs` — the `clap` surface, `ValueEnum`s, `DEFAULT_MODEL`
- `src/error.rs` — `ScrybeError` taxonomy and the stable exit-code map
- `src/color.rs` — the void palette and `--no-color` / `NO_COLOR` handling
- `src/audio/` — input discovery, streaming decode, and 16 kHz mono resample
- `src/model/mod.rs` — the model registry, cache, SHA-256 verification, and memory guard
- `src/engine/mod.rs` — the whisper.cpp engine, quality gating, and the always-on VAD floor
- `src/batch/mod.rs` — the parallel-decode → serial-inference pipeline and run summary
- `src/output/mod.rs` — transcript serialization (`txt`/`srt`/`vtt`/`json`/`tsv`/`csv`) and collision checks
- `src/skills/mod.rs` — the bundled agent-skill registry; embeds `skills/scrybe/SKILL.md`
- `skills/scrybe/SKILL.md` — the agent skill, installable via `npx skills add coroboros/scrybe`
- `tests/cli.rs` — acceptance tests for the CLI contract
- `Cargo.toml` — package metadata, dependency pins, crate lints, and the cargo-dist binary-distribution table
- `.github/workflows/ci.yml` — thin caller of the shared `coroboros/ci` rust-packages pipeline; `.github/workflows/metal-smoke.yml` is the supplemental Metal compile-smoke
- `ci/setup.sh` · `ci/test.env` · `ci/test-setup.sh` — the pipeline hooks: native build deps (CMake/ffmpeg), the `SCRYBE_REQUIRE_*` test env, and the tiny-model pre-fetch

## Rules
- **No panics on user input.** Every user-facing failure routes through `ScrybeError`; `unwrap`/`expect`/`panic` are deny-level clippy lints. Validate at boundaries, return a coded error.
- **Exit codes are a contract.** The `error.rs` code map is stable across releases — never change a code, only add. Argument errors are clap's (exit `2`).
- **Color always paints; the stream decides.** Render via `color::paint` and let `anstream` strip ANSI when output is not a terminal or `NO_COLOR` is set. Honor `--no-color`.
- **Single source for shared values.** The default model is `cli::DEFAULT_MODEL`, referenced by both the clap default and `models list`.
- **One skill source.** The agent skill lives once in `skills/scrybe/SKILL.md`, embedded via `include_str!` for `skills get` and published for `npx skills add`; never duplicate its content.
- Run `cargo fmt && cargo clippy --all-targets && cargo test` before every commit.

## CI overrides
All other Coroboros git conventions apply. Divergences:
- **CI** — consumes the shared `coroboros/ci` rust-packages pipeline; `.github/workflows/ci.yml` is a thin caller of `rust-packages.yml@v0`. The whisper.cpp build deps reach it through the pipeline hooks: `ci/setup.sh` (CMake; ffmpeg on test legs, gated by `CARGO_DIST_TARGET`) and `ci/test.env` + `ci/test-setup.sh` (tiny-model pre-fetch and the `SCRYBE_REQUIRE_*` fail-loud gates). The Metal compile-smoke rides a supplemental `.github/workflows/metal-smoke.yml`. The cargo-deny policy, versioning, CHANGELOG, and release are imposed centrally — no local `deny.toml` or `release-plz.toml`.
- **Branch model** — main-only: feature branch → PR → squash-merge → tag.
