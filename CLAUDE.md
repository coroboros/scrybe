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
- Planned: `whisper-rs` (engine), `symphonia` + `rubato` (decode), `hf-hub` (model cache), `rayon` + `indicatif` (batch UX)

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
- `tests/cli.rs` — acceptance tests for the CLI contract
- `Cargo.toml` — package metadata, dependency pins, and crate lints

## Rules
- **No panics on user input.** Every user-facing failure routes through `ScrybeError`; `unwrap`/`expect`/`panic` are deny-level clippy lints. Validate at boundaries, return a coded error.
- **Exit codes are a contract.** The `error.rs` code map is stable across releases — never change a code, only add. Argument errors are clap's (exit `2`).
- **Color always paints; the stream decides.** Render via `color::paint` and let `anstream` strip ANSI when output is not a terminal or `NO_COLOR` is set. Honor `--no-color`.
- **Single source for shared values.** The default model is `cli::DEFAULT_MODEL`, referenced by both the clap default and `models list`.
- Run `cargo fmt && cargo clippy --all-targets && cargo test` before every commit.
- **NEVER** add a runtime dependency without checking it earns its place over `std` and the existing crates.
