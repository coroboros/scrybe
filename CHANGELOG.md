# Changelog

## v0.1.0 - 01/06/2026

### Features
- CLI surface — `scrybe <paths>` plus the `models list` subcommand, with `--model --lang --task --format --out-dir --jobs --threads --force --dry-run --decoder --no-color --json`
- Void-tinted color layer — honors `NO_COLOR`, `CLICOLOR_FORCE`, the `--no-color` flag, and non-TTY auto-strip via `anstream`
- Structured error taxonomy — `ScrybeError` renders one actionable line per failure with stable exit codes (`10`–`14`, `20`)

### Configuration
- Rust crate scaffold — `rust-toolchain.toml` pinned to `1.96`, `rustfmt.toml`, clippy lints denying `unwrap`/`expect`/`panic`, MIT `LICENSE.md`
