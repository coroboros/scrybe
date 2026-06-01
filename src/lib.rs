//! scrybe library: the reusable pieces behind the CLI — audio ingest, the model
//! registry and cache, and the whisper.cpp engine. The binary in `main.rs` wires
//! these together; tests drive them directly.
//!
//! This surface exists for the `scrybe` binary and its test suite, not as a stable
//! public API: items are `pub` so the integration tests can reach them, and some
//! carry process-global or destructive effects (`color::init`, `model::evict`).
//! Treat it as unstable and exempt from semver until a curated facade is defined.

pub mod audio;
pub mod batch;
pub mod cli;
pub mod color;
pub mod engine;
pub mod error;
pub mod model;
pub mod output;
