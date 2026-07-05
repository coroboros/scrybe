//! Internals behind the `scrybe` binary, exposed for its test suite — not a stable
//! API. Items are `pub` only so integration tests can reach them; some carry
//! process-global or destructive effects (`color::init`, `model::evict`). Treat as
//! unstable, exempt from semver.

pub mod audio;
pub mod batch;
pub mod cli;
pub mod color;
pub mod diarize;
pub mod engine;
pub mod error;
pub mod model;
pub mod output;
pub mod skills;
