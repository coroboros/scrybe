//! scrybe library: the reusable pieces behind the CLI — audio ingest, the model
//! registry and cache, and the whisper.cpp engine. The binary in `main.rs` wires
//! these together; tests drive them directly.

pub mod audio;
pub mod cli;
pub mod color;
pub mod engine;
pub mod error;
pub mod model;
