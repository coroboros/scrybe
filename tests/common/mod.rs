//! Shared helpers for the integration test crates. Lives in a subdirectory so
//! Cargo does not compile it as its own test binary.
#![allow(dead_code)] // each test binary uses a subset
#![allow(clippy::unwrap_used, clippy::expect_used)]

use assert_cmd::Command;

/// A binary invocation with ambient color env neutralized, so a test only sees
/// the color signal it sets itself.
pub fn scrybe() -> Command {
    let mut cmd = Command::cargo_bin("scrybe").expect("scrybe binary builds");
    cmd.env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .env_remove("CLICOLOR_FORCE");
    cmd
}

/// Whether the tiny model is cached. Transcription tests skip cleanly when it is
/// absent (CI fetches it once); shared so every gate uses one convention.
pub fn tiny_cached() -> bool {
    scrybe::model::cached_path(scrybe::cli::Model::Tiny).is_some()
}
