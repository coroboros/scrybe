//! Model-management acceptance tests — the WS-3 contract (network-free subset).
//!
//! Download/SHA/cache behavior is exercised manually against the live hub; here
//! we pin the deterministic surface: the memory guard, translation gating, and
//! the `models` subcommands that need no network.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use predicates::prelude::*;

mod common;
use common::scrybe;

const WAV: &str = "tests/fixtures/audio/tone.wav";

#[test]
fn memory_guard_refuses_oversized_run() {
    // The guard scales with per-job decode buffers (~384 MiB each). 1200 jobs is
    // ~450 GiB of buffers alone — past 85% headroom on any real runner (even a
    // 512 GiB box), so this drives exit 12 without depending on the host's RAM.
    scrybe()
        .args(["--model", "large-v3", "--jobs", "1200", WAV])
        .assert()
        .failure()
        .code(12)
        .stderr(predicate::str::contains("not enough memory"));
}

#[test]
fn turbo_cannot_translate() {
    scrybe()
        .args(["--model", "large-v3-turbo", "--task", "translate", WAV])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("cannot translate"));
}

#[test]
fn models_list_shows_sizes_and_default() {
    scrybe()
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("large-v3-turbo"))
        .stdout(predicate::str::contains("GB"))
        .stdout(predicate::str::contains("(default)"));
}

#[test]
fn models_path_points_at_hub_cache() {
    scrybe()
        .args(["models", "path"])
        .assert()
        .success()
        .stdout(predicate::str::contains("huggingface"));
}

#[test]
fn english_only_model_rejects_foreign_lang() {
    // The capability gate runs before any download, so this is network-free.
    scrybe()
        .args(["--model", "distil-large-v3.5", "--lang", "fr", WAV])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("English-only"));
}

#[test]
fn offline_without_cache_exits_11() {
    // An empty HF_HOME forces a cache miss; --offline must error with exit 11 and
    // make no network call (the offline branch never touches the API).
    let empty_cache = tempfile::tempdir().unwrap();
    scrybe()
        .env("HF_HOME", empty_cache.path())
        .args(["--offline", "--model", "tiny", WAV])
        .assert()
        .failure()
        .code(11)
        .stderr(predicate::str::contains("--offline"));
}
