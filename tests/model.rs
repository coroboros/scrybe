//! Model-management acceptance tests — the WS-3 contract (network-free subset).
//!
//! Covered here, no network: the memory guard, translation gating, the offline
//! cache-hit/miss/corrupt branches, the SHA gate, and the re-download retry *state
//! machine* (`fetch_with_retry`, unit-tested in `model`). What stays manual against
//! the live hub — because an online `get` revalidates over the network: the first
//! download + progress bar (AC1's transfer), resumable transfer (AC2, opaque hf-hub
//! internals), and the live re-download leg of corrupt-cache recovery (AC5).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use predicates::prelude::*;

mod common;
use common::scrybe;

const WAV: &str = "tests/fixtures/audio/tone.wav";

#[test]
fn memory_guard_refuses_oversized_run() {
    // The guard scales with per-job decode buffers (1 GiB each). 1200 jobs is
    // ~1.2 TiB of buffers alone — past 85% headroom on any real runner (even a
    // 512 GiB box), so this drives exit 12 without depending on the host's RAM.
    scrybe()
        .args(["--model", "large-v3", "--jobs", "1200", WAV])
        .assert()
        .failure()
        .code(12)
        .stderr(predicate::str::contains("not enough memory"));
}

#[test]
fn zero_config_oversized_jobs_refused_with_hint() {
    // No --model (auto-resolve) + an absurd job count drives the resolve_model →
    // guard_memory wiring on the zero-config path and pins the actionable OOM hint
    // reaching stderr. Host-independent: 1200 jobs is ~1.2 TiB of decode buffers, so
    // even the smallest model can't fit and the "no model fits" branch fires.
    scrybe()
        .args(["--jobs", "1200", WAV])
        .assert()
        .failure()
        .code(12)
        .stderr(predicate::str::contains("no model fits at 1200"));
}

#[test]
fn json_single_file_is_still_memory_guarded() {
    // The single-file --json branch returns before the batch banner, but resolve →
    // guard runs ahead of it, so an oversized --jobs must still exit 12 on the json
    // fast path — the early return never bypasses the guard. Network-free: the guard
    // fails before any model load.
    scrybe()
        .args(["--model", "large-v3", "--jobs", "1200", "--json", WAV])
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
fn english_only_model_accepts_en_case_insensitively() {
    // The `en` exception is case-insensitive; --dry-run stops before any download.
    scrybe()
        .args([
            "--model",
            "distil-large-v3.5",
            "--lang",
            "EN",
            "--dry-run",
            WAV,
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("English-only").not());
}

#[test]
fn english_only_model_accepts_explicit_auto() {
    // `--lang auto` degenerates to detection (English for an English-only model);
    // it must not be rejected like a foreign language. --dry-run stays network-free.
    scrybe()
        .args([
            "--model",
            "distil-large-v3.5",
            "--lang",
            "auto",
            "--dry-run",
            WAV,
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("English-only").not());
}

#[test]
fn multilingual_model_accepts_non_english_lang() {
    // tiny is multilingual; --lang fr must pass the capability gate. --dry-run
    // stops before any download, so this stays network-free.
    scrybe()
        .args(["--model", "tiny", "--lang", "fr", "--dry-run", WAV])
        .assert()
        .success()
        .stderr(predicate::str::contains("English-only").not());
}

#[test]
fn offline_corrupt_cache_exits_11() {
    // Seed a cache entry whose bytes do not match the pinned SHA, then run
    // --offline. The integrity gate must reject it (exit 11, "checksum mismatch")
    // without any network call — the WS-3 "corrupted cache is not used" contract.
    let hf = tempfile::tempdir().unwrap();
    let repo = hf.path().join("hub/models--ggerganov--whisper.cpp");
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::create_dir_all(repo.join("snapshots/deadbeef")).unwrap();
    std::fs::write(repo.join("refs/main"), b"deadbeef").unwrap();
    std::fs::write(
        repo.join("snapshots/deadbeef/ggml-tiny.bin"),
        b"corrupt, not a real ggml model",
    )
    .unwrap();
    scrybe()
        .env("HF_HOME", hf.path())
        .args(["--offline", "--model", "tiny", WAV])
        .assert()
        .failure()
        .code(11)
        .stderr(predicate::str::contains("checksum mismatch"));
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
