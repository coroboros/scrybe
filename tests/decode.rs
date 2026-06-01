//! Decode-pipeline acceptance tests — the WS-2 contract.
//!
//! Exercised through the binary on committed fixtures (the audio modules are
//! crate-internal): the canonical formats decode to 16 kHz mono, non-audio is
//! skipped, HE-AAC fails loud with exit 10, and the ffmpeg fallback decodes it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use assert_cmd::Command;
use predicates::prelude::*;

fn scrybe() -> Command {
    let mut cmd = Command::cargo_bin("scrybe").unwrap();
    cmd.env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .env_remove("CLICOLOR_FORCE");
    cmd
}

#[test]
fn folder_decodes_all_supported_and_skips_non_audio() {
    scrybe()
        .arg("tests/fixtures/audio")
        .assert()
        .success()
        // Every canonical format is decoded to 16 kHz mono.
        .stdout(predicate::str::contains("tone.wav"))
        .stdout(predicate::str::contains("tone.mp3"))
        .stdout(predicate::str::contains("tone.flac"))
        .stdout(predicate::str::contains("tone.ogg"))
        .stdout(predicate::str::contains("tone.m4a"))
        // 44.1 kHz stereo source → 16 kHz mono.
        .stdout(predicate::str::contains("44100 Hz 2 ch → 16 kHz mono"))
        // The non-audio file is never handed to the decoder.
        .stdout(predicate::str::contains("notes.txt").not());
}

#[test]
fn he_aac_fails_loud_with_exit_10() {
    scrybe()
        .arg("tests/fixtures/aac/he-aac.m4a")
        .assert()
        .failure()
        .code(10)
        .stderr(predicate::str::contains("HE-AAC"))
        .stderr(predicate::str::contains("--decoder ffmpeg"));
}

#[test]
fn aac_lc_decodes_without_false_positive() {
    scrybe()
        .arg("tests/fixtures/aac/lc-aac.m4a")
        .assert()
        .success()
        .stdout(predicate::str::contains("16 kHz mono"));
}

#[test]
fn ffmpeg_fallback_decodes_he_aac() {
    if which_ffmpeg().is_none() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    scrybe()
        .args(["--decoder", "ffmpeg", "tests/fixtures/aac/he-aac.m4a"])
        .assert()
        .success()
        .stdout(predicate::str::contains("16 kHz mono"));
}

fn which_ffmpeg() -> Option<()> {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| ())
}
