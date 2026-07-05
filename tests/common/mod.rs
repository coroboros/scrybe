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

/// The cached tiny-model path, or `None` when it is absent. Transcription tests
/// skip cleanly when missing (CI fetches it once); shared so every gate — bool or
/// path — uses one lookup and one convention.
pub fn tiny_model_path() -> Option<std::path::PathBuf> {
    scrybe::model::cached_path(scrybe::cli::Model::Tiny)
}

/// Whether the tiny model is cached.
pub fn tiny_cached() -> bool {
    tiny_model_path().is_some()
}

/// Whether a usable `ffmpeg` is on PATH; ffmpeg-path tests skip cleanly otherwise.
pub fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// The cached tiny-model path, or `None` to skip — but a hard failure under
/// `SCRYBE_REQUIRE_MODEL` (CI sets it), so a model-gated test can't silently pass while
/// exercising nothing where a model is guaranteed, yet still skips cleanly offline. The
/// skip message lives here once.
pub fn require_model_or_skip() -> Option<std::path::PathBuf> {
    if let Some(path) = tiny_model_path() {
        return Some(path);
    }
    assert!(
        std::env::var_os("SCRYBE_REQUIRE_MODEL").is_none(),
        "tiny model required (SCRYBE_REQUIRE_MODEL is set) but not cached — run `scrybe models pull tiny`",
    );
    eprintln!("skipping: tiny model not cached — run `scrybe models pull tiny`");
    None
}

/// Like [`require_model_or_skip`] for `ffmpeg`: a clean skip when it is absent, but a
/// hard failure under `SCRYBE_REQUIRE_FFMPEG` (CI installs ffmpeg), so the ffmpeg
/// decode path can never silently no-op where it is guaranteed present.
pub fn require_ffmpeg_or_skip() -> bool {
    if ffmpeg_available() {
        return true;
    }
    assert!(
        std::env::var_os("SCRYBE_REQUIRE_FFMPEG").is_none(),
        "ffmpeg required (SCRYBE_REQUIRE_FFMPEG is set) but not on PATH",
    );
    eprintln!("skipping: ffmpeg not on PATH");
    false
}

/// Like [`require_model_or_skip`] for the diarization pair: a clean skip when
/// either model is missing, but a hard failure under `SCRYBE_REQUIRE_DIARIZE`
/// (CI pre-fetches both), so the diarization path can never silently no-op
/// where it is guaranteed present.
pub fn require_diarize_or_skip() -> bool {
    let cached = scrybe::model::diarization_status()
        .iter()
        .all(|(_, _, path)| path.is_some());
    if cached {
        return true;
    }
    assert!(
        std::env::var_os("SCRYBE_REQUIRE_DIARIZE").is_none(),
        "diarization models required (SCRYBE_REQUIRE_DIARIZE is set) but not cached — run `scrybe models pull diarization`",
    );
    eprintln!("skipping: diarization models not cached — run `scrybe models pull diarization`");
    false
}
