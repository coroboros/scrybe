//! Decode-pipeline acceptance tests — the WS-2 contract.
//!
//! Driven directly against the audio library (no model, no transcription): the
//! canonical formats decode to 16 kHz mono, non-audio is skipped, HE-AAC fails
//! loud with exit 10, AAC-LC decodes, and the ffmpeg fallback handles HE-AAC.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use scrybe::audio::{self, AudioPcm};
use scrybe::cli::Decoder;
use scrybe::error::ScrybeError;

mod common;
use common::ffmpeg_available;

fn decode(path: &str, decoder: Decoder) -> Result<AudioPcm, ScrybeError> {
    audio::load_audio(Path::new(path), decoder)
}

#[test]
fn supported_formats_decode_to_16k_mono() {
    for file in ["tone.wav", "tone.mp3", "tone.flac", "tone.ogg", "tone.m4a"] {
        let pcm = decode(&format!("tests/fixtures/audio/{file}"), Decoder::Symphonia).expect(file);
        // 44.1 kHz stereo source down to 16 kHz mono.
        assert_eq!(pcm.source_sample_rate, 44_100, "{file} sample rate");
        assert_eq!(pcm.source_channels, 2, "{file} channels");
        assert!(!pcm.samples.is_empty(), "{file} produced no samples");
        let secs = pcm.duration_secs();
        assert!((0.45..=0.6).contains(&secs), "{file}: {secs}s");
    }
}

#[test]
fn discovery_skips_non_audio() {
    let inputs = [PathBuf::from("tests/fixtures/audio")];
    let found = audio::discover(&inputs, false);
    assert!(
        found
            .iter()
            .any(|p| p.extension().is_some_and(|e| e == "wav"))
    );
    assert!(
        !found
            .iter()
            .any(|p| p.extension().is_some_and(|e| e == "txt"))
    );
    // Exactly the five audio files in the folder (tone.{wav,mp3,flac,ogg,m4a});
    // notes.txt is skipped. A discovery regression that adds/drops one fails here.
    assert_eq!(found.len(), 5, "discovered: {found:?}");
}

#[test]
fn he_aac_fails_loud_with_exit_10() {
    let err = decode("tests/fixtures/aac/he-aac.m4a", Decoder::Symphonia).unwrap_err();
    assert_eq!(err.exit_code(), 10);
    assert!(err.to_string().contains("HE-AAC"), "got: {err}");
}

#[test]
fn aac_lc_decodes_without_false_positive() {
    let pcm = decode("tests/fixtures/aac/lc-aac.m4a", Decoder::Symphonia).expect("lc-aac");
    assert!(!pcm.samples.is_empty());
}

#[test]
fn ffmpeg_fallback_decodes_he_aac() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let pcm = decode("tests/fixtures/aac/he-aac.m4a", Decoder::Ffmpeg).expect("ffmpeg he-aac");
    assert!(!pcm.samples.is_empty());
}

#[test]
fn ffmpeg_rejects_non_audio_with_exit_10() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    // A non-audio file drives ffmpeg's failure branch, mapped to exit 10.
    let err = decode("tests/fixtures/audio/notes.txt", Decoder::Ffmpeg).unwrap_err();
    assert_eq!(err.exit_code(), 10);
    assert!(err.to_string().contains("ffmpeg failed"), "got: {err}");
}

#[test]
fn ffmpeg_decodes_leading_dash_filename() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    // A filename starting with `-` must not be parsed by ffmpeg as an option — the
    // canonicalize-to-absolute-path guard prevents that. Decode must succeed.
    let dir = tempfile::tempdir().unwrap();
    let dash = dir.path().join("-dash.wav");
    std::fs::copy("tests/fixtures/audio/tone.wav", &dash).unwrap();
    let pcm = decode(dash.to_str().unwrap(), Decoder::Ffmpeg).expect("decode -dash.wav");
    assert!(!pcm.samples.is_empty());
}
