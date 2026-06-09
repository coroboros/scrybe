//! Decode-pipeline acceptance tests — the decode contract.
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
use common::require_ffmpeg_or_skip;

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

/// A minimal 16 kHz mono s16 WAV header declaring `data_size` PCM bytes, with no
/// body. symphonia reads `num_frames` from the header before decoding, so a crafted
/// size drives `decode_file`'s ceiling / empty-audio branches without a real fixture.
fn wav_header(data_size: u32) -> Vec<u8> {
    let (sample_rate, channels, bits) = (16_000u32, 1u16, 16u16);
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * u32::from(block_align);
    let mut h = Vec::new();
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data_size).to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&byte_rate.to_le_bytes());
    h.extend_from_slice(&block_align.to_le_bytes());
    h.extend_from_slice(&bits.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_size.to_le_bytes());
    h
}

#[test]
fn crafted_declared_length_does_not_preallocate() {
    // A header declaring ~350M frames with no data body. The streaming decoder never
    // pre-allocates from the declared length (the source is resampled packet by packet,
    // never fully resident), so a crafted huge length cannot force a multi-GB alloc; the
    // header-only file simply yields no audio and fails loud with exit 10 — promptly,
    // not after an OOM.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.wav");
    std::fs::write(&path, wav_header(700_000_000)).unwrap();
    let err = decode(path.to_str().unwrap(), Decoder::Symphonia).unwrap_err();
    assert_eq!(err.exit_code(), 10);
    assert!(
        err.to_string().contains("no decodable audio") || err.to_string().contains("no audio"),
        "got: {err}"
    );
}

#[test]
fn empty_audio_track_fails_loud() {
    // Valid container, zero-length data → decode_file's empty-audio branch.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.wav");
    std::fs::write(&path, wav_header(0)).unwrap();
    let err = decode(path.to_str().unwrap(), Decoder::Symphonia).unwrap_err();
    assert_eq!(err.exit_code(), 10);
    assert!(
        err.to_string().contains("no decodable audio") || err.to_string().contains("no audio"),
        "got: {err}"
    );
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
    if !require_ffmpeg_or_skip() {
        return;
    }
    let pcm = decode("tests/fixtures/aac/he-aac.m4a", Decoder::Ffmpeg).expect("ffmpeg he-aac");
    assert!(!pcm.samples.is_empty());
    // The ffmpeg path emits mono 16 kHz directly, so provenance reflects that shape —
    // the only provenance assertion on the ffmpeg branch (symphonia is covered above).
    assert_eq!(pcm.source_sample_rate, 16_000, "ffmpeg path outputs 16 kHz");
    assert_eq!(pcm.source_channels, 1, "ffmpeg path downmixes to mono");
}

#[test]
fn ffmpeg_rejects_non_audio_with_exit_10() {
    if !require_ffmpeg_or_skip() {
        return;
    }
    // A non-audio file drives ffmpeg's failure branch, mapped to exit 10.
    let err = decode("tests/fixtures/audio/notes.txt", Decoder::Ffmpeg).unwrap_err();
    assert_eq!(err.exit_code(), 10);
    assert!(err.to_string().contains("ffmpeg failed"), "got: {err}");
}

#[test]
fn ffmpeg_decodes_leading_dash_filename() {
    if !require_ffmpeg_or_skip() {
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
