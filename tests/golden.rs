//! Golden-transcript test — the WS-7 contract.
//!
//! Transcribes a committed speech clip with the tiny model on the CPU backend
//! and checks the result against a reference within a word-error-rate tolerance
//! (float output is backend/quant-dependent, so an exact match is wrong). Skips
//! cleanly when the tiny model is not cached, so CI fetches it once and offline
//! runs do not fail.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use scrybe::audio;
use scrybe::cli::{Decoder, Model};
use scrybe::engine::{Engine, TranscribeOptions};

const REFERENCE: &str = "the quick brown fox jumps over the lazy dog";
const WER_TOLERANCE: f64 = 0.34;

#[test]
fn english_clip_within_wer_tolerance() {
    let Some(model_path) = scrybe::model::cached_path(Model::Tiny) else {
        eprintln!("skipping: tiny model not cached — run `scrybe models pull tiny`");
        return;
    };

    let engine = Engine::load(&model_path, None).expect("load tiny model");
    let pcm = audio::load_audio(
        Path::new("tests/fixtures/speech/en.wav"),
        Decoder::Symphonia,
    )
    .expect("decode en.wav");
    let options = TranscribeOptions {
        language: Some("en".to_owned()),
        translate: false,
        threads: 4,
    };
    let transcript = engine
        .transcribe(&pcm.samples, &options, |_| {})
        .expect("transcribe");
    let hypothesis = transcript
        .segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let wer = word_error_rate(REFERENCE, &hypothesis);
    assert!(
        wer <= WER_TOLERANCE,
        "WER {wer:.2} too high; got: {hypothesis:?}"
    );
    // A loose WER bar can pass on near-empty output; anchor on a high-signal word
    // so a catastrophic quality regression still fails.
    let normalized = hypothesis.to_lowercase();
    assert!(
        normalized.contains("fox") || normalized.contains("dog"),
        "transcript lost its anchor words; got: {hypothesis:?}"
    );
}

/// Word error rate: word-level edit distance over reference length, after
/// lowercasing and stripping punctuation.
fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize(reference);
    let hypothesis = normalize(hypothesis);
    if reference.is_empty() {
        return f64::from(u8::from(!hypothesis.is_empty()));
    }
    edit_distance(&reference, &hypothesis) as f64 / reference.len() as f64
}

fn normalize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut curr = vec![0usize; hypothesis.len() + 1];
    for (i, r) in reference.iter().enumerate() {
        curr[0] = i + 1;
        for (j, h) in hypothesis.iter().enumerate() {
            let cost = usize::from(r != h);
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[hypothesis.len()]
}

#[test]
fn wer_helper_is_sane() {
    assert_eq!(word_error_rate("a b c", "a b c"), 0.0);
    assert!((word_error_rate("a b c", "a b") - 1.0 / 3.0).abs() < 1e-9);
    assert_eq!(word_error_rate("Hello, World!", "hello world"), 0.0);
}
