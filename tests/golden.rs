//! Golden-transcript test — the transcription-quality contract.
//!
//! Transcribes a committed speech clip with the tiny model on the CPU backend
//! and checks the result against a reference within a word-error-rate tolerance
//! (float output is backend/quant-dependent, so an exact match is wrong). Skips
//! cleanly when the tiny model is not cached on a developer machine; under CI
//! (`SCRYBE_REQUIRE_MODEL`, where the model is pre-fetched) a missing model is a
//! hard failure, so a green run can never mean "exercised nothing".
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use scrybe::audio;
use scrybe::cli::Decoder;
use scrybe::engine::{Engine, TranscribeOptions};

mod common;
use common::require_model_or_skip;

const REFERENCE: &str = "the quick brown fox jumps over the lazy dog";
const WER_TOLERANCE: f64 = 0.34;

#[test]
fn english_clip_within_wer_tolerance() {
    let Some(model_path) = require_model_or_skip() else {
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
        word_timestamps: false,
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

#[test]
fn vad_floor_transcribes_through_the_engine() {
    // The mandated VAD floor is always on in production (main wires
    // `ensure_vad()` into `Engine::load`). The WER test above loads with VAD off,
    // so this is the only coverage of the engine's VAD arm end-to-end.
    let Some(model_path) = require_model_or_skip() else {
        return;
    };

    let vad_path = scrybe::model::ensure_vad().expect("bundled VAD materializes");
    let engine = Engine::load(&model_path, Some(&vad_path)).expect("load tiny model with VAD");
    let pcm = audio::load_audio(
        Path::new("tests/fixtures/speech/en.wav"),
        Decoder::Symphonia,
    )
    .expect("decode en.wav");
    let options = TranscribeOptions {
        language: Some("en".to_owned()),
        translate: false,
        threads: 4,
        word_timestamps: false,
    };
    let transcript = engine
        .transcribe(&pcm.samples, &options, |_| {})
        .expect("transcribe with VAD enabled");
    let hypothesis = transcript
        .segments
        .iter()
        .map(|s| s.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        hypothesis.contains("fox") || hypothesis.contains("dog"),
        "VAD-enabled transcript lost its anchor words; got: {hypothesis:?}"
    );
}

#[test]
fn translate_task_changes_output_through_the_engine() {
    // The only end-to-end coverage of `--task translate`. Transcribing French keeps
    // French; translating it forces English output — so the two transcripts must
    // differ. A dropped or inverted `set_translate` makes them identical and fails
    // here. Differential (not an exact match) so it does not depend on tiny's weak
    // translation quality.
    let Some(model_path) = require_model_or_skip() else {
        return;
    };

    let engine = Engine::load(&model_path, None).expect("load tiny model");
    let pcm = audio::load_audio(
        Path::new("tests/fixtures/speech/fr.wav"),
        Decoder::Symphonia,
    )
    .expect("decode fr.wav");
    let run = |translate: bool| {
        let options = TranscribeOptions {
            language: Some("fr".to_owned()),
            translate,
            threads: 4,
            word_timestamps: false,
        };
        engine
            .transcribe(&pcm.samples, &options, |_| {})
            .expect("transcribe")
            .segments
            .iter()
            .map(|s| s.text.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned()
    };

    let transcribed = run(false);
    let translated = run(true);
    assert!(!translated.is_empty(), "translation produced no text");
    assert_ne!(
        transcribed, translated,
        "set_translate(true) must change the output (transcribe vs translate)"
    );
}

#[test]
fn auto_detects_and_reports_french() {
    // A non-English clip with no --lang is auto-detected and reported.
    // `language: None` routes through full_lang_id_from_state + get_lang_str (the
    // detect arm), not the echo arm an explicit --lang would take.
    let Some(model_path) = require_model_or_skip() else {
        return;
    };
    let engine = Engine::load(&model_path, None).expect("load tiny model");
    let pcm = audio::load_audio(
        Path::new("tests/fixtures/speech/fr.wav"),
        Decoder::Symphonia,
    )
    .expect("decode fr.wav");
    let options = TranscribeOptions {
        language: None,
        translate: false,
        threads: 4,
        word_timestamps: false,
    };
    let transcript = engine
        .transcribe(&pcm.samples, &options, |_| {})
        .expect("transcribe with auto-detect");
    assert_eq!(
        transcript.language, "fr",
        "French clip must auto-detect as fr"
    );
    assert!(
        !transcript.segments.is_empty(),
        "auto-detect produced no segments"
    );
}

#[test]
fn uppercase_lang_is_normalized_not_silently_collapsed() {
    // The capability gate accepts `--lang EN` case-insensitively; whisper.cpp matches
    // case-sensitively. Without normalization an uppercase code yields a near-empty
    // transcript and a mis-reported language. Assert the engine lowercases it: full
    // transcript preserved, language reported as `en`.
    let Some(model_path) = require_model_or_skip() else {
        return;
    };
    let engine = Engine::load(&model_path, None).expect("load tiny model");
    let pcm = audio::load_audio(
        Path::new("tests/fixtures/speech/en.wav"),
        Decoder::Symphonia,
    )
    .expect("decode en.wav");
    let transcript = engine
        .transcribe(
            &pcm.samples,
            &TranscribeOptions {
                language: Some("EN".to_owned()),
                translate: false,
                threads: 4,
                word_timestamps: false,
            },
            |_| {},
        )
        .expect("transcribe");
    assert_eq!(
        transcript.language, "en",
        "uppercase --lang must normalize to en"
    );
    let hypothesis = transcript
        .segments
        .iter()
        .map(|s| s.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        hypothesis.contains("fox") || hypothesis.contains("dog"),
        "uppercase --lang collapsed the transcript: {hypothesis:?}"
    );
}

#[test]
fn long_trailing_silence_does_not_hallucinate_or_loop() {
    // The headline correctness floor over a long silence gap. The catastrophic
    // failure the floor must prevent is a repetition LOOP / the gap filling with
    // hallucinated text; whisper.cpp can still emit a rare stray token over silence
    // (an inherent limit VAD + the no-speech gate reduce but don't fully erase), so
    // assert the achievable guarantee: no loop and the gap stays near-empty — not a
    // literal zero. Build speech + ~30 s of silence at runtime (no multi-MB fixture)
    // through the production path (VAD on, as main wires it).
    let Some(model_path) = require_model_or_skip() else {
        return;
    };
    let vad_path = scrybe::model::ensure_vad().expect("bundled VAD materializes");
    let engine = Engine::load(&model_path, Some(&vad_path)).expect("load tiny model with VAD");
    let speech = audio::load_audio(
        Path::new("tests/fixtures/speech/en.wav"),
        Decoder::Symphonia,
    )
    .expect("decode en.wav");
    let silence = audio::load_audio(
        Path::new("tests/fixtures/speech/silence.wav"),
        Decoder::Symphonia,
    )
    .expect("decode silence.wav");
    let speech_secs = speech.samples.len() as f64 / 16_000.0;
    let reps = (30.0 / silence.duration_secs()).ceil() as usize;
    let mut samples = speech.samples;
    for _ in 0..reps {
        samples.extend_from_slice(&silence.samples); // ~30 s of real silence
    }

    let transcript = engine
        .transcribe(
            &samples,
            &TranscribeOptions {
                language: Some("en".to_owned()),
                translate: false,
                threads: 4,
                word_timestamps: false,
            },
            |_| {},
        )
        .expect("transcribe");

    // The ~30 s gap stays near-empty: a stray token may slip through, but the floor
    // must keep the silence from filling with text (a loop would emit dozens here).
    let in_silence = transcript
        .segments
        .iter()
        .filter(|s| s.start >= speech_secs + 1.0)
        .count();
    assert!(
        in_silence <= 2,
        "silence over-hallucinated: {in_silence} segments in the gap"
    );
    // No repetition loop: no transcript text repeats more than twice (a loop would
    // repeat one phrase many times across the gap).
    let mut counts = std::collections::HashMap::new();
    for segment in &transcript.segments {
        *counts.entry(segment.text.trim()).or_insert(0_usize) += 1;
    }
    assert!(
        counts.values().all(|&n| n <= 2),
        "repetition loop detected: {counts:?}"
    );
}

#[test]
fn empty_pcm_surfaces_as_transcription_failed_exit_16() {
    // The only exit code whose production path was asserted only synthetically.
    // Empty PCM is the deterministic way to force a real `full()` fault: against the
    // pinned whisper-rs it returns NoSamples → run_error → TranscriptionFailed (16),
    // so a future change re-routing runtime faults to 13/15 would fail here.
    let Some(model_path) = require_model_or_skip() else {
        return;
    };
    let engine = Engine::load(&model_path, None).expect("load tiny model");
    let options = TranscribeOptions {
        language: Some("en".to_owned()),
        translate: false,
        threads: 4,
        word_timestamps: false,
    };
    let exit = engine
        .transcribe(&[], &options, |_| {})
        .err()
        .map(|e| e.exit_code());
    assert_eq!(
        exit,
        Some(16),
        "empty PCM must surface as TranscriptionFailed"
    );
}

#[test]
fn word_timestamps_populate_aligned_words() {
    // The only end-to-end coverage of the token→word grouping. With word_timestamps
    // on, segments carry per-word timing; each word is well-formed (end >= start) with
    // non-empty text, and the joined words recover the anchor content. word_timestamps
    // off (asserted by the other tests via empty `words`) keeps the cost off non-JSON.
    let Some(model_path) = require_model_or_skip() else {
        return;
    };
    let engine = Engine::load(&model_path, None).expect("load tiny model");
    let pcm = audio::load_audio(
        Path::new("tests/fixtures/speech/en.wav"),
        Decoder::Symphonia,
    )
    .expect("decode en.wav");
    let transcript = engine
        .transcribe(
            &pcm.samples,
            &TranscribeOptions {
                language: Some("en".to_owned()),
                translate: false,
                threads: 4,
                word_timestamps: true,
            },
            |_| {},
        )
        .expect("transcribe");
    let words: Vec<_> = transcript.segments.iter().flat_map(|s| &s.words).collect();
    assert!(!words.is_empty(), "word timestamps must populate words");
    for word in &words {
        assert!(word.end >= word.start, "word end before start: {word:?}");
        assert!(!word.text.trim().is_empty(), "word text must be non-empty");
    }
    let joined = words
        .iter()
        .map(|w| w.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        joined.contains("fox") || joined.contains("dog"),
        "words lost the anchor content: {joined:?}"
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
