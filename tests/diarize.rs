//! End-to-end diarization acceptance on committed fixtures. Skips cleanly
//! when the diarization models are not cached; `SCRYBE_REQUIRE_DIARIZE=1`
//! (set in CI) turns the skip into a hard failure so the covered path can
//! never silently no-op.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use scrybe::audio;
use scrybe::cli::Decoder;
use scrybe::diarize::{DiarizeOptions, Diarizer, Turn};
use scrybe::model;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/speech")
        .join(name)
}

/// Both diarization model paths from the local cache, or a clean skip.
fn require_models_or_skip() -> Option<(PathBuf, PathBuf)> {
    let cached = model::ensure_segmentation(true)
        .ok()
        .zip(model::ensure_embedding(true).ok());
    if cached.is_none() {
        assert!(
            std::env::var_os("SCRYBE_REQUIRE_DIARIZE").is_none(),
            "SCRYBE_REQUIRE_DIARIZE is set but the diarization models are not cached"
        );
        eprintln!("skipping: diarization models not cached");
    }
    cached
}

fn diarize_fixture(name: &str, options: &DiarizeOptions) -> Option<Vec<Turn>> {
    let (seg, emb) = require_models_or_skip()?;
    let pcm = audio::load_audio(&fixture(name), Decoder::Symphonia).expect("fixture decodes");
    let mut diarizer = Diarizer::load(&seg, &emb).expect("models load");
    Some(
        diarizer
            .diarize(&pcm.samples, options)
            .expect("diarization runs"),
    )
}

#[test]
fn two_voice_conversation_yields_two_alternating_speakers() {
    let Some(turns) = diarize_fixture("two-speakers.wav", &DiarizeOptions::default()) else {
        return;
    };
    assert!(!turns.is_empty(), "no turns on a 24 s conversation");

    let num_speakers = turns.iter().map(|t| t.speaker).max().unwrap() + 1;
    assert_eq!(
        num_speakers, 2,
        "expected exactly 2 speakers, turns: {turns:?}"
    );

    // Each speaker holds a real share of the conversation.
    for speaker in 0..num_speakers {
        let speech: f64 = turns
            .iter()
            .filter(|t| t.speaker == speaker)
            .map(|t| t.end - t.start)
            .sum();
        assert!(
            speech > 4.0,
            "speaker {speaker} only credited {speech:.2} s"
        );
    }

    // Turns stay inside the file and are well-formed.
    for turn in &turns {
        assert!(turn.start < turn.end, "inverted turn {turn:?}");
        assert!(turn.end < 25.5, "turn past end of audio {turn:?}");
    }

    // The first audible voice keeps id 0 (first-appearance ordering) and the
    // conversation actually alternates at least once.
    assert_eq!(turns[0].speaker, 0);
    assert!(
        turns.windows(2).any(|w| w[0].speaker != w[1].speaker),
        "speakers never alternate: {turns:?}"
    );
}

#[test]
fn pinning_the_speaker_count_is_honored() {
    let options = DiarizeOptions {
        num_speakers: Some(2),
    };
    let Some(turns) = diarize_fixture("two-speakers.wav", &options) else {
        return;
    };
    let num_speakers = turns.iter().map(|t| t.speaker).max().unwrap() + 1;
    assert_eq!(num_speakers, 2);
}

#[test]
fn single_voice_yields_one_speaker() {
    let Some(turns) = diarize_fixture("en.wav", &DiarizeOptions::default()) else {
        return;
    };
    assert!(!turns.is_empty(), "no turns on a speech fixture");
    assert!(
        turns.iter().all(|t| t.speaker == 0),
        "single voice split into several speakers: {turns:?}"
    );
}

#[test]
fn silence_yields_no_turns() {
    let Some(turns) = diarize_fixture("silence.wav", &DiarizeOptions::default()) else {
        return;
    };
    assert!(turns.is_empty(), "phantom speakers on silence: {turns:?}");
}

#[test]
fn matches_the_python_reference_within_one_frame() {
    // tests/reference/diarize_reference.py is an independent implementation
    // of the same pyannote 3.1 algorithm (numpy/scipy/onnxruntime); its
    // committed output pins this port numerically. One segmentation frame
    // (10/589 s ≈ 17 ms) of boundary headroom absorbs ONNX kernel drift
    // across platforms; anything looser would mask real regressions.
    const BOUNDARY_TOLERANCE: f64 = 0.02;

    let Some(turns) = diarize_fixture("two-speakers.wav", &DiarizeOptions::default()) else {
        return;
    };
    let expected: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(fixture("two-speakers.expected.json")).unwrap(),
    )
    .unwrap();

    assert_eq!(
        turns.len(),
        expected.len(),
        "turn count diverges: {turns:?}"
    );
    for (turn, reference) in turns.iter().zip(&expected) {
        assert_eq!(
            turn.speaker,
            reference["speaker"].as_u64().unwrap() as usize,
            "speaker sequence diverges at {turn:?}"
        );
        let start = reference["start"].as_f64().unwrap();
        let end = reference["end"].as_f64().unwrap();
        assert!(
            (turn.start - start).abs() <= BOUNDARY_TOLERANCE
                && (turn.end - end).abs() <= BOUNDARY_TOLERANCE,
            "boundary drift beyond one frame: {turn:?} vs [{start}, {end}]"
        );
    }
}

#[test]
fn identical_runs_are_identical() {
    let Some(first) = diarize_fixture("two-speakers.wav", &DiarizeOptions::default()) else {
        return;
    };
    let second = diarize_fixture("two-speakers.wav", &DiarizeOptions::default()).unwrap();
    assert_eq!(first, second, "diarization must be deterministic");
}
