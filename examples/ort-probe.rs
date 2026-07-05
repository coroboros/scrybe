//! Proves the ONNX Runtime substrate end-to-end on one target: fetch the
//! SHA-pinned speaker-segmentation model, decode a wav through scrybe's own
//! pipeline, run one inference, and check the output shape — routing every
//! failure through `ScrybeError` so the exit code stays on contract.
//!
//! Run it twice: the first run may download the model (progress on stderr);
//! the second must complete with stderr fully silent.

use std::path::PathBuf;

use ort::session::Session;
use ort::value::TensorRef;
use scrybe::audio::{self, TARGET_SAMPLE_RATE};
use scrybe::cli::Decoder;
use scrybe::error::ScrybeError;
use scrybe::model;

/// One segmentation window: the model consumes fixed 10 s frames at 16 kHz.
const WINDOW_SAMPLES: usize = 10 * TARGET_SAMPLE_RATE as usize;
/// The powerset head emits 7 classes (silence, 3 single speakers, 3 pairs).
const POWERSET_CLASSES: usize = 7;

fn main() {
    match run() {
        Ok(summary) => println!("{summary}"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(e.exit_code());
        }
    }
}

fn run() -> Result<String, ScrybeError> {
    let wav = std::env::args().nth(1).map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/speech/en.wav"),
        PathBuf::from,
    );

    let model_path = model::ensure_segmentation(false)?;
    let mut samples = audio::load_audio(&wav, Decoder::Symphonia)?.samples;
    samples.resize(WINDOW_SAMPLES, 0.0);

    let load_err = |e: ort::Error| ScrybeError::ModelLoadFailed {
        path: model_path.clone(),
        detail: e.to_string(),
    };
    let mut session = Session::builder()
        .map_err(&load_err)?
        .commit_from_file(&model_path)
        .map_err(&load_err)?;

    let run_err = |detail: String| ScrybeError::TranscriptionFailed { detail };
    let out_name = session
        .outputs()
        .first()
        .ok_or_else(|| run_err("model reports no outputs".to_owned()))?
        .name()
        .to_owned();

    let input = ndarray::Array3::from_shape_vec((1, 1, WINDOW_SAMPLES), samples)
        .map_err(|e| run_err(e.to_string()))?;

    let started = std::time::Instant::now();
    let outputs = session
        .run(ort::inputs![
            TensorRef::from_array_view(input.view()).map_err(|e| run_err(e.to_string()))?
        ])
        .map_err(|e| run_err(e.to_string()))?;
    let elapsed = started.elapsed();

    let logits = outputs[out_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(|e| run_err(e.to_string()))?;

    let shape = logits.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] != POWERSET_CLASSES {
        return Err(run_err(format!("unexpected output shape {shape:?}")));
    }
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in logits.iter() {
        if !v.is_finite() {
            return Err(run_err("non-finite value in model output".to_owned()));
        }
        min = min.min(v);
        max = max.max(v);
    }

    Ok(format!(
        "ort probe ok: output {shape:?}, range [{min:.3}, {max:.3}], inference {} ms, model {}",
        elapsed.as_millis(),
        model_path.display(),
    ))
}
