//! ONNX Runtime plumbing for the two diarization models. Failures at the
//! model boundary are validated and mapped — a bad artifact must surface as a
//! coded error, never a panic or a C++ abort.

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use super::fbank::NUM_MEL_BINS;
use super::frames::ChunkActivity;
use super::powerset::decode_frame;
use super::{NUM_FRAMES, POWERSET_CLASSES, WINDOW_SAMPLES};
use crate::error::ScrybeError;

pub(crate) fn load(path: &Path) -> Result<Session, ScrybeError> {
    let load_err = |e: ort::Error| ScrybeError::ModelLoadFailed {
        path: path.to_path_buf(),
        detail: e.to_string(),
    };
    Session::builder()
        .map_err(load_err)?
        .commit_from_file(path)
        .map_err(load_err)
}

fn run_err(e: impl std::fmt::Display) -> ScrybeError {
    ScrybeError::DiarizationFailed {
        detail: e.to_string(),
    }
}

/// First output name of a session (both models expose exactly one output).
fn first_output(session: &Session) -> Result<String, ScrybeError> {
    Ok(session
        .outputs()
        .first()
        .ok_or_else(|| run_err("model reports no outputs"))?
        .name()
        .to_owned())
}

/// Run the segmentation model on one 10 s window and decode the powerset
/// head to per-frame binary speaker activity.
pub(crate) fn segmentation_activity(
    session: &mut Session,
    window: &[f32],
) -> Result<ChunkActivity, ScrybeError> {
    debug_assert_eq!(window.len(), WINDOW_SAMPLES);
    let out_name = first_output(session)?;
    // Borrow the window directly as a [1,1,N] tensor — no copy, no ndarray.
    let outputs = session
        .run(ort::inputs![
            TensorRef::from_array_view(([1, 1, window.len()], window)).map_err(run_err)?
        ])
        .map_err(run_err)?;
    let logits = outputs[out_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(run_err)?;

    let shape = logits.shape().to_vec();
    if shape != [1, NUM_FRAMES, POWERSET_CLASSES] {
        return Err(run_err(format!(
            "unexpected segmentation output shape {shape:?}"
        )));
    }
    let flat: Vec<f32> = logits.iter().copied().collect();
    Ok(flat
        .chunks_exact(POWERSET_CLASSES)
        .map(|frame| {
            let mut scores = [0.0_f32; POWERSET_CLASSES];
            scores.copy_from_slice(frame);
            decode_frame(&scores)
        })
        .collect())
}

/// Run the embedding model on the selected fbank frames of one local speaker.
pub(crate) fn embed(
    session: &mut Session,
    features: &[[f32; NUM_MEL_BINS]],
) -> Result<Vec<f32>, ScrybeError> {
    let out_name = first_output(session)?;
    // `as_flattened` reinterprets &[[f32; 80]] as &[f32] — no copy.
    let outputs = session
        .run(ort::inputs![
            TensorRef::from_array_view((
                [1, features.len(), NUM_MEL_BINS],
                features.as_flattened()
            ))
            .map_err(run_err)?
        ])
        .map_err(run_err)?;
    let embedding = outputs[out_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(run_err)?;

    let row: Vec<f32> = embedding.iter().copied().collect();
    if row.is_empty() || row.iter().any(|v| !v.is_finite()) {
        return Err(run_err("embedding model returned a non-finite vector"));
    }
    Ok(row)
}
