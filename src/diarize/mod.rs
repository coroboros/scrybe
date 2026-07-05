//! Offline speaker diarization: a Rust port of the pyannote
//! `speaker-diarization-3.1` pipeline on ONNX Runtime.
//!
//! Pipeline: 10 s sliding-window segmentation (powerset decode) → one speaker
//! embedding per (chunk, local speaker), masked by the segmentation → global
//! agglomerative clustering (centroid linkage) → per-frame reconstruction →
//! speaker turns. Pure over the audio: takes 16 kHz mono f32 PCM and model
//! paths, does no I/O of its own.

mod chunk;
mod clustering;
mod embedding;
mod fbank;
mod frames;
mod powerset;
mod session;

use std::path::Path;

use crate::audio::TARGET_SAMPLE_RATE;
use crate::error::ScrybeError;

/// One segmentation window: the model consumes fixed 10 s frames at 16 kHz.
pub const WINDOW_SAMPLES: usize = 10 * TARGET_SAMPLE_RATE as usize;
/// Hop between windows: 1 s (10% of the window, the pipeline default).
pub(crate) const STEP_SAMPLES: usize = WINDOW_SAMPLES / 10;
/// Segmentation output frames per 10 s window.
pub(crate) const NUM_FRAMES: usize = 589;
/// Local speaker capacity of the segmentation model.
pub(crate) const NUM_LOCAL_SPEAKERS: usize = 3;
/// The powerset head emits 7 classes (silence, 3 singles, 3 pairs).
pub const POWERSET_CLASSES: usize = 7;

/// A diarized speaker turn on the global timeline. `speaker` is a dense,
/// zero-based global index ordered by first appearance; output formats decide
/// how to label it.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

/// Caller-tunable knobs. `num_speakers` pins the exact cluster count;
/// `None` lets the calibrated distance threshold decide.
#[derive(Debug, Clone, Default)]
pub struct DiarizeOptions {
    pub num_speakers: Option<usize>,
}

/// The two loaded diarization models, reused across files.
pub struct Diarizer {
    segmentation: ort::session::Session,
    embedding: ort::session::Session,
}

impl Diarizer {
    /// Load both models from verified local paths.
    pub fn load(segmentation: &Path, embedding: &Path) -> Result<Self, ScrybeError> {
        Ok(Self {
            segmentation: session::load(segmentation)?,
            embedding: session::load(embedding)?,
        })
    }

    /// Diarize 16 kHz mono f32 PCM into speaker turns. Returns an empty list
    /// when nothing is attributable (silence, or too little clean speech to
    /// embed any speaker).
    pub fn diarize(
        &mut self,
        samples: &[f32],
        options: &DiarizeOptions,
    ) -> Result<Vec<Turn>, ScrybeError> {
        let plan = chunk::ChunkPlan::new(samples.len());
        let num_chunks = plan.num_chunks();

        let mut chunks: Vec<frames::ChunkActivity> = Vec::with_capacity(num_chunks);
        let mut embeddings: Vec<Option<Vec<f32>>> =
            Vec::with_capacity(num_chunks * NUM_LOCAL_SPEAKERS);
        let mut active: Vec<bool> = Vec::with_capacity(num_chunks * NUM_LOCAL_SPEAKERS);

        for i in 0..num_chunks {
            let window = plan.window(samples, i);
            let activity = session::segmentation_activity(&mut self.segmentation, &window)?;
            let features = fbank::fbank_cmn(&window);

            for spk in 0..NUM_LOCAL_SPEAKERS {
                let is_active = activity.iter().any(|frame| frame[spk]);
                active.push(is_active);
                if !is_active {
                    embeddings.push(None);
                    continue;
                }
                let mask = embedding::choose_mask(&activity, spk);
                let selected = embedding::select_frames(&features, &mask);
                if selected.len() < embedding::MIN_FBANK_FRAMES {
                    embeddings.push(None);
                    continue;
                }
                embeddings.push(Some(session::embed(&mut self.embedding, &selected)?));
            }
            chunks.push(activity);
        }

        let outcome = clustering::cluster_speakers(&embeddings, &active, options.num_speakers);
        if outcome.num_clusters == 0 {
            return Ok(Vec::new());
        }

        let mut count = frames::speaker_count(&chunks);
        if let Some(n) = options.num_speakers {
            let cap = n as u32;
            for c in count.iter_mut() {
                *c = (*c).min(cap);
            }
        }

        let binary = frames::reconstruct(&chunks, &outcome.labels, outcome.num_clusters, &count);
        Ok(frames::to_turns(&binary))
    }
}
