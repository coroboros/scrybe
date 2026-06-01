//! The whisper.cpp transcription engine (via whisper-rs).
//!
//! The backend is chosen at compile time: CPU by default, `metal`/`cuda`/
//! `vulkan` when their cargo feature is enabled. The correctness floor lives
//! here — `condition_on_previous_text` is disabled to break repetition loops,
//! whisper.cpp's default no-speech/logprob/entropy thresholds and temperature
//! fallback are kept, and segments whose no-speech probability is high are
//! dropped so silence does not hallucinate text.

use std::fmt;
use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::error::ScrybeError;

/// Segments with a no-speech probability above this are treated as silence and
/// dropped from the transcript.
const NO_SPEECH_DROP: f32 = 0.6;

/// The compiled inference backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Cpu => "CPU",
            Self::Metal => "Metal",
            Self::Cuda => "CUDA",
            Self::Vulkan => "Vulkan",
        };
        f.write_str(name)
    }
}

/// The backend this binary was compiled with.
pub const fn active_backend() -> Backend {
    if cfg!(feature = "metal") {
        Backend::Metal
    } else if cfg!(feature = "cuda") {
        Backend::Cuda
    } else if cfg!(feature = "vulkan") {
        Backend::Vulkan
    } else {
        Backend::Cpu
    }
}

/// One transcript segment with its timing.
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// A finished transcript plus the language that was used or detected.
pub struct Transcript {
    pub segments: Vec<Segment>,
    pub language: String,
}

/// How to run one transcription.
pub struct TranscribeOptions {
    /// Source language code, or `None` to auto-detect.
    pub language: Option<String>,
    /// Translate to English instead of transcribing in the source language.
    pub translate: bool,
    /// CPU threads for decoding.
    pub threads: usize,
}

/// A loaded model, reusable across files.
pub struct Engine {
    ctx: WhisperContext,
}

impl Engine {
    /// Load a ggml model from disk onto the active backend.
    pub fn load(model_path: &Path) -> Result<Self, ScrybeError> {
        // Route whisper.cpp/GGML's chatty stderr into the (uninstalled) log
        // backend, which silences it.
        whisper_rs::install_logging_hooks();
        let mut params = WhisperContextParameters::default();
        params.use_gpu(active_backend() != Backend::Cpu);
        let ctx = WhisperContext::new_with_params(model_path, params)
            .map_err(|e| load_error(e.to_string()))?;
        Ok(Self { ctx })
    }

    /// Transcribe 16 kHz mono f32 PCM. whisper.cpp handles long-audio windowing
    /// internally; quality gating and no-speech filtering happen here.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        opts: &TranscribeOptions,
    ) -> Result<Transcript, ScrybeError> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| run_error(e.to_string()))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(opts.threads.max(1) as i32);
        params.set_translate(opts.translate);
        // condition_on_previous_text = false: the single most effective guard
        // against repetition loops on long audio.
        params.set_no_context(true);
        // Quality gating floor (whisper.cpp's defaults, set explicitly): fall back
        // through rising temperatures, and drop low-confidence / high-entropy /
        // silent output rather than hallucinate.
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        params.set_no_speech_thold(NO_SPEECH_DROP);
        params.set_logprob_thold(-1.0);
        params.set_entropy_thold(2.4);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_token_timestamps(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        let language = opts.language.as_deref().unwrap_or("auto");
        params.set_language(Some(language));

        state
            .full(params, pcm)
            .map_err(|e| run_error(e.to_string()))?;

        let mut segments = Vec::new();
        let count = state.full_n_segments();
        for index in 0..count {
            let Some(segment) = state.get_segment(index) else {
                continue;
            };
            // Hallucination gate: skip segments whisper marks as likely silence.
            if segment.no_speech_probability() > NO_SPEECH_DROP {
                continue;
            }
            let text = segment
                .to_str_lossy()
                .map(|cow| cow.trim().to_owned())
                .unwrap_or_default();
            if text.is_empty() {
                continue;
            }
            segments.push(Segment {
                start: segment.start_timestamp() as f64 / 100.0,
                end: segment.end_timestamp() as f64 / 100.0,
                text,
            });
        }

        let detected = match opts.language.as_deref() {
            Some(lang) if lang != "auto" => lang.to_owned(),
            _ => {
                let id = state.full_lang_id_from_state();
                whisper_rs::get_lang_str(id).unwrap_or(language).to_owned()
            }
        };

        Ok(Transcript {
            segments,
            language: detected,
        })
    }
}

fn load_error(detail: String) -> ScrybeError {
    if active_backend() == Backend::Cpu {
        ScrybeError::ModelDownloadFailed {
            model: "<loaded model>".to_owned(),
            detail: format!("could not load model: {detail}"),
        }
    } else {
        ScrybeError::GpuInitFailed { detail }
    }
}

fn run_error(detail: String) -> ScrybeError {
    ScrybeError::GpuInitFailed {
        detail: format!("transcription failed: {detail}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_build_reports_cpu() {
        assert_eq!(active_backend(), Backend::Cpu);
        assert_eq!(Backend::Cpu.to_string(), "CPU");
        assert_eq!(Backend::Metal.to_string(), "Metal");
    }
}
