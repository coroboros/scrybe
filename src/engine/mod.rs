//! The whisper.cpp transcription engine (via whisper-rs).
//!
//! The backend is chosen at compile time: CPU by default, `metal`/`cuda`/
//! `vulkan` when their cargo feature is enabled. The correctness floor lives
//! here — `condition_on_previous_text` is disabled to break repetition loops,
//! whisper.cpp's default no-speech/logprob/entropy thresholds and temperature
//! fallback are kept, and segments whose no-speech probability is high are
//! dropped so silence does not hallucinate text. Long-audio windowing is
//! delegated to whisper.cpp (its internal 30 s windows) plus Silero VAD
//! segmentation — the deliberate substitute for hand-rolled overlapped chunking.

use std::fmt;
use std::path::{Path, PathBuf};

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
#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// A finished transcript plus the language that was used or detected.
#[derive(Debug, Clone)]
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

/// A loaded model, reusable across files, plus the optional VAD model.
pub struct Engine {
    ctx: WhisperContext,
    vad_model_path: Option<PathBuf>,
}

impl Engine {
    /// Load a ggml model from disk onto the active backend. `vad_model_path`, when
    /// present, enables Silero voice-activity segmentation per transcription.
    pub fn load(model_path: &Path, vad_model_path: Option<&Path>) -> Result<Self, ScrybeError> {
        // Route whisper.cpp/GGML's chatty stderr into the (uninstalled) log
        // backend, which silences it.
        whisper_rs::install_logging_hooks();
        let mut params = WhisperContextParameters::default();
        params.use_gpu(active_backend() != Backend::Cpu);
        let ctx = WhisperContext::new_with_params(model_path, params)
            .map_err(|e| load_error(model_path, e.to_string()))?;
        Ok(Self {
            ctx,
            vad_model_path: vad_model_path.map(Path::to_path_buf),
        })
    }

    /// Transcribe 16 kHz mono f32 PCM. whisper.cpp handles long-audio windowing
    /// internally; quality gating and no-speech filtering happen here.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        opts: &TranscribeOptions,
        progress: impl FnMut(i32) + 'static,
    ) -> Result<Transcript, ScrybeError> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| run_error(e.to_string()))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // Clamp before the narrowing cast so a huge --threads can't wrap negative.
        params.set_n_threads(opts.threads.clamp(1, i32::MAX as usize) as i32);
        params.set_translate(opts.translate);
        // condition_on_previous_text = false: the single most effective guard
        // against repetition loops on long audio.
        params.set_no_context(true);
        // Quality gating floor (whisper.cpp's defaults, set explicitly): fall back
        // through rising temperatures, and drop low-confidence / high-entropy /
        // silent output rather than hallucinate.
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        // Passed to whisper.cpp too, but `filter_segments` is the authoritative
        // silence gate regardless of how the engine acts on this threshold.
        params.set_no_speech_thold(NO_SPEECH_DROP);
        params.set_logprob_thold(-1.0);
        // entropy_thold is whisper.cpp's name for the compression-ratio gate.
        params.set_entropy_thold(2.4);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        // Voice-activity segmentation when the Silero model is present — the
        // spec's mandated correctness floor; the no-speech filter below stays as a
        // second layer. enable_vad panics unless a path is set first, so it is
        // gated on a valid path.
        if let Some(vad) = self.vad_model_path.as_deref().and_then(Path::to_str) {
            params.set_vad_model_path(Some(vad));
            params.enable_vad(true);
        }

        let language = opts.language.as_deref().unwrap_or("auto");
        params.set_language(Some(language));
        params.set_progress_callback_safe(progress);

        state
            .full(params, pcm)
            .map_err(|e| run_error(e.to_string()))?;

        let count = state.full_n_segments();
        let raw = (0..count).filter_map(|index| {
            let segment = state.get_segment(index)?;
            let text = segment
                .to_str_lossy()
                .map(|cow| cow.into_owned())
                .unwrap_or_default();
            Some((
                segment.no_speech_probability(),
                text,
                segment.start_timestamp(),
                segment.end_timestamp(),
            ))
        });
        let segments = filter_segments(raw);

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

/// Apply the hallucination gate (drop segments whisper marks as likely silence),
/// trim and skip empty text, and map centisecond timestamps to seconds. Pure (no
/// FFI) so the correctness floor is unit-testable without a model.
fn filter_segments(raw: impl IntoIterator<Item = (f32, String, i64, i64)>) -> Vec<Segment> {
    raw.into_iter()
        .filter_map(|(no_speech, text, start_cs, end_cs)| {
            if no_speech > NO_SPEECH_DROP {
                return None;
            }
            let text = text.trim().to_owned();
            if text.is_empty() {
                return None;
            }
            Some(Segment {
                start: start_cs as f64 / 100.0,
                end: end_cs as f64 / 100.0,
                text,
            })
        })
        .collect()
}

fn load_error(path: &Path, detail: String) -> ScrybeError {
    if active_backend() == Backend::Cpu {
        // A SHA-verified file that fails to load is a corrupt/incompatible ggml,
        // not a download or GPU fault.
        ScrybeError::ModelLoadFailed {
            path: path.to_path_buf(),
            detail,
        }
    } else {
        ScrybeError::GpuInitFailed { detail }
    }
}

fn run_error(detail: String) -> ScrybeError {
    // `create_state`/`full` failures are runtime compute faults: the context — and
    // thus the backend — already initialized in `Engine::load`. So this is never an
    // init fault on any backend; `GpuInitFailed` is reserved for `load_error`.
    ScrybeError::TranscriptionFailed { detail }
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

    #[test]
    fn cpu_engine_errors_are_not_labelled_gpu() {
        // On the default CPU build, load and runtime failures must not surface as
        // GPU faults (exit 13): a corrupt model is 15, a compute failure is 16.
        assert_eq!(
            load_error(Path::new("/m.bin"), "x".to_owned()).exit_code(),
            15
        );
        assert_eq!(run_error("x".to_owned()).exit_code(), 16);
    }

    #[test]
    fn filter_segments_enforces_the_no_speech_gate() {
        let raw = vec![
            (0.9_f32, "loud silence".to_owned(), 0, 100), // > 0.6 → dropped
            (0.6_f32, "boundary kept".to_owned(), 100, 250), // == 0.6 → kept (`>` semantics)
            (0.1_f32, "   ".to_owned(), 250, 300),        // whitespace → skipped
            (0.1_f32, "  hello  ".to_owned(), 300, 450),  // trimmed and kept
        ];
        let segs = filter_segments(raw);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "boundary kept");
        assert_eq!(segs[0].start, 1.0); // 100 cs / 100
        assert_eq!(segs[0].end, 2.5);
        assert_eq!(segs[1].text, "hello"); // trimmed
        assert_eq!(segs[1].start, 3.0);
    }
}
