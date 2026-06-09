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

/// No-speech probability above this drops the segment as silence.
const NO_SPEECH_DROP: f32 = 0.6;

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

#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// Per-word timing, populated only when [`TranscribeOptions::word_timestamps`]
    /// is set (JSON output); empty otherwise.
    pub words: Vec<Word>,
}

#[derive(Debug, Clone)]
pub struct Word {
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
    pub language: Option<String>,
    /// Translate to English instead of transcribing in the source language.
    pub translate: bool,
    pub threads: usize,
    /// Emit per-word timing (enables whisper token timestamps). Set for JSON output;
    /// off otherwise, since the other formats carry only segment-level timing.
    pub word_timestamps: bool,
}

/// A loaded model, reusable across files.
pub struct Engine {
    ctx: WhisperContext,
    vad_model_path: Option<PathBuf>,
}

impl Engine {
    /// Load a ggml model from disk onto the active backend. `vad_model_path`, when
    /// present, enables Silero voice-activity segmentation per transcription.
    pub fn load(model_path: &Path, vad_model_path: Option<&Path>) -> Result<Self, ScrybeError> {
        // Route whisper.cpp/GGML's chatty stderr into the (uninstalled) log
        // backend, which silences it. A process-global effect, so install it once
        // even when multiple engines are loaded (e.g. across tests).
        static LOG_HOOKS: std::sync::Once = std::sync::Once::new();
        LOG_HOOKS.call_once(whisper_rs::install_logging_hooks);
        let mut params = WhisperContextParameters::default();
        params.use_gpu(active_backend() != Backend::Cpu);
        let ctx = WhisperContext::new_with_params(model_path, params)
            .map_err(|e| load_error(active_backend(), model_path, e.to_string()))?;
        Ok(Self {
            ctx,
            vad_model_path: vad_model_path.map(Path::to_path_buf),
        })
    }

    /// Transcribe 16 kHz mono f32 PCM.
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
        // Per-token timing for the optional JSON `words` field; whisper skips the DTW
        // timing work entirely when this is off, so non-JSON runs pay nothing.
        params.set_token_timestamps(opts.word_timestamps);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        // VAD is the first silence layer (no-speech filter_segments is the second).
        // enable_vad panics unless a path is set first, so gate on a valid path.
        if let Some(vad) = self.vad_model_path.as_deref().and_then(Path::to_str) {
            params.set_vad_model_path(Some(vad));
            params.enable_vad(true);
        }

        let normalized = normalize_language(opts.language.as_deref());
        let language = normalized.as_deref().unwrap_or("auto");
        params.set_language(Some(language));
        params.set_progress_callback_safe(progress);

        state
            .full(params, pcm)
            .map_err(|e| run_error(e.to_string()))?;

        let count = state.full_n_segments();
        let collect_words = opts.word_timestamps;
        let raw = (0..count).filter_map(|index| {
            let segment = state.get_segment(index)?;
            let text = segment
                .to_str_lossy()
                .map(|cow| cow.into_owned())
                .unwrap_or_default();
            let words = if collect_words {
                let tokens = (0..segment.n_tokens()).filter_map(|token_index| {
                    let token = segment.get_token(token_index)?;
                    let text = token.to_str_lossy().ok()?.into_owned();
                    let data = token.token_data();
                    Some((text, data.t0, data.t1))
                });
                group_words(tokens)
            } else {
                Vec::new()
            };
            Some((
                segment.no_speech_probability(),
                text,
                segment.start_timestamp(),
                segment.end_timestamp(),
                words,
            ))
        });
        let segments = filter_segments(raw);

        let detected = match normalized.as_deref() {
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
fn filter_segments(
    raw: impl IntoIterator<Item = (f32, String, i64, i64, Vec<Word>)>,
) -> Vec<Segment> {
    raw.into_iter()
        .filter_map(|(no_speech, text, start_cs, end_cs, words)| {
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
                words,
            })
        })
        .collect()
}

/// Group whisper tokens (`(text, t0_cs, t1_cs)`, centisecond timestamps) into timed
/// words. A token whose text begins with a space opens a new word — whisper renders
/// the SentencePiece word boundary that way — and following spaceless tokens extend
/// it. Bracketed special tokens (`[_BEG_]`, `[_TT_..]`) and blank tokens are skipped.
/// Pure, so the grouping is unit-testable without a model.
fn group_words(tokens: impl IntoIterator<Item = (String, i64, i64)>) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    for (text, t0, t1) in tokens {
        if text.starts_with('[') && text.ends_with(']') {
            continue;
        }
        if text.trim().is_empty() {
            continue;
        }
        let start = t0 as f64 / 100.0;
        let end = t1 as f64 / 100.0;
        match words.last_mut() {
            // A spaceless token continues the current word (subword piece).
            Some(word) if !text.starts_with(' ') => {
                word.text.push_str(&text);
                word.end = end;
            }
            _ => words.push(Word {
                start,
                end,
                text: text.trim_start().to_owned(),
            }),
        }
    }
    words
}

/// Normalize a requested language to what whisper.cpp expects: lowercased (its keys
/// are lowercase ISO codes, matched case-sensitively), with an empty/whitespace code
/// treated as unspecified (auto-detect). Without this an accepted `--lang EN` / `""`
/// reaches whisper as an unknown code and silently collapses the transcript. All
/// valid codes are lowercase, so lowercasing can't corrupt a real one.
fn normalize_language(lang: Option<&str>) -> Option<String> {
    lang.map(str::to_ascii_lowercase)
        .filter(|code| !code.trim().is_empty())
}

fn load_error(backend: Backend, path: &Path, detail: String) -> ScrybeError {
    if backend == Backend::Cpu {
        // A SHA-verified file that fails to load is a corrupt/incompatible ggml,
        // not a download or GPU fault.
        ScrybeError::ModelLoadFailed {
            path: path.to_path_buf(),
            detail,
        }
    } else {
        // GPU build: a failed context creation is where GPU init actually happens.
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
        // Both load_error arms, independent of the compiled backend: a corrupt model
        // is 15 on CPU and 13 (GPU init) on a GPU build; a runtime fault is 16.
        assert_eq!(
            load_error(Backend::Cpu, Path::new("/m.bin"), "x".to_owned()).exit_code(),
            15
        );
        assert_eq!(
            load_error(Backend::Metal, Path::new("/m.bin"), "x".to_owned()).exit_code(),
            13
        );
        assert_eq!(run_error("x".to_owned()).exit_code(), 16);
    }

    #[test]
    fn normalize_language_lowercases_and_blanks_to_auto() {
        // Lowercased so whisper's case-sensitive matcher accepts it; empty/whitespace
        // becomes None (auto-detect) so an accepted blank code can't collapse output.
        assert_eq!(normalize_language(Some("EN")).as_deref(), Some("en"));
        assert_eq!(normalize_language(Some("Fr")).as_deref(), Some("fr"));
        assert_eq!(normalize_language(Some("auto")).as_deref(), Some("auto"));
        assert_eq!(normalize_language(Some("")), None);
        assert_eq!(normalize_language(Some("   ")), None);
        assert_eq!(normalize_language(None), None);
    }

    #[test]
    fn filter_segments_enforces_the_no_speech_gate() {
        let raw = vec![
            (0.9_f32, "loud silence".to_owned(), 0, 100, vec![]), // > 0.6 → dropped
            (0.6_f32, "boundary kept".to_owned(), 100, 250, vec![]), // == 0.6 → kept (`>` semantics)
            (0.1_f32, "   ".to_owned(), 250, 300, vec![]),           // whitespace → skipped
            (0.1_f32, "  hello  ".to_owned(), 300, 450, vec![]),     // trimmed and kept
        ];
        let segs = filter_segments(raw);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "boundary kept");
        assert_eq!(segs[0].start, 1.0); // 100 cs / 100
        assert_eq!(segs[0].end, 2.5);
        assert_eq!(segs[1].text, "hello"); // trimmed
        assert_eq!(segs[1].start, 3.0);
    }

    #[test]
    fn group_words_merges_subword_tokens_and_skips_specials() {
        // Whisper emits a leading space at each word boundary and splits words into
        // subword tokens; specials are bracketed. Pin: " quick" + "er" → one word
        // "quicker" spanning both token times; " fox" → its own word; the `[_TT_..]`
        // special and the blank token are dropped.
        let tokens = vec![
            ("[_BEG_]".to_owned(), 0, 0),
            (" quick".to_owned(), 0, 30),
            ("er".to_owned(), 30, 50),
            (" fox".to_owned(), 50, 80),
            (" ".to_owned(), 80, 80),
            ("[_TT_100]".to_owned(), 80, 80),
        ];
        let words = group_words(tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "quicker");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 0.5); // extends to the second token's t1 (50 cs)
        assert_eq!(words[1].text, "fox");
        assert_eq!(words[1].start, 0.5);
        assert_eq!(words[1].end, 0.8);
    }
}
