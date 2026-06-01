//! Structured failure taxonomy with stable, documented exit codes.
//!
//! Every runtime failure is one [`ScrybeError`] rendered as a single actionable
//! line, and each variant maps to a fixed exit code so callers can branch on
//! `$?`. Argument errors (bad flags, unknown values) are owned by `clap`, and
//! configuration/usage errors (unknown model capability, output collision, no
//! input paths) are deliberately clap-aligned: they print their own line and
//! exit `2` outside this enum rather than inventing a code per case.

use std::fmt;
use std::path::{Path, PathBuf};

/// A user-facing failure carrying enough context to render an actionable line.
/// Exit codes are stable across releases — only ever add, never renumber.
#[derive(Debug)]
pub enum ScrybeError {
    /// Audio uses a codec the decoder cannot handle (e.g. HE-AAC/SBR).
    UnsupportedCodec { path: PathBuf, detail: String },
    /// A model could not be fetched or read from cache.
    ModelDownloadFailed { model: String, detail: String },
    /// A cached model file could not be loaded (corrupt or incompatible ggml).
    ModelLoadFailed { path: PathBuf, detail: String },
    /// The chosen model plus job count exceeds available memory.
    OutOfMemory { detail: String },
    /// The GPU backend failed to initialize.
    GpuInitFailed { detail: String },
    /// Inference failed at runtime on the CPU backend (state/decode failure).
    TranscriptionFailed { detail: String },
    /// An input path does not exist.
    FileNotFound { path: PathBuf },
    /// Some files in a batch failed while others succeeded.
    PartialBatchFailure { failed: usize, total: usize },
    /// The run was stopped early (Ctrl-C) before every file was processed.
    Interrupted { completed: usize, total: usize },
    /// An unexpected I/O failure, such as writing an output file.
    Io { detail: String },
}

impl ScrybeError {
    /// Build an [`UnsupportedCodec`](Self::UnsupportedCodec) for `path`.
    pub fn unsupported_codec(path: &Path, detail: impl Into<String>) -> Self {
        Self::UnsupportedCodec {
            path: path.to_path_buf(),
            detail: detail.into(),
        }
    }

    /// The process exit code for this failure. Stable across releases.
    pub const fn exit_code(&self) -> i32 {
        match self {
            Self::UnsupportedCodec { .. } => 10,
            Self::ModelDownloadFailed { .. } => 11,
            Self::OutOfMemory { .. } => 12,
            Self::GpuInitFailed { .. } => 13,
            Self::FileNotFound { .. } => 14,
            Self::ModelLoadFailed { .. } => 15,
            Self::TranscriptionFailed { .. } => 16,
            Self::PartialBatchFailure { .. } | Self::Interrupted { .. } => 20,
            Self::Io { .. } => 1,
        }
    }
}

impl fmt::Display for ScrybeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCodec { path, detail } => write!(
                f,
                "unsupported codec in {}: {detail}. Re-encode with `ffmpeg -i \"{}\" out.wav`, or retry with `--decoder ffmpeg`.",
                path.display(),
                path.display(),
            ),
            Self::ModelDownloadFailed { model, detail } => write!(
                f,
                "could not obtain model `{model}`: {detail}. Check the network or pre-fetch with `scrybe models pull {model}`.",
            ),
            Self::ModelLoadFailed { path, detail } => write!(
                f,
                "could not load model {}: {detail}. The file may be a corrupt or incompatible ggml — re-fetch with `scrybe models pull` or pick a different `--model`.",
                path.display(),
            ),
            Self::OutOfMemory { detail } => write!(
                f,
                "not enough memory: {detail}. Choose a smaller `--model` or lower `--jobs`.",
            ),
            Self::GpuInitFailed { detail } => write!(
                f,
                "GPU backend failed to start: {detail}. Re-run with `--jobs 1` or use a CPU build.",
            ),
            Self::TranscriptionFailed { detail } => write!(
                f,
                "transcription failed: {detail}. Try a smaller `--model`, or re-fetch the model with `scrybe models pull`.",
            ),
            Self::FileNotFound { path } => {
                write!(f, "no such file or directory: {}", path.display())
            }
            Self::PartialBatchFailure { failed, total } => write!(
                f,
                "{failed} of {total} files failed; the rest completed. See the per-file lines above.",
            ),
            Self::Interrupted { completed, total } => write!(
                f,
                "interrupted: {completed} of {total} files processed before stopping; no files were lost.",
            ),
            Self::Io { detail } => write!(f, "I/O error: {detail}"),
        }
    }
}

impl std::error::Error for ScrybeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_the_documented_contract() {
        let cases: [(ScrybeError, i32); 10] = [
            (
                ScrybeError::Io {
                    detail: String::new(),
                },
                1,
            ),
            (
                ScrybeError::UnsupportedCodec {
                    path: PathBuf::new(),
                    detail: String::new(),
                },
                10,
            ),
            (
                ScrybeError::ModelDownloadFailed {
                    model: String::new(),
                    detail: String::new(),
                },
                11,
            ),
            (
                ScrybeError::OutOfMemory {
                    detail: String::new(),
                },
                12,
            ),
            (
                ScrybeError::GpuInitFailed {
                    detail: String::new(),
                },
                13,
            ),
            (
                ScrybeError::FileNotFound {
                    path: PathBuf::new(),
                },
                14,
            ),
            (
                ScrybeError::ModelLoadFailed {
                    path: PathBuf::new(),
                    detail: String::new(),
                },
                15,
            ),
            (
                ScrybeError::TranscriptionFailed {
                    detail: String::new(),
                },
                16,
            ),
            (
                ScrybeError::PartialBatchFailure {
                    failed: 1,
                    total: 2,
                },
                20,
            ),
            (
                ScrybeError::Interrupted {
                    completed: 1,
                    total: 2,
                },
                20,
            ),
        ];
        for (err, code) in cases {
            assert_eq!(err.exit_code(), code, "{err:?}");
        }
    }
}
