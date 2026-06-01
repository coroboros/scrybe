//! Structured failure taxonomy with stable, documented exit codes.
//!
//! Every user-facing failure is one [`ScrybeError`] rendered as a single
//! actionable line, and each variant maps to a fixed exit code so callers can
//! branch on `$?`. Argument errors (bad flags, unknown values) are owned by
//! `clap`, which exits `2`.

use std::fmt;
use std::path::PathBuf;

// Variants beyond `FileNotFound` are constructed by the decode, model-cache,
// engine, and batch stages added in later milestones; their exit codes are
// fixed here so the contract is stable from the first release.
#[allow(dead_code)]
/// A user-facing failure carrying enough context to render an actionable line.
#[derive(Debug)]
pub enum ScrybeError {
    /// Audio uses a codec the decoder cannot handle (e.g. HE-AAC/SBR).
    UnsupportedCodec { path: PathBuf, detail: String },
    /// A model could not be fetched or read from cache.
    ModelDownloadFailed { model: String, detail: String },
    /// The chosen model plus job count exceeds available memory.
    OutOfMemory { detail: String },
    /// The GPU backend failed to initialize.
    GpuInitFailed { detail: String },
    /// An input path does not exist.
    FileNotFound { path: PathBuf },
    /// Some files in a batch failed while others succeeded.
    PartialBatchFailure { failed: usize, total: usize },
}

impl ScrybeError {
    /// The process exit code for this failure. Stable across releases.
    pub const fn exit_code(&self) -> i32 {
        match self {
            Self::UnsupportedCodec { .. } => 10,
            Self::ModelDownloadFailed { .. } => 11,
            Self::OutOfMemory { .. } => 12,
            Self::GpuInitFailed { .. } => 13,
            Self::FileNotFound { .. } => 14,
            Self::PartialBatchFailure { .. } => 20,
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
            Self::OutOfMemory { detail } => write!(
                f,
                "not enough memory: {detail}. Choose a smaller `--model` or lower `--jobs`.",
            ),
            Self::GpuInitFailed { detail } => write!(
                f,
                "GPU backend failed to start: {detail}. Re-run with `--jobs 1` or use a CPU build.",
            ),
            Self::FileNotFound { path } => {
                write!(f, "no such file or directory: {}", path.display())
            }
            Self::PartialBatchFailure { failed, total } => write!(
                f,
                "{failed} of {total} files failed; the rest completed. See the per-file lines above.",
            ),
        }
    }
}

impl std::error::Error for ScrybeError {}
