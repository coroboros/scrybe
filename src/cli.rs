//! The `clap` command surface: the default transcribe action plus the `models`
//! subcommand. Restricted-choice flags are `ValueEnum`s, so an invalid value is
//! rejected with the list of valid ones.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

/// The model chosen when `--model` is omitted. Single source for both the clap
/// default and the marker in `models list`.
pub const DEFAULT_MODEL: Model = Model::LargeV3Turbo;

/// scrybe — collapse sound into signal. Transcribe audio to text, offline.
#[derive(Debug, Parser)]
#[command(name = "scrybe", version, about, long_about = None, propagate_version = true)]
pub struct Cli {
    /// Audio files or directories to transcribe.
    #[arg(value_name = "PATHS")]
    pub paths: Vec<PathBuf>,

    /// Recurse into subdirectories when a path is a directory.
    #[arg(long)]
    pub recursive: bool,

    /// Whisper model to use; defaults to the largest that fits detected RAM.
    #[arg(long, value_enum)]
    pub model: Option<Model>,

    /// Source language code (e.g. `en`, `fr`); auto-detected when omitted.
    #[arg(long, value_name = "LANG")]
    pub lang: Option<String>,

    /// Transcribe in the source language, or translate to English.
    #[arg(long, value_enum, default_value_t = Task::Transcribe)]
    pub task: Task,

    /// Output formats, comma-separated (e.g. `--format srt,json`).
    #[arg(long, value_enum, value_delimiter = ',', default_value = "txt")]
    pub format: Vec<Format>,

    /// Write outputs here instead of next to each input.
    #[arg(long, value_name = "DIR")]
    pub out_dir: Option<PathBuf>,

    /// Files processed concurrently; device-aware when omitted.
    #[arg(long, value_name = "N")]
    pub jobs: Option<usize>,

    /// CPU threads per inference job; device-aware when omitted.
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Reprocess inputs even when an up-to-date output exists.
    #[arg(long)]
    pub force: bool,

    /// Print the resolved plan without transcribing.
    #[arg(long)]
    pub dry_run: bool,

    /// Audio decoder backend.
    #[arg(long, value_enum, default_value_t = Decoder::Symphonia)]
    pub decoder: Decoder,

    /// Disable colored output.
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Emit JSON; forces JSON output, overriding `--format`. A single input streams
    /// to stdout, multiple inputs write `.json` sidecars.
    #[arg(long)]
    pub json: bool,

    /// Use only cached models; never access the network.
    #[arg(long)]
    pub offline: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

impl Cli {
    /// The formats a run will actually write. `--json` forces JSON output,
    /// overriding `--format`. De-duplicated (order-preserving) so `--format txt,txt`
    /// doesn't write the same file twice. Single source for the policy so the plan
    /// banner, the collision check, and the writers all agree.
    pub fn effective_formats(&self) -> Vec<Format> {
        let requested = if self.json {
            vec![Format::Json]
        } else {
            self.format.clone()
        };
        let mut unique = Vec::with_capacity(requested.len());
        for format in requested {
            if !unique.contains(&format) {
                unique.push(format);
            }
        }
        unique
    }
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Manage Whisper models on disk.
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
}

/// Actions under `scrybe models`.
#[derive(Debug, Subcommand)]
pub enum ModelsAction {
    /// List the known Whisper models, sizes, and which are cached.
    List,
    /// Download a model into the cache.
    Pull { model: Model },
    /// Remove a cached model.
    Remove { model: Model },
    /// Print the model cache directory.
    Path,
}

/// The Whisper model family scrybe can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Model {
    #[value(name = "tiny")]
    Tiny,
    #[value(name = "base")]
    Base,
    #[value(name = "small")]
    Small,
    #[value(name = "large-v3")]
    LargeV3,
    #[value(name = "large-v3-turbo")]
    LargeV3Turbo,
    #[value(name = "distil-large-v3.5")]
    DistilLargeV35,
}

/// What the engine should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Task {
    /// Transcribe in the source language.
    Transcribe,
    /// Translate speech to English.
    Translate,
}

/// Transcript serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Format {
    Txt,
    Srt,
    Vtt,
    Json,
    Tsv,
}

/// Audio decoder backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Decoder {
    Symphonia,
    Ffmpeg,
}

/// `Display` via each enum's `ValueEnum` name, so resolved config and the model
/// list print the exact value users pass on the command line.
macro_rules! display_via_value_enum {
    ($($t:ty),+ $(,)?) => {$(
        impl std::fmt::Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.to_possible_value() {
                    Some(value) => f.write_str(value.get_name()),
                    // Unreachable: every derived variant names a value. Render the
                    // Debug name rather than silently emitting nothing.
                    None => write!(f, "{self:?}"),
                }
            }
        }
    )+};
}

display_via_value_enum!(Model, Task, Format, Decoder);

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn effective_formats_dedups_preserving_order() {
        let cli = Cli::try_parse_from(["scrybe", "--format", "txt,srt,txt", "x.wav"]).unwrap();
        assert_eq!(cli.effective_formats(), vec![Format::Txt, Format::Srt]);
        // --json overrides --format entirely.
        let json = Cli::try_parse_from(["scrybe", "--json", "--format", "srt", "x.wav"]).unwrap();
        assert_eq!(json.effective_formats(), vec![Format::Json]);
    }

    #[test]
    fn model_display_round_trips_through_value_enum() {
        // Pins the macro's invariant: every variant names a value, so the Display
        // never hits its unreachable Debug-name arm. A `#[value(skip)]` or rename
        // that breaks the round-trip fails here.
        for model in Model::value_variants() {
            assert_eq!(
                model.to_string(),
                model.to_possible_value().unwrap().get_name()
            );
        }
    }
}
