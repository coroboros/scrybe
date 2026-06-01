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

    /// Whisper model to use.
    #[arg(long, value_enum, default_value_t = DEFAULT_MODEL)]
    pub model: Model,

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

    /// Emit a single file's transcript as JSON on stdout.
    #[arg(long)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
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
    /// List the known Whisper models.
    List,
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
                    None => Ok(()),
                }
            }
        }
    )+};
}

display_via_value_enum!(Model, Task, Format, Decoder);
