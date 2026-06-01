//! Transcript serialization: `txt`, `srt`, `vtt`, `json`, `tsv`.
//!
//! Subtitle timestamps are sanitized so they are never negative and never
//! overlap. JSON is a stable, versioned schema for downstream tooling.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::cli::Format;
use crate::engine::{Segment, Transcript};
use crate::error::ScrybeError;

/// Bumped when the JSON schema changes incompatibly.
const JSON_SCHEMA_VERSION: u32 = 1;

/// Metadata recorded alongside the transcript in JSON output.
pub struct Meta<'a> {
    pub model: &'a str,
    pub duration: f64,
}

/// Render a transcript to one format as a string.
pub fn render(transcript: &Transcript, format: Format, meta: &Meta<'_>) -> String {
    match format {
        Format::Txt => render_txt(transcript),
        Format::Srt => render_srt(transcript),
        Format::Vtt => render_vtt(transcript),
        Format::Json => render_json(transcript, meta),
        Format::Tsv => render_tsv(transcript),
    }
}

/// Write the transcript in each requested format next to `input` (or into
/// `out_dir`), returning the paths written.
pub fn write_outputs(
    transcript: &Transcript,
    input: &Path,
    formats: &[Format],
    out_dir: Option<&Path>,
    meta: &Meta<'_>,
) -> Result<Vec<PathBuf>, ScrybeError> {
    let mut written = Vec::new();
    for &format in formats {
        let path = output_path(input, format, out_dir);
        std::fs::write(&path, render(transcript, format, meta)).map_err(|e| ScrybeError::Io {
            detail: format!("could not write {}: {e}", path.display()),
        })?;
        written.push(path);
    }
    Ok(written)
}

/// Whether every requested output for `input` already exists and is at least as
/// new as the input — the signal to skip re-transcribing unless `--force`.
pub fn outputs_up_to_date(input: &Path, formats: &[Format], out_dir: Option<&Path>) -> bool {
    let Ok(input_mtime) = input.metadata().and_then(|m| m.modified()) else {
        return false;
    };
    formats.iter().all(|&format| {
        output_path(input, format, out_dir)
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|out_mtime| out_mtime >= input_mtime)
    })
}

/// The output path for one format: `<stem>.<ext>` in `out_dir` or beside the input.
fn output_path(input: &Path, format: Format, out_dir: Option<&Path>) -> PathBuf {
    let stem = input.file_stem().unwrap_or(input.as_os_str());
    let dir = out_dir.or_else(|| input.parent()).unwrap_or(Path::new("."));
    let mut name = stem.to_os_string();
    name.push(".");
    name.push(format.to_string());
    dir.join(name)
}

fn render_txt(transcript: &Transcript) -> String {
    let mut out = String::new();
    for segment in &transcript.segments {
        out.push_str(&segment.text);
        out.push('\n');
    }
    out
}

fn render_srt(transcript: &Transcript) -> String {
    let mut out = String::new();
    for (index, segment) in sanitized(&transcript.segments).into_iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            index + 1,
            timestamp(segment.start, ','),
            timestamp(segment.end, ','),
            segment.text,
        ));
    }
    out
}

fn render_vtt(transcript: &Transcript) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for segment in sanitized(&transcript.segments) {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            timestamp(segment.start, '.'),
            timestamp(segment.end, '.'),
            segment.text,
        ));
    }
    out
}

fn render_tsv(transcript: &Transcript) -> String {
    let mut out = String::from("start\tend\ttext\n");
    for segment in sanitized(&transcript.segments) {
        out.push_str(&format!(
            "{}\t{}\t{}\n",
            (segment.start * 1000.0).round() as i64,
            (segment.end * 1000.0).round() as i64,
            segment.text,
        ));
    }
    out
}

fn render_json(transcript: &Transcript, meta: &Meta<'_>) -> String {
    #[derive(Serialize)]
    struct Doc<'a> {
        schema_version: u32,
        model: &'a str,
        language: &'a str,
        duration: f64,
        segments: Vec<Seg<'a>>,
    }
    #[derive(Serialize)]
    struct Seg<'a> {
        start: f64,
        end: f64,
        text: &'a str,
    }

    let doc = Doc {
        schema_version: JSON_SCHEMA_VERSION,
        model: meta.model,
        language: &transcript.language,
        duration: meta.duration,
        segments: sanitized(&transcript.segments)
            .into_iter()
            .map(|s| Seg {
                start: s.start,
                end: s.end,
                text: s.text,
            })
            .collect(),
    };
    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_owned())
}

/// A timestamp-sanitized view of a segment: borrows the text, owns the times.
struct Timed<'a> {
    start: f64,
    end: f64,
    text: &'a str,
}

/// Clamp timestamps non-negative, keep `end >= start`, and push each start to at
/// least the previous end so subtitle cues never overlap.
fn sanitized(segments: &[Segment]) -> Vec<Timed<'_>> {
    let mut out = Vec::with_capacity(segments.len());
    let mut prev_end = 0.0_f64;
    for segment in segments {
        let start = segment.start.max(0.0).max(prev_end);
        let end = segment.end.max(start);
        prev_end = end;
        out.push(Timed {
            start,
            end,
            text: &segment.text,
        });
    }
    out
}

/// Format seconds as `HH:MM:SS<sep>mmm` (`,` for SRT, `.` for VTT).
fn timestamp(seconds: f64, sep: char) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn transcript() -> Transcript {
        Transcript {
            language: "en".to_owned(),
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 1.5,
                    text: "Hello world".to_owned(),
                },
                Segment {
                    start: 1.5,
                    end: 3.0,
                    text: "second line".to_owned(),
                },
            ],
        }
    }

    #[test]
    fn timestamp_format() {
        assert_eq!(timestamp(0.0, ','), "00:00:00,000");
        assert_eq!(timestamp(3661.25, '.'), "01:01:01.250");
        assert_eq!(timestamp(-5.0, ','), "00:00:00,000");
    }

    #[test]
    fn srt_has_no_negative_or_overlapping_timestamps() {
        let messy = Transcript {
            language: "en".to_owned(),
            segments: vec![
                Segment {
                    start: -1.0,
                    end: 2.0,
                    text: "a".to_owned(),
                },
                Segment {
                    start: 1.0,
                    end: 0.5,
                    text: "b".to_owned(),
                },
            ],
        };
        let srt = render_srt(&messy);
        // Negative start clamped to 0; second cue pushed to the first cue's end
        // (no overlap), with end >= start. The `-->` arrow is the only dash.
        assert!(srt.contains("00:00:00,000 --> 00:00:02,000"));
        assert!(srt.contains("00:00:02,000 --> 00:00:02,000"));
        assert_eq!(srt.matches('-').count(), 2 * 2, "only arrows carry dashes");
    }

    #[test]
    fn json_is_valid_and_versioned() {
        let json = render_json(
            &transcript(),
            &Meta {
                model: "tiny",
                duration: 3.0,
            },
        );
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["language"], "en");
        assert_eq!(value["model"], "tiny");
        assert_eq!(value["segments"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn txt_is_one_segment_per_line() {
        assert_eq!(render_txt(&transcript()), "Hello world\nsecond line\n");
    }
}
