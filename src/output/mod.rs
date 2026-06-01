//! Transcript serialization: `txt`, `srt`, `vtt`, `json`, `tsv`.
//!
//! Subtitle timestamps are sanitized so they are never negative and never
//! overlap. JSON is a stable, versioned schema for downstream tooling.

use std::fmt::Write as _;
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

/// Find two distinct inputs that would write the same output file (e.g. `a.wav`
/// and `a.m4a` → `a.txt`, or `--out-dir` flattening equal-stem files from
/// different folders). Returns an actionable message for the first collision, or
/// `None` when every output is unique — so the caller can fail loud rather than
/// silently overwrite.
pub fn first_collision(
    inputs: &[PathBuf],
    formats: &[Format],
    out_dir: Option<&Path>,
) -> Option<String> {
    let mut seen: std::collections::HashMap<PathBuf, &Path> = std::collections::HashMap::new();
    for input in inputs {
        for &format in formats {
            let out = output_path(input, format, out_dir);
            match seen.get(&out) {
                Some(&prev) if prev != input.as_path() => {
                    return Some(format!(
                        "{} and {} both write {}",
                        prev.display(),
                        input.display(),
                        out.display(),
                    ));
                }
                _ => {
                    seen.insert(out, input);
                }
            }
        }
    }
    None
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
        // Writing into a String is infallible; the trailing blank line ends the cue.
        let _ = writeln!(
            out,
            "{}\n{} --> {}\n{}\n",
            index + 1,
            timestamp(segment.start, ','),
            timestamp(segment.end, ','),
            segment.text,
        );
    }
    out
}

fn render_vtt(transcript: &Transcript) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for segment in sanitized(&transcript.segments) {
        let _ = writeln!(
            out,
            "{} --> {}\n{}\n",
            timestamp(segment.start, '.'),
            timestamp(segment.end, '.'),
            segment.text,
        );
    }
    out
}

fn render_tsv(transcript: &Transcript) -> String {
    let mut out = String::from("start\tend\ttext\n");
    for segment in sanitized(&transcript.segments) {
        let _ = writeln!(
            out,
            "{}\t{}\t{}",
            (segment.start * 1000.0).round() as i64,
            (segment.end * 1000.0).round() as i64,
            segment.text,
        );
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
    // Serializing these owned plain types is infallible; the arm is unreachable.
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
    fn sanitized_clamps_and_orders_timestamps() {
        // The shared timestamp guard behind every format (SRT only asserts it via
        // rendered text). Negative start clamped, end >= start, cues non-overlapping.
        let messy = [
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
        ];
        let timed = sanitized(&messy);
        assert_eq!(timed[0].start, 0.0, "negative start clamped");
        assert!(timed[0].end >= timed[0].start);
        assert!(timed[1].start >= timed[0].end, "cues never overlap");
        assert!(timed[1].end >= timed[1].start);
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

    #[test]
    fn detects_output_collisions() {
        let wav = PathBuf::from("/x/talk.wav");
        let m4a = PathBuf::from("/x/talk.m4a");
        // Same stem, different container → both write talk.txt.
        assert!(first_collision(&[wav.clone(), m4a], &[Format::Txt], None).is_some());
        // Distinct stems → no collision.
        let other = PathBuf::from("/x/other.wav");
        assert!(first_collision(&[wav, other], &[Format::Txt], None).is_none());
        // --out-dir flattening equal-stem files from different folders.
        let ep1 = PathBuf::from("/ep01/audio.mp3");
        let ep2 = PathBuf::from("/ep02/audio.mp3");
        assert!(first_collision(&[ep1, ep2], &[Format::Json], Some(Path::new("/out"))).is_some());
    }

    #[test]
    fn vtt_has_header_and_dot_separator() {
        let vtt = render_vtt(&transcript());
        assert!(vtt.starts_with("WEBVTT\n\n"), "missing header:\n{vtt}");
        assert!(vtt.contains("00:00:00.000 --> 00:00:01.500"));
        assert!(vtt.contains("Hello world"));
    }

    #[test]
    fn tsv_has_header_and_millisecond_rows() {
        let tsv = render_tsv(&transcript());
        assert!(tsv.starts_with("start\tend\ttext\n"));
        assert!(tsv.contains("0\t1500\tHello world"));
        assert!(tsv.contains("1500\t3000\tsecond line"));
    }

    #[test]
    fn output_path_extension_and_directory() {
        let input = Path::new("/a/b/clip.wav");
        assert_eq!(
            output_path(input, Format::Txt, None),
            PathBuf::from("/a/b/clip.txt")
        );
        assert_eq!(
            output_path(input, Format::Json, Some(Path::new("/out"))),
            PathBuf::from("/out/clip.json"),
        );
        // Same stem, different formats → different files.
        assert_ne!(
            output_path(input, Format::Srt, None),
            output_path(input, Format::Vtt, None),
        );
        // A no-extension input keeps its whole name as the stem.
        assert_eq!(
            output_path(Path::new("/a/clip"), Format::Tsv, None),
            PathBuf::from("/a/clip.tsv")
        );
    }

    #[test]
    fn outputs_up_to_date_tracks_mtimes_and_missing() {
        use std::time::{Duration, SystemTime};
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("clip.wav");
        std::fs::write(&input, b"x").unwrap();
        let formats = [Format::Txt];

        // No output yet → stale.
        assert!(!outputs_up_to_date(&input, &formats, None));

        let out = output_path(&input, Format::Txt, None);
        std::fs::write(&out, b"y").unwrap();
        let touch = |path: &Path, when: SystemTime| {
            std::fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(when)
                .unwrap();
        };

        // Output older than input → stale.
        touch(&out, SystemTime::now() - Duration::from_secs(60));
        assert!(!outputs_up_to_date(&input, &formats, None));

        // Output newer than input → up to date.
        touch(&out, SystemTime::now() + Duration::from_secs(60));
        assert!(outputs_up_to_date(&input, &formats, None));

        // A nonexistent input has no metadata → stale (never skip).
        assert!(!outputs_up_to_date(
            Path::new("/no/such/input.wav"),
            &formats,
            None
        ));
    }

    #[test]
    fn outputs_up_to_date_respects_out_dir() {
        use std::time::{Duration, SystemTime};
        let in_dir = tempfile::tempdir().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let input = in_dir.path().join("clip.wav");
        std::fs::write(&input, b"x").unwrap();
        let formats = [Format::Txt];

        // No output in the out-dir → stale.
        assert!(!outputs_up_to_date(&input, &formats, Some(out_dir.path())));

        // Present in the out-dir and newer than the input → up to date.
        let out = output_path(&input, Format::Txt, Some(out_dir.path()));
        std::fs::write(&out, b"y").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&out)
            .unwrap()
            .set_modified(SystemTime::now() + Duration::from_secs(60))
            .unwrap();
        assert!(outputs_up_to_date(&input, &formats, Some(out_dir.path())));
    }
}
