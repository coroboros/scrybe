//! Transcript serialization: `txt`, `srt`, `vtt`, `json`, `tsv`, `csv`.
//!
//! Subtitle timestamps are sanitized so they are never negative and never
//! overlap. JSON is a stable, versioned schema for downstream tooling.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::cli::Format;
use crate::engine::{Segment, Transcript, Word};
use crate::error::ScrybeError;

/// Bumped when the JSON schema changes incompatibly.
const JSON_SCHEMA_VERSION: u32 = 1;

/// Metadata recorded alongside the transcript in JSON output.
pub struct Meta<'a> {
    pub model: &'a str,
    pub duration: f64,
    /// Whether `--diarize` ran. Drives the tsv/csv speaker column so a batch
    /// keeps one schema even when a file yields no speakers (empty cells, not a
    /// dropped column).
    pub diarized: bool,
}

pub fn render(transcript: &Transcript, format: Format, meta: &Meta<'_>) -> String {
    match format {
        Format::Txt => render_txt(transcript),
        Format::Srt => render_srt(transcript),
        Format::Vtt => render_vtt(transcript),
        Format::Json => render_json(transcript, meta),
        Format::Tsv => render_tsv(transcript, meta.diarized),
        Format::Csv => render_csv(transcript, meta.diarized),
    }
}

/// Write the transcript in each requested format next to `input` (or into
/// `out_dir`), returning the paths written. Each file lands via a same-dir
/// temp + atomic rename, so a hard kill mid-write can never leave a truncated
/// transcript behind. Rename replaces the destination inode: a symlinked
/// output path is swapped for a regular file — durability over in-place
/// overwrite, the right default for generated sidecars.
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
        let io_error = |e: std::io::Error| ScrybeError::Io {
            detail: format!("could not write {}: {e}", path.display()),
        };
        let mut tmp = path.clone().into_os_string();
        tmp.push(format!(".{}.tmp", std::process::id()));
        let tmp = PathBuf::from(tmp);
        // Write then rename; on any failure remove the temp so a full disk or a
        // permission fault never strands `<name>.<pid>.tmp` litter.
        let write_then_rename = std::fs::write(&tmp, render(transcript, format, meta))
            .and_then(|()| std::fs::rename(&tmp, &path));
        if let Err(e) = write_then_rename {
            let _ = std::fs::remove_file(&tmp);
            return Err(io_error(e));
        }
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

/// Find two distinct inputs that would write the same output file, returning an
/// actionable message for the first collision (else `None`) so the caller fails loud
/// rather than silently overwriting. Keyed by case-folded path: paths differing only
/// in case (`Audio.txt` vs `audio.txt`) collide on macOS/Windows. The ASCII fold
/// approximates their case rules; being wrong only over-refuses, never loses a transcript.
pub fn first_collision(
    inputs: &[PathBuf],
    formats: &[Format],
    out_dir: Option<&Path>,
) -> Option<String> {
    let mut seen: std::collections::HashMap<String, (PathBuf, &Path)> =
        std::collections::HashMap::new();
    for input in inputs {
        let input = input.as_path();
        for &format in formats {
            let out = output_path(input, format, out_dir);
            let key = out.to_string_lossy().to_lowercase();
            match seen.get(&key) {
                Some((prev_out, prev_in)) if *prev_in != input => {
                    let message = if *prev_out == out {
                        format!(
                            "{} and {} both write {}",
                            prev_in.display(),
                            input.display(),
                            out.display(),
                        )
                    } else {
                        format!(
                            "{} and {} write {} and {}, which collide on a case-insensitive filesystem",
                            prev_in.display(),
                            input.display(),
                            prev_out.display(),
                            out.display(),
                        )
                    };
                    return Some(message);
                }
                _ => {
                    seen.entry(key).or_insert((out, input));
                }
            }
        }
    }
    None
}

/// `<stem>.<ext>` in `out_dir`, else beside the input.
fn output_path(input: &Path, format: Format, out_dir: Option<&Path>) -> PathBuf {
    let stem = input.file_stem().unwrap_or(input.as_os_str());
    let dir = out_dir.or_else(|| input.parent()).unwrap_or(Path::new("."));
    let mut name = stem.to_os_string();
    name.push(".");
    name.push(format.to_string());
    dir.join(name)
}

/// Machine-format speaker label (`SPEAKER_00`), the WhisperX-compatible
/// spelling downstream tooling greps for.
fn speaker_label(speaker: usize) -> String {
    format!("SPEAKER_{speaker:02}")
}

/// Human-format speaker name (`Speaker 1`), used by txt/srt prefixes and VTT
/// voice tags.
fn speaker_name(speaker: usize) -> String {
    format!("Speaker {}", speaker + 1)
}

fn render_txt(transcript: &Transcript) -> String {
    let mut out = String::new();
    for segment in &transcript.segments {
        if let Some(speaker) = segment.speaker {
            out.push_str(&speaker_name(speaker));
            out.push_str(": ");
        }
        out.push_str(&segment.text);
        out.push('\n');
    }
    out
}

fn render_srt(transcript: &Transcript) -> String {
    let mut out = String::new();
    for (index, segment) in sanitized(&transcript.segments).into_iter().enumerate() {
        let text = match segment.speaker {
            Some(speaker) => format!("{}: {}", speaker_name(speaker), segment.text),
            None => segment.text.to_owned(),
        };
        // Writing into a String is infallible.
        let _ = writeln!(
            out,
            "{}\n{} --> {}\n{}\n",
            index + 1,
            timestamp(segment.start, ','),
            timestamp(segment.end, ','),
            text,
        );
    }
    out
}

fn render_vtt(transcript: &Transcript) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for segment in sanitized(&transcript.segments) {
        let text = match segment.speaker {
            // The WebVTT voice span: players label and style the speaker.
            Some(speaker) => format!("<v {}>{}", speaker_name(speaker), segment.text),
            None => segment.text.to_owned(),
        };
        let _ = writeln!(
            out,
            "{} --> {}\n{}\n",
            timestamp(segment.start, '.'),
            timestamp(segment.end, '.'),
            text,
        );
    }
    out
}

fn render_tsv(transcript: &Transcript, diarized: bool) -> String {
    let mut out = String::from(if diarized {
        "start\tend\ttext\tspeaker\n"
    } else {
        "start\tend\ttext\n"
    });
    for segment in sanitized(&transcript.segments) {
        let _ = write!(
            out,
            "{}\t{}\t{}",
            (segment.start * 1000.0).round() as i64,
            (segment.end * 1000.0).round() as i64,
            segment.text,
        );
        if diarized {
            let _ = write!(
                out,
                "\t{}",
                segment.speaker.map(speaker_label).unwrap_or_default()
            );
        }
        out.push('\n');
    }
    out
}

fn render_csv(transcript: &Transcript, diarized: bool) -> String {
    let mut out = String::from(if diarized {
        "start,end,text,speaker\n"
    } else {
        "start,end,text\n"
    });
    for segment in sanitized(&transcript.segments) {
        let _ = write!(
            out,
            "{},{},{}",
            (segment.start * 1000.0).round() as i64,
            (segment.end * 1000.0).round() as i64,
            csv_field(segment.text),
        );
        if diarized {
            let _ = write!(
                out,
                ",{}",
                segment.speaker.map(speaker_label).unwrap_or_default()
            );
        }
        out.push('\n');
    }
    out
}

/// Quote a CSV text field per RFC 4180: wrap in double quotes and double any inner
/// quote, so a comma, quote, or newline in the transcript cannot break the columns.
fn csv_field(text: &str) -> String {
    format!("\"{}\"", text.replace('"', "\"\""))
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
        // Optional, additive — present only with `--diarize`, so existing
        // consumers and `schema_version` are unaffected when absent.
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker: Option<String>,
        // Optional, additive — present only with word timestamps (JSON output), so
        // existing consumers and `schema_version` are unaffected when absent.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        words: Vec<WordOut<'a>>,
    }
    #[derive(Serialize)]
    struct WordOut<'a> {
        start: f64,
        end: f64,
        text: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker: Option<String>,
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
                speaker: s.speaker.map(speaker_label),
                words: s
                    .words
                    .iter()
                    .map(|w| WordOut {
                        start: w.start,
                        end: w.end,
                        text: &w.text,
                        speaker: w.speaker.map(speaker_label),
                    })
                    .collect(),
            })
            .collect(),
    };
    // Serializing these owned plain types is infallible; the arm is unreachable.
    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_owned())
}

/// A timestamp-sanitized view of a segment: borrows the text and words, owns the times.
struct Timed<'a> {
    start: f64,
    end: f64,
    text: &'a str,
    words: &'a [Word],
    speaker: Option<usize>,
}

/// Clamp timestamps non-negative, keep `end >= start`, and push each start to at
/// least the previous end so subtitle cues never overlap. Per-word timing is carried
/// through untouched (only JSON reads it; the subtitle/text writers ignore it).
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
            words: &segment.words,
            speaker: segment.speaker,
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
                    speaker: None,
                    words: vec![
                        Word {
                            start: 0.0,
                            end: 0.7,
                            text: "Hello".to_owned(),
                            speaker: None,
                        },
                        Word {
                            start: 0.7,
                            end: 1.5,
                            text: "world".to_owned(),
                            speaker: None,
                        },
                    ],
                },
                Segment {
                    start: 1.5,
                    end: 3.0,
                    text: "second line".to_owned(),
                    words: vec![],
                    speaker: None,
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
                    words: vec![],
                    speaker: None,
                },
                Segment {
                    start: 1.0,
                    end: 0.5,
                    text: "b".to_owned(),
                    words: vec![],
                    speaker: None,
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
                words: vec![],
                speaker: None,
            },
            Segment {
                start: 1.0,
                end: 0.5,
                text: "b".to_owned(),
                words: vec![],
                speaker: None,
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
                diarized: false,
            },
        );
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["language"], "en");
        assert_eq!(value["model"], "tiny");
        assert_eq!(value["duration"], 3.0);
        assert_eq!(value["segments"].as_array().unwrap().len(), 2);
        // Pin the per-segment keys too: a rename or dropped field is an incompatible
        // schema change and must force a JSON_SCHEMA_VERSION bump, not pass silently.
        let seg = &value["segments"][0];
        assert_eq!(seg["start"], 0.0);
        assert_eq!(seg["end"], 1.5);
        assert_eq!(seg["text"], "Hello world");
        // Word-level timing is emitted when present, as an additive array...
        let words = seg["words"]
            .as_array()
            .expect("first segment carries words");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0]["text"], "Hello");
        assert_eq!(words[0]["start"], 0.0);
        assert_eq!(words[0]["end"], 0.7);
        // ...and omitted entirely when a segment has none, so the schema stays stable.
        assert!(
            value["segments"][1].get("words").is_none(),
            "a word-less segment must not emit an empty words array"
        );
    }

    #[test]
    fn txt_is_one_segment_per_line() {
        assert_eq!(render_txt(&transcript()), "Hello world\nsecond line\n");
    }

    #[test]
    fn detects_output_collisions() {
        let wav = PathBuf::from("/x/talk.wav");
        let m4a = PathBuf::from("/x/talk.m4a");
        assert!(first_collision(&[wav.clone(), m4a], &[Format::Txt], None).is_some());
        let other = PathBuf::from("/x/other.wav");
        assert!(first_collision(&[wav, other], &[Format::Txt], None).is_none());
        let ep1 = PathBuf::from("/ep01/audio.mp3");
        let ep2 = PathBuf::from("/ep02/audio.mp3");
        assert!(first_collision(&[ep1, ep2], &[Format::Json], Some(Path::new("/out"))).is_some());
    }

    #[test]
    fn detects_case_only_collisions() {
        // Two inputs whose outputs differ only in case (Audio.txt vs audio.txt) would
        // overwrite each other on a case-insensitive filesystem (macOS/Windows). The
        // guard must catch it and say *why*, distinct from the exact-collision message.
        let upper = PathBuf::from("/in/Audio.wav");
        let lower = PathBuf::from("/in/audio.m4a");
        let msg = first_collision(&[upper, lower], &[Format::Txt], None)
            .expect("case-only outputs must be flagged");
        assert!(
            msg.contains("case-insensitive"),
            "case collision must explain the trigger: {msg}"
        );
        // Outputs that differ beyond case are independent — no false positive.
        let a = PathBuf::from("/in/alpha.wav");
        let b = PathBuf::from("/in/beta.wav");
        assert!(first_collision(&[a, b], &[Format::Txt], None).is_none());
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
        let tsv = render_tsv(&transcript(), false);
        assert!(tsv.starts_with("start\tend\ttext\n"));
        assert!(tsv.contains("0\t1500\tHello world"));
        assert!(tsv.contains("1500\t3000\tsecond line"));
    }

    #[test]
    fn csv_has_header_and_quotes_fields() {
        let csv = render_csv(&transcript(), false);
        assert!(csv.starts_with("start,end,text\n"));
        assert!(csv.contains("0,1500,\"Hello world\""));
        // A comma or quote in the text must not break the columns.
        assert_eq!(csv_field("a, \"b\""), "\"a, \"\"b\"\"\"");
    }

    #[test]
    fn tabular_speaker_column_follows_the_flag_not_the_data() {
        // Column presence tracks --diarize, not whether speakers were found, so a
        // batch keeps one schema: a diarized-but-speakerless transcript still emits
        // the column (empty cells), never a narrower row than its siblings.
        let speakerless = transcript(); // both segments have speaker: None
        let tsv = render_tsv(&speakerless, true);
        assert!(tsv.starts_with("start\tend\ttext\tspeaker\n"));
        assert!(
            tsv.lines().nth(1).unwrap().ends_with('\t'),
            "empty speaker cell"
        );
        let csv = render_csv(&speakerless, true);
        assert!(csv.starts_with("start,end,text,speaker\n"));
        assert!(
            csv.lines().nth(1).unwrap().ends_with(','),
            "empty speaker cell"
        );
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
        assert_ne!(
            output_path(input, Format::Srt, None),
            output_path(input, Format::Vtt, None),
        );
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

        touch(&out, SystemTime::now() - Duration::from_secs(60));
        assert!(!outputs_up_to_date(&input, &formats, None));

        touch(&out, SystemTime::now() + Duration::from_secs(60));
        assert!(outputs_up_to_date(&input, &formats, None));

        // Missing input → stale, never skip.
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

        assert!(!outputs_up_to_date(&input, &formats, Some(out_dir.path())));

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
