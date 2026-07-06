//! Attach diarized speakers to a transcript: per segment and per word, the
//! speaker with the largest cumulative overlap wins (the WhisperX
//! convention). Segments are never split at speaker changes; a span no turn
//! overlaps keeps no speaker rather than guessing the nearest one — that is
//! how hallucinated text on silence stays unattributed.

use super::Turn;
use crate::engine::Transcript;

/// Cumulative overlap between `[start, end)` and each speaker's turns; the
/// speaker with the most overlap, or `None` when nothing overlaps.
fn dominant_speaker(turns: &[Turn], start: f64, end: f64) -> Option<usize> {
    let mut overlap: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for turn in turns {
        let shared = turn.end.min(end) - turn.start.max(start);
        if shared > 0.0 {
            *overlap.entry(turn.speaker).or_insert(0.0) += shared;
        }
    }
    overlap
        .into_iter()
        // Ties resolve to the lower speaker id, deterministically.
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(speaker, _)| speaker)
}

/// Label every segment (and every word, when present) with its dominant
/// diarized speaker.
pub fn assign_speakers(transcript: &mut Transcript, turns: &[Turn]) {
    for segment in &mut transcript.segments {
        segment.speaker = dominant_speaker(turns, segment.start, segment.end);
        for word in &mut segment.words {
            word.speaker = dominant_speaker(turns, word.start, word.end);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Segment, Word};

    fn turn(start: f64, end: f64, speaker: usize) -> Turn {
        Turn {
            start,
            end,
            speaker,
        }
    }

    fn segment(start: f64, end: f64) -> Segment {
        Segment {
            start,
            end,
            text: String::new(),
            words: vec![],
            speaker: None,
        }
    }

    #[test]
    fn segment_takes_the_speaker_with_most_overlap() {
        // Segment [0, 4): speaker 0 covers 1 s, speaker 1 covers 2.5 s.
        let turns = [turn(0.0, 1.0, 0), turn(1.5, 4.0, 1)];
        let mut transcript = Transcript {
            language: "en".into(),
            segments: vec![segment(0.0, 4.0)],
        };
        assign_speakers(&mut transcript, &turns);
        assert_eq!(transcript.segments[0].speaker, Some(1));
    }

    #[test]
    fn split_turns_of_one_speaker_accumulate() {
        // Speaker 0 twice for 1 s each vs speaker 1 once for 1.5 s: cumulative
        // overlap must win, not the single longest turn.
        let turns = [turn(0.0, 1.0, 0), turn(3.0, 4.0, 0), turn(1.0, 2.5, 1)];
        let mut transcript = Transcript {
            language: "en".into(),
            segments: vec![segment(0.0, 4.0)],
        };
        assign_speakers(&mut transcript, &turns);
        assert_eq!(transcript.segments[0].speaker, Some(0));
    }

    #[test]
    fn no_overlap_leaves_no_speaker() {
        // Hallucination-on-silence class: text where nobody was diarized must
        // never inherit the nearest speaker.
        let turns = [turn(10.0, 12.0, 0)];
        let mut transcript = Transcript {
            language: "en".into(),
            segments: vec![segment(0.0, 2.0)],
        };
        assign_speakers(&mut transcript, &turns);
        assert_eq!(transcript.segments[0].speaker, None);
    }

    #[test]
    fn words_are_labeled_individually() {
        // A segment spanning a speaker change keeps one dominant label, but
        // its words split correctly across the boundary.
        let turns = [turn(0.0, 2.0, 0), turn(2.0, 5.0, 1)];
        let mut seg = segment(0.0, 5.0);
        seg.words = vec![
            Word {
                start: 0.2,
                end: 1.8,
                text: "early".into(),
                speaker: None,
            },
            Word {
                start: 2.2,
                end: 4.8,
                text: "late".into(),
                speaker: None,
            },
        ];
        let mut transcript = Transcript {
            language: "en".into(),
            segments: vec![seg],
        };
        assign_speakers(&mut transcript, &turns);
        let seg = &transcript.segments[0];
        assert_eq!(seg.speaker, Some(1), "dominant speaker covers 3 of 5 s");
        assert_eq!(seg.words[0].speaker, Some(0));
        assert_eq!(seg.words[1].speaker, Some(1));
    }
}
