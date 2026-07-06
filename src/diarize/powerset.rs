//! Powerset decode for the segmentation model's 7-class head.
//!
//! Class order follows pyannote's `Powerset.build_mapping` (lexicographic
//! `itertools.combinations` over set sizes 0..=2 for 3 speakers):
//! {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}. The model ends in LogSoftmax, so
//! argmax over the 7 log-probabilities selects the active-speaker set.

use super::{NUM_LOCAL_SPEAKERS, POWERSET_CLASSES};

/// Powerset class → per-speaker activity, in pyannote's class order.
const MAPPING: [[bool; NUM_LOCAL_SPEAKERS]; POWERSET_CLASSES] = [
    [false, false, false],
    [true, false, false],
    [false, true, false],
    [false, false, true],
    [true, true, false],
    [true, false, true],
    [false, true, true],
];

/// Decode one frame: argmax over the 7 class scores → binary speaker activity.
/// Ties resolve to the first maximum, matching `torch.argmax`/`np.argmax`.
pub(crate) fn decode_frame(scores: &[f32; POWERSET_CLASSES]) -> [bool; NUM_LOCAL_SPEAKERS] {
    let mut best = 0;
    for (k, &v) in scores.iter().enumerate().skip(1) {
        if v > scores[best] {
            best = k;
        }
    }
    MAPPING[best]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_matches_pyannote_class_order() {
        // Pin the table itself: silence, singles in speaker order, then pairs
        // (0,1), (0,2), (1,2) — the lexicographic combinations order the
        // segmentation model was trained with. A permutation here would
        // scramble every downstream speaker assignment.
        let active: Vec<Vec<usize>> = MAPPING
            .iter()
            .map(|row| (0..3).filter(|&s| row[s]).collect())
            .collect();
        let expected: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ];
        assert_eq!(active, expected);
    }

    #[test]
    fn decode_picks_argmax_class() {
        let mut scores = [-10.0_f32; POWERSET_CLASSES];
        scores[4] = -0.1; // {0,1}
        assert_eq!(decode_frame(&scores), [true, true, false]);

        scores = [-10.0; POWERSET_CLASSES];
        scores[0] = -0.5; // silence wins
        assert_eq!(decode_frame(&scores), [false, false, false]);
    }

    #[test]
    fn decode_breaks_ties_on_first_maximum() {
        // np.argmax / torch.argmax return the first occurrence on ties.
        let scores = [-1.0_f32; POWERSET_CLASSES];
        assert_eq!(decode_frame(&scores), [false, false, false]);
    }
}
