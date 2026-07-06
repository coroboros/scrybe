//! The global frame grid and everything computed on it: instantaneous
//! speaker count, cluster-aligned reconstruction, and the final speaker
//! turns. Ports pyannote 3.1's conventions exactly: a 10/589 s frame grid
//! anchored at 0, `closest_frame` rounding, `np.rint` half-even rounding for
//! the count, sum (not average) support for reconstruction, and segment
//! boundaries on frame midpoints.

use super::{NUM_FRAMES, NUM_LOCAL_SPEAKERS, STEP_SAMPLES, Turn, WINDOW_SAMPLES};
use crate::audio::TARGET_SAMPLE_RATE;

/// One segmentation window, decoded to per-frame binary speaker activity.
pub(crate) type ChunkActivity = Vec<[bool; NUM_LOCAL_SPEAKERS]>;

/// pyannote 3.1's frame grid: 589 frames per 10 s window, starting at 0.
const FRAME_DURATION: f64 = WINDOW_SAMPLES as f64 / TARGET_SAMPLE_RATE as f64 / NUM_FRAMES as f64;
/// Window hop in seconds (1 s).
const STEP_SECS: f64 = STEP_SAMPLES as f64 / TARGET_SAMPLE_RATE as f64;

/// `SlidingWindow.closest_frame` for the frame grid: `rint(t/d - 0.5)`,
/// half-even like `np.rint`, floored at 0.
fn closest_frame(t: f64) -> usize {
    (t / FRAME_DURATION - 0.5).round_ties_even().max(0.0) as usize
}

/// Global frame count for `num_chunks` sliding windows (pyannote
/// `Inference.aggregate`'s output length).
pub(crate) fn num_global_frames(num_chunks: usize) -> usize {
    closest_frame(10.0 + (num_chunks.saturating_sub(1)) as f64 * STEP_SECS) + 1
}

/// Start frame of chunk `i` on the global grid.
fn chunk_start_frame(i: usize) -> usize {
    closest_frame(i as f64 * STEP_SECS)
}

/// Per-frame instantaneous speaker count: overlap-average of the per-chunk
/// speaker sums (uniform weights), rounded half-even.
pub(crate) fn speaker_count(chunks: &[ChunkActivity]) -> Vec<u32> {
    let total = num_global_frames(chunks.len());
    let mut sum = vec![0.0_f64; total];
    let mut den = vec![0.0_f64; total];
    for (i, chunk) in chunks.iter().enumerate() {
        let start = chunk_start_frame(i);
        for (f, frame) in chunk.iter().enumerate() {
            let speakers = frame.iter().filter(|&&on| on).count() as f64;
            sum[start + f] += speakers;
            den[start + f] += 1.0;
        }
    }
    sum.iter()
        .zip(&den)
        .map(|(&s, &d)| {
            if d == 0.0 {
                0
            } else {
                (s / d).round_ties_even() as u32
            }
        })
        .collect()
}

/// Rebuild the global binary (frame × cluster) activity: per chunk each
/// cluster column is the max over its local speakers, chunks contribute by
/// NaN-masked sum (absent clusters contribute nothing), and per frame the
/// top-`count` clusters by summed support win. Columns are padded past
/// `num_clusters` when the count demands more simultaneous speakers than
/// clustering produced (pyannote does the same).
pub(crate) fn reconstruct(
    chunks: &[ChunkActivity],
    labels: &[i32],
    num_clusters: usize,
    count: &[u32],
) -> Vec<Vec<bool>> {
    let max_count = count.iter().copied().max().unwrap_or(0) as usize;
    let num_columns = num_clusters.max(max_count);
    let total = count.len();

    let mut support = vec![vec![0.0_f64; num_columns]; total];
    for (i, chunk) in chunks.iter().enumerate() {
        let start = chunk_start_frame(i);
        let mut locals_by_cluster = vec![Vec::new(); num_clusters];
        for s in 0..NUM_LOCAL_SPEAKERS {
            let label = labels[i * NUM_LOCAL_SPEAKERS + s];
            if label >= 0 {
                locals_by_cluster[label as usize].push(s);
            }
        }
        for (k, locals) in locals_by_cluster.iter().enumerate() {
            if locals.is_empty() {
                continue; // absent cluster: masked out, adds nothing
            }
            for (f, frame) in chunk.iter().enumerate() {
                if locals.iter().any(|&s| frame[s]) {
                    support[start + f][k] += 1.0;
                }
            }
        }
    }

    let mut binary = vec![vec![false; num_columns]; total];
    // Reused across frames: the comparator is a strict total order, so re-sorting
    // the permuted buffer yields the same result as a fresh `0..num_columns`.
    let mut order: Vec<usize> = (0..num_columns).collect();
    for (t, row) in support.iter().enumerate() {
        // Descending support; ties resolve to the lower column index, like
        // np.argsort's stable sort on negated values.
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        for &k in order.iter().take(count[t] as usize) {
            binary[t][k] = true;
        }
    }
    binary
}

/// Frame runs → speaker turns, with boundaries on frame midpoints
/// (pyannote `Binarize` with `min_duration_on = min_duration_off = 0`).
/// Speakers are renumbered densely by first appearance in time.
pub(crate) fn to_turns(binary: &[Vec<bool>]) -> Vec<Turn> {
    let num_columns = binary.first().map_or(0, Vec::len);
    let midpoint = |i: usize| (i as f64 + 0.5) * FRAME_DURATION;

    let mut turns = Vec::new();
    for k in 0..num_columns {
        let mut start = midpoint(0);
        let mut is_active = binary[0][k];
        let mut last = start;
        for (t, row) in binary.iter().enumerate().skip(1) {
            last = midpoint(t);
            if is_active && !row[k] {
                turns.push(Turn {
                    start,
                    end: last,
                    speaker: k,
                });
                is_active = false;
            } else if !is_active && row[k] {
                start = last;
                is_active = true;
            }
        }
        if is_active {
            turns.push(Turn {
                start,
                end: last,
                speaker: k,
            });
        }
    }

    // (start, end, speaker) — the reference's key; `end` breaks same-start ties
    // before the id so first-appearance renumbering matches pyannote.
    turns.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.speaker.cmp(&b.speaker))
    });

    // Dense speaker ids ordered by first appearance.
    let mut remap = std::collections::HashMap::new();
    let mut next = 0usize;
    for turn in turns.iter_mut() {
        let id = *remap.entry(turn.speaker).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        turn.speaker = id;
    }
    turns
}

#[cfg(test)]
mod tests {
    use super::super::clustering::NO_CLUSTER;
    use super::*;

    fn silent_chunk() -> ChunkActivity {
        vec![[false; NUM_LOCAL_SPEAKERS]; NUM_FRAMES]
    }

    #[test]
    fn frame_grid_matches_pyannote_values() {
        // One 10 s chunk → exactly 589 frames; each extra 1 s step adds ~59.
        assert_eq!(num_global_frames(1), 589);
        assert_eq!(num_global_frames(2), 648);
        assert_eq!(chunk_start_frame(0), 0);
        assert_eq!(chunk_start_frame(1), 58);
        assert_eq!(chunk_start_frame(2), 117);
    }

    #[test]
    fn count_averages_across_overlapping_chunks() {
        // Chunk 0 sees one speaker everywhere; chunk 1 sees two. In the
        // overlap the average 1.5 rounds half-even to 2.
        let mut c0 = silent_chunk();
        for frame in c0.iter_mut() {
            frame[0] = true;
        }
        let mut c1 = silent_chunk();
        for frame in c1.iter_mut() {
            frame[0] = true;
            frame[1] = true;
        }
        let count = speaker_count(&[c0, c1]);
        assert_eq!(count.len(), 648);
        assert_eq!(count[0], 1, "only chunk 0 covers the run-in");
        assert_eq!(count[100], 2, "overlap averages 1.5 → half-even 2");
        assert_eq!(count[600], 2, "only chunk 1 covers the tail");
    }

    #[test]
    fn reconstruct_keeps_top_count_clusters() {
        // One chunk, local speakers 0 and 1 active on the same frames but
        // mapped to clusters 0 and 1; count = 1 keeps only one of them.
        let mut chunk = silent_chunk();
        for frame in chunk.iter_mut().take(100) {
            frame[0] = true;
            frame[1] = true;
        }
        let labels = vec![0, 1, NO_CLUSTER];
        // Count mirrors the segmentation (as in the real pipeline): one
        // speaker on the active frames, zero elsewhere.
        let mut count = vec![0u32; NUM_FRAMES];
        for c in count.iter_mut().take(100) {
            *c = 1;
        }
        let binary = reconstruct(&[chunk], &labels, 2, &count);
        let active_frames = binary.iter().filter(|row| row.iter().any(|&b| b)).count();
        assert_eq!(active_frames, 100);
        for row in binary.iter().take(100) {
            assert_eq!(
                row.iter().filter(|&&b| b).count(),
                1,
                "count caps active clusters"
            );
        }
    }

    #[test]
    fn zero_count_silences_everything() {
        let mut chunk = silent_chunk();
        for frame in chunk.iter_mut() {
            frame[0] = true;
        }
        let binary = reconstruct(
            &[chunk],
            &[0, NO_CLUSTER, NO_CLUSTER],
            1,
            &vec![0u32; NUM_FRAMES],
        );
        assert!(binary.iter().all(|row| row.iter().all(|&b| !b)));
    }

    #[test]
    fn turns_use_frame_midpoints_and_first_appearance_order() {
        // Cluster 1 speaks first (frames 10..20), cluster 0 second (30..40):
        // first-appearance numbering flips them.
        let total = 50;
        let mut binary = vec![vec![false; 2]; total];
        for row in binary.iter_mut().take(20).skip(10) {
            row[1] = true;
        }
        for row in binary.iter_mut().take(40).skip(30) {
            row[0] = true;
        }
        let turns = to_turns(&binary);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, 0, "first speaker in time gets id 0");
        assert_eq!(turns[1].speaker, 1);
        let expected_start = (10.0 + 0.5) * FRAME_DURATION;
        let expected_end = (20.0 + 0.5) * FRAME_DURATION;
        assert!((turns[0].start - expected_start).abs() < 1e-9);
        assert!((turns[0].end - expected_end).abs() < 1e-9);
    }

    #[test]
    fn silence_produces_no_turns() {
        let binary = vec![vec![false; 2]; 100];
        assert!(to_turns(&binary).is_empty());
    }
}
