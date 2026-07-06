//! Sliding-window chunking, matching pyannote's `Inference.slide`: full 10 s
//! windows every 1 s, plus one zero-padded tail window whenever the last
//! samples don't land exactly on a window boundary. Files shorter than one
//! window become a single zero-padded chunk.

use super::{STEP_SAMPLES, WINDOW_SAMPLES};

/// The chunk layout for `num_samples` of 16 kHz mono audio.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ChunkPlan {
    pub full_chunks: usize,
    pub has_tail: bool,
}

impl ChunkPlan {
    pub(crate) fn new(num_samples: usize) -> Self {
        let full_chunks = if num_samples >= WINDOW_SAMPLES {
            (num_samples - WINDOW_SAMPLES) / STEP_SAMPLES + 1
        } else {
            0
        };
        let has_tail = num_samples < WINDOW_SAMPLES
            || !(num_samples - WINDOW_SAMPLES).is_multiple_of(STEP_SAMPLES);
        Self {
            full_chunks,
            has_tail,
        }
    }

    pub(crate) fn num_chunks(&self) -> usize {
        self.full_chunks + usize::from(self.has_tail)
    }

    /// Start sample of chunk `i` (tail chunk included).
    pub(crate) fn start(&self, i: usize) -> usize {
        i * STEP_SAMPLES
    }

    /// Copy chunk `i` of `samples` into a fixed window, zero-padding past the
    /// end of the audio.
    pub(crate) fn window(&self, samples: &[f32], i: usize) -> Vec<f32> {
        let start = self.start(i);
        let end = (start + WINDOW_SAMPLES).min(samples.len());
        let mut window = vec![0.0; WINDOW_SAMPLES];
        window[..end - start].copy_from_slice(&samples[start..end]);
        window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_window_has_no_tail() {
        let plan = ChunkPlan::new(WINDOW_SAMPLES);
        assert_eq!(
            plan,
            ChunkPlan {
                full_chunks: 1,
                has_tail: false
            }
        );
    }

    #[test]
    fn short_audio_is_one_padded_chunk() {
        let plan = ChunkPlan::new(WINDOW_SAMPLES / 2);
        assert_eq!(
            plan,
            ChunkPlan {
                full_chunks: 0,
                has_tail: true
            }
        );
        assert_eq!(plan.num_chunks(), 1);

        let window = plan.window(&vec![1.0; WINDOW_SAMPLES / 2], 0);
        assert_eq!(window.len(), WINDOW_SAMPLES);
        assert!(window[..WINDOW_SAMPLES / 2].iter().all(|&v| v == 1.0));
        assert!(window[WINDOW_SAMPLES / 2..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn one_extra_sample_adds_a_padded_tail() {
        let plan = ChunkPlan::new(WINDOW_SAMPLES + 1);
        assert_eq!(
            plan,
            ChunkPlan {
                full_chunks: 1,
                has_tail: true
            }
        );
        // The tail starts one step in and is padded to a full window.
        assert_eq!(plan.start(1), STEP_SAMPLES);
    }

    #[test]
    fn step_aligned_audio_has_no_tail() {
        // 20 s: (320000 - 160000) / 16000 + 1 = 11 full windows, remainder 0.
        let plan = ChunkPlan::new(2 * WINDOW_SAMPLES);
        assert_eq!(
            plan,
            ChunkPlan {
                full_chunks: 11,
                has_tail: false
            }
        );
    }

    #[test]
    fn empty_audio_is_one_silent_chunk() {
        // Degenerate but must not panic: zero samples → a single all-zero window.
        let plan = ChunkPlan::new(0);
        assert_eq!(plan.num_chunks(), 1);
        let window = plan.window(&[], 0);
        assert!(window.iter().all(|&v| v == 0.0));
    }
}
