//! Per-(chunk, local-speaker) mask selection for embedding extraction:
//! prefer the speaker's overlap-free frames, fall back to all their frames
//! when too little clean speech exists, then select the matching fbank rows
//! (pyannote's ONNX embedding path: fbank over the full window, hard frame
//! selection through the nearest-upsampled mask).

use super::fbank::NUM_MEL_BINS;
use super::frames::ChunkActivity;
use super::{NUM_FRAMES, NUM_LOCAL_SPEAKERS};

/// Minimum active segmentation frames for the overlap-free mask to be
/// trusted (pyannote derives this from a runtime probe of the embedding
/// model; sherpa hardcodes 10 — adopted here and pinned by the reference
/// fixtures).
const MIN_CLEAN_SEG_FRAMES: usize = 10;
/// Minimum selected fbank frames (~0.25 s of speech) to attempt an
/// embedding; below this, resnet34 statistics pooling is unreliable and the
/// slot is dropped from clustering instead.
pub(crate) const MIN_FBANK_FRAMES: usize = 25;

/// The 589-frame activity mask to embed speaker `spk` with: overlap-free
/// frames when enough exist, all their frames otherwise.
pub(crate) fn choose_mask(chunk: &ChunkActivity, spk: usize) -> Vec<bool> {
    debug_assert!(spk < NUM_LOCAL_SPEAKERS);
    let clean: Vec<bool> = chunk
        .iter()
        .map(|frame| frame[spk] && frame.iter().filter(|&&on| on).count() < 2)
        .collect();
    if clean.iter().filter(|&&on| on).count() > MIN_CLEAN_SEG_FRAMES {
        clean
    } else {
        chunk.iter().map(|frame| frame[spk]).collect()
    }
}

/// Keep the fbank rows whose nearest segmentation frame is active
/// (`torch.nn.functional.interpolate(mode="nearest")` index mapping).
pub(crate) fn select_frames(
    features: &[[f32; NUM_MEL_BINS]],
    mask: &[bool],
) -> Vec<[f32; NUM_MEL_BINS]> {
    debug_assert_eq!(mask.len(), NUM_FRAMES);
    let total = features.len();
    features
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let src = i * NUM_FRAMES / total;
            mask[src.min(NUM_FRAMES - 1)]
        })
        .map(|(_, row)| *row)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_with(rows: &[(usize, [bool; NUM_LOCAL_SPEAKERS])]) -> ChunkActivity {
        let mut chunk = vec![[false; NUM_LOCAL_SPEAKERS]; NUM_FRAMES];
        for &(i, frame) in rows {
            chunk[i] = frame;
        }
        chunk
    }

    #[test]
    fn clean_mask_excludes_overlap_when_plentiful() {
        // Speaker 0: 30 solo frames + 20 overlapped with speaker 1.
        let mut rows = Vec::new();
        for i in 0..30 {
            rows.push((i, [true, false, false]));
        }
        for i in 30..50 {
            rows.push((i, [true, true, false]));
        }
        let chunk = chunk_with(&rows);
        let mask = choose_mask(&chunk, 0);
        assert_eq!(
            mask.iter().filter(|&&on| on).count(),
            30,
            "overlap frames excluded"
        );
    }

    #[test]
    fn falls_back_to_full_mask_when_clean_speech_is_scarce() {
        // Speaker 0: 5 solo frames (≤ the gate) + 40 overlapped.
        let mut rows = Vec::new();
        for i in 0..5 {
            rows.push((i, [true, false, false]));
        }
        for i in 5..45 {
            rows.push((i, [true, true, false]));
        }
        let chunk = chunk_with(&rows);
        let mask = choose_mask(&chunk, 0);
        assert_eq!(
            mask.iter().filter(|&&on| on).count(),
            45,
            "fallback keeps every frame"
        );
    }

    #[test]
    fn select_frames_maps_nearest_upsampled_indices() {
        // 998 fbank rows against the 589-frame mask: activate segmentation
        // frames [100, 200) and check the selected fbank row span.
        let features = vec![[0.0; NUM_MEL_BINS]; 998];
        let mut mask = vec![false; NUM_FRAMES];
        for m in mask.iter_mut().take(200).skip(100) {
            *m = true;
        }
        let selected = select_frames(&features, &mask);
        // fbank row i maps to seg frame floor(i*589/998): active for
        // i in [ceil(100*998/589), ceil(200*998/589)) = [170, 339).
        assert_eq!(selected.len(), 339 - 170);
    }

    #[test]
    fn empty_mask_selects_nothing() {
        let features = vec![[0.0; NUM_MEL_BINS]; 998];
        let mask = vec![false; NUM_FRAMES];
        assert!(select_frames(&features, &mask).is_empty());
    }
}
