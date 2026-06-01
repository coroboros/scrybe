//! Resample mono f32 PCM to 16 kHz with rubato's FFT synchronous resampler.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use super::TARGET_SAMPLE_RATE;

/// Resample a mono signal from `src_rate` to 16 kHz. A no-op when already at the
/// target rate. Single-channel data is laid out as a flat (interleaved-by-1)
/// buffer. Errors are returned as a message; the caller attaches the path.
pub fn to_16k_mono(mono: &[f32], src_rate: u32) -> Result<Vec<f32>, String> {
    if src_rate == TARGET_SAMPLE_RATE {
        return Ok(mono.to_vec());
    }
    if mono.is_empty() {
        return Ok(Vec::new());
    }

    let fail = |detail: String| {
        format!("resample {src_rate} Hz → {TARGET_SAMPLE_RATE} Hz failed: {detail}")
    };

    let chunk = 1024;
    let sub_chunks = 2;
    let channels = 1;
    let mut resampler = Fft::<f32>::new(
        src_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        chunk,
        sub_chunks,
        channels,
        FixedSync::Both,
    )
    .map_err(|e| fail(e.to_string()))?;

    let in_frames = mono.len();
    let input =
        InterleavedSlice::new(mono, channels, in_frames).map_err(|e| fail(e.to_string()))?;

    let out_capacity = resampler.process_all_needed_output_len(in_frames);
    let mut out = vec![0.0f32; out_capacity];
    let mut output = InterleavedSlice::new_mut(&mut out, channels, out_capacity)
        .map_err(|e| fail(e.to_string()))?;

    let (_in_done, out_done) = resampler
        .process_all_into_buffer(&input, &mut output, in_frames, None)
        .map_err(|e| fail(e.to_string()))?;
    out.truncate(out_done);
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn downsamples_to_expected_length() {
        // 1 s of 44.1 kHz → ~16 k samples at 16 kHz (±2% for filter delay trim).
        let src_rate = 44_100;
        let input = vec![0.0f32; src_rate as usize];
        let out = to_16k_mono(&input, src_rate).unwrap();
        let expected = TARGET_SAMPLE_RATE as f64;
        let ratio = out.len() as f64 / expected;
        assert!(
            (0.98..=1.02).contains(&ratio),
            "len {} not ~{expected}",
            out.len()
        );
    }

    #[test]
    fn passthrough_at_target_rate() {
        let input = vec![0.1f32, 0.2, 0.3];
        assert_eq!(to_16k_mono(&input, TARGET_SAMPLE_RATE).unwrap(), input);
    }
}
