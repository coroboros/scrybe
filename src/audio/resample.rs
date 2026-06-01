//! Resample mono f32 PCM to 16 kHz with rubato's FFT synchronous resampler.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use super::TARGET_SAMPLE_RATE;

/// Resample a mono signal from `src_rate` to 16 kHz. A no-op when already at the
/// target rate (returns the input buffer unchanged). Single-channel data is laid
/// out as a flat (interleaved-by-1) buffer. Errors are returned as a message; the
/// caller attaches the path.
pub fn to_16k_mono(mono: Vec<f32>, src_rate: u32) -> Result<Vec<f32>, String> {
    if src_rate == TARGET_SAMPLE_RATE {
        return Ok(mono);
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
        InterleavedSlice::new(&mono, channels, in_frames).map_err(|e| fail(e.to_string()))?;

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
        let out = to_16k_mono(input, src_rate).unwrap();
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
        assert_eq!(
            to_16k_mono(input.clone(), TARGET_SAMPLE_RATE).unwrap(),
            input
        );
    }

    #[test]
    fn handles_empty_and_sub_chunk_input() {
        // Empty input short-circuits to an empty buffer.
        assert!(to_16k_mono(Vec::new(), 48_000).unwrap().is_empty());
        // Fewer samples than the FFT chunk size must resample without panicking.
        let out = to_16k_mono(vec![0.1f32; 10], 48_000).expect("sub-chunk input");
        assert!(
            out.len() <= 10,
            "tiny input yields a tiny output: {}",
            out.len()
        );
    }

    #[test]
    fn preserves_tone_frequency() {
        // One second of a 1 kHz tone at 48 kHz → 16 kHz: the frequency (≈1000
        // positive-going zero crossings) must survive, not just the length.
        let src_rate = 48_000u32;
        let freq = 1000.0f32;
        let input: Vec<f32> = (0..src_rate)
            .map(|n| (2.0 * std::f32::consts::PI * freq * n as f32 / src_rate as f32).sin())
            .collect();
        let out = to_16k_mono(input, src_rate).unwrap();
        let cycles = out.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        assert!(
            (900..=1100).contains(&cycles),
            "expected ~1000 cycles, got {cycles}"
        );
    }
}
