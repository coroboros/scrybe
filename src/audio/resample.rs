//! Resample mono f32 PCM to 16 kHz with rubato's FFT synchronous resampler.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use super::TARGET_SAMPLE_RATE;

/// Resample a mono signal from `src_rate` to 16 kHz. A no-op when already at the
/// target rate (returns the input buffer unchanged). Single-channel data is laid
/// out as a flat (interleaved-by-1) buffer. Errors are returned as a message; the
/// caller attaches the path.
pub fn to_16k_mono(mono: Vec<f32>, src_rate: u32) -> Result<Vec<f32>, String> {
    to_16k_mono_capped(mono, src_rate, crate::model::DECODE_BUFFER)
}

/// `to_16k_mono` with an injectable output-byte ceiling, so the overflow Err arm is
/// testable with a tiny ceiling instead of a multi-GB allocation.
fn to_16k_mono_capped(
    mono: Vec<f32>,
    src_rate: u32,
    ceiling_bytes: u64,
) -> Result<Vec<f32>, String> {
    if src_rate == TARGET_SAMPLE_RATE {
        return Ok(mono);
    }
    if mono.is_empty() {
        return Ok(Vec::new());
    }

    let fail = |detail: String| {
        format!("resample {src_rate} Hz → {TARGET_SAMPLE_RATE} Hz failed: {detail}")
    };

    // rubato FFT block-size tuning: 1024-frame blocks split into 2 sub-chunks trade
    // latency for throughput sensibly for whole-file offline resampling. Mono input.
    const FFT_CHUNK: usize = 1024;
    const FFT_SUB_CHUNKS: usize = 2;
    const CHANNELS: usize = 1;
    let mut resampler = Fft::<f32>::new(
        src_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        FFT_CHUNK,
        FFT_SUB_CHUNKS,
        CHANNELS,
        FixedSync::Both,
    )
    .map_err(|e| fail(e.to_string()))?;

    let in_frames = mono.len();
    let input =
        InterleavedSlice::new(&mono, CHANNELS, in_frames).map_err(|e| fail(e.to_string()))?;

    let out_capacity = resampler.process_all_needed_output_len(in_frames);
    // A low source rate upsamples a modest raw buffer into a much larger output;
    // bound it before the allocation so a crafted rate can't force a multi-GB alloc
    // the raw-PCM ceiling alone doesn't catch.
    if resample_output_too_large(out_capacity, ceiling_bytes) {
        return Err(fail(
            "resampled output exceeds the in-memory limit; use a shorter clip or `--decoder ffmpeg`"
                .to_owned(),
        ));
    }
    let mut out = vec![0.0f32; out_capacity];
    let mut output = InterleavedSlice::new_mut(&mut out, CHANNELS, out_capacity)
        .map_err(|e| fail(e.to_string()))?;

    let (_in_done, out_done) = resampler
        .process_all_into_buffer(&input, &mut output, in_frames, None)
        .map_err(|e| fail(e.to_string()))?;
    out.truncate(out_done);
    Ok(out)
}

/// Whether a resample would allocate an output buffer beyond `ceiling_bytes` — the
/// binding constraint when upsampling (output larger than the raw input the decode
/// ceiling bounds).
fn resample_output_too_large(out_frames: usize, ceiling_bytes: u64) -> bool {
    const F32_BYTES: u64 = 4;
    (out_frames as u64).saturating_mul(F32_BYTES) > ceiling_bytes
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
    fn upsamples_below_target_rate() {
        // 1 s of 8 kHz → ~16k samples: output larger than input, exercising the
        // real resampler's upsample path (8 kHz telephony is a common input).
        let src_rate = 8_000;
        let input = vec![0.0f32; src_rate as usize];
        let out = to_16k_mono(input, src_rate).unwrap();
        let ratio = out.len() as f64 / f64::from(TARGET_SAMPLE_RATE);
        assert!((0.98..=1.02).contains(&ratio), "len {} not ~16k", out.len());
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
    fn resample_output_ceiling_bounds_upsample_alloc() {
        assert!(!resample_output_too_large(10, 1000)); // 40 bytes ≤ 1000
        assert!(resample_output_too_large(1000, 1000)); // 4000 > 1000
        assert!(resample_output_too_large(usize::MAX, 1000)); // saturates, never wraps
    }

    #[test]
    fn upsample_exceeding_ceiling_fails_loud() {
        // A small 8 kHz input upsamples to ~2x; a tiny ceiling makes the Err arm fire
        // before any large allocation, carrying the actionable message.
        let err = to_16k_mono_capped(vec![0.1f32; 100], 8_000, 256).unwrap_err();
        assert!(
            err.contains("in-memory limit") || err.contains("--decoder ffmpeg"),
            "got: {err}"
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
