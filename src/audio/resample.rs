//! Stream mono f32 PCM to 16 kHz with rubato's FFT synchronous resampler.
//!
//! The decoder pushes source-rate mono as packets arrive; only the 16 kHz output
//! accumulates, so the full-resolution source is never resident. The output is
//! bounded by a ceiling (derived from the per-job memory budget); beyond it the
//! decode fails loud, with `--decoder ffmpeg` as the alternative for a multi-hour
//! clip. A source already at 16 kHz is a passthrough — no resampler, no copy beyond
//! the accumulating buffer.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};

use super::TARGET_SAMPLE_RATE;

const F32_BYTES: u64 = 4;

/// Ceiling on resampled 16 kHz-mono output, in samples — the only large per-job
/// allocation now that decode streams. Derived from `model::DECODE_BUFFER` (the
/// per-job memory reservation) so the resident output cannot exceed what the guard
/// reserved. ~4.6 hours of 16 kHz mono.
pub(crate) const MAX_OUTPUT_SAMPLES: usize = (crate::model::DECODE_BUFFER / F32_BYTES) as usize;

// rubato FFT block-size tuning: ~1024-frame chunks split into 2 sub-chunks trade
// latency for throughput sensibly for whole-file offline resampling. Mono input,
// fixed input size so each call consumes a constant chunk and produces variable
// output — the natural shape for streaming.
const FFT_CHUNK: usize = 1024;
const FFT_SUB_CHUNKS: usize = 2;
const CHANNELS: usize = 1;

/// How long the input buffer may grow past the read cursor before it is compacted,
/// bounding the streaming window to a small multiple of the chunk size.
const COMPACT_AFTER_CHUNKS: usize = 64;

#[derive(Debug)]
pub enum ResampleError {
    Overflow,
    Failed(String),
}

/// Streaming mono → 16 kHz resampler. Push source-rate mono as it decodes; `finish`
/// flushes the filter tail and returns the 16 kHz output.
pub struct Resampler16k {
    /// `None` when the source is already 16 kHz (passthrough).
    inner: Option<Fft<f32>>,
    chunk: usize,
    out_max: usize,
    delay: usize,
    ratio: f64,
    in_buf: Vec<f32>,
    in_pos: usize,
    scratch: Vec<f32>,
    out: Vec<f32>,
    total_in: usize,
    ceiling: usize,
}

impl Resampler16k {
    pub fn new(src_rate: u32) -> Result<Self, ResampleError> {
        Self::with_ceiling(src_rate, MAX_OUTPUT_SAMPLES)
    }

    /// `new` with an injectable output-sample ceiling, so the overflow arm is testable
    /// with a tiny ceiling instead of a multi-GB allocation.
    fn with_ceiling(src_rate: u32, ceiling: usize) -> Result<Self, ResampleError> {
        if src_rate == TARGET_SAMPLE_RATE {
            return Ok(Self::passthrough(ceiling));
        }
        let inner = Fft::<f32>::new(
            src_rate as usize,
            TARGET_SAMPLE_RATE as usize,
            FFT_CHUNK,
            FFT_SUB_CHUNKS,
            CHANNELS,
            FixedSync::Input,
        )
        .map_err(|e| ResampleError::Failed(e.to_string()))?;
        let chunk = inner.input_frames_next();
        let out_max = inner.output_frames_max();
        let delay = inner.output_delay();
        Ok(Self {
            inner: Some(inner),
            chunk,
            out_max,
            delay,
            ratio: f64::from(TARGET_SAMPLE_RATE) / f64::from(src_rate),
            in_buf: Vec::new(),
            in_pos: 0,
            scratch: vec![0.0; out_max],
            out: Vec::new(),
            total_in: 0,
            ceiling,
        })
    }

    fn passthrough(ceiling: usize) -> Self {
        Self {
            inner: None,
            chunk: 0,
            out_max: 0,
            delay: 0,
            ratio: 1.0,
            in_buf: Vec::new(),
            in_pos: 0,
            scratch: Vec::new(),
            out: Vec::new(),
            total_in: 0,
            ceiling,
        }
    }

    pub fn push(&mut self, mono: &[f32]) -> Result<(), ResampleError> {
        if self.inner.is_none() {
            self.out.extend_from_slice(mono);
            return self.check_ceiling();
        }
        self.in_buf.extend_from_slice(mono);
        self.total_in += mono.len();
        while self.in_buf.len() - self.in_pos >= self.chunk {
            self.process(None)?;
            self.in_pos += self.chunk;
            if self.in_pos >= self.chunk * COMPACT_AFTER_CHUNKS {
                self.in_buf.drain(..self.in_pos);
                self.in_pos = 0;
            }
        }
        Ok(())
    }

    /// Flush the filter tail and return the 16 kHz output. Mirrors rubato's
    /// `process_all_into_buffer` — the body must track its delay/trim semantics.
    pub fn finish(mut self) -> Result<Vec<f32>, ResampleError> {
        if self.inner.is_none() {
            return Ok(self.out);
        }
        let expected = (self.ratio * self.total_in as f64).ceil() as usize;
        let remaining = self.in_buf.len() - self.in_pos;
        if remaining > 0 {
            self.in_buf.resize(self.in_pos + self.chunk, 0.0);
            self.process(Some(remaining))?;
            self.in_pos += self.chunk;
        }
        let target = self.delay + expected;
        while self.out.len() < target {
            self.in_buf.resize(self.in_pos + self.chunk, 0.0);
            self.in_buf[self.in_pos..self.in_pos + self.chunk].fill(0.0);
            let before = self.out.len();
            self.process(Some(0))?;
            self.in_pos += self.chunk;
            if self.out.len() == before {
                break; // resampler produced nothing; stop rather than spin
            }
        }
        let trim = self.delay.min(self.out.len());
        self.out.drain(..trim);
        self.out.truncate(expected);
        Ok(self.out)
    }

    /// Process exactly one input chunk at the read cursor; `partial` marks how many of
    /// the chunk's frames are real (the rest silence) for the tail and zero-pump.
    fn process(&mut self, partial: Option<usize>) -> Result<(), ResampleError> {
        let produced = {
            let inner = self
                .inner
                .as_mut()
                .ok_or_else(|| ResampleError::Failed("no resampler".to_owned()))?;
            let input = InterleavedSlice::new(
                &self.in_buf[self.in_pos..self.in_pos + self.chunk],
                CHANNELS,
                self.chunk,
            )
            .map_err(|e| ResampleError::Failed(e.to_string()))?;
            let mut output = InterleavedSlice::new_mut(&mut self.scratch, CHANNELS, self.out_max)
                .map_err(|e| ResampleError::Failed(e.to_string()))?;
            let indexing = partial.map(|len| Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: Some(len),
                active_channels_mask: None,
            });
            inner
                .process_into_buffer(&input, &mut output, indexing.as_ref())
                .map_err(|e| ResampleError::Failed(e.to_string()))?
                .1
        };
        self.out.extend_from_slice(&self.scratch[..produced]);
        self.check_ceiling()
    }

    fn check_ceiling(&self) -> Result<(), ResampleError> {
        if self.out.len() > self.ceiling {
            Err(ResampleError::Overflow)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Drives the same streaming path the decoder uses, so correctness tests double as streaming coverage.
    fn resample(mono: Vec<f32>, src_rate: u32) -> Result<Vec<f32>, ResampleError> {
        let mut r = Resampler16k::new(src_rate)?;
        r.push(&mono)?;
        r.finish()
    }

    #[test]
    fn downsamples_to_expected_length() {
        // 1 s of 44.1 kHz → ~16 k samples at 16 kHz (±2% for filter delay trim).
        let src_rate = 44_100;
        let input = vec![0.0f32; src_rate as usize];
        let out = resample(input, src_rate).unwrap();
        let ratio = out.len() as f64 / f64::from(TARGET_SAMPLE_RATE);
        assert!((0.98..=1.02).contains(&ratio), "len {} not ~16k", out.len());
    }

    #[test]
    fn upsamples_below_target_rate() {
        // 1 s of 8 kHz → ~16k samples: output larger than input (telephony upsample).
        let src_rate = 8_000;
        let input = vec![0.0f32; src_rate as usize];
        let out = resample(input, src_rate).unwrap();
        let ratio = out.len() as f64 / f64::from(TARGET_SAMPLE_RATE);
        assert!((0.98..=1.02).contains(&ratio), "len {} not ~16k", out.len());
    }

    #[test]
    fn passthrough_at_target_rate() {
        let input = vec![0.1f32, 0.2, 0.3];
        assert_eq!(resample(input.clone(), TARGET_SAMPLE_RATE).unwrap(), input);
    }

    #[test]
    fn streaming_in_chunks_matches_one_shot() {
        // The streaming contract: feeding the input in small packets must yield the
        // same output as one push. A chunking/tail/delay bug would diverge here.
        let src_rate = 44_100u32;
        let freq = 440.0f32;
        let input: Vec<f32> = (0..src_rate)
            .map(|n| (2.0 * std::f32::consts::PI * freq * n as f32 / src_rate as f32).sin())
            .collect();
        let one_shot = resample(input.clone(), src_rate).unwrap();

        let mut streamed = Resampler16k::new(src_rate).unwrap();
        for packet in input.chunks(333) {
            streamed.push(packet).unwrap();
        }
        let streamed = streamed.finish().unwrap();
        assert_eq!(streamed, one_shot, "streamed output must equal one-shot");
    }

    #[test]
    fn upsample_exceeding_ceiling_fails_loud() {
        // A small 8 kHz input upsamples to ~2x; a tiny ceiling (64 samples) makes the
        // Overflow arm fire before any large allocation.
        let mut r = Resampler16k::with_ceiling(8_000, 64).unwrap();
        let err = r.push(&vec![0.1f32; 4_000]).err().or_else(|| {
            Resampler16k::with_ceiling(8_000, 64)
                .unwrap()
                .finish()
                .err()
        });
        assert!(
            matches!(err, Some(ResampleError::Overflow)),
            "expected Overflow"
        );
    }

    #[test]
    fn handles_empty_and_sub_chunk_input() {
        assert!(resample(Vec::new(), 48_000).unwrap().is_empty());
        // Fewer samples than the FFT chunk size resample without panicking.
        let out = resample(vec![0.1f32; 10], 48_000).expect("sub-chunk input");
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
        let out = resample(input, src_rate).unwrap();
        let cycles = out.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        assert!(
            (900..=1100).contains(&cycles),
            "expected ~1000 cycles, got {cycles}"
        );
    }
}
