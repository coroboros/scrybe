//! Kaldi-style 80-dim log-mel filterbank features for the wespeaker embedding
//! model, matching the reference recipe pyannote copied from wespeaker's
//! `infer_onnx.py`: waveform × 32768, dither 0, DC removal, pre-emphasis
//! 0.97, symmetric Hamming window, 25 ms / 10 ms frames padded to a 512-point
//! FFT, power spectrum, mel bins spanning 20 Hz to Nyquist, `log(max(e, ε))`,
//! and per-utterance cepstral mean subtraction.

use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

use crate::audio::TARGET_SAMPLE_RATE;

pub(crate) const NUM_MEL_BINS: usize = 80;
const FRAME_LENGTH: usize = 400; // 25 ms @ 16 kHz
const FRAME_SHIFT: usize = 160; // 10 ms
const FFT_SIZE: usize = 512; // frame length rounded up to a power of two
const PREEMPHASIS: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;
/// Kaldi's `high_freq = 0.0` resolves to the Nyquist frequency.
const HIGH_FREQ: f32 = TARGET_SAMPLE_RATE as f32 / 2.0;
/// Kaldi floors filterbank energies at f32 epsilon before the log.
const ENERGY_FLOOR: f32 = f32::EPSILON;

/// Compute mean-normalized fbank features for 16 kHz mono f32 PCM in
/// [-1, 1]. Returns one 80-dim row per frame (`snip_edges` frame count:
/// `(n - 400)/160 + 1`); fewer than 400 samples yield no frames.
pub(crate) fn fbank_cmn(samples: &[f32]) -> Vec<[f32; NUM_MEL_BINS]> {
    if samples.len() < FRAME_LENGTH {
        return Vec::new();
    }
    let num_frames = (samples.len() - FRAME_LENGTH) / FRAME_SHIFT + 1;

    let window: Vec<f32> = (0..FRAME_LENGTH)
        .map(|i| {
            0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_LENGTH - 1) as f32).cos()
        })
        .collect();
    let mel_banks = mel_banks();

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut fft_in = vec![0.0_f32; FFT_SIZE];
    let mut fft_out = vec![Complex::default(); FFT_SIZE / 2 + 1];
    let mut scratch = fft.make_scratch_vec();

    let mut features = vec![[0.0_f32; NUM_MEL_BINS]; num_frames];
    let mut frame = [0.0_f32; FRAME_LENGTH];
    for (out, start) in features.iter_mut().zip((0..).step_by(FRAME_SHIFT)) {
        // Waveform scaled to int16 range, as the reference recipe feeds kaldi.
        for (dst, &src) in frame.iter_mut().zip(&samples[start..start + FRAME_LENGTH]) {
            *dst = src * 32768.0;
        }

        let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
        for v in frame.iter_mut() {
            *v -= mean;
        }

        // Kaldi pre-emphasis: right-to-left in place; the first sample uses itself.
        for i in (1..FRAME_LENGTH).rev() {
            frame[i] -= PREEMPHASIS * frame[i - 1];
        }
        frame[0] -= PREEMPHASIS * frame[0];

        for (v, w) in frame.iter_mut().zip(&window) {
            *v *= w;
        }

        fft_in[..FRAME_LENGTH].copy_from_slice(&frame);
        fft_in[FRAME_LENGTH..].fill(0.0);
        // realfft only fails on wrong buffer lengths, which are fixed here.
        if fft
            .process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
            .is_err()
        {
            return Vec::new();
        }

        let mut power = [0.0_f32; FFT_SIZE / 2 + 1];
        for (p, c) in power.iter_mut().zip(&fft_out) {
            *p = c.re * c.re + c.im * c.im;
        }

        for (bin, bank) in out.iter_mut().zip(&mel_banks) {
            let energy: f32 = bank.iter().map(|&(k, w)| power[k] * w).sum();
            *bin = energy.max(ENERGY_FLOOR).ln();
        }
    }

    // Per-utterance cepstral mean subtraction, applied before any frame
    // selection (the reference computes fbank + CMN over the full window).
    let mut mean = [0.0_f32; NUM_MEL_BINS];
    for row in &features {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= num_frames as f32;
    }
    for row in features.iter_mut() {
        for (v, m) in row.iter_mut().zip(&mean) {
            *v -= m;
        }
    }
    features
}

/// Kaldi mel scale.
fn mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Sparse triangular mel filters: for each of the 80 bins, the (FFT bin,
/// weight) pairs over the first `FFT_SIZE/2` spectrum bins (Kaldi's banks
/// exclude the Nyquist bin).
fn mel_banks() -> Vec<Vec<(usize, f32)>> {
    let fft_bin_width = TARGET_SAMPLE_RATE as f32 / FFT_SIZE as f32;
    let mel_low = mel(LOW_FREQ);
    let mel_high = mel(HIGH_FREQ);
    let mel_delta = (mel_high - mel_low) / (NUM_MEL_BINS + 1) as f32;

    (0..NUM_MEL_BINS)
        .map(|b| {
            let left = mel_low + b as f32 * mel_delta;
            let center = left + mel_delta;
            let right = center + mel_delta;
            let mut bank = Vec::new();
            for k in 0..FFT_SIZE / 2 {
                let m = mel(k as f32 * fft_bin_width);
                if m > left && m < right {
                    let w = if m <= center {
                        (m - left) / mel_delta
                    } else {
                        (right - m) / mel_delta
                    };
                    bank.push((k, w));
                }
            }
            bank
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_follows_snip_edges() {
        assert!(fbank_cmn(&vec![0.0; 399]).is_empty());
        assert_eq!(fbank_cmn(&vec![0.1; 400]).len(), 1);
        assert_eq!(fbank_cmn(&vec![0.1; 16_000]).len(), 98);
        // The 10 s segmentation window yields the fbank length the embedding
        // mask is interpolated onto.
        assert_eq!(fbank_cmn(&vec![0.1; 160_000]).len(), 998);
    }

    #[test]
    fn pure_tone_energy_lands_in_the_right_mel_region() {
        // 1 kHz sine over the first half only — CMN cancels stationary
        // content, so the tone must be absent part of the time for its bin to
        // stand out. mel(1000) ≈ 1000 → nearest filter center is bin ≈ 27.
        let samples: Vec<f32> = (0..32_000)
            .map(|i| {
                if i < 16_000 {
                    (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.5
                } else {
                    0.0
                }
            })
            .collect();
        let features = fbank_cmn(&samples);
        let row = &features[10]; // well inside the tone half
        let peak = (0..NUM_MEL_BINS).max_by(|&a, &b| {
            row[a]
                .partial_cmp(&row[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let peak = peak.unwrap_or(0);
        assert!(
            (23..=31).contains(&peak),
            "1 kHz peak landed at mel bin {peak}"
        );
    }

    #[test]
    fn cmn_zeroes_the_utterance_mean() {
        let samples: Vec<f32> = (0..8_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.3)
            .collect();
        let features = fbank_cmn(&samples);
        for bin in 0..NUM_MEL_BINS {
            let mean: f32 = features.iter().map(|r| r[bin]).sum::<f32>() / features.len() as f32;
            assert!(mean.abs() < 1e-3, "bin {bin} mean {mean} not centered");
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let samples: Vec<f32> = (0..8_000)
            .map(|i| ((i % 331) as f32 / 331.0) - 0.5)
            .collect();
        assert_eq!(fbank_cmn(&samples), fbank_cmn(&samples));
    }
}
