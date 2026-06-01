//! Audio ingest: discover inputs, decode to PCM, resample to 16 kHz mono.
//!
//! Whisper expects 16 kHz mono f32 PCM. [`load_audio`] turns any supported file
//! into exactly that, failing loud on unsupported codecs.

mod decode;
mod discovery;
mod resample;

use std::path::Path;

pub use discovery::discover;

use crate::cli::Decoder;
use crate::error::ScrybeError;

/// Whisper's required input sample rate.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// 16 kHz mono f32 PCM plus provenance about the decoded source.
pub struct AudioPcm {
    pub samples: Vec<f32>,
    pub source_sample_rate: u32,
    pub source_channels: u16,
}

impl AudioPcm {
    /// Duration in seconds at the target sample rate.
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / f64::from(TARGET_SAMPLE_RATE)
    }
}

/// Decode `path` and resample to 16 kHz mono f32 PCM via the chosen backend.
pub fn load_audio(path: &Path, decoder: Decoder) -> Result<AudioPcm, ScrybeError> {
    let decoded = match decoder {
        Decoder::Symphonia => decode::decode_file(path)?,
        Decoder::Ffmpeg => decode::decode_via_ffmpeg(path)?,
    };
    let mono = downmix(&decoded.samples, decoded.channels);
    let samples = resample::to_16k_mono(&mono, decoded.sample_rate).map_err(|detail| {
        ScrybeError::UnsupportedCodec {
            path: path.to_path_buf(),
            detail,
        }
    })?;
    Ok(AudioPcm {
        samples,
        source_sample_rate: decoded.sample_rate,
        source_channels: decoded.channels,
    })
}

/// Average interleaved channels down to a single mono channel.
fn downmix(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let channels = usize::from(channels.max(1));
    if channels == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn downmix_averages_stereo_to_mono() {
        // L/R interleaved: (1.0,0.0),(0.0,1.0),(0.5,0.5) → 0.5,0.5,0.5
        let stereo = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        assert_eq!(downmix(&stereo, 2), vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn downmix_mono_is_passthrough() {
        let mono = [0.1, 0.2, 0.3];
        assert_eq!(downmix(&mono, 1), mono);
    }
}
