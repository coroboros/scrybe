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

/// 16 kHz mono f32 PCM plus provenance about the decoded source. The provenance
/// fields are a decode-observability contract the acceptance tests assert against
/// (source rate/channels), not used by the transcription path itself.
#[derive(Debug)]
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
    let source_sample_rate = decoded.sample_rate;
    let source_channels = decoded.channels;
    let mono = downmix(decoded.samples, source_channels);
    let samples = resample::to_16k_mono(mono, source_sample_rate)
        .map_err(|detail| ScrybeError::unsupported_codec(path, detail))?;
    Ok(AudioPcm {
        samples,
        source_sample_rate,
        source_channels,
    })
}

/// Average interleaved channels down to a single mono channel. Takes ownership so
/// the already-mono common case returns the buffer without a copy.
fn downmix(interleaved: Vec<f32>, channels: u16) -> Vec<f32> {
    let channels = usize::from(channels.max(1));
    if channels == 1 {
        return interleaved;
    }
    // Both decode paths yield whole interleaved frames — symphonia emits
    // frames × channels samples, the ffmpeg path forces mono (`-ac 1`) — so the
    // buffer length is a multiple of `channels` and `chunks_exact` discards nothing
    // real. A lone trailing sample only appears on malformed input, where dropping
    // the incomplete frame is safer than fabricating one.
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
        let stereo = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        assert_eq!(downmix(stereo, 2), vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn downmix_mono_is_passthrough() {
        let mono = vec![0.1, 0.2, 0.3];
        assert_eq!(downmix(mono.clone(), 1), mono);
    }

    #[test]
    fn downmix_averages_three_channels() {
        // One 3-channel frame [1,2,3] → mean 2.0; pins the general (>2 channels) arm.
        assert_eq!(downmix(vec![1.0, 2.0, 3.0], 3), vec![2.0]);
    }

    #[test]
    fn downmix_drops_trailing_partial_frame() {
        // `chunks_exact` ignores a lone trailing sample that doesn't complete a
        // stereo frame — documenting that intended behavior.
        let stereo = vec![1.0, 0.0, 0.0, 1.0, 0.5];
        assert_eq!(downmix(stereo, 2), vec![0.5, 0.5]);
    }
}
