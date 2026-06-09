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

/// Decode `path` to 16 kHz mono f32 PCM via the chosen backend.
pub fn load_audio(path: &Path, decoder: Decoder) -> Result<AudioPcm, ScrybeError> {
    match decoder {
        // symphonia streams decode → downmix → resample to 16 kHz mono itself, so the
        // source is never fully resident.
        Decoder::Symphonia => decode::decode_file(path),
        // ffmpeg already emits 16 kHz mono.
        Decoder::Ffmpeg => decode::decode_via_ffmpeg(path),
    }
}

/// Average one decoded packet's interleaved channels into mono, written into `out`
/// (cleared first, so the scratch buffer is reused across a streaming decode). Mono
/// input copies straight through.
fn downmix_into(interleaved: &[f32], channels: u16, out: &mut Vec<f32>) {
    let channels = usize::from(channels.max(1));
    out.clear();
    if channels == 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    // A decoded packet always holds whole interleaved frames (a multiple of
    // `channels`), so `chunks_exact` discards nothing real; a lone trailing sample
    // only appears on malformed input, where dropping the incomplete frame is safer
    // than fabricating one.
    out.extend(
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32),
    );
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn downmix(interleaved: &[f32], channels: u16) -> Vec<f32> {
        let mut out = Vec::new();
        downmix_into(interleaved, channels, &mut out);
        out
    }

    #[test]
    fn downmix_averages_stereo_to_mono() {
        // L/R interleaved: (1.0,0.0),(0.0,1.0),(0.5,0.5) → 0.5,0.5,0.5
        let stereo = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        assert_eq!(downmix(&stereo, 2), vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn downmix_mono_is_passthrough() {
        let mono = [0.1, 0.2, 0.3];
        assert_eq!(downmix(&mono, 1), mono.to_vec());
    }

    #[test]
    fn downmix_averages_three_channels() {
        // One 3-channel frame [1,2,3] → mean 2.0; pins the general (>2 channels) arm.
        assert_eq!(downmix(&[1.0, 2.0, 3.0], 3), vec![2.0]);
    }

    #[test]
    fn downmix_drops_trailing_partial_frame() {
        // `chunks_exact` ignores a lone trailing sample that doesn't complete a
        // stereo frame — documenting that intended behavior.
        let stereo = [1.0, 0.0, 0.0, 1.0, 0.5];
        assert_eq!(downmix(&stereo, 2), vec![0.5, 0.5]);
    }

    #[test]
    fn downmix_into_clears_the_scratch_between_packets() {
        // The reused scratch must not accumulate across packets.
        let mut out = vec![9.0, 9.0, 9.0];
        downmix_into(&[1.0, 0.0, 0.0, 1.0], 2, &mut out);
        assert_eq!(out, vec![0.5, 0.5], "previous contents must be cleared");
    }
}
