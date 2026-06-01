//! Decode audio files to interleaved f32 PCM.
//!
//! The default path is pure-Rust symphonia. HE-AAC/SBR is detected up front and
//! rejected with an actionable message (symphonia is AAC-LC only). The optional
//! ffmpeg path shells out for codecs symphonia cannot handle.

use std::path::Path;
use std::process::Command;

use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use super::TARGET_SAMPLE_RATE;
use crate::error::ScrybeError;

/// Interleaved f32 PCM plus the source stream's rate and channel count.
pub struct Decoded {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Decode with symphonia. Fails loud (no silent/garbled output) on unsupported
/// codecs, including HE-AAC, pointing the user at the ffmpeg escape.
pub fn decode_file(path: &Path) -> Result<Decoded, ScrybeError> {
    let unsupported = |detail: String| ScrybeError::UnsupportedCodec {
        path: path.to_path_buf(),
        detail,
    };

    let file = std::fs::File::open(path).map_err(|e| unsupported(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| unsupported(format!("could not read container: {e}")))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| unsupported("no audio track found".to_owned()))?;
    let track_id = track.id;

    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| unsupported("missing audio codec parameters".to_owned()))?;

    let sample_rate = audio_params
        .sample_rate
        .ok_or_else(|| unsupported("unknown sample rate".to_owned()))?;
    let channels = audio_params
        .channels
        .as_ref()
        .map_or(1, |ch| ch.count() as u16)
        .max(1);

    if let Some(extra) = audio_params.extra_data.as_deref()
        && is_he_aac_asc(extra)
    {
        return Err(unsupported(
            "HE-AAC/SBR is not supported by the built-in decoder".to_owned(),
        ));
    }

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| unsupported(format!("no decoder for this codec: {e}")))?;

    let mut samples = Vec::new();
    let mut chunk: Vec<f32> = Vec::new();
    loop {
        match format.next_packet() {
            Ok(Some(packet)) => {
                if packet.track_id != track_id {
                    continue;
                }
                match decoder.decode(&packet) {
                    Ok(decoded) => append_f32(&decoded, &mut chunk, &mut samples),
                    Err(SymphoniaError::DecodeError(_) | SymphoniaError::IoError(_)) => continue,
                    Err(e) => return Err(unsupported(format!("decode error: {e}"))),
                }
            }
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => break,
            Err(SymphoniaError::IoError(_)) => break,
            Err(e) => return Err(unsupported(format!("read error: {e}"))),
        }
    }

    if samples.is_empty() {
        return Err(unsupported("file contained no decodable audio".to_owned()));
    }
    Ok(Decoded {
        samples,
        sample_rate,
        channels,
    })
}

fn append_f32(decoded: &GenericAudioBufferRef<'_>, chunk: &mut Vec<f32>, out: &mut Vec<f32>) {
    decoded.copy_to_vec_interleaved(chunk);
    out.extend_from_slice(chunk);
}

/// Decode by shelling out to a system `ffmpeg`, producing 16 kHz mono f32 PCM
/// directly. The escape hatch for codecs symphonia cannot handle (e.g. HE-AAC).
pub fn decode_via_ffmpeg(path: &Path) -> Result<Decoded, ScrybeError> {
    let unsupported = |detail: String| ScrybeError::UnsupportedCodec {
        path: path.to_path_buf(),
        detail,
    };

    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar"])
        .arg(TARGET_SAMPLE_RATE.to_string())
        .arg("pipe:1")
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                unsupported("`--decoder ffmpeg` requested but ffmpeg is not on PATH".to_owned())
            } else {
                unsupported(format!("failed to run ffmpeg: {e}"))
            }
        })?;

    if !output.status.success() {
        return Err(unsupported(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    let samples = output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return Err(unsupported("ffmpeg produced no audio".to_owned()));
    }
    Ok(Decoded {
        samples,
        sample_rate: TARGET_SAMPLE_RATE,
        channels: 1,
    })
}

/// Big-endian bit reader over the AudioSpecificConfig byte stream.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn at(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    fn read(&mut self, n: usize) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            value = (value << 1) | u32::from(bit);
            self.pos += 1;
        }
        Some(value)
    }
}

/// The MPEG-4 audio object type, decoding the 5-bit value plus the `31` escape
/// to a 6-bit extended type (`32 + value`).
fn read_object_type(reader: &mut BitReader<'_>) -> Option<u32> {
    let aot = reader.read(5)?;
    if aot == 31 {
        Some(32 + reader.read(6)?)
    } else {
        Some(aot)
    }
}

/// SBR sync extension marker inside an AudioSpecificConfig (ISO 14496-3).
const SBR_SYNC_EXTENSION: u32 = 0x2B7;

/// Detect HE-AAC / HE-AACv2 from an AudioSpecificConfig. Covers both signalings:
/// explicit hierarchical (base object type 5 = SBR, 29 = PS) and backward-
/// compatible (base type 2 + the `0x2B7` sync extension declaring SBR/PS), which
/// is what Apple's encoder emits and what symphonia silently mis-decodes.
fn is_he_aac_asc(asc: &[u8]) -> bool {
    let mut reader = BitReader::at(asc, 0);
    match read_object_type(&mut reader) {
        Some(5 | 29) => return true,
        Some(_) => {}
        None => return false,
    }
    let bits = asc.len() * 8;
    for start in 0..bits.saturating_sub(16) {
        let mut reader = BitReader::at(asc, start);
        if reader.read(11) == Some(SBR_SYNC_EXTENSION)
            && matches!(read_object_type(&mut reader), Some(5 | 29))
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn detects_explicit_sbr_and_ps() {
        // HE-AAC (SBR): base object type 5 → top 5 bits 00101 → 0x28.
        assert!(is_he_aac_asc(&[0x28, 0x00]));
        // HE-AACv2 (PS): base object type 29 → 11101 → 0xE8.
        assert!(is_he_aac_asc(&[0xE8, 0x00]));
    }

    #[test]
    fn detects_backward_compatible_sbr() {
        // Real Apple HE-AAC ASC: base AOT 2 + 0x2B7 sync extension + ext AOT 5.
        assert!(is_he_aac_asc(&[0x14, 0x10, 0x56, 0xe5, 0xa8]));
    }

    #[test]
    fn passes_plain_aac_lc() {
        // AAC-LC, no SBR extension: 0x12 0x10 and 0x12 0x90 must not trip.
        assert!(!is_he_aac_asc(&[0x12, 0x10]));
        assert!(!is_he_aac_asc(&[0x12, 0x90]));
        assert!(!is_he_aac_asc(&[]));
    }
}
