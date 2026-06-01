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

/// Bytes per decoded f32 sample.
const SAMPLE_BYTES: u64 = 4;

/// Ceiling on a source's raw decoded PCM. Beyond this, the whole-file decode
/// would risk exhausting memory before the resample frees it; we fail loud
/// instead. Long/high-rate inputs should use `--jobs 1` or be pre-converted.
const MAX_SOURCE_PCM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Interleaved f32 PCM plus the source stream's rate and channel count.
pub struct Decoded {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Decode with symphonia. Fails loud (no silent/garbled output) on unsupported
/// codecs, including HE-AAC, pointing the user at the ffmpeg escape.
pub fn decode_file(path: &Path) -> Result<Decoded, ScrybeError> {
    let unsupported = |detail: String| ScrybeError::unsupported_codec(path, detail);

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
    let source_frames = track.num_frames;

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

    // Reject sources whose raw PCM would exhaust memory before the resample frees
    // it (the memory guard models the post-resample buffer, not this transient).
    if let Some(frames) = source_frames {
        let bytes = frames
            .saturating_mul(u64::from(channels))
            .saturating_mul(SAMPLE_BYTES);
        if bytes > MAX_SOURCE_PCM_BYTES {
            return Err(unsupported(format!(
                "audio is too large to decode in memory (~{} raw); use `--jobs 1`, a shorter clip, or `--decoder ffmpeg`",
                crate::model::human_size(bytes),
            )));
        }
    }

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

    // Pre-size from the (guard-bounded) frame count to avoid ~log2(N) reallocs on
    // long files; the loop also caps growth for sources that don't declare length.
    let mut samples = Vec::with_capacity(presize_capacity(source_frames, channels));
    let max_samples = (MAX_SOURCE_PCM_BYTES / SAMPLE_BYTES) as usize;
    let mut chunk: Vec<f32> = Vec::new();
    loop {
        match format.next_packet() {
            Ok(Some(packet)) => {
                if packet.track_id != track_id {
                    continue;
                }
                match decoder.decode(&packet) {
                    Ok(decoded) => {
                        append_f32(&decoded, &mut chunk, &mut samples);
                        if samples.len() > max_samples {
                            return Err(unsupported(
                                "audio exceeds the in-memory decode limit; use `--jobs 1`, a shorter clip, or `--decoder ffmpeg`".to_owned(),
                            ));
                        }
                    }
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

/// Initial capacity for the decode buffer. Pre-sizes to the declared frame count
/// to avoid realloc storms on long files, but caps the speculative allocation so
/// a crafted container header (huge declared length from a tiny file) can't force
/// a multi-GB up-front alloc — the in-loop ceiling still bounds growth from the
/// bytes actually decoded.
fn presize_capacity(source_frames: Option<u64>, channels: u16) -> usize {
    /// ~64 MB of f32 — enough to pre-size ~17 min of 16 kHz mono before growth.
    const PRESIZE_CAP_SAMPLES: usize = 16 * 1024 * 1024;
    match source_frames {
        Some(frames) => (frames as usize)
            .saturating_mul(channels as usize)
            .min(PRESIZE_CAP_SAMPLES),
        None => 0,
    }
}

fn append_f32(decoded: &GenericAudioBufferRef<'_>, chunk: &mut Vec<f32>, out: &mut Vec<f32>) {
    decoded.copy_to_vec_interleaved(chunk);
    out.extend_from_slice(chunk);
}

/// Decode by shelling out to a system `ffmpeg`, producing 16 kHz mono f32 PCM
/// directly. The escape hatch for codecs symphonia cannot handle (e.g. HE-AAC).
pub fn decode_via_ffmpeg(path: &Path) -> Result<Decoded, ScrybeError> {
    let unsupported = |detail: String| ScrybeError::unsupported_codec(path, detail);

    // Canonicalize so a leading-dash path can't be parsed by ffmpeg as an option.
    let input = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&input)
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

    // Same in-memory ceiling as the symphonia path; the stdout bytes are already
    // f32le PCM, so the byte length is exact. Reject before the second alloc.
    if output.stdout.len() as u64 > MAX_SOURCE_PCM_BYTES {
        return Err(unsupported(
            "audio is too large to decode in memory; use a shorter clip".to_owned(),
        ));
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
    // The SBR sync extension + extension object type span 16 bits, so the last
    // valid start is `bits - 16` (inclusive). Reads past the end return None.
    let bits = asc.len() * 8;
    for start in 0..=bits.saturating_sub(16) {
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
    fn detects_sbr_in_the_final_16_bits() {
        // 0x2B7 sync + ext object type 5 occupying exactly the last 16 bits —
        // regression for the scan's upper bound (previously exclusive, so missed).
        assert!(is_he_aac_asc(&[0x12, 0x10, 0x56, 0xe5]));
    }

    #[test]
    fn presize_capacity_caps_crafted_headers() {
        // A small declared length pre-sizes exactly (2 ch interleaved).
        assert_eq!(presize_capacity(Some(1000), 2), 2000);
        // A crafted huge declared length is capped, not trusted up front.
        assert_eq!(presize_capacity(Some(u64::MAX), 2), 16 * 1024 * 1024);
        // No declared length → no speculative allocation.
        assert_eq!(presize_capacity(None, 1), 0);
    }

    #[test]
    fn passes_plain_aac_lc() {
        // AAC-LC, no SBR extension: 0x12 0x10 and 0x12 0x90 must not trip.
        assert!(!is_he_aac_asc(&[0x12, 0x10]));
        assert!(!is_he_aac_asc(&[0x12, 0x90]));
        assert!(!is_he_aac_asc(&[]));
    }
}
