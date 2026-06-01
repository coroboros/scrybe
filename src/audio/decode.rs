//! Decode audio files to interleaved f32 PCM.
//!
//! The default path is pure-Rust symphonia. HE-AAC/SBR is detected up front and
//! rejected with an actionable message (symphonia is AAC-LC only). The optional
//! ffmpeg path shells out for codecs symphonia cannot handle.

use std::io::Read as _;
use std::path::Path;
use std::process::{Command, Stdio};

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

/// Ceiling on one source's raw decoded PCM, before the resample frees it. Derived
/// from the per-job decode reservation (`model::DECODE_BUFFER`) so the two cannot
/// desync: with the batch pool running at most `jobs` decodes at once, each capped
/// here, the aggregate stays within what `guard_memory` reserved. Beyond it we fail
/// loud; very large sources should use `--decoder ffmpeg` (streams to 16 kHz mono).
const MAX_SOURCE_PCM_BYTES: u64 = crate::model::DECODE_BUFFER;

/// The ceiling expressed in f32 samples, for the in-loop/streaming sample counts.
const MAX_SOURCE_SAMPLES: usize = (MAX_SOURCE_PCM_BYTES / SAMPLE_BYTES) as usize;

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
    if let Some(frames) = source_frames
        && exceeds_decode_ceiling(frames, channels)
    {
        return Err(unsupported(format!(
            "audio is too large to decode in memory (~{} raw); use `--jobs 1`, a shorter clip, or `--decoder ffmpeg`",
            crate::model::human_size(raw_pcm_bytes(frames, channels)),
        )));
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
                        if samples.len() > MAX_SOURCE_SAMPLES {
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
            // A chained/multi-segment stream changes track params mid-file. Rather
            // than silently truncate to the first segment, fail loud and point at
            // the ffmpeg path, which handles it.
            Err(SymphoniaError::ResetRequired) => {
                return Err(unsupported(
                    "chained or multi-segment stream is not supported; retry with `--decoder ffmpeg`"
                        .to_owned(),
                ));
            }
            // End of stream is signalled as an unexpected-EOF I/O error; treat only
            // that as the end. Any other I/O error is a genuine read failure — fail
            // loud rather than silently truncate the transcript.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::IoError(e)) => {
                return Err(unsupported(format!("read error: {e}")));
            }
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

/// Why streaming f32le decode stopped early.
#[derive(Debug)]
enum StreamError {
    /// The running sample count exceeded the caller's ceiling.
    Overflow,
    /// A read from the source failed.
    Io(std::io::Error),
}

/// Decode an f32-little-endian byte stream into samples, reassembling values that
/// straddle reads and bailing with `Overflow` once `max_samples` is passed. Pure
/// over any `Read`, so the byte reassembly and the ceiling are unit-testable
/// without spawning a subprocess; the caller owns any process teardown.
fn read_f32le_stream<R: std::io::Read>(
    mut src: R,
    max_samples: usize,
) -> Result<Vec<f32>, StreamError> {
    let mut samples: Vec<f32> = Vec::new();
    let mut pending: Vec<u8> = Vec::new(); // < 4 leftover bytes spanning reads
    let mut buf = [0u8; 1 << 16];
    loop {
        let read = src.read(&mut buf).map_err(StreamError::Io)?;
        if read == 0 {
            break;
        }
        pending.extend_from_slice(&buf[..read]);
        let full = pending.len() / 4 * 4;
        samples.extend(
            pending[..full]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        pending.drain(..full);
        if samples.len() > max_samples {
            return Err(StreamError::Overflow);
        }
    }
    Ok(samples)
}

/// Raw interleaved f32 PCM bytes a source of `frames` × `channels` would decode to.
fn raw_pcm_bytes(frames: u64, channels: u16) -> u64 {
    frames
        .saturating_mul(u64::from(channels))
        .saturating_mul(SAMPLE_BYTES)
}

/// Whether a source's raw PCM would exceed the in-memory decode ceiling. Derived
/// from `MAX_SOURCE_PCM_BYTES` so the security-relevant fail-loud branch is
/// testable and cannot silently desync from the budget.
fn exceeds_decode_ceiling(frames: u64, channels: u16) -> bool {
    raw_pcm_bytes(frames, channels) > MAX_SOURCE_PCM_BYTES
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

/// Append one decoded packet's interleaved samples to `out`. `chunk` is reused
/// scratch — `copy_to_vec_interleaved` overwrites it in full each call — so the
/// per-packet allocation is amortized across the file.
fn append_f32(decoded: &GenericAudioBufferRef<'_>, chunk: &mut Vec<f32>, out: &mut Vec<f32>) {
    decoded.copy_to_vec_interleaved(chunk);
    out.extend_from_slice(chunk);
}

/// Decode by shelling out to a system `ffmpeg`, producing 16 kHz mono f32 PCM
/// directly. The escape hatch for codecs symphonia cannot handle (e.g. HE-AAC).
pub fn decode_via_ffmpeg(path: &Path) -> Result<Decoded, ScrybeError> {
    let unsupported = |detail: String| ScrybeError::unsupported_codec(path, detail);

    // `-i` consumes its next argument literally (ffmpeg has no `--` end-of-options
    // marker), so a leading-dash name is already safe; canonicalizing to an absolute
    // path is defense-in-depth, and falls back to the raw path if it fails.
    let input = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&input)
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar"])
        .arg(TARGET_SAMPLE_RATE.to_string())
        .arg("pipe:1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                unsupported("`--decoder ffmpeg` requested but ffmpeg is not on PATH".to_owned())
            } else {
                unsupported(format!("failed to run ffmpeg: {e}"))
            }
        })?;

    let Some(mut stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(unsupported("could not capture ffmpeg output".to_owned()));
    };

    // Stream stdout, decoding complete f32le samples as they arrive and bailing the
    // moment the running total would exceed the ceiling — so peak memory stays
    // bounded by `MAX_SOURCE_PCM_BYTES` like the symphonia path, instead of
    // buffering the whole (possibly huge) stream first. `-v error` keeps stderr
    // tiny, so draining stdout before reading it cannot deadlock. The byte loop is
    // a pure, unit-tested helper; the child kill/wait stays here.
    let samples = match read_f32le_stream(&mut stdout, MAX_SOURCE_SAMPLES) {
        Ok(samples) => samples,
        Err(StreamError::Overflow) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(unsupported(
                "audio is too large to decode in memory; use a shorter clip or `--jobs 1`"
                    .to_owned(),
            ));
        }
        Err(StreamError::Io(e)) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(unsupported(format!("reading ffmpeg output failed: {e}")));
        }
    };

    let status = child
        .wait()
        .map_err(|e| unsupported(format!("waiting for ffmpeg failed: {e}")))?;
    if !status.success() {
        let mut err = String::new();
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_string(&mut err);
        }
        return Err(unsupported(format!("ffmpeg failed: {}", err.trim())));
    }
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
///
/// The backward-compatible extension sits at a determinate bit offset, right after
/// the GASpecificConfig — so the config is parsed structurally to reach it rather
/// than scanning the whole payload, which could match `0x2B7` coincidentally
/// inside a valid AAC-LC config and false-reject it.
fn is_he_aac_asc(asc: &[u8]) -> bool {
    let mut r = BitReader::at(asc, 0);
    let Some(aot) = read_object_type(&mut r) else {
        return false;
    };
    if aot == 5 || aot == 29 {
        return true;
    }
    // samplingFrequencyIndex (15 escapes to an explicit 24-bit rate).
    let Some(sfi) = r.read(4) else { return false };
    if sfi == 0x0f && r.read(24).is_none() {
        return false;
    }
    // GASpecificConfig has a determinate length only with an explicit channel
    // configuration (1..=7). Config 0 carries a variable program_config_element;
    // a backward-compatible SBR extension does not occur there in practice, so it
    // is treated as plain AAC-LC (recoverable via `--decoder ffmpeg` if ever wrong).
    let Some(channels) = r.read(4) else {
        return false;
    };
    if !(1..=7).contains(&channels) {
        return false;
    }
    // Minimal GASpecificConfig for standalone AAC: frameLengthFlag,
    // dependsOnCoreCoder (+14-bit coreCoderDelay when set), extensionFlag.
    if r.read(1).is_none() {
        return false;
    }
    match r.read(1) {
        Some(0) => {}
        Some(_) if r.read(14).is_some() => {}
        _ => return false,
    }
    if r.read(1).is_none() {
        return false;
    }
    // syncExtensionType at its determinate position, then the extension AOT.
    r.read(11) == Some(SBR_SYNC_EXTENSION) && matches!(read_object_type(&mut r), Some(5 | 29))
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
    fn detects_sbr_with_no_trailing_bytes() {
        // Backward-compatible SBR (0x2B7 + ext AOT 5) ending exactly at the ASC
        // boundary, no padding after the extension.
        assert!(is_he_aac_asc(&[0x12, 0x10, 0x56, 0xe5]));
    }

    #[test]
    fn channel_config_zero_is_not_false_rejected() {
        // AAC-LC with channelConfiguration 0 (program config element). The tail
        // bytes embed 0x2B7 + AOT 5, so the old whole-buffer scan false-rejected
        // it as HE-AAC; structural parsing stops at the channel config and treats
        // it as plain AAC-LC. (Same bytes as the positive fixture, channels → 0.)
        assert!(!is_he_aac_asc(&[0x14, 0x00, 0x56, 0xe5, 0xa8]));
    }

    /// A `Read` that yields at most `chunk` bytes per call, so a 4-byte f32 can
    /// straddle reads — exercising the carry path.
    struct ChunkedReader<'a> {
        data: &'a [u8],
        pos: usize,
        chunk: usize,
    }

    impl std::io::Read for ChunkedReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.data[self.pos..].len().min(self.chunk).min(buf.len());
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn read_f32le_stream_reassembles_split_samples() {
        // Two f32 split across 3-byte reads, so both straddle a read boundary.
        let bytes: Vec<u8> = 1.5f32
            .to_le_bytes()
            .into_iter()
            .chain((-2.25f32).to_le_bytes())
            .collect();
        let reader = ChunkedReader {
            data: &bytes,
            pos: 0,
            chunk: 3,
        };
        let out = read_f32le_stream(reader, usize::MAX).unwrap();
        assert_eq!(out, vec![1.5, -2.25]);
    }

    #[test]
    fn read_f32le_stream_bails_on_overflow() {
        let bytes = vec![0u8; 12]; // three f32
        let reader = ChunkedReader {
            data: &bytes,
            pos: 0,
            chunk: 5,
        };
        assert!(matches!(
            read_f32le_stream(reader, 2),
            Err(StreamError::Overflow)
        ));
    }

    #[test]
    fn read_f32le_stream_empty_is_empty() {
        let reader = ChunkedReader {
            data: &[],
            pos: 0,
            chunk: 4,
        };
        assert!(read_f32le_stream(reader, 16).unwrap().is_empty());
    }

    #[test]
    fn decode_ceiling_rejects_oversized_sources() {
        // Just over the ceiling (in frames) is rejected; a normal clip passes.
        let frames_at_ceiling = MAX_SOURCE_PCM_BYTES / SAMPLE_BYTES; // mono
        assert!(!exceeds_decode_ceiling(frames_at_ceiling, 1));
        assert!(exceeds_decode_ceiling(frames_at_ceiling + 1, 1));
        // Channels multiply the raw size: half the frames in stereo still exceeds.
        assert!(exceeds_decode_ceiling(frames_at_ceiling / 2 + 1, 2));
        // A crafted huge frame count saturates rather than wrapping → rejected.
        assert!(exceeds_decode_ceiling(u64::MAX, 8));
        // A short clip is fine.
        assert!(!exceeds_decode_ceiling(16_000 * 60, 1)); // 1 min mono
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
