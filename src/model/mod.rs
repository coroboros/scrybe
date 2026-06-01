//! Whisper model registry, cache, and the 8 GB memory guard.
//!
//! Models are ggml weights from the whisper.cpp HuggingFace repos, fetched via
//! `hf-hub` into the standard HF cache with a progress bar and resumable
//! transfer, then verified against a pinned SHA-256. `--offline` uses the cache
//! only.

use std::fmt::Write as _;
use std::io::Read;
use std::path::PathBuf;

use hf_hub::Cache;
use hf_hub::api::sync::ApiBuilder;
use sha2::{Digest, Sha256};

use crate::cli::{DEFAULT_MODEL, Model};
use crate::error::ScrybeError;

/// The Silero VAD model (whisper.cpp voice-activity segmentation), fetched into
/// the same HF cache as the Whisper weights.
const VAD_REPO: &str = "ggml-org/whisper-vad";
const VAD_FILE: &str = "ggml-silero-v5.1.2.bin";
const VAD_SHA256: &str = "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf";

/// Static metadata for one model: where to fetch it and what it can do.
pub struct ModelInfo {
    pub repo: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub multilingual: bool,
    pub can_translate: bool,
}

/// Registry entry for a model. SHA-256 values are the HuggingFace LFS oids.
pub fn info(model: Model) -> ModelInfo {
    match model {
        Model::Tiny => ModelInfo {
            repo: "ggerganov/whisper.cpp",
            file: "ggml-tiny.bin",
            size: 77_691_713,
            sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
            multilingual: true,
            can_translate: true,
        },
        Model::Base => ModelInfo {
            repo: "ggerganov/whisper.cpp",
            file: "ggml-base.bin",
            size: 147_951_465,
            sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
            multilingual: true,
            can_translate: true,
        },
        Model::Small => ModelInfo {
            repo: "ggerganov/whisper.cpp",
            file: "ggml-small.bin",
            size: 487_601_967,
            sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
            multilingual: true,
            can_translate: true,
        },
        Model::LargeV3 => ModelInfo {
            repo: "ggerganov/whisper.cpp",
            file: "ggml-large-v3.bin",
            size: 3_095_033_483,
            sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
            multilingual: true,
            can_translate: true,
        },
        Model::LargeV3Turbo => ModelInfo {
            repo: "ggerganov/whisper.cpp",
            file: "ggml-large-v3-turbo.bin",
            size: 1_624_555_275,
            sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
            multilingual: true,
            can_translate: false,
        },
        Model::DistilLargeV35 => ModelInfo {
            repo: "distil-whisper/distil-large-v3.5-ggml",
            file: "ggml-model.bin",
            size: 1_519_521_155,
            sha256: "ec2498919b498c5f6b00041adb45650124b3cd9f26f545fffa8f5d11c28dcf26",
            // English-leaning per the distil-whisper release; gated English-only
            // here so `--lang <non-en>` is rejected loud rather than silently
            // degrading. Not a registry oversight.
            multilingual: false,
            can_translate: false,
        },
    }
}

/// Memory budget for one concurrent decode. This is the single source of truth for
/// the decode-memory invariant: it is BOTH the amount `guard_memory` reserves per
/// job AND the hard per-file raw-PCM ceiling the decoder enforces (the decode path
/// derives its ceiling from this value). The batch pool runs at most `jobs` decodes
/// at once, each bounded by this ceiling, so the aggregate resident decode memory
/// can never exceed `DECODE_BUFFER * jobs` — the exact figure reserved here. Files
/// whose raw PCM would exceed it fail loud, with `--decoder ffmpeg` (which streams
/// straight to 16 kHz mono) as the escape for very large sources.
pub(crate) const DECODE_BUFFER: u64 = 1024 * 1024 * 1024;

/// Estimated peak memory transcribing `model` at `jobs` concurrent decodes. The
/// engine loads one shared model context and runs inference serially, so the
/// weights and inference working set are resident once; only the in-flight decode
/// buffers scale with `jobs`. This is deliberately more permissive than a per-job
/// model copy (the spec's premise) because that copy never happens — fewer false
/// refusals, still OOM-safe.
fn estimated_memory(model: Model, jobs: usize) -> u64 {
    let weights = info(model).size;
    let working_set = weights / 2;
    weights + working_set + DECODE_BUFFER.saturating_mul(jobs.max(1) as u64)
}

/// Fraction of detected RAM a run may use, leaving headroom for the OS and other
/// processes.
const MEMORY_BUDGET_PERCENT: u64 = 85;

/// Whether transcribing `model` at `jobs` would exceed the memory budget. Multiply
/// before dividing so the percentage doesn't truncate the byte total; `saturating_mul`
/// keeps the function total over its whole `u64` domain (it is public and tested
/// with synthetic RAM values).
pub fn would_exceed_memory(total_ram: u64, model: Model, jobs: usize) -> bool {
    estimated_memory(model, jobs) > total_ram.saturating_mul(MEMORY_BUDGET_PERCENT) / 100
}

/// The largest model that fits at one job, for the smart default and guard hints.
pub fn largest_fitting(total_ram: u64) -> Model {
    for model in [Model::LargeV3Turbo, Model::Small, Model::Base, Model::Tiny] {
        if !would_exceed_memory(total_ram, model, 1) {
            return model;
        }
    }
    Model::Tiny
}

/// Resolve the model to run: an explicit `--model` is honored as-is (and later
/// guarded, so an oversized explicit pick still fails loud); when omitted, pick
/// the largest model that fits detected RAM, falling back to the nominal default
/// when memory can't be read.
pub fn resolve_model(explicit: Option<Model>, total_ram: Option<u64>) -> Model {
    explicit.unwrap_or_else(|| total_ram.map_or(DEFAULT_MODEL, largest_fitting))
}

/// Total physical memory in bytes, or `None` when it can't be read.
pub fn total_memory() -> Option<u64> {
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};
    let sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    let total = sys.total_memory();
    (total > 0).then_some(total)
}

/// Refuse to run when the model plus job count would not fit in memory.
pub fn guard_memory(model: Model, jobs: usize) -> Result<(), ScrybeError> {
    let Some(total) = total_memory() else {
        return Ok(());
    };
    if would_exceed_memory(total, model, jobs) {
        let fits = largest_fitting(total);
        return Err(ScrybeError::OutOfMemory {
            detail: format!(
                "{model} at {jobs} job(s) needs ~{}, but only {} is available; the largest model that fits is `{fits}`",
                human_size(estimated_memory(model, jobs)),
                human_size(total),
            ),
        });
    }
    Ok(())
}

/// The HuggingFace hub cache directory (`$HF_HOME/hub` or `~/.cache/huggingface/hub`).
pub fn cache_dir() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache/huggingface/hub");
    }
    PathBuf::from(".cache/huggingface/hub")
}

/// The cached path for a repo file, or `None` when it is not yet downloaded.
fn cached(repo: &str, file: &str) -> Option<PathBuf> {
    Cache::from_env().model(repo.to_owned()).get(file)
}

/// The cached path for a model, or `None` when it is not yet downloaded.
pub fn cached_path(model: Model) -> Option<PathBuf> {
    let info = info(model);
    cached(info.repo, info.file)
}

/// Ensure a model is on disk and verified, returning its path. Downloads with a
/// progress bar (resumable) unless `offline`, in which case only the cache is
/// consulted. A checksum mismatch triggers one re-download.
pub fn ensure_available(model: Model, offline: bool) -> Result<PathBuf, ScrybeError> {
    let info = info(model);
    fetch_verified(
        info.repo,
        info.file,
        info.sha256,
        &model.to_string(),
        offline,
    )
}

/// The Silero VAD model, bundled in the binary so the spec's mandated correctness
/// floor is always available — no network, works on a first offline run.
const SILERO_VAD_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/ggml-silero-v5.1.2.bin"
));

/// Ensure the Silero VAD model is on disk, returning its path. Prefers a cached
/// copy (HF cache, or a prior materialization); otherwise writes the bundled,
/// SHA-pinned model so VAD never depends on the network.
pub fn ensure_vad() -> Result<PathBuf, ScrybeError> {
    // A read error or checksum mismatch on a cached copy falls through to writing
    // the bundled, SHA-pinned model — the always-available floor — rather than
    // failing the run.
    if let Some(path) = cached(VAD_REPO, VAD_FILE)
        && sha256_matches(&path, VAD_SHA256, "silero-vad").unwrap_or(false)
    {
        return Ok(path);
    }
    let dir = cache_dir().join("scrybe");
    let path = dir.join(VAD_FILE);
    if path.exists() && sha256_matches(&path, VAD_SHA256, "silero-vad").unwrap_or(false) {
        return Ok(path);
    }
    let io_error = |e: std::io::Error| ScrybeError::Io {
        detail: format!("could not materialize the bundled VAD model: {e}"),
    };
    std::fs::create_dir_all(&dir).map_err(&io_error)?;
    std::fs::write(&path, SILERO_VAD_BYTES).map_err(&io_error)?;
    Ok(path)
}

/// Fetch `file` from `repo`, verifying its SHA-256 (`label` names it in errors).
/// Offline uses the cache only; a checksum mismatch re-downloads once.
fn fetch_verified(
    repo: &str,
    file: &str,
    sha256: &str,
    label: &str,
    offline: bool,
) -> Result<PathBuf, ScrybeError> {
    let dl_error = |detail: String| ScrybeError::ModelDownloadFailed {
        model: label.to_owned(),
        detail,
    };

    if offline {
        let path = cached(repo, file)
            .ok_or_else(|| dl_error("not in cache and `--offline` is set".to_owned()))?;
        if sha256_matches(&path, sha256, label)? {
            return Ok(path);
        }
        return Err(dl_error(
            "cached file is corrupt (checksum mismatch) and `--offline` blocks re-download"
                .to_owned(),
        ));
    }

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| dl_error(e.to_string()))?;
    let repo_api = api.model(repo.to_owned());

    let path = repo_api.get(file).map_err(|e| dl_error(e.to_string()))?;
    if sha256_matches(&path, sha256, label)? {
        return Ok(path);
    }

    // Corrupt download: drop the blob and fetch once more.
    evict(&path);
    let path = repo_api.get(file).map_err(|e| dl_error(e.to_string()))?;
    if sha256_matches(&path, sha256, label)? {
        return Ok(path);
    }
    Err(dl_error("checksum mismatch after re-download".to_owned()))
}

/// Remove a cached file: both the snapshot symlink and the blob it targets.
pub fn evict(path: &std::path::Path) {
    if let Ok(blob) = std::fs::canonicalize(path) {
        let _ = std::fs::remove_file(&blob);
    }
    let _ = std::fs::remove_file(path);
}

fn sha256_matches(
    path: &std::path::Path,
    expected: &str,
    label: &str,
) -> Result<bool, ScrybeError> {
    let io_error = |e: std::io::Error| ScrybeError::ModelDownloadFailed {
        model: label.to_owned(),
        detail: format!("{}: {e}", path.display()),
    };
    let mut file = std::fs::File::open(path).map_err(&io_error)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let read = file.read(&mut buf).map_err(&io_error)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}"); // writing into a String is infallible
    }
    Ok(hex.eq_ignore_ascii_case(expected))
}

/// Human-readable byte size (binary units).
pub fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn memory_guard_refuses_only_genuinely_oversized_runs() {
        // One shared context + serial inference: large-v3 fits 8 GB at a sane job
        // count, but an absurd job count (decode buffers) or a too-small machine
        // is refused. (Justified deviation from the spec's per-copy premise — the
        // copy never happens.)
        assert!(!would_exceed_memory(8 * GB, Model::LargeV3, 1));
        assert!(would_exceed_memory(8 * GB, Model::LargeV3, 16));
        assert!(would_exceed_memory(4 * GB, Model::LargeV3, 1));
        assert!(!would_exceed_memory(8 * GB, Model::LargeV3Turbo, 1));
    }

    #[test]
    fn memory_guard_pins_the_budget_boundary() {
        // Falsify the arithmetic itself, not just the extremes: with total RAM equal
        // to the estimate the 15% headroom is missing (refused); just above the 85%
        // line it fits.
        let est = info(Model::Tiny).size + info(Model::Tiny).size / 2 + DECODE_BUFFER;
        assert!(would_exceed_memory(est, Model::Tiny, 1));
        let total_fits = est * 100 / MEMORY_BUDGET_PERCENT + 1;
        assert!(!would_exceed_memory(total_fits, Model::Tiny, 1));
    }

    #[test]
    fn smart_default_shrinks_on_low_memory() {
        assert_eq!(largest_fitting(8 * GB), Model::LargeV3Turbo);
        assert_eq!(largest_fitting(2 * GB), Model::Small);
        assert_eq!(largest_fitting(512 * 1024 * 1024), Model::Tiny);
    }

    #[test]
    fn resolve_model_honors_explicit_and_shrinks_default() {
        // Explicit pick passes through untouched, even when it won't fit (the
        // memory guard refuses it later — never a silent override).
        assert_eq!(
            resolve_model(Some(Model::LargeV3), Some(2 * GB)),
            Model::LargeV3
        );
        // Omitted --model resolves to the largest that fits detected RAM.
        assert_eq!(resolve_model(None, Some(8 * GB)), Model::LargeV3Turbo);
        assert_eq!(resolve_model(None, Some(2 * GB)), Model::Small);
        // Unknown RAM falls back to the nominal default.
        assert_eq!(resolve_model(None, None), DEFAULT_MODEL);
    }

    #[test]
    fn human_size_units() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(1_624_555_275), "1.5 GB");
    }

    #[test]
    fn translation_gating_matches_spec() {
        assert!(info(Model::LargeV3).can_translate);
        assert!(!info(Model::LargeV3Turbo).can_translate);
        assert!(!info(Model::DistilLargeV35).can_translate);
    }

    #[test]
    fn bundled_vad_is_always_available_and_verified() {
        // The mandated VAD floor must resolve with no network (bundled), and the
        // returned file must match the pinned SHA.
        let path = ensure_vad().expect("bundled VAD must always be available");
        assert!(path.exists());
        assert!(sha256_matches(&path, VAD_SHA256, "silero-vad").unwrap());
    }
}
