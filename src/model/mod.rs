//! Whisper model registry, cache, and the 8 GB memory guard.
//!
//! Models are ggml weights from the whisper.cpp HuggingFace repos, fetched via
//! `hf-hub` into the standard HF cache with a progress bar and resumable
//! transfer, then verified against a pinned SHA-256. `--offline` uses the cache
//! only.

use std::io::Read;
use std::path::PathBuf;

use hf_hub::Cache;
use hf_hub::api::sync::ApiBuilder;
use sha2::{Digest, Sha256};

use crate::cli::Model;
use crate::error::ScrybeError;

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
            multilingual: false,
            can_translate: false,
        },
    }
}

/// Peak RAM for one inference job: weights plus ~⅓ runtime overhead and a fixed
/// working-set allowance.
fn job_memory(model: Model) -> u64 {
    let size = info(model).size;
    size + size / 3 + 256 * 1024 * 1024
}

/// Whether running `jobs` copies of `model` would exceed safe memory, leaving a
/// 15% headroom on the detected total.
pub fn would_exceed_memory(total_ram: u64, model: Model, jobs: usize) -> bool {
    let needed = job_memory(model).saturating_mul(jobs.max(1) as u64);
    needed > total_ram / 100 * 85
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
                "{model} × {jobs} job(s) needs ~{}, but only {} is available; the largest model that fits is `{fits}`",
                human_size(job_memory(model).saturating_mul(jobs.max(1) as u64)),
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

/// The cached path for a model, or `None` when it is not yet downloaded.
pub fn cached_path(model: Model) -> Option<PathBuf> {
    let info = info(model);
    Cache::from_env().model(info.repo.to_owned()).get(info.file)
}

/// Ensure the model is on disk and verified, returning its path. Downloads with
/// a progress bar (resumable) unless `offline`, in which case only the cache is
/// consulted. A checksum mismatch triggers one re-download.
pub fn ensure_available(model: Model, offline: bool) -> Result<PathBuf, ScrybeError> {
    let info = info(model);
    let dl_error = |detail: String| ScrybeError::ModelDownloadFailed {
        model: model.to_string(),
        detail,
    };

    if offline {
        let path = cached_path(model)
            .ok_or_else(|| dl_error("not in cache and `--offline` is set".to_owned()))?;
        if sha256_matches(&path, info.sha256)? {
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
    let repo = api.model(info.repo.to_owned());

    let path = repo.get(info.file).map_err(|e| dl_error(e.to_string()))?;
    if sha256_matches(&path, info.sha256)? {
        return Ok(path);
    }

    // Corrupt download: drop the blob and fetch once more.
    if let Ok(real) = std::fs::canonicalize(&path) {
        let _ = std::fs::remove_file(&real);
    }
    let path = repo.get(info.file).map_err(|e| dl_error(e.to_string()))?;
    if sha256_matches(&path, info.sha256)? {
        return Ok(path);
    }
    Err(dl_error("checksum mismatch after re-download".to_owned()))
}

fn sha256_matches(path: &std::path::Path, expected: &str) -> Result<bool, ScrybeError> {
    let mut file = std::fs::File::open(path).map_err(|e| ScrybeError::ModelDownloadFailed {
        model: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let read = file
            .read(&mut buf)
            .map_err(|e| ScrybeError::ModelDownloadFailed {
                model: path.display().to_string(),
                detail: e.to_string(),
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
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
    fn large_v3_with_two_jobs_refused_on_8gb() {
        // The spec's headline guard: large-v3 (~2.9 GB) × 2 must not fit in 8 GB.
        assert!(would_exceed_memory(8 * GB, Model::LargeV3, 2));
        // Turbo at one job fits comfortably.
        assert!(!would_exceed_memory(8 * GB, Model::LargeV3Turbo, 1));
    }

    #[test]
    fn smart_default_shrinks_on_low_memory() {
        assert_eq!(largest_fitting(8 * GB), Model::LargeV3Turbo);
        assert_eq!(largest_fitting(2 * GB), Model::Small);
        assert_eq!(largest_fitting(512 * 1024 * 1024), Model::Tiny);
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
}
