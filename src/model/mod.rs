//! Whisper model registry, cache, and the memory guard (85% of detected RAM).
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
// The VAD filename feeds both the cache lookup and the bundled-bytes include path;
// a macro single-sources it because concat!/include_bytes! take a literal, not a const.
macro_rules! vad_file {
    () => {
        "ggml-silero-v5.1.2.bin"
    };
}
const VAD_FILE: &str = vad_file!();
const VAD_SHA256: &str = "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf";

/// Static metadata for one model: where to fetch it and what it can do.
pub struct ModelInfo {
    pub repo: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub multilingual: bool,
    /// Translation is gated to `large-v3` (the rest translate only poorly or not at all).
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
            can_translate: false,
        },
        Model::Base => ModelInfo {
            repo: "ggerganov/whisper.cpp",
            file: "ggml-base.bin",
            size: 147_951_465,
            sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
            multilingual: true,
            can_translate: false,
        },
        Model::Small => ModelInfo {
            repo: "ggerganov/whisper.cpp",
            file: "ggml-small.bin",
            size: 487_601_967,
            sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
            multilingual: true,
            can_translate: false,
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

/// Per-job decode-memory budget, and the single source of truth for the decode
/// ceiling: it is BOTH the amount `guard_memory` reserves per job AND the hard
/// per-file ceiling on the 16 kHz-mono output (`audio::resample::MAX_OUTPUT_SAMPLES`
/// derives from it, so the two cannot desync). Decode streams — each packet is
/// downmixed and resampled as it arrives — so the full-resolution source is never
/// resident; the only large per-job buffer is the 16 kHz output, bounded here. The
/// batch pool runs at most `jobs` decodes at once, so resident decode memory tracks
/// `DECODE_BUFFER * jobs`. A clip whose 16 kHz output would exceed this (~4.6 hours)
/// fails loud, with `--decoder ffmpeg` as the alternative.
pub(crate) const DECODE_BUFFER: u64 = 1024 * 1024 * 1024;

/// Resident memory for `model`'s loaded context: weights plus a ~half-weights
/// inference working set, held once (one shared context, serial inference). The
/// single source for this derivation, shared by the peak-memory estimate and the
/// max-jobs solver so they cannot disagree. Saturating so an outsized model size
/// can't overflow.
fn resident_memory(model: Model) -> u64 {
    let weights = info(model).size;
    weights.saturating_add(weights / 2)
}

/// Estimated peak memory transcribing `model` at `jobs` concurrent decodes. The
/// engine loads one shared model context and runs inference serially, so the
/// weights and inference working set are resident once; only the in-flight decode
/// buffers scale with `jobs`. This is deliberately more permissive than a per-job
/// model copy because that copy never happens — fewer false refusals, still
/// OOM-safe.
fn estimated_memory(model: Model, jobs: usize) -> u64 {
    // Saturating throughout so an absurd `--jobs` can't overflow (debug panic /
    // release wrap); the result clamps to u64::MAX, which the guard reads as
    // "won't fit".
    resident_memory(model).saturating_add(DECODE_BUFFER.saturating_mul(jobs.max(1) as u64))
}

/// Fraction of detected RAM a run may use, leaving headroom for the OS and other
/// processes.
const MEMORY_BUDGET_PERCENT: u64 = 85;

/// Bytes a run may use: `MEMORY_BUDGET_PERCENT`% of detected RAM. Single source so
/// every guard agrees. Multiply before dividing so the percentage doesn't truncate
/// the byte total; `saturating_mul` keeps it total over the whole `u64` domain (the
/// guards are tested with synthetic RAM values).
const fn memory_budget(total_ram: u64) -> u64 {
    total_ram.saturating_mul(MEMORY_BUDGET_PERCENT) / 100
}

/// Whether transcribing `model` at `jobs` would exceed the memory budget.
pub(crate) fn would_exceed_memory(total_ram: u64, model: Model, jobs: usize) -> bool {
    estimated_memory(model, jobs) > memory_budget(total_ram)
}

/// The largest model that fits at `jobs` concurrent jobs, for the smart default and
/// guard hints. Selecting at the job count the run will actually use means a
/// self-chosen default can never be refused by its own guard.
pub(crate) fn largest_fitting(total_ram: u64, jobs: usize) -> Model {
    for model in [Model::LargeV3Turbo, Model::Small, Model::Base, Model::Tiny] {
        if !would_exceed_memory(total_ram, model, jobs) {
            return model;
        }
    }
    Model::Tiny
}

/// Resolve the model to run: an explicit `--model` is honored as-is (and later
/// guarded, so an oversized explicit pick still fails loud); when omitted, pick the
/// largest model that fits detected RAM *at the resolved job count*, falling back to
/// the nominal default when memory can't be read.
pub fn resolve_model(explicit: Option<Model>, total_ram: Option<u64>, jobs: usize) -> Model {
    explicit.unwrap_or_else(|| total_ram.map_or(DEFAULT_MODEL, |ram| largest_fitting(ram, jobs)))
}

/// The most jobs at which `model` still fits `total_ram` (at least 1). The decode
/// reservation grows `DECODE_BUFFER` per job, so `cores/2` on a big box can exceed
/// RAM on its own; the auto job count is clamped to this against the smallest model
/// so a zero-config run is never refused by its own guard.
pub(crate) fn max_jobs_fitting(total_ram: u64, model: Model) -> usize {
    let budget = memory_budget(total_ram);
    let resident = resident_memory(model);
    ((budget.saturating_sub(resident) / DECODE_BUFFER) as usize).max(1)
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

/// Refuse to run when the model plus job count would not fit in memory. Takes the
/// detected total so the whole run reads RAM once.
pub fn guard_memory(model: Model, jobs: usize, total: Option<u64>) -> Result<(), ScrybeError> {
    let Some(total) = total else {
        return Ok(());
    };
    if would_exceed_memory(total, model, jobs) {
        let need = human_size(estimated_memory(model, jobs));
        let have = human_size(total);
        // Recommend a model that fits at the SAME job count, so the hint never names
        // the model just refused. On a machine so small nothing fits at this count,
        // `largest_fitting` falls back to the smallest model that still doesn't fit —
        // don't quote it; point at lowering jobs instead.
        let fits = largest_fitting(total, jobs);
        let detail = if would_exceed_memory(total, fits, jobs) {
            format!(
                "{model} at {jobs} job(s) needs ~{need}, but only {have} is available; no model fits at {jobs} job(s)"
            )
        } else {
            format!(
                "{model} at {jobs} job(s) needs ~{need}, but only {have} is available; the largest model that fits is `{fits}`"
            )
        };
        return Err(ScrybeError::OutOfMemory { detail });
    }
    Ok(())
}

/// The HuggingFace hub cache directory (`$HF_HOME/hub` or `~/.cache/huggingface/hub`).
/// Derived from hf-hub's own resolution so it agrees with where downloads actually
/// land on every platform — `dirs::home_dir()` resolves `%USERPROFILE%` on Windows,
/// where `HOME` is usually unset. Single source: the lookup, the download, `models
/// path`, and the bundled-VAD directory all share this root.
pub fn cache_dir() -> PathBuf {
    Cache::from_env().path().join("hub")
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

/// The Silero VAD model, bundled in the binary so the mandated correctness floor
/// is always available — no network, works on a first offline run.
const SILERO_VAD_BYTES: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/", vad_file!()));

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
    materialize_vad(&cache_dir().join("scrybe"))
}

/// Materialize the bundled VAD model into `dir`, reusing an already-present,
/// SHA-matching copy without rewriting. Split out (dir-parameterized) so the
/// reuse branch is testable against an isolated directory.
fn materialize_vad(dir: &std::path::Path) -> Result<PathBuf, ScrybeError> {
    let path = dir.join(VAD_FILE);
    if path.exists() && sha256_matches(&path, VAD_SHA256, "silero-vad").unwrap_or(false) {
        return Ok(path);
    }
    let io_error = |e: std::io::Error| ScrybeError::Io {
        detail: format!("could not materialize the bundled VAD model: {e}"),
    };
    std::fs::create_dir_all(dir).map_err(&io_error)?;
    // Write to a per-process temp file, then atomically rename it into place, so a
    // concurrent or interrupted run never reads a half-written model (which would
    // fail the SHA gate or load as corrupt). Rename within one directory is atomic;
    // the loser of a race just replaces the file with byte-identical contents.
    let tmp = dir.join(format!("{VAD_FILE}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, SILERO_VAD_BYTES).map_err(&io_error)?;
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        io_error(e)
    })?;
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

    // `from_env`, not `new`: it builds on `Cache::from_env()` (HF_HOME-aware) and
    // honors HF_ENDPOINT, so downloads land in the same tree `cached()`/`cache_dir()`
    // read. `new` hardcodes ~/.cache/huggingface and would split pull from lookup.
    let api = ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|e| dl_error(e.to_string()))?;
    let repo_api = api.model(repo.to_owned());

    fetch_with_retry(
        || repo_api.get(file).map_err(|e| dl_error(e.to_string())),
        |path| sha256_matches(path, sha256, label),
        evict,
        || dl_error("checksum mismatch after re-download".to_owned()),
    )
}

/// Fetch-verify with one re-download on checksum mismatch: fetch, verify; on
/// mismatch evict the corrupt blob and fetch once more, verify; if it still
/// mismatches, fail loud. Pure over the fetch/verify/evict effects so the retry
/// ordering (evict before re-fetch, exactly one retry, terminal error) is
/// unit-testable without touching the network.
fn fetch_with_retry(
    mut fetch: impl FnMut() -> Result<PathBuf, ScrybeError>,
    verify: impl Fn(&std::path::Path) -> Result<bool, ScrybeError>,
    mut evict: impl FnMut(&std::path::Path),
    mismatch: impl FnOnce() -> ScrybeError,
) -> Result<PathBuf, ScrybeError> {
    let path = fetch()?;
    if verify(&path)? {
        return Ok(path);
    }
    evict(&path);
    let path = fetch()?;
    if verify(&path)? {
        return Ok(path);
    }
    Err(mismatch())
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
    // Names a read fault as a download failure (a cached blob we can't read is, to
    // the user, a fetch problem); distinct from `materialize_vad`'s `io_error`, which
    // builds the generic `Io` variant.
    let read_error = |e: std::io::Error| ScrybeError::ModelDownloadFailed {
        model: label.to_owned(),
        detail: format!("{}: {e}", path.display()),
    };
    let mut file = std::fs::File::open(path).map_err(&read_error)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let read = file.read(&mut buf).map_err(&read_error)?;
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
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
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
        // is refused. (Justified deviation from a per-copy estimate — the copy
        // never happens.)
        assert!(!would_exceed_memory(8 * GB, Model::LargeV3, 1));
        assert!(would_exceed_memory(8 * GB, Model::LargeV3, 16));
        assert!(would_exceed_memory(4 * GB, Model::LargeV3, 1));
        assert!(!would_exceed_memory(8 * GB, Model::LargeV3Turbo, 1));
    }

    #[test]
    fn total_memory_reads_a_sane_value() {
        // The one production RAM probe; it drives model and jobs selection. A sysinfo
        // bump that changes units/refresh semantics (KiB vs bytes, or yields 0) must
        // fail loudly here, not silently degrade auto-selection. Conservative floor so
        // it never flakes on a small CI runner.
        let total = total_memory().expect("a real test host reports total RAM");
        assert!(
            total >= 256 * 1024 * 1024,
            "implausibly low total RAM: {total}"
        );
    }

    #[test]
    fn fetch_with_retry_pins_the_redownload_state_machine() {
        use std::cell::Cell;
        let err = || ScrybeError::ModelDownloadFailed {
            model: "t".to_owned(),
            detail: "mismatch".to_owned(),
        };

        // Clean hit: one fetch, verify true, never evict.
        let (fetches, evicts) = (Cell::new(0), Cell::new(0));
        let r = fetch_with_retry(
            || {
                fetches.set(fetches.get() + 1);
                Ok(PathBuf::from("/a"))
            },
            |_| Ok(true),
            |_| evicts.set(evicts.get() + 1),
            err,
        );
        assert!(r.is_ok());
        assert_eq!((fetches.get(), evicts.get()), (1, 0));

        // Mismatch then good: evict between, exactly two fetches, Ok.
        let (fetches, evicts) = (Cell::new(0), Cell::new(0));
        let r = fetch_with_retry(
            || {
                fetches.set(fetches.get() + 1);
                Ok(PathBuf::from("/a"))
            },
            |_| Ok(fetches.get() == 2), // false on the first verify, true on the second
            |_| evicts.set(evicts.get() + 1),
            err,
        );
        assert!(r.is_ok());
        assert_eq!((fetches.get(), evicts.get()), (2, 1));

        // Mismatch twice: two fetches, one evict, terminal error.
        let (fetches, evicts) = (Cell::new(0), Cell::new(0));
        let r = fetch_with_retry(
            || {
                fetches.set(fetches.get() + 1);
                Ok(PathBuf::from("/a"))
            },
            |_| Ok(false),
            |_| evicts.set(evicts.get() + 1),
            err,
        );
        assert!(matches!(r, Err(ScrybeError::ModelDownloadFailed { .. })));
        assert_eq!((fetches.get(), evicts.get()), (2, 1));
    }

    #[test]
    fn sha256_matches_verifies_content_case_insensitively() {
        // The integrity gate for every downloaded binary. Known-answer vector
        // (SHA-256 of "abc"), checked lowercase and uppercase, plus a mismatch —
        // so a regression that always returns true (disabling verification) fails.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("data");
        std::fs::write(&file, b"abc").unwrap();
        const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(
            sha256_matches(&file, ABC, "t").unwrap(),
            "correct lowercase hex"
        );
        assert!(
            sha256_matches(&file, &ABC.to_uppercase(), "t").unwrap(),
            "case-insensitive match"
        );
        assert!(
            !sha256_matches(&file, &"0".repeat(64), "t").unwrap(),
            "mismatching hex is rejected"
        );
    }

    #[test]
    fn memory_guard_saturates_on_absurd_jobs() {
        // A pathological job count must refuse, never overflow-panic (no-panic rule).
        assert!(would_exceed_memory(u64::MAX, Model::Tiny, usize::MAX));
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
        assert_eq!(largest_fitting(8 * GB, 1), Model::LargeV3Turbo);
        assert_eq!(largest_fitting(2 * GB, 1), Model::Small);
        assert_eq!(largest_fitting(512 * 1024 * 1024, 1), Model::Tiny);
        // More jobs add decode buffers, so the largest that fits shrinks: 8 GiB at
        // 6 jobs can't hold turbo's weights plus 6 GiB of buffers.
        assert_ne!(largest_fitting(8 * GB, 6), Model::LargeV3Turbo);
    }

    #[test]
    fn resolve_model_honors_explicit_and_shrinks_default() {
        // Explicit pick passes through untouched, even when it won't fit (the
        // memory guard refuses it later — never a silent override).
        assert_eq!(
            resolve_model(Some(Model::LargeV3), Some(2 * GB), 1),
            Model::LargeV3
        );
        // Omitted --model resolves to the largest that fits detected RAM.
        assert_eq!(resolve_model(None, Some(8 * GB), 1), Model::LargeV3Turbo);
        assert_eq!(resolve_model(None, Some(2 * GB), 1), Model::Small);
        // Unknown RAM falls back to the nominal default.
        assert_eq!(resolve_model(None, None, 1), DEFAULT_MODEL);
    }

    #[test]
    fn oom_hint_never_names_a_model_that_does_not_fit() {
        // When something fits, the hint names a fitting model.
        let err = guard_memory(Model::LargeV3, 3, Some(8 * GB)).unwrap_err();
        assert!(err.to_string().contains("fits is `"), "{err}");
        // When nothing fits at this RAM (even tiny exceeds budget), the hint must not
        // quote a model that was just refused — it points at lowering jobs instead.
        let err = guard_memory(Model::LargeV3, 1, Some(GB)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no model fits"), "{msg}");
        assert!(
            !msg.contains("fits is `"),
            "named a non-fitting model: {msg}"
        );
    }

    #[test]
    fn smart_default_is_never_refused_by_its_own_guard() {
        // Regression: the default is selected at the job count it will run at, so a
        // zero-config run can't pick a model the guard then rejects (the jobs=1-only
        // selection used to refuse itself on high-core / low-RAM CPU boxes).
        // Invariant: whenever the machine can run *anything* at this job count (tiny
        // fits), the resolved default fits too. If even tiny can't fit, refusing is
        // correct, so that case is excluded.
        for &ram in &[4 * GB, 6 * GB, 8 * GB, 16 * GB] {
            for &jobs in &[1usize, 2, 4, 6, 12] {
                if would_exceed_memory(ram, Model::Tiny, jobs) {
                    continue;
                }
                let model = resolve_model(None, Some(ram), jobs);
                assert!(
                    !would_exceed_memory(ram, model, jobs),
                    "default {model} refused at {ram} bytes / {jobs} jobs though tiny fits"
                );
            }
        }
    }

    #[test]
    fn human_size_units() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(1_624_555_275), "1.5 GB");
    }

    #[test]
    fn translation_gating_matches_spec() {
        // Only large-v3 translates; the rest are gated off (rejected before download).
        assert!(info(Model::LargeV3).can_translate);
        assert!(!info(Model::Tiny).can_translate);
        assert!(!info(Model::Base).can_translate);
        assert!(!info(Model::Small).can_translate);
        assert!(!info(Model::LargeV3Turbo).can_translate);
        assert!(!info(Model::DistilLargeV35).can_translate);
    }

    #[cfg(unix)]
    #[test]
    fn evict_removes_both_the_snapshot_symlink_and_its_blob() {
        // The HF cache stores a snapshot symlink → blobs/<sha>; evict must delete
        // both, not just the link (else the blob leaks and the cache never shrinks).
        let dir = tempfile::tempdir().unwrap();
        let blob = dir.path().join("blob");
        std::fs::write(&blob, b"x").unwrap();
        let snapshot = dir.path().join("snapshot");
        std::os::unix::fs::symlink(&blob, &snapshot).unwrap();
        evict(&snapshot);
        assert!(!snapshot.exists(), "snapshot symlink not removed");
        assert!(!blob.exists(), "blob (canonical target) not removed");
    }

    #[test]
    fn materialize_vad_reuses_an_existing_valid_copy() {
        // First call writes the bundled model; the second must take the cache-hit
        // early return and reuse it (mtime unchanged), not rewrite. Isolated tempdir
        // so it's deterministic and parallel-safe.
        let dir = tempfile::tempdir().unwrap();
        let first = materialize_vad(dir.path()).expect("first materialization");
        let mtime = first.metadata().unwrap().modified().unwrap();
        let second = materialize_vad(dir.path()).expect("reuse");
        assert_eq!(first, second);
        assert_eq!(
            second.metadata().unwrap().modified().unwrap(),
            mtime,
            "a valid cached VAD must be reused, not rewritten"
        );
        assert!(sha256_matches(&second, VAD_SHA256, "silero-vad").unwrap());
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
