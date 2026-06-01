//! Parallel batch orchestration and the run summary.
//!
//! Files are decoded in parallel on a bounded worker pool and fed through a
//! bounded channel into a single serial inference stage, so the (GPU) engine is
//! never starved while CPU decode runs ahead. Up-to-date outputs are skipped,
//! Ctrl-C stops gracefully after the in-flight file, and a colored table
//! summarizes the run.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::cli::{Decoder, Format};
use crate::color;
use crate::engine::{Backend, Engine, TranscribeOptions};
use crate::error::ScrybeError;
use crate::{audio, output};

/// Fallback parallelism when the platform cannot report it.
const DEFAULT_PARALLELISM: usize = 4;

/// The machine's usable parallelism, or a sane default.
pub fn detected_parallelism() -> usize {
    std::thread::available_parallelism().map_or(DEFAULT_PARALLELISM, |n| n.get())
}

/// The real-time factor (`audio / wall`) as a display string, or `None` when no
/// wall time has elapsed. Single source for the `×RT` token shown live and in the
/// run summary.
fn realtime_factor(audio_secs: f64, wall_secs: f64) -> Option<String> {
    (wall_secs > 0.0).then(|| format!("{:.1}×RT", audio_secs / wall_secs))
}

/// Resolve the concurrent-file count for the active backend. A single GPU
/// command queue serializes inference, so more than two concurrent jobs only
/// contend — clamp and say so. On CPU, default to half the cores.
pub fn resolve_jobs(requested: Option<usize>, backend: Backend) -> (usize, Option<String>) {
    let cores = detected_parallelism();
    match backend {
        Backend::Cpu => (requested.unwrap_or((cores / 2).max(1)).max(1), None),
        _ => {
            let want = requested.unwrap_or(1).max(1);
            if want > 2 {
                (
                    2,
                    Some(format!(
                        "{want} jobs requested; clamped to 2 — one GPU serializes inference"
                    )),
                )
            } else {
                (want, None)
            }
        }
    }
}

/// Everything one batch run needs beyond the files and engine.
pub struct Config<'a> {
    pub decoder: Decoder,
    pub options: TranscribeOptions,
    pub formats: &'a [Format],
    pub out_dir: Option<&'a std::path::Path>,
    pub model: &'a str,
    pub force: bool,
    pub jobs: usize,
}

/// What the parallel decode stage hands the serial inference stage per file.
enum DecodeMsg {
    Pcm(audio::AudioPcm),
    Skip,
    Failed(ScrybeError),
}

enum Outcome {
    Done {
        outputs: Vec<PathBuf>,
        language: String,
    },
    Skipped,
    Failed(String),
}

struct FileResult {
    name: String,
    duration: f64,
    wall: f64,
    outcome: Outcome,
}

/// Run the batch. Returns the process exit code: 0 all clear, 20 partial failure.
pub fn run(engine: &Engine, files: &[PathBuf], cfg: &Config<'_>) -> Result<i32, ScrybeError> {
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&interrupted);
        // First Ctrl-C requests a graceful stop; ignore if a handler already exists.
        let _ = ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst));
    }

    let multi = MultiProgress::new();
    let aggregate = multi.add(ProgressBar::new(files.len() as u64));
    aggregate.set_style(
        ProgressStyle::with_template("{bar:30} {pos}/{len} files  {elapsed_precise}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.jobs)
        .build()
        .map_err(|e| ScrybeError::Io {
            detail: e.to_string(),
        })?;

    // The index identifies the file; the consumer borrows `files[index]`, so no
    // path is cloned through the channel. Capacity 1 stages just the next file —
    // enough to keep inference fed — so peak resident decode buffers stay within
    // the memory guard's per-job budget rather than doubling with the channel.
    let (tx, rx) = sync_channel::<(usize, DecodeMsg)>(1);
    let mut results: Vec<Option<FileResult>> = (0..files.len()).map(|_| None).collect();

    std::thread::scope(|scope| {
        // Producer: decode in parallel, skipping inputs whose outputs are current.
        let producer_interrupt = Arc::clone(&interrupted);
        scope.spawn(move || {
            pool.install(|| {
                files.par_iter().enumerate().for_each(|(index, file)| {
                    if producer_interrupt.load(Ordering::SeqCst) {
                        return;
                    }
                    let msg = if !cfg.force
                        && output::outputs_up_to_date(file, cfg.formats, cfg.out_dir)
                    {
                        DecodeMsg::Skip
                    } else {
                        match audio::load_audio(file, cfg.decoder) {
                            Ok(pcm) => DecodeMsg::Pcm(pcm),
                            Err(e) => DecodeMsg::Failed(e),
                        }
                    };
                    let _ = tx.send((index, msg));
                });
            });
        });

        // Consumer: serial inference + write.
        for (index, decoded) in rx {
            if interrupted.load(Ordering::SeqCst) {
                break;
            }
            let file = &files[index];
            let name = file.file_name().map_or_else(
                || file.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            let bar = multi.add(ProgressBar::new(100));
            bar.set_style(
                ProgressStyle::with_template("  {prefix:.dim} {bar:24} {percent:>3}% {msg}")
                    .unwrap_or_else(|_| ProgressStyle::default_bar()),
            );
            bar.set_prefix(name.clone());
            bar.enable_steady_tick(Duration::from_millis(120));

            let started = Instant::now();
            let (duration, outcome) = transcribe_one(engine, file, decoded, cfg, &bar);
            bar.finish_and_clear();
            aggregate.inc(1);

            results[index] = Some(FileResult {
                name,
                duration,
                wall: started.elapsed().as_secs_f64(),
                outcome,
            });
        }
    });
    aggregate.finish_and_clear();

    let interrupted = interrupted.load(Ordering::SeqCst);
    print_summary(&results, interrupted);
    batch_exit_code(&results, interrupted, files.len())
}

/// Decide the run's exit result from the per-file outcomes: partial failure when
/// any file failed, a distinct interrupted result when Ctrl-C stopped the run
/// before every file ran, otherwise success.
fn batch_exit_code(
    results: &[Option<FileResult>],
    interrupted: bool,
    total: usize,
) -> Result<i32, ScrybeError> {
    let processed = results.iter().flatten().count();
    let failed = results
        .iter()
        .flatten()
        .filter(|r| matches!(r.outcome, Outcome::Failed(_)))
        .count();
    if failed > 0 {
        Err(ScrybeError::PartialBatchFailure {
            failed,
            total: processed,
        })
    } else if interrupted && processed < total {
        Err(ScrybeError::Interrupted {
            completed: processed,
            total,
        })
    } else {
        Ok(0)
    }
}

fn transcribe_one(
    engine: &Engine,
    file: &std::path::Path,
    decoded: DecodeMsg,
    cfg: &Config<'_>,
    bar: &ProgressBar,
) -> (f64, Outcome) {
    let pcm = match decoded {
        DecodeMsg::Pcm(pcm) => pcm,
        DecodeMsg::Skip => return (0.0, Outcome::Skipped),
        DecodeMsg::Failed(e) => return (0.0, Outcome::Failed(e.to_string())),
    };
    let duration = pcm.duration_secs();
    // Drive the per-file bar from whisper's progress callback, with live ×RT.
    let progress_bar = bar.clone();
    let started = Instant::now();
    let on_progress = move |percent: i32| {
        let percent = percent.clamp(0, 100);
        progress_bar.set_position(percent as u64);
        let done = f64::from(percent) / 100.0 * duration;
        if let Some(rt) = realtime_factor(done, started.elapsed().as_secs_f64()) {
            progress_bar.set_message(rt);
        }
    };
    let transcript = match engine.transcribe(&pcm.samples, &cfg.options, on_progress) {
        Ok(t) => t,
        Err(e) => return (duration, Outcome::Failed(e.to_string())),
    };
    let meta = output::Meta {
        model: cfg.model,
        duration,
    };
    match output::write_outputs(&transcript, file, cfg.formats, cfg.out_dir, &meta) {
        Ok(outputs) => (
            duration,
            Outcome::Done {
                outputs,
                language: transcript.language,
            },
        ),
        Err(e) => (duration, Outcome::Failed(e.to_string())),
    }
}

fn print_summary(results: &[Option<FileResult>], interrupted: bool) {
    let mut done = 0;
    let mut skipped = 0;
    let mut failed = 0;
    let mut total_audio = 0.0;
    let mut total_wall = 0.0;

    for result in results.iter().flatten() {
        let (status, detail) = match &result.outcome {
            Outcome::Done { outputs, language } => {
                done += 1;
                total_audio += result.duration;
                total_wall += result.wall;
                let rt =
                    realtime_factor(result.duration, result.wall).unwrap_or_else(|| "—".to_owned());
                (
                    color::paint(color::SUCCESS, "ok"),
                    format!(
                        "[{language}] {:.1}s in {:.1}s ({rt}) → {}",
                        result.duration,
                        result.wall,
                        outputs
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                )
            }
            Outcome::Skipped => {
                skipped += 1;
                (color::paint(color::DIM, "skip"), "up to date".to_owned())
            }
            Outcome::Failed(reason) => {
                failed += 1;
                (color::paint(color::ERROR, "fail"), reason.clone())
            }
        };
        anstream::eprintln!(
            "  {status:>4}  {}  {}",
            result.name,
            color::paint(color::DIM, &detail)
        );
    }

    let speed =
        realtime_factor(total_audio, total_wall).map_or(String::new(), |rt| format!(" · {rt}"));
    let footer = format!(
        "{done} done · {skipped} skipped · {failed} failed{speed}{}",
        if interrupted { " · interrupted" } else { "" }
    );
    anstream::eprintln!("{}", color::paint(color::ACCENT, &footer));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_jobs_clamped_to_two() {
        let (jobs, note) = resolve_jobs(Some(4), Backend::Metal);
        assert_eq!(jobs, 2);
        assert!(note.is_some());

        let (jobs, note) = resolve_jobs(Some(1), Backend::Metal);
        assert_eq!(jobs, 1);
        assert!(note.is_none());
    }

    #[test]
    fn cpu_jobs_default_and_override() {
        assert!(resolve_jobs(None, Backend::Cpu).0 >= 1);
        assert_eq!(resolve_jobs(Some(3), Backend::Cpu).0, 3);
    }

    fn result(outcome: Outcome) -> Option<FileResult> {
        Some(FileResult {
            name: "f".to_owned(),
            duration: 0.0,
            wall: 0.0,
            outcome,
        })
    }

    #[test]
    fn exit_code_partial_failure_is_20() {
        let results = [
            result(Outcome::Done {
                outputs: vec![],
                language: "en".to_owned(),
            }),
            result(Outcome::Failed("bad".to_owned())),
        ];
        assert!(matches!(
            batch_exit_code(&results, false, 2),
            Err(ScrybeError::PartialBatchFailure {
                failed: 1,
                total: 2
            })
        ));
    }

    #[test]
    fn exit_code_all_done_is_zero() {
        let results = [
            result(Outcome::Done {
                outputs: vec![],
                language: "en".to_owned(),
            }),
            result(Outcome::Skipped),
        ];
        assert!(matches!(batch_exit_code(&results, false, 2), Ok(0)));
    }

    #[test]
    fn exit_code_interrupted_partial_is_distinct() {
        // One file done, one never processed, run interrupted.
        let results = [
            result(Outcome::Done {
                outputs: vec![],
                language: "en".to_owned(),
            }),
            None,
        ];
        assert!(matches!(
            batch_exit_code(&results, true, 2),
            Err(ScrybeError::Interrupted {
                completed: 1,
                total: 2
            })
        ));
    }
}
