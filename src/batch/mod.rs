//! Parallel batch orchestration and the run summary.
//!
//! Files decode in parallel on a bounded pool and feed a single serial inference
//! stage over a bounded channel, so the (GPU) engine is never starved by CPU decode.
//! Up-to-date outputs are skipped; Ctrl-C stops gracefully after the in-flight file.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::cli::{Decoder, Format};
use crate::color;
use crate::diarize::{self, DiarizeOptions, Diarizer};
use crate::engine::{Backend, Engine, TranscribeOptions};
use crate::error::ScrybeError;
use crate::{audio, output};

/// Fallback parallelism when the platform cannot report it.
const DEFAULT_PARALLELISM: usize = 4;

/// The machine's usable parallelism, or `DEFAULT_PARALLELISM`. Always >= 1.
pub fn detected_parallelism() -> usize {
    std::thread::available_parallelism().map_or(DEFAULT_PARALLELISM, |n| n.get())
}

/// A progress-bar style from a template, falling back to the default bar if the
/// template fails to parse. Single source for the two batch bars.
fn styled_bar(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template).unwrap_or_else(|_| ProgressStyle::default_bar())
}

/// The real-time factor (`audio / wall`) as a display string, or `None` when no
/// wall time has elapsed. Single source for the `×RT` token shown live and in the
/// run summary.
fn realtime_factor(audio_secs: f64, wall_secs: f64) -> Option<String> {
    (wall_secs > 0.0).then(|| format!("{:.1}×RT", audio_secs / wall_secs))
}

/// Default concurrent-file count on CPU: half the cores (at least one), so decode
/// overlaps inference without oversubscribing.
fn cpu_default_jobs(cores: usize) -> usize {
    (cores / 2).max(1)
}

/// The auto CPU job count: half the cores, but clamped so its decode-buffer
/// reservation can't by itself exceed the memory budget — otherwise a zero-config
/// run on a high-core / modest-RAM box is refused by its own guard before decoding
/// anything. An explicit `--jobs N` is never clamped here (user intent fails loud).
fn auto_cpu_jobs(cores: usize, total_ram: Option<u64>) -> usize {
    let base = cpu_default_jobs(cores);
    match total_ram {
        Some(ram) => base.min(crate::model::max_jobs_fitting(ram, crate::cli::Model::Tiny)),
        None => base,
    }
    .max(1)
}

/// Resolve the concurrent-file count for the active backend. A single GPU
/// command queue serializes inference, so more than two concurrent jobs only
/// contend — clamp and say so. On CPU, default to half the cores (RAM-clamped).
pub fn resolve_jobs(
    requested: Option<usize>,
    backend: Backend,
    total_ram: Option<u64>,
) -> (usize, Option<String>) {
    let cores = detected_parallelism();
    match backend {
        Backend::Cpu => (
            requested
                .unwrap_or_else(|| auto_cpu_jobs(cores, total_ram))
                .max(1),
            None,
        ),
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

/// Skippable when not forced and every requested output is up to date.
fn decide_skip(
    file: &std::path::Path,
    force: bool,
    formats: &[Format],
    out_dir: Option<&std::path::Path>,
) -> bool {
    !force && output::outputs_up_to_date(file, formats, out_dir)
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

/// The process-global interrupt flag the SIGINT handler sets. The handler is
/// installed once and references this flag, so every `run` observes Ctrl-C — binding
/// the handler to a per-run flag would leave a second run deaf to SIGINT.
static INTERRUPT: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// The shared interrupt flag, installing the SIGINT handler on first use. First Ctrl-C
/// requests a graceful stop; a pre-existing handler (e.g. another caller's) is left in
/// place.
fn interrupt_flag() -> &'static Arc<AtomicBool> {
    INTERRUPT.get_or_init(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let handler_flag = Arc::clone(&flag);
        let _ = ctrlc::set_handler(move || handler_flag.store(true, Ordering::SeqCst));
        flag
    })
}

/// Run the batch. Returns the process exit code: 0 all clear, 20 partial failure.
/// Safe to call more than once per process: the SIGINT handler is installed once over
/// a shared flag, reset at the start of each run, so every run observes Ctrl-C.
pub fn run(
    engine: &Engine,
    files: &[PathBuf],
    cfg: &Config<'_>,
    mut diarize: Option<(&mut Diarizer, DiarizeOptions)>,
) -> Result<i32, ScrybeError> {
    let diarize_active = diarize.is_some();
    let interrupted = Arc::clone(interrupt_flag());
    interrupted.store(false, Ordering::SeqCst);

    let multi = MultiProgress::new();
    let aggregate = multi.add(ProgressBar::new(files.len() as u64));
    aggregate.set_style(styled_bar("{bar:30} {pos}/{len} files  {elapsed_precise}"));

    let mut results: Vec<Option<FileResult>> = (0..files.len()).map(|_| None).collect();

    // Pass the file index, not the path, through the channel — nothing is cloned.
    drive_pipeline(
        files.len(),
        cfg.jobs,
        &interrupted,
        |index| {
            let file = &files[index];
            if decide_skip(file, cfg.force, cfg.formats, cfg.out_dir) {
                DecodeMsg::Skip
            } else {
                match audio::load_audio(file, cfg.decoder) {
                    Ok(pcm) => DecodeMsg::Pcm(pcm),
                    Err(e) => DecodeMsg::Failed(e),
                }
            }
        },
        |index, decoded| {
            let file = &files[index];
            let name = file.file_name().map_or_else(
                || file.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            let bar = multi.add(ProgressBar::new(100));
            bar.set_style(styled_bar("  {prefix:.dim} {bar:24} {percent:>3}% {msg}"));
            bar.set_prefix(name.clone());
            bar.enable_steady_tick(Duration::from_millis(120));

            let started = Instant::now();
            let result = transcribe_one(
                engine,
                file,
                decoded,
                cfg,
                &bar,
                diarize.as_mut().map(|(d, o)| (&mut **d, &*o)),
                &interrupted,
            );
            bar.finish_and_clear();
            aggregate.inc(1);

            // `None` = interrupted between transcription and diarization: the
            // file wrote nothing and must not count as processed.
            if let Some((duration, outcome)) = result {
                results[index] = Some(FileResult {
                    name,
                    duration,
                    wall: started.elapsed().as_secs_f64(),
                    outcome,
                });
            }
        },
    )?;
    aggregate.finish_and_clear();

    let interrupted = interrupted.load(Ordering::SeqCst);
    print_summary(&results, interrupted);
    // The freshness check is mtime-only and knows nothing about options: a
    // re-run that adds --diarize would silently skip everything. Say so.
    if diarize_active {
        let skipped = results
            .iter()
            .flatten()
            .filter(|r| matches!(r.outcome, Outcome::Skipped))
            .count();
        if skipped > 0 {
            anstream::eprintln!(
                "{}",
                color::paint(
                    color::WARN,
                    &format!(
                        "{skipped} up-to-date output(s) skipped without speakers — rerun with --force to add them"
                    ),
                )
            );
        }
    }
    batch_exit_code(&results, interrupted, files.len())
}

/// Run `decode` in parallel on a `jobs`-wide pool, feeding a single serial `consume`
/// over a capacity-1 channel. On interrupt the consumer drains rather than breaks: a
/// bare break leaves producers parked on the full `tx.send`, hanging `thread::scope`.
/// Generic over the payload so the orchestration is unit-testable without a model.
fn drive_pipeline<M: Send>(
    count: usize,
    jobs: usize,
    interrupted: &AtomicBool,
    decode: impl Fn(usize) -> M + Send + Sync,
    mut consume: impl FnMut(usize, M),
) -> Result<(), ScrybeError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(|e| ScrybeError::Io {
            detail: e.to_string(),
        })?;
    let (tx, rx) = sync_channel::<(usize, M)>(1);

    std::thread::scope(|scope| {
        scope.spawn(move || {
            pool.install(|| {
                (0..count).into_par_iter().for_each(|index| {
                    if interrupted.load(Ordering::SeqCst) {
                        return;
                    }
                    let _ = tx.send((index, decode(index)));
                });
            });
        });

        for (index, item) in rx {
            // Drain, never break: see the deadlock note above.
            if interrupted.load(Ordering::SeqCst) {
                continue;
            }
            consume(index, item);
        }
    });
    Ok(())
}

/// The run's exit result, with a distinct Interrupted error when Ctrl-C stopped
/// the run before every file ran.
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
        Err(ScrybeError::PartialBatchFailure { failed, processed })
    } else if interrupted && processed < total {
        Err(ScrybeError::Interrupted {
            completed: processed,
            total,
        })
    } else {
        Ok(0)
    }
}

/// Transcribe (and optionally diarize) one decoded file. Returns `None` when
/// the run was interrupted between the transcription and diarization stages —
/// nothing was written, so the file must not count as processed.
#[allow(clippy::too_many_arguments)]
fn transcribe_one(
    engine: &Engine,
    file: &std::path::Path,
    decoded: DecodeMsg,
    cfg: &Config<'_>,
    bar: &ProgressBar,
    diarize: Option<(&mut Diarizer, &DiarizeOptions)>,
    interrupted: &AtomicBool,
) -> Option<(f64, Outcome)> {
    let pcm = match decoded {
        DecodeMsg::Pcm(pcm) => pcm,
        DecodeMsg::Skip => return Some((0.0, Outcome::Skipped)),
        DecodeMsg::Failed(e) => return Some((0.0, Outcome::Failed(e.to_string()))),
    };
    let duration = pcm.duration_secs();
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
    let mut transcript = match engine.transcribe(&pcm.samples, &cfg.options, on_progress) {
        Ok(t) => t,
        Err(e) => return Some((duration, Outcome::Failed(e.to_string()))),
    };
    if let Some((diarizer, options)) = diarize {
        // A Ctrl-C between the stages stops before diarization rather than
        // writing output the user would read as diarized.
        if interrupted.load(Ordering::SeqCst) {
            return None;
        }
        bar.set_message("diarizing");
        match diarizer.diarize(&pcm.samples, options) {
            Ok(turns) => diarize::assign_speakers(&mut transcript, &turns),
            Err(e) => return Some((duration, Outcome::Failed(e.to_string()))),
        }
    }
    let meta = output::Meta {
        model: cfg.model,
        duration,
    };
    match output::write_outputs(&transcript, file, cfg.formats, cfg.out_dir, &meta) {
        Ok(outputs) => Some((
            duration,
            Outcome::Done {
                outputs,
                language: transcript.language,
            },
        )),
        Err(e) => Some((duration, Outcome::Failed(e.to_string()))),
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn gpu_jobs_clamped_to_two() {
        let (jobs, note) = resolve_jobs(Some(4), Backend::Metal, None);
        assert_eq!(jobs, 2);
        assert!(note.is_some());

        let (jobs, note) = resolve_jobs(Some(1), Backend::Metal, None);
        assert_eq!(jobs, 1);
        assert!(note.is_none());
    }

    #[test]
    fn cpu_jobs_default_and_override() {
        assert!(resolve_jobs(None, Backend::Cpu, None).0 >= 1);
        // An explicit --jobs is honored as-is, never RAM-clamped (user intent).
        assert_eq!(
            resolve_jobs(Some(3), Backend::Cpu, Some(8 * 1024 * 1024 * 1024)).0,
            3
        );
    }

    #[test]
    fn auto_cpu_jobs_clamps_to_what_ram_allows() {
        const GB: u64 = 1024 * 1024 * 1024;
        // 8 GiB can't reserve 8 decode buffers, so the heuristic clamps below half-cores.
        assert!(auto_cpu_jobs(16, Some(8 * GB)) < cpu_default_jobs(16));
        assert_eq!(auto_cpu_jobs(16, None), cpu_default_jobs(16));
    }

    #[test]
    fn zero_config_cpu_run_is_never_refused() {
        // The honest guarantee: across core/RAM grids, the auto (jobs, model) pair a
        // flag-free CPU run resolves to is never rejected by guard_memory. No skip
        // clause — the auto job count is RAM-clamped so a fit always exists.
        const GB: u64 = 1024 * 1024 * 1024;
        for &cores in &[2usize, 8, 16, 24, 32, 64] {
            for &ram in &[2 * GB, 4 * GB, 8 * GB, 16 * GB] {
                let jobs = auto_cpu_jobs(cores, Some(ram));
                let model = crate::model::resolve_model(None, Some(ram), jobs);
                assert!(
                    !crate::model::would_exceed_memory(ram, model, jobs),
                    "zero-config refused: {cores} cores / {ram} bytes → {jobs} jobs, {model}"
                );
            }
        }
    }

    #[test]
    fn realtime_factor_formats_and_guards_zero_wall() {
        assert!(realtime_factor(10.0, 0.0).is_none());
        assert_eq!(realtime_factor(20.0, 10.0).as_deref(), Some("2.0×RT"));
    }

    #[test]
    fn cpu_default_is_half_the_cores() {
        assert_eq!(cpu_default_jobs(8), 4);
        assert_eq!(cpu_default_jobs(2), 1);
        assert_eq!(cpu_default_jobs(1), 1); // never zero
    }

    #[test]
    fn decide_skip_honors_force_and_freshness() {
        use std::time::{Duration, SystemTime};
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("clip.wav");
        std::fs::write(&input, b"x").unwrap();
        let formats = [Format::Txt];
        let out = dir.path().join("clip.txt");

        assert!(!decide_skip(&input, false, &formats, Some(dir.path())));

        std::fs::write(&out, b"y").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&out)
            .unwrap()
            .set_modified(SystemTime::now() + Duration::from_secs(60))
            .unwrap();
        assert!(decide_skip(&input, false, &formats, Some(dir.path())));

        assert!(!decide_skip(&input, true, &formats, Some(dir.path())));
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
                processed: 2
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
    fn exit_code_interrupted_after_completion_is_zero() {
        // Interrupt requested, but every file already finished → clean success.
        let results = [result(Outcome::Done {
            outputs: vec![],
            language: "en".to_owned(),
        })];
        assert!(matches!(batch_exit_code(&results, true, 1), Ok(0)));
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

    #[test]
    fn interrupt_mid_pipeline_drains_without_deadlock() {
        // Regression for the Ctrl-C deadlock: the old interrupt `break` left producers
        // parked on the capacity-1 channel, hanging thread::scope. The watchdog turns a
        // regression hang into a clean failure instead of wedging the run.
        use std::sync::atomic::AtomicUsize;
        use std::sync::mpsc;

        let interrupted = Arc::new(AtomicBool::new(false));
        let consumed = Arc::new(AtomicUsize::new(0));
        let (done_tx, done_rx) = mpsc::channel();

        let drive_flag = Arc::clone(&interrupted);
        let drive_consumed = Arc::clone(&consumed);
        let worker = std::thread::spawn(move || {
            let result = drive_pipeline(
                1000,
                4,
                &drive_flag,
                |index| index,
                |_index, _item| {
                    // Slow consumer so producers park on the full channel before the interrupt.
                    drive_consumed.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(5));
                },
            );
            let _ = done_tx.send(result.is_ok());
        });

        // Interrupt once the pipeline is full and producers are parked on the channel.
        std::thread::sleep(Duration::from_millis(80));
        interrupted.store(true, Ordering::SeqCst);

        // None on timeout (deadlock), Some(true) on a clean drained return.
        let outcome = done_rx.recv_timeout(Duration::from_secs(10)).ok();
        assert_eq!(
            outcome,
            Some(true),
            "drive_pipeline deadlocked on interrupt instead of draining (or errored)"
        );
        worker.join().expect("worker thread joins");
        assert!(
            consumed.load(Ordering::SeqCst) < 1000,
            "interrupt did not stop new work"
        );
    }
}
