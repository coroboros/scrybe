//! scrybe entry point: parse the CLI, set up color, dispatch, and translate any
//! failure into its stable exit code.

use clap::{Parser, ValueEnum};

use scrybe::cli::{self, Cli, Command, Model, ModelsAction, Task};
use scrybe::error::ScrybeError;
use scrybe::{audio, batch, color, engine, model, output};

/// Exit code for argument/usage problems, matching clap's own convention.
const USAGE_ERROR: i32 = 2;

fn main() {
    std::process::exit(run());
}

/// Parse, configure color, dispatch. Returns the process exit code.
fn run() -> i32 {
    let cli = Cli::parse();
    color::init(cli.no_color);

    match dispatch(&cli) {
        Ok(code) => code,
        Err(err) => {
            print_error(&err);
            err.exit_code()
        }
    }
}

fn dispatch(cli: &Cli) -> Result<i32, ScrybeError> {
    match &cli.command {
        Some(Command::Models { action }) => {
            run_models(action, cli.offline)?;
            Ok(0)
        }
        None => transcribe(cli),
    }
}

/// The default action. The engine arrives in a later milestone; here we resolve
/// inputs (failing loud on a missing path) and print the plan.
fn transcribe(cli: &Cli) -> Result<i32, ScrybeError> {
    if cli.paths.is_empty() {
        print_usage_hint();
        return Ok(USAGE_ERROR);
    }

    for path in &cli.paths {
        if !path.exists() {
            return Err(ScrybeError::FileNotFound { path: path.clone() });
        }
    }

    // Resolve concurrency and the effective model up front so the memory guard
    // checks what will actually run. Inference is serial (one shared context), so
    // `--jobs` widens decode-ahead, not the number of concurrent inferences.
    let backend = engine::active_backend();
    let (jobs, clamp_note) = batch::resolve_jobs(cli.jobs, backend);
    let model = model::resolve_model(cli.model, model::total_memory());
    model::guard_memory(model, jobs)?;

    if let Some(code) = validate_model_capabilities(model, cli) {
        return Ok(code);
    }

    if let Some(dir) = cli.out_dir.as_deref() {
        std::fs::create_dir_all(dir).map_err(|e| ScrybeError::Io {
            detail: format!("could not create out-dir {}: {e}", dir.display()),
        })?;
    }

    let files = audio::discover(&cli.paths, cli.recursive);
    let formats = cli.effective_formats();
    print_plan(cli, model, &formats);

    if files.is_empty() {
        anstream::eprintln!(
            "{}",
            color::paint(color::WARN, "no audio files found in the given paths.")
        );
        return Ok(0);
    }

    // Fail loud rather than silently overwrite when two inputs map to one output.
    // Checked before the dry-run gate and model load, so a doomed run fails fast
    // and the guard is reachable without a model on disk.
    if let Some(collision) = output::first_collision(&files, &formats, cli.out_dir.as_deref()) {
        return Ok(usage_error(&format!(
            "output collision — {collision}. Use distinct names or an `--out-dir` per source."
        )));
    }

    if cli.dry_run {
        for file in &files {
            anstream::println!(
                "  {}  {}",
                file.display(),
                color::paint(color::DIM, "(dry-run)")
            );
        }
        return Ok(0);
    }

    let model_path = model::ensure_available(model, cli.offline)?;
    // VAD is the mandated correctness floor and is bundled, so it is always on.
    let vad_path = model::ensure_vad()?;
    let engine = engine::Engine::load(&model_path, Some(&vad_path))?;

    // Inference runs one file at a time, so the active job gets every core;
    // `--jobs` only widens the decode-ahead pool, never the inference threads.
    let threads = cli
        .threads
        .unwrap_or_else(|| batch::detected_parallelism().max(1));
    let options = engine::TranscribeOptions {
        language: cli.lang.clone(),
        translate: cli.task == Task::Translate,
        threads,
    };
    let model_name = model.to_string();

    // `--json` on a single file streams to stdout for piping; no batch UI.
    if cli.json && files.len() == 1 {
        let pcm = audio::load_audio(&files[0], cli.decoder)?;
        let transcript = engine.transcribe(&pcm.samples, &options, |_| {})?;
        let meta = output::Meta {
            model: &model_name,
            duration: pcm.duration_secs(),
        };
        anstream::println!("{}", output::render(&transcript, formats[0], &meta));
        return Ok(0);
    }
    if cli.json {
        anstream::eprintln!(
            "{}",
            color::paint(
                color::WARN,
                "--json with multiple inputs: writing .json sidecars (stdout streaming is single-file only)",
            )
        );
    }

    anstream::eprintln!(
        "{}",
        color::paint(
            color::DIM,
            &format!("backend {backend} · model {model} · {jobs} job(s)"),
        )
    );
    if let Some(note) = clamp_note {
        anstream::eprintln!("{}", color::paint(color::WARN, &note));
    }

    let config = batch::Config {
        decoder: cli.decoder,
        options,
        formats: &formats,
        out_dir: cli.out_dir.as_deref(),
        model: &model_name,
        force: cli.force,
        jobs,
    };
    batch::run(&engine, &files, &config)
}

/// Reject model + task/language combinations the model cannot serve, before any
/// work runs. These are configuration errors, so they exit like a usage error.
fn validate_model_capabilities(model: Model, cli: &Cli) -> Option<i32> {
    let info = model::info(model);
    if cli.task == Task::Translate && !info.can_translate {
        return Some(usage_error(&format!(
            "model `{model}` cannot translate; use a translation-capable model such as `large-v3`",
        )));
    }
    if let Some(lang) = cli.lang.as_deref()
        && !lang.eq_ignore_ascii_case("en")
        && !info.multilingual
    {
        return Some(usage_error(&format!(
            "model `{model}` is English-only; it cannot transcribe `--lang {lang}`",
        )));
    }
    None
}

fn run_models(action: &ModelsAction, offline: bool) -> Result<(), ScrybeError> {
    match action {
        ModelsAction::List => {
            list_models();
            Ok(())
        }
        ModelsAction::Pull { model } => {
            let path = model::ensure_available(*model, offline)?;
            anstream::println!(
                "{} {}",
                color::paint(color::SUCCESS, "pulled"),
                path.display()
            );
            Ok(())
        }
        ModelsAction::Remove { model } => {
            match model::cached_path(*model) {
                Some(path) => {
                    model::evict(&path);
                    anstream::println!("{} {model}", color::paint(color::SUCCESS, "removed"));
                }
                None => {
                    anstream::println!("{} {model} is not cached", color::paint(color::DIM, "—"))
                }
            }
            Ok(())
        }
        ModelsAction::Path => {
            anstream::println!("{}", model::cache_dir().display());
            Ok(())
        }
    }
}

fn list_models() {
    anstream::println!(
        "{}  {}",
        color::paint(color::ACCENT, "Whisper models"),
        color::paint(color::DIM, &format!("(default: {})", cli::DEFAULT_MODEL)),
    );
    for model in Model::value_variants() {
        let info = model::info(*model);
        let cached = if model::cached_path(*model).is_some() {
            color::paint(color::SUCCESS, "cached")
        } else {
            color::paint(color::DIM, "—")
        };
        let default = if *model == cli::DEFAULT_MODEL {
            color::paint(color::DIM, "  (default)")
        } else {
            String::new()
        };
        anstream::println!(
            "  {:<18} {:>9}  {cached}{default}",
            model.to_string(),
            model::human_size(info.size),
        );
    }
}

/// Print the fully-resolved invocation. Reports every flag so the plan is the
/// single source of truth for what a run would do.
fn print_plan(cli: &Cli, model: Model, formats: &[cli::Format]) {
    let optional =
        |value: &Option<usize>| value.map_or_else(|| "auto".to_owned(), |n| n.to_string());
    let lang = cli.lang.as_deref().unwrap_or("auto");
    let formats = formats
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let out_dir = cli
        .out_dir
        .as_ref()
        .map_or_else(|| "sidecar".to_owned(), |dir| dir.display().to_string());

    let config = format!(
        "model={} task={} lang={} format={} jobs={} threads={} out-dir={} decoder={} recursive={} force={} json={} offline={} dry-run={}",
        model,
        cli.task,
        lang,
        formats,
        optional(&cli.jobs),
        optional(&cli.threads),
        out_dir,
        cli.decoder,
        cli.recursive,
        cli.force,
        cli.json,
        cli.offline,
        cli.dry_run,
    );
    // Status banner on stderr, keeping stdout clean for `--json` piping.
    anstream::eprintln!(
        "{}  {}",
        color::paint(color::ACCENT, "scrybe"),
        color::paint(color::DIM, &config),
    );
}

fn print_error(err: &ScrybeError) {
    eprint_error(&err.to_string());
}

fn eprint_error(message: &str) {
    anstream::eprintln!("{} {message}", color::paint(color::ERROR, "error:"));
}

/// Print a configuration/usage error and return the clap-aligned exit code. These
/// failures live outside `ScrybeError` by design (see `error.rs`); this keeps the
/// print-then-exit-2 contract in one place.
fn usage_error(message: &str) -> i32 {
    eprint_error(message);
    USAGE_ERROR
}

fn print_usage_hint() {
    anstream::eprintln!(
        "{}: no input paths. Pass an audio file or folder, or run {} for options.",
        color::paint(color::ACCENT, "scrybe"),
        color::paint(color::DIM, "scrybe --help"),
    );
}
