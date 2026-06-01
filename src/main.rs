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

    // Refuse before doing work if the model + job count cannot fit in memory.
    model::guard_memory(cli.model, cli.jobs.unwrap_or(1))?;

    if let Some(code) = validate_model_capabilities(cli) {
        return Ok(code);
    }

    let files = audio::discover(&cli.paths, cli.recursive);
    print_plan(cli);

    if files.is_empty() {
        anstream::eprintln!(
            "{}",
            color::paint(color::WARN, "no audio files found in the given paths.")
        );
        return Ok(0);
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

    let model_path = model::ensure_available(cli.model, cli.offline)?;
    let engine = engine::Engine::load(&model_path)?;

    let options = engine::TranscribeOptions {
        language: cli.lang.clone(),
        translate: cli.task == Task::Translate,
        threads: cli.threads.unwrap_or_else(default_threads),
    };
    let model_name = cli.model.to_string();

    // `--json` on a single file streams to stdout for piping; no batch UI.
    if cli.json && files.len() == 1 {
        let file = &files[0];
        let pcm = audio::load_audio(file, cli.decoder)?;
        let transcript = engine.transcribe(&pcm.samples, &options)?;
        let meta = output::Meta {
            model: &model_name,
            duration: pcm.duration_secs(),
        };
        anstream::println!("{}", output::render(&transcript, cli::Format::Json, &meta));
        return Ok(0);
    }

    let backend = engine::active_backend();
    let (jobs, clamp_note) = batch::resolve_jobs(cli.jobs, backend);
    anstream::eprintln!(
        "{}",
        color::paint(
            color::DIM,
            &format!("backend {backend} · model {} · {jobs} job(s)", cli.model),
        )
    );
    if let Some(note) = clamp_note {
        anstream::eprintln!("{}", color::paint(color::WARN, &note));
    }

    let formats: Vec<cli::Format> = if cli.json {
        vec![cli::Format::Json]
    } else {
        cli.format.clone()
    };
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

/// Default decode threads: the machine's parallelism, or 4 when unknown.
fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// Reject model + task/language combinations the model cannot serve, before any
/// work runs. These are configuration errors, so they exit like a usage error.
fn validate_model_capabilities(cli: &Cli) -> Option<i32> {
    let info = model::info(cli.model);
    if cli.task == Task::Translate && !info.can_translate {
        eprint_error(&format!(
            "model `{}` cannot translate; use a translation-capable model such as `large-v3`",
            cli.model
        ));
        return Some(USAGE_ERROR);
    }
    if let Some(lang) = cli.lang.as_deref()
        && !lang.eq_ignore_ascii_case("en")
        && !info.multilingual
    {
        eprint_error(&format!(
            "model `{}` is English-only; it cannot transcribe `--lang {lang}`",
            cli.model
        ));
        return Some(USAGE_ERROR);
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
                    if let Ok(real) = std::fs::canonicalize(&path) {
                        let _ = std::fs::remove_file(&real);
                    }
                    let _ = std::fs::remove_file(&path);
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
fn print_plan(cli: &Cli) {
    let optional =
        |value: &Option<usize>| value.map_or_else(|| "auto".to_owned(), |n| n.to_string());
    let lang = cli.lang.as_deref().unwrap_or("auto");
    let formats = cli
        .format
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
        cli.model,
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

fn print_usage_hint() {
    anstream::eprintln!(
        "{}: no input paths. Pass an audio file or folder, or run {} for options.",
        color::paint(color::ACCENT, "scrybe"),
        color::paint(color::DIM, "scrybe --help"),
    );
}
