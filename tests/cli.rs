//! CLI surface acceptance tests — the WS-1 contract.
//!
//! Pins the help surface, color on/off in both directions, the invalid-model
//! error, the no-panic guarantee, and the file-not-found exit code. `assert_cmd`
//! captures output through a pipe (never a TTY), so a bare run strips color
//! exactly as a real pipe would; `CLICOLOR_FORCE=1` stands in for a color-capable
//! terminal to assert the positive direction.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests may unwrap; the binary may not

use predicates::prelude::*;

mod common;
use common::{scrybe, tiny_cached};

/// ANSI escape introducer — its presence means color was emitted.
const ESC: &str = "\u{1b}";

#[test]
fn help_lists_every_flag_and_subcommand_and_exits_zero() {
    scrybe()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"))
        .stdout(predicate::str::contains("--lang"))
        .stdout(predicate::str::contains("--task"))
        .stdout(predicate::str::contains("--format"))
        .stdout(predicate::str::contains("--out-dir"))
        .stdout(predicate::str::contains("--jobs"))
        .stdout(predicate::str::contains("--threads"))
        .stdout(predicate::str::contains("--force"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--decoder"))
        .stdout(predicate::str::contains("--no-color"))
        .stdout(predicate::str::contains("--json"))
        .stdout(predicate::str::contains("--offline"))
        .stdout(predicate::str::contains("--recursive"))
        .stdout(predicate::str::contains("models"));
}

#[test]
fn piped_stdout_has_no_ansi() {
    scrybe()
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains(ESC).not());
}

#[test]
fn no_color_env_strips_ansi() {
    scrybe()
        .env("NO_COLOR", "1")
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains(ESC).not());
}

#[test]
fn no_color_flag_strips_ansi() {
    scrybe()
        .args(["--no-color", "models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains(ESC).not());
}

#[test]
fn clicolor_force_emits_ansi() {
    scrybe()
        .env("CLICOLOR_FORCE", "1")
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains(ESC));
}

#[test]
fn invalid_model_lists_valid_models_and_exits_nonzero() {
    scrybe()
        .args(["--model", "bogus"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("tiny"))
        .stderr(predicate::str::contains("large-v3-turbo"))
        .stderr(predicate::str::contains("distil-large-v3.5"));
}

#[test]
fn bad_numeric_input_does_not_panic() {
    scrybe()
        .args(["--jobs", "abc", "Cargo.toml"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("panicked").not());
}

#[test]
fn missing_input_path_exits_file_not_found() {
    scrybe()
        .arg("definitely-not-a-real-file.xyz")
        .assert()
        .failure()
        .code(14)
        .stderr(predicate::str::contains("no such file"));
}

#[test]
fn json_single_file_streams_clean_stdout() {
    if !tiny_cached() {
        eprintln!("skipping: tiny model not cached");
        return;
    }
    // stdout must be only the JSON document — the status banner goes to stderr.
    // No --lang, so "en" here also exercises the auto-detect language arm.
    scrybe()
        .args(["--model", "tiny", "--json", "tests/fixtures/speech/en.wav"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"schema_version\": 1"))
        .stdout(predicate::str::contains("\"language\": \"en\""))
        .stdout(predicate::str::contains("model=").not());
}

#[test]
fn out_dir_redirects_output_away_from_input() {
    if !tiny_cached() {
        eprintln!("skipping: tiny model not cached");
        return;
    }
    let out = tempfile::tempdir().unwrap();
    scrybe()
        .args(["--model", "tiny", "--format", "txt", "--out-dir"])
        .arg(out.path())
        .arg("tests/fixtures/speech/en.wav")
        .assert()
        .success();
    assert!(
        out.path().join("en.txt").exists(),
        "output should land in --out-dir"
    );
    assert!(
        !std::path::Path::new("tests/fixtures/speech/en.txt").exists(),
        "output must not be written beside the input",
    );
}

#[test]
fn dry_run_lists_files_without_writing() {
    let out = tempfile::tempdir().unwrap();
    scrybe()
        .args(["--dry-run", "--format", "txt", "--out-dir"])
        .arg(out.path())
        .arg("tests/fixtures/speech/en.wav")
        .assert()
        .success()
        .stdout(predicate::str::contains("dry-run"));
    assert!(
        !out.path().join("en.txt").exists(),
        "dry-run must not write output"
    );
}

#[test]
fn mixed_batch_reports_failure_and_exits_20() {
    if !tiny_cached() {
        eprintln!("skipping: tiny model not cached");
        return;
    }
    let out = tempfile::tempdir().unwrap();
    scrybe()
        .args(["--model", "tiny", "--format", "txt", "--out-dir"])
        .arg(out.path())
        .args([
            "tests/fixtures/speech/en.wav",
            "tests/fixtures/aac/he-aac.m4a",
        ])
        .assert()
        .failure()
        .code(20)
        .stderr(predicate::str::contains("en.wav"))
        .stderr(predicate::str::contains("he-aac.m4a"));
    // The good file completes despite the bad one (no abort-on-first-failure).
    assert!(
        out.path().join("en.txt").exists(),
        "the good file should still be written"
    );
}

#[test]
fn skips_up_to_date_output_unless_forced() {
    if !tiny_cached() {
        eprintln!("skipping: tiny model not cached");
        return;
    }
    let out = tempfile::tempdir().unwrap();
    let transcribe = |force: bool| {
        let mut cmd = scrybe();
        cmd.args(["--model", "tiny", "--format", "txt"]);
        if force {
            cmd.arg("--force");
        }
        cmd.arg("--out-dir")
            .arg(out.path())
            .arg("tests/fixtures/speech/en.wav");
        cmd
    };
    transcribe(false)
        .assert()
        .success()
        .stderr(predicate::str::contains("ok"));
    // Second run: output is current → skipped.
    transcribe(false)
        .assert()
        .success()
        .stderr(predicate::str::contains("up to date"));
    // --force reprocesses.
    transcribe(true)
        .assert()
        .success()
        .stderr(predicate::str::contains("ok"));
}

#[test]
fn silence_produces_no_transcript() {
    if !tiny_cached() {
        eprintln!("skipping: tiny model not cached");
        return;
    }
    let out = tempfile::tempdir().unwrap();
    scrybe()
        .args(["--model", "tiny", "--format", "txt", "--out-dir"])
        .arg(out.path())
        .arg("tests/fixtures/speech/silence.wav")
        .assert()
        .success();
    let text = std::fs::read_to_string(out.path().join("silence.txt"))
        .expect("silence.txt must be written");
    assert!(
        text.trim().is_empty(),
        "silence must not hallucinate, got: {text:?}"
    );
}

#[test]
fn json_multi_file_writes_sidecars_into_created_out_dir() {
    if !tiny_cached() {
        eprintln!("skipping: tiny model not cached");
        return;
    }
    let parent = tempfile::tempdir().unwrap();
    let out = parent.path().join("fresh"); // does not exist yet → exercises create_dir_all
    // --format txt is passed but --json must override it: only .json is written.
    scrybe()
        .args(["--model", "tiny", "--json", "--format", "txt", "--out-dir"])
        .arg(&out)
        .args([
            "tests/fixtures/speech/en.wav",
            "tests/fixtures/speech/silence.wav",
        ])
        .assert()
        .success()
        // Multiple inputs → .json sidecars (not stdout streaming).
        .stdout(predicate::str::contains("schema_version").not())
        .stderr(predicate::str::contains("writing .json sidecars"));
    assert!(
        out.join("en.json").exists(),
        "en.json sidecar in the created dir"
    );
    assert!(
        out.join("silence.json").exists(),
        "silence.json sidecar in the created dir"
    );
    assert!(
        !out.join("en.txt").exists(),
        "--json overrides --format txt: no .txt sidecar"
    );
}

#[cfg(unix)]
#[test]
fn ctrl_c_stops_gracefully_with_partial_exit() {
    if !tiny_cached() {
        eprintln!("skipping: tiny model not cached");
        return;
    }
    use std::process::{Command as Proc, Stdio};
    use std::time::{Duration, Instant};

    let work = tempfile::tempdir().unwrap();
    let inputs = work.path().join("in");
    let out = work.path().join("out");
    std::fs::create_dir_all(&inputs).unwrap();
    let total = 16;
    for i in 0..total {
        std::fs::copy(
            "tests/fixtures/speech/en.wav",
            inputs.join(format!("clip{i:02}.wav")),
        )
        .unwrap();
    }

    // Two formats so we can assert the per-file output set is never half-written.
    let mut child = Proc::new(env!("CARGO_BIN_EXE_scrybe"))
        .args(["--model", "tiny", "--format", "txt,srt", "--out-dir"])
        .arg(&out)
        .arg(&inputs)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn scrybe");

    let count_ext = |ext: &'static str| {
        std::fs::read_dir(&out)
            .map(|d| {
                d.flatten()
                    .filter(|e| e.path().extension().is_some_and(|x| x == ext))
                    .count()
            })
            .unwrap_or(0)
    };

    // Once the run has produced its first output, interrupt mid-batch.
    let start = Instant::now();
    while count_ext("txt") == 0 && start.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(50));
    }
    // ctrlc catches SIGINT and requests a graceful stop (no kill -9).
    Proc::new("kill")
        .arg("-INT")
        .arg(child.id().to_string())
        .status()
        .expect("send SIGINT");

    let status = child.wait().expect("wait for scrybe");
    assert_eq!(status.code(), Some(20), "interrupted run must exit 20");
    let produced = count_ext("txt");
    assert!(
        (1..total).contains(&produced),
        "partial completion expected, got {produced}/{total}"
    );
    // Graceful stop finishes the in-flight file's whole output set, so every
    // completed stem has both sidecars — never a half-written set.
    assert_eq!(
        produced,
        count_ext("srt"),
        "each completed file must write both txt and srt, or neither"
    );
    // And every produced sidecar is fully written, not truncated mid-cue.
    let mut srt_seen = 0;
    for entry in std::fs::read_dir(&out).unwrap().flatten() {
        let path = entry.path();
        match path.extension().and_then(|x| x.to_str()) {
            Some("srt") => {
                srt_seen += 1;
                assert!(
                    std::fs::read_to_string(&path).unwrap().contains("-->"),
                    "every completed .srt must be a well-formed cue file: {path:?}"
                );
            }
            Some("txt") => assert!(
                !std::fs::read_to_string(&path).unwrap().is_empty(),
                "every completed .txt must be non-empty: {path:?}"
            ),
            _ => {}
        }
    }
    assert!(srt_seen >= 1, "expected at least one completed .srt");
}

#[test]
fn recursive_flag_controls_directory_descent() {
    // Model-free: --dry-run exercises the clap→discover seam without inference.
    let root = tempfile::tempdir().unwrap();
    std::fs::copy("tests/fixtures/speech/en.wav", root.path().join("top.wav")).unwrap();
    let sub = root.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    std::fs::copy("tests/fixtures/speech/en.wav", sub.join("nested.wav")).unwrap();

    // Without --recursive: top-level file listed, nested file absent.
    scrybe()
        .arg("--dry-run")
        .arg(root.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("top.wav"))
        .stdout(predicate::str::contains("nested.wav").not());

    // With --recursive: the nested file appears too.
    scrybe()
        .args(["--dry-run", "--recursive"])
        .arg(root.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("top.wav"))
        .stdout(predicate::str::contains("nested.wav"));
}

#[test]
fn colliding_outputs_fail_fast_with_usage_error() {
    // Same stem, different containers → both map to tone.txt. Caught before model
    // load, so this needs no cached model and writes nothing.
    scrybe()
        .args([
            "--format",
            "txt",
            "tests/fixtures/audio/tone.wav",
            "tests/fixtures/audio/tone.mp3",
        ])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("output collision"));
    assert!(
        !std::path::Path::new("tests/fixtures/audio/tone.txt").exists(),
        "a doomed run must not write any output"
    );
}

#[test]
fn default_writes_sidecar_next_to_input() {
    if !tiny_cached() {
        eprintln!("skipping: tiny model not cached");
        return;
    }
    // No --out-dir: the sidecar lands next to the input. Copy into a tempdir so
    // the real fixtures dir stays clean.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("en.wav");
    std::fs::copy("tests/fixtures/speech/en.wav", &input).unwrap();
    scrybe()
        .args(["--model", "tiny", "--format", "txt"])
        .arg(&input)
        .assert()
        .success();
    assert!(
        dir.path().join("en.txt").exists(),
        "default writes the sidecar next to the input"
    );
}

#[test]
fn uncreatable_out_dir_exits_io_error() {
    // Parent of --out-dir is a regular file, so create_dir_all fails before any
    // model load — the only path that surfaces exit 1 (Io) through the binary.
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"x").unwrap();
    scrybe()
        .args(["--format", "txt", "--out-dir"])
        .arg(blocker.join("sub")) // a directory under a regular file
        .arg("tests/fixtures/speech/en.wav")
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("could not create out-dir"));
}

#[test]
fn json_flag_is_reflected_in_the_plan_banner() {
    // --json forces JSON output; the plan banner must report format=json, not the
    // raw --format default. Model-free via --dry-run.
    scrybe()
        .args(["--dry-run", "--json", "tests/fixtures/speech/en.wav"])
        .assert()
        .success()
        .stderr(predicate::str::contains("format=json"));
}

#[test]
fn threads_override_is_reflected_in_the_plan_banner() {
    scrybe()
        .args([
            "--dry-run",
            "--threads",
            "2",
            "tests/fixtures/speech/en.wav",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("threads=2"));
    scrybe()
        .args(["--dry-run", "tests/fixtures/speech/en.wav"])
        .assert()
        .success()
        .stderr(predicate::str::contains("threads=auto"));
}

#[test]
fn no_input_paths_exits_usage_error() {
    scrybe()
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("no input paths"));
}

#[test]
fn empty_directory_reports_no_audio_found() {
    let empty = tempfile::tempdir().unwrap();
    scrybe()
        .arg("--dry-run")
        .arg(empty.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("no audio files found"));
}

#[test]
fn models_remove_uncached_reports_not_cached() {
    // Point the cache at an empty dir so the model is guaranteed uncached; no
    // network, no download.
    let cache = tempfile::tempdir().unwrap();
    scrybe()
        .env("HF_HOME", cache.path())
        .args(["models", "remove", "tiny"])
        .assert()
        .success()
        .stdout(predicate::str::contains("is not cached"));
}

#[test]
fn models_list_shows_every_model() {
    scrybe()
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tiny"))
        .stdout(predicate::str::contains("base"))
        .stdout(predicate::str::contains("small"))
        .stdout(predicate::str::contains("large-v3"))
        .stdout(predicate::str::contains("large-v3-turbo"))
        .stdout(predicate::str::contains("distil-large-v3.5"));
}
