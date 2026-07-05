//! CLI surface acceptance tests — the CLI contract.
//!
//! Pins the help surface, color on/off in both directions, the invalid-model
//! error, the no-panic guarantee, and the file-not-found exit code. `assert_cmd`
//! captures output through a pipe (never a TTY), so a bare run strips color
//! exactly as a real pipe would; `CLICOLOR_FORCE=1` stands in for a color-capable
//! terminal to assert the positive direction.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests may unwrap; the binary may not

use predicates::prelude::*;

mod common;
use common::{require_ffmpeg_or_skip, require_model_or_skip, scrybe};

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
        .stdout(predicate::str::contains("models"))
        .stdout(predicate::str::contains("skills"))
        // The Agents footer points an AI agent at the bundled skill.
        .stdout(predicate::str::contains("Agents:"))
        .stdout(predicate::str::contains("npx skills add coroboros/scrybe"));
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
fn non_numeric_jobs_is_a_clap_usage_error() {
    // `--jobs abc` is rejected by clap before scrybe runs — pin that exit-2 contract.
    scrybe()
        .args(["--jobs", "abc", "Cargo.toml"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn zero_jobs_is_clamped_not_panicked() {
    // `--jobs 0` passes clap and reaches resolve_jobs; it must clamp to 1, never
    // divide-by-zero or panic. Model-free via --dry-run.
    scrybe()
        .args(["--jobs", "0", "--dry-run", "tests/fixtures/speech/en.wav"])
        .assert()
        .success()
        .stderr(predicate::str::contains("panicked").not());
}

#[test]
fn malformed_audio_fails_without_panicking() {
    if require_model_or_skip().is_none() {
        return;
    }
    // Garbage bytes with an audio extension reach the decoder (scrybe's own path),
    // which must fail loud (exit 20, partial batch) — never panic past the no-panic
    // lints.
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.wav");
    std::fs::write(&bad, b"definitely not a wav file").unwrap();
    scrybe()
        .args(["--model", "tiny", "--format", "txt", "--out-dir"])
        .arg(dir.path())
        .arg(&bad)
        .assert()
        .failure()
        .code(20)
        .stderr(predicate::str::contains("panicked").not());
}

#[test]
fn single_file_unsupported_codec_exits_10() {
    if require_model_or_skip().is_none() {
        return;
    }
    // The single-file --json path decodes directly (not via batch), so a decode
    // failure propagates UnsupportedCodec → exit 10 at the process boundary — the
    // only end-to-end check of that mapping (the batched path re-wraps it as 20).
    scrybe()
        .args(["--model", "tiny", "--json", "tests/fixtures/aac/he-aac.m4a"])
        .assert()
        .failure()
        .code(10)
        .stderr(predicate::str::contains("--decoder ffmpeg"));
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
    if require_model_or_skip().is_none() {
        return;
    }
    // stdout must be only the JSON document — the status banner goes to stderr.
    // No --lang, so "en" here also exercises the auto-detect language arm.
    let output = scrybe()
        .args(["--model", "tiny", "--json", "tests/fixtures/speech/en.wav"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains("model="),
        "status banner leaked into stdout:\n{stdout}"
    );
    // Parse the document so the Meta wiring (resolved model + real duration) is
    // pinned, not just substrings — a swapped/wrong/hardcoded field then fails.
    let doc: serde_json::Value = serde_json::from_str(&stdout).expect("stdout is valid JSON");
    assert_eq!(doc["schema_version"], 1);
    assert_eq!(doc["language"], "en"); // auto-detect arm
    assert_eq!(doc["model"], "tiny"); // resolved model name, wired from the run
    assert!(
        doc["duration"].as_f64().is_some_and(|d| d > 0.0),
        "duration not the decoded length: {}",
        doc["duration"]
    );
}

#[test]
fn out_dir_redirects_output_away_from_input() {
    if require_model_or_skip().is_none() {
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
    if require_model_or_skip().is_none() {
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
    // ...and the failed file writes nothing — the "fail loud, never silent/garbled"
    // guarantee: a decode failure must not leave a truncated/empty sidecar.
    assert!(
        !out.path().join("he-aac.txt").exists(),
        "a failed file must write no output"
    );
}

#[test]
fn skips_up_to_date_output_unless_forced() {
    if require_model_or_skip().is_none() {
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
    let mtime = || {
        std::fs::metadata(out.path().join("en.txt"))
            .and_then(|m| m.modified())
            .unwrap()
    };
    transcribe(false)
        .assert()
        .success()
        .stderr(predicate::str::contains("ok"));
    let first = mtime();
    // Second run: output is current → skipped, and the file is not rewritten.
    transcribe(false)
        .assert()
        .success()
        .stderr(predicate::str::contains("up to date"));
    assert_eq!(mtime(), first, "a skipped file must not be rewritten");
    // --force reprocesses.
    transcribe(true)
        .assert()
        .success()
        .stderr(predicate::str::contains("ok"));
}

#[test]
fn silence_produces_no_transcript() {
    if require_model_or_skip().is_none() {
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
    if require_model_or_skip().is_none() {
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
    if require_model_or_skip().is_none() {
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
    if require_model_or_skip().is_none() {
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
fn decoder_ffmpeg_wires_through_the_binary() {
    if !require_ffmpeg_or_skip() {
        return;
    }
    if require_model_or_skip().is_none() {
        return;
    }
    // he-aac.m4a needs ffmpeg (symphonia rejects it), so success proves the
    // clap→Config→load_audio(ffmpeg) seam end to end.
    let out = tempfile::tempdir().unwrap();
    scrybe()
        .args([
            "--model",
            "tiny",
            "--decoder",
            "ffmpeg",
            "--format",
            "txt",
            "--out-dir",
        ])
        .arg(out.path())
        .arg("tests/fixtures/aac/he-aac.m4a")
        .assert()
        .success();
    assert!(
        out.path().join("he-aac.txt").exists(),
        "ffmpeg-decoded output should be written"
    );
}

#[test]
fn vtt_and_tsv_formats_write_well_formed_sidecars() {
    if require_model_or_skip().is_none() {
        return;
    }
    // The vtt/tsv/csv writers with no end-to-end coverage: drive them through
    // clap → effective_formats → write_outputs and check the files parse.
    let out = tempfile::tempdir().unwrap();
    scrybe()
        .args(["--model", "tiny", "--format", "vtt,tsv,csv", "--out-dir"])
        .arg(out.path())
        .arg("tests/fixtures/speech/en.wav")
        .assert()
        .success();
    let vtt = std::fs::read_to_string(out.path().join("en.vtt")).expect("en.vtt written");
    assert!(vtt.starts_with("WEBVTT"), "vtt header missing:\n{vtt}");
    assert!(vtt.contains("-->"), "vtt has no cue:\n{vtt}");
    let tsv = std::fs::read_to_string(out.path().join("en.tsv")).expect("en.tsv written");
    assert!(
        tsv.starts_with("start\tend\ttext\n"),
        "tsv header missing:\n{tsv}"
    );
    assert!(
        tsv.lines()
            .nth(1)
            .is_some_and(|row| row.matches('\t').count() == 2),
        "tsv has no tab-separated data row:\n{tsv}"
    );
    let csv = std::fs::read_to_string(out.path().join("en.csv")).expect("en.csv written");
    assert!(
        csv.starts_with("start,end,text\n"),
        "csv header missing:\n{csv}"
    );
    assert!(
        csv.lines().nth(1).is_some_and(|row| row.contains('"')),
        "csv has no quoted data row:\n{csv}"
    );
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
fn models_remove_evicts_a_cached_model() {
    // Seed a cached snapshot (remove doesn't SHA-verify, so any bytes trigger the
    // evict arm), then assert `models remove` reports "removed" and deletes it.
    let hf = tempfile::tempdir().unwrap();
    let repo = hf.path().join("hub/models--ggerganov--whisper.cpp");
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::create_dir_all(repo.join("snapshots/rev")).unwrap();
    std::fs::write(repo.join("refs/main"), b"rev").unwrap();
    let snapshot = repo.join("snapshots/rev/ggml-tiny.bin");
    std::fs::write(&snapshot, b"cached bytes").unwrap();
    scrybe()
        .env("HF_HOME", hf.path())
        .args(["models", "remove", "tiny"])
        .assert()
        .success()
        .stdout(predicate::str::contains("removed"));
    assert!(!snapshot.exists(), "remove must evict the cached snapshot");
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

#[test]
fn skills_list_shows_the_bundled_skill() {
    scrybe()
        .args(["skills", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("scrybe"))
        .stdout(predicate::str::contains("npx skills add coroboros/scrybe"));
}

#[test]
fn skills_get_prints_the_skill_markdown_verbatim() {
    // `get` emits the embedded SKILL.md as-is on a clean, unstyled stdout so an
    // agent can pipe or read it. The frontmatter and a section heading prove it is
    // the real file, not a stub; no ANSI proves it is unstyled.
    scrybe()
        .args(["skills", "get", "scrybe"])
        .assert()
        .success()
        .stdout(predicate::str::contains("name: scrybe"))
        .stdout(predicate::str::contains("## Install"))
        .stdout(predicate::str::contains(ESC).not());
}

#[test]
fn skills_get_defaults_to_scrybe() {
    // No name → the sole bundled skill, matching `get scrybe`.
    scrybe()
        .args(["skills", "get"])
        .assert()
        .success()
        .stdout(predicate::str::contains("name: scrybe"));
}

#[test]
fn skills_get_unknown_name_exits_usage_error() {
    scrybe()
        .args(["skills", "get", "bogus"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("unknown skill `bogus`"))
        .stderr(predicate::str::contains("scrybe"));
}

#[test]
fn speakers_without_diarize_is_a_usage_error() {
    scrybe()
        .args(["--speakers", "2", "tests/fixtures/speech/en.wav"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("--diarize"));
}

#[test]
fn dry_run_diarize_reports_model_status_without_downloading() {
    // A fresh, empty HF cache: the plan must name both diarization models as
    // pending downloads and fetch nothing.
    let cache = tempfile::tempdir().unwrap();
    scrybe()
        .env("HF_HOME", cache.path())
        .args(["--dry-run", "--diarize", "tests/fixtures/speech/en.wav"])
        .assert()
        .success()
        .stdout(predicate::str::contains("diarization/segmentation"))
        .stdout(predicate::str::contains("diarization/embedding"))
        .stdout(predicate::str::contains("not cached"));
    let downloaded = std::fs::read_dir(cache.path())
        .map(|entries| entries.count())
        .unwrap_or(0);
    assert_eq!(downloaded, 0, "--dry-run must not touch the network");
}

#[test]
fn offline_diarize_without_cache_exits_11_with_the_pull_hint() {
    let cache = tempfile::tempdir().unwrap();
    scrybe()
        .env("HF_HOME", cache.path())
        .args(["--offline", "--diarize", "tests/fixtures/speech/en.wav"])
        .assert()
        .failure()
        .code(11)
        .stderr(predicate::str::contains("models pull diarization"));
}

#[test]
fn models_list_includes_the_diarization_pair() {
    scrybe()
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("diarization"))
        .stdout(predicate::str::contains("--diarize"));
}

#[test]
fn diarize_json_carries_two_speaker_labels_end_to_end() {
    // The whole CLI path on the committed two-voice fixture: transcription
    // (tiny) + diarization + merge + JSON rendering.
    if common::require_model_or_skip().is_none() || !common::require_diarize_or_skip() {
        return;
    }
    let output = scrybe()
        .args([
            "--diarize",
            "--json",
            "tests/fixtures/speech/two-speakers.wav",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&output.get_output().stdout).into_owned();
    let doc: serde_json::Value = serde_json::from_str(&stdout).expect("clean JSON on stdout");
    assert_eq!(doc["schema_version"], 1, "additive fields keep the schema");
    let speakers: std::collections::HashSet<&str> = doc["segments"]
        .as_array()
        .expect("segments array")
        .iter()
        .filter_map(|s| s["speaker"].as_str())
        .collect();
    assert!(
        speakers.contains("SPEAKER_00") && speakers.contains("SPEAKER_01"),
        "expected two speaker labels, got {speakers:?}"
    );
}

#[test]
fn undiarized_json_never_carries_speaker_keys() {
    if common::require_model_or_skip().is_none() {
        return;
    }
    let output = scrybe()
        .args(["--json", "tests/fixtures/speech/en.wav"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&output.get_output().stdout).into_owned();
    let doc: serde_json::Value = serde_json::from_str(&stdout).expect("clean JSON on stdout");
    for segment in doc["segments"].as_array().expect("segments array") {
        assert!(
            segment.get("speaker").is_none(),
            "speaker key must be absent without --diarize: {segment}"
        );
    }
}
