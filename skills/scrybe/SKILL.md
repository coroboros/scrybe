---
name: scrybe
description: Transcribe speech to text offline with scrybe — a fast, self-contained Whisper CLI (no Python, no system ffmpeg). Reads common audio (wav, mp3, flac, ogg) and mp4/m4a files, one at a time or a whole folder, writes txt, srt, vtt, json, tsv, or csv, and can translate foreign-language speech to English in the same pass. Use whenever someone wants a transcript, subtitles, captions, or speech-to-text from a local recording, audio file, or video. Triggers on "transcribe this", "pull the text out of this", "get subtitles for this video", "speech to text", "caption this recording", "what's said in this audio".
---

# scrybe

Transcribes speech to text offline with Whisper — one file or a whole folder, from
the terminal. A single Rust binary decodes wav, mp3, flac, ogg, and mp4/m4a natively
(no system `ffmpeg`), runs whisper.cpp (Metal on Apple Silicon, CPU everywhere), and
writes the formats you ask for. Models download once into the Hugging Face cache;
after that nothing leaves the machine.

## Install

scrybe is a CLI binary. If `scrybe` is not on `PATH`, install it — pick the path
that matches the toolchain already on the machine, then continue:

```sh
brew install coroboros/tap/scrybe   # macOS — preferred
npx @coroboros/scrybe               # Node toolchains
cargo binstall scrybe               # prebuilt binary via cargo
```

Building from source needs a C/C++ toolchain and CMake (whisper.cpp is compiled by
`whisper-rs-sys`); the prebuilt paths above need neither. Verify with
`scrybe --version` before transcribing.

## Use

Run `scrybe --help` for the full surface. Common invocations:

```sh
scrybe talk.mp3                     # transcribe one file → talk.txt beside it
scrybe ./recordings                 # a whole folder
scrybe ./recordings --recursive     # and its subfolders
scrybe talk.mp3 --format srt,vtt    # subtitles instead of plain text
scrybe talk.mp3 --json              # stream a JSON transcript to stdout
scrybe ./in --out-dir ./out         # write outputs to ./out, not beside inputs
scrybe --dry-run ./in               # resolve the plan without transcribing
scrybe --offline ./in               # cached models only, no network
```

Defaults are zero-config: omit `--model` and `--jobs` and scrybe picks the largest
model and the concurrency that fit detected RAM. Pass `--model large-v3` for the
most accurate model, or `--task translate` (large-v3 only) to translate to English.
Set `--lang fr` to skip language auto-detection.

## Inputs

scrybe reads `.wav`, `.mp3`, `.flac`, `.ogg`, and the `.mp4`/`.m4a` containers
(AAC-LC, ALAC) directly — an mp4 video transcribes from its audio track with no
extra step. HE-AAC/SBR inside a recognized container fails loud rather than
mis-decoding (exit `10`); pass `--decoder ffmpeg` to decode it through a system
`ffmpeg`, or re-encode it. Other video containers (`.mkv`, `.mov`, `.webm`) are not
recognized as audio inputs and are skipped rather than transcribed, so extract the
audio first: `ffmpeg -i in.mkv out.wav`, then transcribe `out.wav`.

## Parse a transcript

For programmatic use, prefer `--json`. A single input streams the document to
stdout (the status banner stays on stderr, so stdout is clean to pipe); multiple
inputs write `.json` sidecars. The schema is stable and versioned:

```json
{
  "schema_version": 1,
  "model": "large-v3-turbo",
  "language": "en",
  "duration": 12.84,
  "segments": [
    { "start": 0.0, "end": 2.4, "text": "the quick brown fox" }
  ]
}
```

Timestamps are in seconds. Each segment carries an optional `words` array of
per-word timing, present only with JSON output.

## Exit codes

scrybe fails loud with a stable exit code rather than emitting garbled output —
branch on `$?`:

- `0` success · `2` usage error (bad flag or value)
- `10` unsupported codec (e.g. HE-AAC/SBR — re-encode or pass `--decoder ffmpeg`)
- `11` model download failed · `12` out of memory · `14` file not found
- `20` partial batch failure, or interrupted before completion

On out of memory, retry with a smaller `--model` or lower `--jobs`. On a partial
batch, the per-file lines on stderr name which inputs failed; the rest completed.
