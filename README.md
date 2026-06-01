# scrybe

> Collapse sound into signal.

A fast, offline Whisper transcription CLI. Point it at a file or a folder; it
transcribes audio to text from the terminal — Metal on Apple Silicon, CPU
everywhere, no Python and no system `ffmpeg`.

## Status

Early. The command surface, color layer, and error model are in place. The
audio decode, model cache, transcription engine, and batch UX land in the next
milestones — today's build resolves inputs and prints the plan.

## Install

From source, until the release channels (Homebrew, `npx`, `cargo binstall`,
curl) are wired:

```sh
cargo install --path .
```

## Usage

```sh
scrybe ./recordings          # transcribe a folder
scrybe talk.mp3 --format srt # one file, SubRip output
scrybe --dry-run ./in        # resolve the plan without transcribing
scrybe models list           # show the model family
```

Run `scrybe --help` for the full flag list.

## Models

| Model | Notes |
|-------|-------|
| `tiny` / `base` / `small` | small and fast, lower accuracy |
| `large-v3` | most accurate, translation-capable |
| `large-v3-turbo` | default — near-`large-v3` accuracy, much faster |
| `distil-large-v3.5` | distilled, fast, English-leaning |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | success |
| `2` | usage error (bad flag or value) |
| `10` | unsupported codec |
| `11` | model download failed |
| `12` | out of memory |
| `13` | GPU init failed |
| `14` | file not found |
| `20` | partial batch failure |

## License

MIT © Coroboros
