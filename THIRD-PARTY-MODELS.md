# Third-party models

scrybe downloads these models at runtime (SHA-256-pinned, into the Hugging Face
cache) or embeds them in the binary. Each remains under its own license;
attributions below.

## Whisper (transcription)

- **Whisper** — OpenAI. MIT. ggml conversions fetched from
  [`ggerganov/whisper.cpp`](https://huggingface.co/ggerganov/whisper.cpp).
- **Distil-Whisper large-v3.5** — Hugging Face. MIT. Fetched from
  [`distil-whisper/distil-large-v3.5-ggml`](https://huggingface.co/distil-whisper/distil-large-v3.5-ggml).
- **Silero VAD v5.1.2** — Silero Team. MIT. Embedded in the binary
  (`assets/ggml-silero-v5.1.2.bin`, via
  [`ggml-org/whisper-vad`](https://huggingface.co/ggml-org/whisper-vad)).

## Diarization (`--diarize`)

- **pyannote segmentation-3.0** — Hervé Bredin ([pyannote.audio](https://github.com/pyannote/pyannote-audio)). MIT.
  ONNX export fetched from
  [`onnx-community/pyannote-segmentation-3.0`](https://huggingface.co/onnx-community/pyannote-segmentation-3.0).
  The diarization pipeline in `src/diarize/` ports the pyannote.audio 3.1.1
  speaker-diarization recipe (MIT) to Rust.
- **WeSpeaker ResNet34-LM (VoxCeleb)** — WeNet Community. CC-BY-4.0 (upstream
  code Apache-2.0). ONNX export fetched from
  [`onnx-community/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM).

## Runtime notices

- **ONNX Runtime** — Microsoft. MIT. Statically linked; its binary
  distribution embeds Eigen (MPL-2.0, unmodified) and other permissively
  licensed components — see the
  [ONNX Runtime ThirdPartyNotices](https://github.com/microsoft/onnxruntime/blob/main/ThirdPartyNotices.txt).
- **whisper.cpp / ggml** — Georgi Gerganov and contributors. MIT. Compiled
  from source and statically linked via `whisper-rs`.
