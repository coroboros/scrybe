#!/usr/bin/env bash
# Measure transcription speed (×RT = audio ÷ wall, higher is faster) per model.
# Usage: bench/run.sh [model ...]   (default: tiny base large-v3-turbo)
# Add --features metal to the build below to measure the GPU path.
set -euo pipefail
cd "$(dirname "$0")/.."

SAMPLE="bench/sample-60s.wav"
if [ ! -f "$SAMPLE" ]; then
  # ~56 s, 16 kHz mono — the 3 s test clip looped, so no large fixture is committed.
  ffmpeg -y -stream_loop 20 -i tests/fixtures/speech/en.wav -ac 1 -ar 16000 "$SAMPLE" 2>/dev/null
fi

cargo build --release
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(tiny base large-v3-turbo)
out="$(mktemp -d)"

printf '%-20s %s\n' "model" "xRT"
for m in "${models[@]}"; do
  rt="$(./target/release/scrybe --model "$m" --force --format txt --out-dir "$out" "$SAMPLE" 2>&1 \
    | grep -oE '[0-9.]+×RT' | tail -1)"
  printf '%-20s %s\n' "$m" "${rt:-error}"
done
