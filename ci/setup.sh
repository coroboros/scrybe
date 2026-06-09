#!/usr/bin/env bash
# Native build + decode dependencies for scrybe, invoked by coroboros/ci's
# rust/native-deps composite. cmake builds the vendored whisper.cpp (whisper-rs-sys);
# ffmpeg backs the decoder-fallback tests. CARGO_DIST_TARGET is set on a dist-build
# leg and empty on host preflight; cargo-dist 0.32 builds every target on a native
# runner (incl. aarch64 on ubuntu-*-arm), so only cmake is needed — no cross toolchain.
set -euo pipefail

dist_build=false
if [ -n "${CARGO_DIST_TARGET:-}" ]; then
  dist_build=true
fi

# cmake is needed on every leg that compiles the crate — preinstalled on the macOS and
# Windows runners, so only Linux installs it. ffmpeg is only needed where `cargo test`
# runs, so the dist-build legs skip it.
case "${RUNNER_OS:-}" in
  Linux)
    sudo apt-get update
    if [ "${dist_build}" = true ]; then
      sudo apt-get install -y cmake
    else
      sudo apt-get install -y cmake ffmpeg
    fi
    ;;
  Windows)
    [ "${dist_build}" = true ] || choco install ffmpeg -y --no-progress
    ;;
  macOS)
    [ "${dist_build}" = true ] || brew install ffmpeg
    ;;
  *)
    echo "::warning::unknown RUNNER_OS '${RUNNER_OS:-}' — skipping native dep install"
    ;;
esac
