#!/usr/bin/env bash
# Test fixtures for scrybe, invoked by coroboros/ci's rust/test-deps before `cargo test`.
# Pre-fetches the tiny Whisper model into the hf-hub cache so the golden test runs on the
# CPU backend with no per-run download; SCRYBE_REQUIRE_MODEL (ci/test.env) then turns a
# cache miss into a hard failure instead of a skip.
set -euo pipefail

cargo run --quiet -- models pull tiny
