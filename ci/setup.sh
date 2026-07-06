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

# --- ONNX Runtime provisioning (the `ort` crate's native lib) --------------------
# Three dist targets can't use pyke's prebuilt static libs: x86_64-apple-darwin has
# none since ORT 1.23, and the Linux libs need glibc >= 2.38 (which would raise the
# shipped floor). For those, point ort-sys at a self-built static ORT via
# ORT_LIB_PATH; the other targets keep pyke's SHA-verified download. Source of truth
# is the pinned ORT source — the release asset is a SHA-256-verified cache of that
# build, with a from-source fallback so a missing asset never blocks a release.
ORT_VERSION="1.24.2"
ORT_RELEASE_URL="https://github.com/coroboros/scrybe/releases/download/ort-static-${ORT_VERSION}"

ort_sha256() {
  case "$1" in
    x86_64-apple-darwin) echo "14467010fb9932af8e213af47e948549b5ad34c9e94a6bbd83826ce147ae7d5f" ;;
    x86_64-unknown-linux-gnu) echo "1eaa455be809e5ead41875629fe8206fdd4691aba2704d4c1e0111bbea3eb871" ;;
    aarch64-unknown-linux-gnu) echo "08c7fdaddadfb1c2b17fb43af63cb70976e7d914be6c76c3d5b5af7e92b823e5" ;;
    *) return 1 ;;
  esac
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

build_ort_from_source() {
  local target="$1" dest="$2" build_dir
  case "$target" in
    *-apple-darwin) build_dir="build/MacOS/Release" ;;
    *-linux-gnu) build_dir="build/Linux/Release" ;;
    *) echo "::error::no from-source ORT recipe for ${target}"; return 1 ;;
  esac
  git clone --depth 1 --branch "v${ORT_VERSION}" --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime ort-src
  (
    cd ort-src
    MACOSX_DEPLOYMENT_TARGET="13.4" ./build.sh --config Release --parallel --skip_tests \
      --compile_no_warning_as_error --skip_submodule_sync \
      --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
  )
  # re2 is fetched but only test targets depend on it, so `all` never builds it —
  # ort-sys links it unconditionally. Build it explicitly.
  cmake --build "ort-src/${build_dir}" --target re2
  rsync -am --include='*/' --include='*.a' --exclude='*' "ort-src/${build_dir}/" "${dest}/"
}

provision_ort() {
  local target="$1" sha dest archive
  sha="$(ort_sha256 "$target")" || return 0 # pyke serves this target; nothing to do
  dest="${RUNNER_TEMP:-/tmp}/ort-static"
  archive="${dest}/ort-static-${target}.tar.gz"
  mkdir -p "$dest"

  if curl -fsSL "${ORT_RELEASE_URL}/ort-static-${target}.tar.gz" -o "$archive" \
    && [ "$(sha256_of "$archive")" = "$sha" ]; then
    tar -C "$dest" -xzf "$archive"
    echo "::notice::ONNX Runtime ${ORT_VERSION} for ${target} — verified prebuilt archive"
  else
    echo "::warning::prebuilt ORT unavailable or checksum mismatch for ${target}; building from source"
    build_ort_from_source "$target" "$dest"
  fi
  echo "ORT_LIB_PATH=${dest}" >> "${GITHUB_ENV:-/dev/null}"
}

if [ "${dist_build}" = true ]; then
  provision_ort "${CARGO_DIST_TARGET}"
fi
