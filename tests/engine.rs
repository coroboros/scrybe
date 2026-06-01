//! Engine load-failure contract. A corrupt/garbage model file must surface through
//! a real failing `Engine::load` — `ModelLoadFailed` (exit 15) on the CPU backend,
//! `GpuInitFailed` (exit 13) on a GPU build, where context creation is GPU init.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use scrybe::engine::Engine;

/// Drive a real failing load and return its exit code.
fn garbage_load_exit() -> Option<i32> {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.bin");
    std::fs::write(&bad, b"not a ggml model").unwrap();
    // `.err()` avoids requiring `Engine: Debug` for `expect_err`.
    Engine::load(&bad, None).err().map(|e| e.exit_code())
}

#[cfg(not(any(feature = "metal", feature = "cuda", feature = "vulkan")))]
#[test]
fn loading_a_garbage_model_exits_15() {
    assert_eq!(
        garbage_load_exit(),
        Some(15),
        "corrupt ggml on CPU → ModelLoadFailed, not a panic"
    );
}

#[cfg(any(feature = "metal", feature = "cuda", feature = "vulkan"))]
#[test]
fn loading_a_garbage_model_exits_13_on_gpu_build() {
    assert_eq!(
        garbage_load_exit(),
        Some(13),
        "corrupt ggml on a GPU build → GpuInitFailed, not a panic"
    );
}
