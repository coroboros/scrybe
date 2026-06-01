//! Engine load-failure contract. A corrupt/garbage model file must surface as
//! `ModelLoadFailed` (exit 15) on the CPU backend — proving the exit code through
//! a real failing operation, not just the pure error-to-code mapping.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use scrybe::engine::Engine;

#[test]
fn loading_a_garbage_model_exits_15() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.bin");
    std::fs::write(&bad, b"not a ggml model").unwrap();
    // `.err()` avoids requiring `Engine: Debug` for `expect_err`.
    let exit = Engine::load(&bad, None).err().map(|e| e.exit_code());
    assert_eq!(
        exit,
        Some(15),
        "corrupt ggml → ModelLoadFailed, not a panic"
    );
}
