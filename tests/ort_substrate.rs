//! ONNX Runtime substrate contract: the statically linked runtime initializes,
//! and a corrupt model file surfaces as a Rust error mapped onto the stable
//! exit-code contract — never a process kill from the C++ layer.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ort::session::Session;
use scrybe::error::ScrybeError;

#[test]
fn ort_initializes_on_this_target() {
    // Builder creation boots the ORT environment: proves the statically linked
    // runtime is present and callable on this target.
    Session::builder().expect("ONNX Runtime must initialize");
}

#[test]
fn corrupt_model_is_an_error_not_a_process_kill() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.onnx");
    std::fs::write(&bad, b"definitely not an onnx protobuf").unwrap();

    let result = Session::builder().unwrap().commit_from_file(&bad);
    assert!(result.is_err(), "corrupt model must surface as Err");

    let mapped = ScrybeError::ModelLoadFailed {
        path: bad,
        detail: result.unwrap_err().to_string(),
    };
    assert_eq!(
        mapped.exit_code(),
        15,
        "substrate failures ride the existing contract"
    );
}
