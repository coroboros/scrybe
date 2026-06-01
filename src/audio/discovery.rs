//! Input discovery: expand the given paths into a stable, de-duplicated list of
//! audio files, optionally recursing into directories. Content is validated when
//! a file is decoded; here we filter by extension so non-audio (e.g. `.txt`) is
//! never handed to the decoder.

use std::path::{Path, PathBuf};

/// Extensions treated as audio, limited to what the decoder can actually handle.
const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "flac", "m4a", "m4b", "mp4", "ogg", "oga"];

fn is_audio(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .is_some_and(|ext| AUDIO_EXTENSIONS.contains(&ext.as_str()))
}

/// Expand `inputs` into audio files: a file is kept if it looks like audio, a
/// directory is scanned (recursively when `recursive`). Sorted + de-duplicated
/// for deterministic ordering across runs.
pub fn discover(inputs: &[PathBuf], recursive: bool) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for input in inputs {
        if input.is_dir() {
            scan_dir(input, recursive, &mut found);
        } else if is_audio(input) {
            found.push(input.clone());
        }
    }
    found.sort();
    found.dedup();
    found
}

fn scan_dir(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                scan_dir(&path, recursive, out);
            }
        } else if is_audio(&path) {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn filters_non_audio_by_extension() {
        // Track the constant so a dropped extension (e.g. the m4b/mp4/oga aliases)
        // fails here instead of silently shrinking what reaches the decoder.
        for ext in AUDIO_EXTENSIONS {
            assert!(
                is_audio(Path::new(&format!("clip.{ext}"))),
                "{ext} must be audio"
            );
        }
        assert!(
            is_audio(Path::new("a/b.MP3")),
            "extension match is case-insensitive"
        );
        assert!(!is_audio(Path::new("notes.txt")));
        assert!(!is_audio(Path::new("no_extension")));
    }

    #[test]
    fn recursion_dedup_and_stable_ordering() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("a.wav"), b"x").unwrap();
        std::fs::write(root.path().join("notes.txt"), b"x").unwrap();
        std::fs::create_dir(root.path().join("sub")).unwrap();
        std::fs::write(root.path().join("sub").join("b.mp3"), b"x").unwrap();
        let root_path = root.path().to_path_buf();

        let flat = discover(std::slice::from_ref(&root_path), false);
        assert!(flat.iter().any(|p| p.ends_with("a.wav")));
        assert!(
            !flat.iter().any(|p| p.ends_with("b.mp3")),
            "must not recurse without --recursive"
        );
        assert!(
            !flat.iter().any(|p| p.ends_with("notes.txt")),
            "non-audio is skipped"
        );

        let deep = discover(std::slice::from_ref(&root_path), true);
        assert!(
            deep.iter().any(|p| p.ends_with("b.mp3")),
            "recurses with --recursive"
        );

        // The same directory passed twice yields a sorted, de-duplicated list.
        let dup = discover(&[root_path.clone(), root_path], true);
        let mut expected = dup.clone();
        expected.sort();
        expected.dedup();
        assert_eq!(
            dup, expected,
            "discover output must be sorted and de-duplicated"
        );
    }
}
