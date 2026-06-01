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
    use super::*;

    #[test]
    fn filters_non_audio_by_extension() {
        assert!(is_audio(Path::new("a/b.MP3")));
        assert!(is_audio(Path::new("clip.flac")));
        assert!(!is_audio(Path::new("notes.txt")));
        assert!(!is_audio(Path::new("no_extension")));
    }
}
