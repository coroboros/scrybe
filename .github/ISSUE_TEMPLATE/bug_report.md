---
name: Bug report
about: A reproducible defect in scrybe
title: ""
labels: bug
---

## What happened

A clear description of the bug and the expected behavior.

## Reproduce

```sh
# the exact command
scrybe ...
```

- Exit code: `$?`
- Input: format/codec, sample rate, channels (a small sample helps)

## Environment

- scrybe version: `scrybe --version`
- OS + arch:
- Backend: CPU / Metal / CUDA / Vulkan
- Model: `--model ...` (or auto)

## Logs

The error line and any stderr output (run without `--no-color` stripping).
