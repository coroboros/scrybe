# Security Policy

## Supported versions

scrybe is pre-1.0. Security fixes land on the latest release only; there is no
backport window until a stable line is cut.

## Reporting a vulnerability

Report privately — do not open a public issue for a security problem.

- Open a [GitHub security advisory](https://github.com/coroboros/scrybe/security/advisories/new), or
- email `ob@coroboros.com`.

Include the version (`scrybe --version`), platform, and a reproduction — ideally
the input file or a minimal sample that triggers it. Expect an acknowledgement
within a few days and a fix or mitigation plan once the report is confirmed.

## Scope

scrybe parses untrusted media files, so its handling of that content is the attack
surface:

- **The audio decode path** — the built-in decoder streams to 16 kHz mono, so a
  source is never fully resident; a per-file ceiling caps the output against
  decompression bombs, and HE-AAC is rejected by a structural AudioSpecificConfig
  parser rather than a byte scan. A crash, OOM, or garbled output on crafted media
  is in scope.
- **The ffmpeg escape** — `--decoder ffmpeg` shells out to a system `ffmpeg`,
  invoked with a canonicalized absolute path and `-nostdin`, so a leading-dash or
  crafted name cannot inject options or coax it into reading the parent's stdin. An
  option-injection or process-control bypass through this path is in scope.
- **Model integrity** — weights and the VAD model are fetched over the network and
  verified against a pinned SHA-256 (re-fetched once on mismatch, cache-only under
  `--offline`). A corrupt or substituted model that loads is in scope.
- **Panic-safety** — `unsafe` is forbidden (`unsafe_code = "forbid"`) and
  `unwrap`/`expect`/`panic` are deny-level lints; a crash on a crafted file is in
  scope.

Model output quality (transcription accuracy) is not a security issue.
