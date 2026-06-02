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

The primary attack surface is the audio decode path: scrybe parses untrusted media
files. It is defended in depth — no `unsafe` code (`unsafe_code = "forbid"`), no
panics on user input (`unwrap`/`expect`/`panic` are deny-level lints), per-file
memory ceilings against decompression bombs, a structural HE-AAC parser rather than a
byte scan, and a subprocess `ffmpeg` invocation that cannot be coaxed into option
injection. Decode, model-integrity (SHA-256-pinned downloads), and panic-safety
findings are in scope. Model output quality (transcription accuracy) is not a
security issue.
