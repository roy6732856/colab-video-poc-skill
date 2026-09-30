# Validation record

Checked 2026-09-30 on macOS with Python 3.14 and official Colab CLI 0.7.4.

- **39 local tests passed:** 14 installer tests, 8 media verification tests and
  17 orchestration tests. Media fixtures use real ffmpeg/ffprobe. Allocation,
  interruption, watchdog, partial completion and cleanup failures use a fake
  provider, with no live GPU allocation.
- Independent forward-testing found a cleanup-reserve gap and acceptance of
  wrong-duration output. Regressions now cover a 30-second stop timeout plus a
  one-second retry under both unit and wall-time limits, explicit guard-overrun
  reporting, and H3 frame-aligned duration checks. Stop attempts get a 75-second
  reserve; provider release remains subject to external latency/failure.
- Source and installed skill passed the system skill-creator format validator.
  All seven installed skill files matched the source by SHA-256.
- The installed CLI planned two clips with a 20-minute, estimated 3-unit cap
  without allocating a VM. Reviewed inputs and resolved jobs are checksum-bound.
- `bootstrap` fetched and verified the pinned upstream revision and Notebook
  checksum outside the public package. `doctor` queried an existing official
  OAuth setup and reported no active allocation; no credentials were read into
  this package or its public records.
- The new media verifier checked the existing 8-second POC clip: 1376×768,
  192 frames at 24 fps, full audio/video decode, non-silent audio and a temporal
  contact sheet. An existing local Whisper checkpoint recognized the expected
  Chinese words after documented normalization. No new generation was needed.

The new orchestration adapter has **not** completed a live-GPU end-to-end run.
Local passing tests do not establish current GPU availability, remote setup,
generation success, a provider-enforced billing cap or visual/voice acceptance.
The original live POC baseline is documented separately in
[verification.md](skills/colab-video-poc/references/verification.md).

CI executes syntax and offline tests on Linux/macOS and Python 3.11/3.12.
Its result covers the checked commit; it does not authenticate or rent a GPU.
