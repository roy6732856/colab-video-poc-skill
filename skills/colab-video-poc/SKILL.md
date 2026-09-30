---
name: colab-video-poc
description: Generate and verify short MiniMax H3 videos from local reference images on Google Colab, with a compute-unit budget, batch reuse, and automatic GPU release. Use for a Colab video POC or repeatable H3 generation workflow.
---

# Colab Video POC

Use `scripts/workflow.py` from this skill's directory. This original adapter
downloads a pinned [killkli H3 runner](https://github.com/killkli/minimax-h3-colab-skill)
into a local cache; upstream code is not bundled or relicensed.

## Prepare

1. Run `python3 scripts/workflow.py doctor --json`. Check dependencies, OAuth
   availability, live balance and existing sessions. For missing setup, read
   [references/setup.md](references/setup.md). Authentication stays with the
   official CLI; never request tokens in chat or put credentials in jobs/logs.
2. Inspect the reference images and keep their order: `<Picture 1>` is the first
   uploaded image. Pass a supplied prompt file unchanged. For a new prompt and
   manifest format, read [references/jobs.md](references/jobs.md).
3. Defaults for a first POC are one 8-second clip, G4, 45 minutes and an estimated
   maximum of 8 compute units for the whole batch. Respect the user's scope and
   limits. Put already authorized clips in one batch to reuse model state;
   do not add billed refinement clips to a one-clip request.
4. Run `python3 scripts/workflow.py bootstrap`, then
   `python3 scripts/workflow.py plan --manifest /absolute/jobs.json --output-dir /absolute/run-folder --gpu G4 --max-minutes 45 --max-units 8`.
   Set the GPU and both limits to the user's agreed values in **both** plan and run.
   This checks every input and output collision before renting a GPU. Use a fresh
   run folder to preserve earlier results. Planning does not allocate a VM.

## Generate and finish

When generation is authorized, execute within the agreed limits:

```bash
python3 scripts/workflow.py run --manifest /absolute/jobs.json \
  --output-dir /absolute/run-folder --gpu G4 --max-minutes 45 --max-units 8
```

The adapter creates its own named session, reads the live usage rate, guards the
complete runtime, runs the upstream batch, preserves completed clips, collects
execution evidence, and stops its VM before local media checks. It refuses to
start alongside an existing allocation because the account-wide rate cannot
reliably identify this job's cost. Do not stop unrelated sessions to clear that check.

Read `result.json` and per-video QA reports. Show the playable MP4 and inspect the
temporal contact sheet. For dialogue, use an existing local Whisper checkpoint
when available (`--whisper-model /absolute/small.pt`); otherwise report speech
content as unverified. ASR does not accept accent, emotion, naturalness or precise
lip sync. Missing/silent audio, decode failure and speech mismatches remain visible.
Each clip's measured duration is checked against H3's native frame-aligned target.

Report output paths, observed content, resolution/duration, preparation and
execution times, displayed balance change, time-times-rate estimate and cleanup.
Leave final settlement unknown. Deliver earlier completed clips if a later job fails.

## Operational boundaries

- G4 and A100 high-memory succeeded in the baseline; availability/rates can change.
  A100 defaults to high memory. Choose an explicit alternative after allocation
  failure; never silently switch providers or retry a timed-out kernel.
- Ctrl+C and watchdog cleanup stop only the owned runtime. If release cannot be
  confirmed, report its name and stop command instead of claiming zero consumption.
  The guard reserves 75 seconds for cleanup; provider latency can still exceed it.
  Report `budget_guard.limit_exceeded` and keep provider settlement unverified.
- This pipeline is Base/Turbo 768p with native audio; it does not implement the
  complete hosted 2K workflow. Larger VRAM does not establish the fastest GPU.
- Skill reuse removes repeated setup decisions. Fresh VMs still need large
  model downloads; the local code cache is not a persistent remote model cache.
- Keep inputs, outputs, credentials and private logs outside the public package.
  For failure handling and the verified baseline, read
  [references/verification.md](references/verification.md).
