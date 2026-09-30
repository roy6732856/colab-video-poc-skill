# Verification scope and baseline

On 2026-09-30, upstream commit `7768ebfb1627546595f37e9acfd2d353937c6cd8`
successfully produced one reference-image clip:

| Observation | Result |
| --- | --- |
| GPU/host | G4 RTX PRO 6000 Blackwell, nominal 96GB VRAM; 176.885 GiB host RAM |
| Pipeline | Ref2VA FP8 scaled, CUDA 12.8, official 4-step Turbo LoRA |
| Output | 1376×768, 8 seconds, 192 frames, 24 fps, 32 kHz stereo AAC |
| Model files | 44.415 GB total; not peak VRAM |
| Preparation | Approximately 443 seconds before submission |
| Sampling | Approximately 81 seconds, including sampler initialization |
| ComfyUI task | 111.82 seconds including encoders/decoders/output |
| VM allocation to cleanup readback | Approximately 630 seconds |
| Rate/balance | 8.9 CU/hr; balance 80.00 → 79.04; time estimate 1.56 units |
| Content | ASR matched the Chinese line; sampled appearance consistent; cut near 4.417s |

This establishes one-clip feasibility of the original pipeline, not a live GPU
test of every new adapter path, a warm-run benchmark or an A100/G4 speed ranking.
A100 high-memory allocated 80 GiB VRAM; H100 was denied at that time. Availability
must be checked for each run. The public package excludes the original test inputs,
video, private paths, account details and raw session logs.

QA includes complete decoding, audio/video streams, dimensions/timing, non-silent
audio measurement, temporal frames and optional cached-local ASR. The agent must
inspect visual adherence. Accent, emotion, naturalness, voice identity and precise
lip sync need separate acceptance. Preserve mismatches and defects in the report.

Balance display can lag settlement or differ from elapsed-time estimates. Show
both and leave final charges unknown. Watchdog limits are estimated consumption
bounds. Always confirm release of the owned VM and expose cleanup failures.

This adapter implements Base/Turbo output. The complete MiniMax system includes
hosted Context-IR and a separate 2K regeneration path, neither implemented here.

Sources: [MiniMax model/workflow](https://huggingface.co/MiniMaxAI/MiniMax-H3),
[Comfy H3 guidance](https://docs.comfy.org/tutorials/video/minimax/minimax-h3).
