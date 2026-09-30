# Jobs and prompts

Input paths are absolute or relative to the manifest's directory:

```json
{"jobs": [{
  "id": "intro",
  "title": "One short concept clip",
  "reference_images": ["reference.png"],
  "prompt_file": "prompt.txt",
  "duration_seconds": 8,
  "seed": 20260930,
  "expected_dialogue": "今天，就讓創意動起來。"
}]}
```

One to twenty jobs; each needs 1–9 nonempty images, UTF-8 prompt file (or inline
`prompt`) and 4–15 seconds. IDs are unique letters/digits/hyphens/underscores.
The plan resolves an optional seed so execution uses the reviewed value. Outputs
default to `ID.mp4`; an `output_path` must stay inside the run folder and must not
already exist. Outputs may not collide. All paths sent upstream are absolute.

Preserve supplied prompts. For a new Ref2VA prompt, write:

```text
subject_definitions:
S1: The person in <Picture 1>, with their visible clothes and features.

summary:
Describe the whole short clip and intended story.

retention_analysis:
Appearance, object and setting details to retain from each reference.

detailed_description:
[Shot 1] Describe framing, action and camera movement from 0 seconds.
S1 says, <d>[Chinese] 今天，就讓創意動起來。</d>
[Shot 2] At 00:04.500, the camera cuts to a clearly different angle. Describe the
same person and continuing action with relevant appearance consistent.

overall_soundscape:
Describe voice, ambience and audible actions.

non_diegetic_music:
N/A
```

The original notebook defaults to 1376×768 and four Ref2VA sampling steps; this
adapter preserves them. Current Comfy guidance prefers 1344×768; changing that
configuration needs its own verification.

Frames align to `17k+5` at 24 fps. Eight seconds gives exactly 192 frames; four
gives 107 (4.458 seconds), fifteen gives 362 (15.083 seconds). Accepted duration
is not a frame-accurate guarantee. Prompted cuts are semantic instructions.
QA checks the measured duration against that aligned target, allowing two frames
of timestamp tolerance. A substantially shorter or longer output needs review.

For optional local speech QA, set `expected_dialogue` and pass `--whisper-model`
with an existing checkpoint FILE. No ASR model is downloaded automatically.
