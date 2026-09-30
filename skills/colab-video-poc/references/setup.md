# Setup and recovery

Local hosts: macOS/Linux. Python 3.11+, Git, ffmpeg and ffprobe are required;
inference runs on Colab. Install the official CLI using a compatible Python:

```bash
uv tool install --python 3.12 google-colab-cli==0.7.4
```

For PATH issues, run `uv tool update-shell` or pass the full executable path with
`workflow.py --colab /absolute/path/to/colab`. The adapter does not replace tools.

First-time Google authentication is an interactive official CLI operation:

```bash
colab --auth=oauth2 usage
```

Let the user complete the official browser/terminal flow. There is no `colab login`
subcommand in the tested 0.7.4 version. Do not paste codes into manifests/issues.
The CLI manages credentials in its standard configuration directory. Doctor/run
will not launch an unrequested first-time OAuth flow. Existing ADC users can
explicitly set `--auth adc` before the wrapper subcommand.

`doctor --json` checks current access without allocating a VM. `bootstrap` caches
only the small upstream Git repository at a pinned revision, outside the package;
it does not download weights or request a GPU.

Each `run` owns a session named `h3poc-...`. Its result records that exact name.
If release cannot be confirmed, recover that allocation with:

```bash
colab --auth=oauth2 stop --session SESSION_FROM_RESULT
colab --auth=oauth2 sessions
colab --auth=oauth2 usage
```

Closing a browser/terminal does not prove the VM was released. Check allocations
and usage after stopping. Never stop another session merely to get a zero count.

HTTP 429 led to Xet download retries in the baseline. A frozen progress display
alone does not prove a dead kernel; check file growth or sanitized diagnostics.
Do not duplicate a generation while its kernel may remain busy. This version
bounds runtime rather than introducing unverified download/accelerator patches.

Primary sources: [Colab CLI](https://github.com/googlecolab/google-colab-cli),
[resource limits](https://research.google.com/colaboratory/faq.html#resource-limits),
[Hugging Face download settings](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
