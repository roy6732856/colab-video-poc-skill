"""Opt-in pre-staging child. Allocation/release/watchdog belong to workflow.py."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from workflow import CACHE_MANIFEST, SCRIPT_DIR, drive_cache_profile, read_json, save, verify_checkout

REMOTE_REPORT = "/content/h3-cache-report.json"


def setup_source(upstream: Path) -> str:
    verify_checkout(upstream)
    notebook = json.loads((upstream / "assets/MiniMax_H3_Turbo_Colab.ipynb").read_text())
    cells = notebook["cells"]
    if cells[3]["cell_type"] != "code" or any(c["cell_type"] == "code" for c in cells[:3]):
        raise ValueError("Pinned notebook first setup cell is not at index 3")
    return "".join(cells[3]["source"]) + '\nassert INFERENCE_MODE == "reference" and REF2VA_VARIANT == "fp8_scaled", "Unsupported Drive cache model variant"\n'


def remote_source(root: str, *, mount_root: str = "/content/drive",
                  stage_root: str = "/content/ComfyUI/models",
                  manifest_path: str = "/content/h3-cache-manifest.json",
                  report_path: str = REMOTE_REPORT) -> str:
    # This executes in the SAME kernel as setup, so its selected variant is checked.
    return f'''import os, json, sys, torch
from pathlib import Path
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
sys.path.insert(0, "/content")
import model_cache
assert os.path.ismount({mount_root!r}), "Drive mount not confirmed; persistence unknown"
assert torch.cuda.is_available(), "GPU is required for staging"
assert INFERENCE_MODE == "reference" and REF2VA_VARIANT == "fp8_scaled", "Unsupported cache model variant"
assert Path({stage_root!r}).parent.is_dir(), "ComfyUI setup must finish before staging"
profile = json.loads(Path({manifest_path!r}).read_text())
model_cache.validate_manifest(profile)
root = Path({root!r})
# Parent mount assertion precedes mkdir; the helper itself never creates Drive roots.
root.mkdir(parents=True, exist_ok=True)
model_cache.prepare_models(profile, root, stage_root={stage_root!r}, report_path={report_path!r})
'''


def flush_source(mount_root: str = "/content/drive") -> str:
    return f'''import os
from google.colab import drive
assert os.path.ismount({mount_root!r}), "Drive mount missing before flush; persistence unknown"
drive.flush_and_unmount()
assert not os.path.ismount({mount_root!r}), "Drive flush/unmount not confirmed; persistence unknown"
'''


def launch(upstream: Path, cli, root: str, profile_sha256: str,
           runner_command: list[str], *, execute=os.execv) -> None:
    """Prepare one existing owned session; replace child with normal runner on success."""
    def option(name):
        return runner_command[runner_command.index(name) + 1]
    session, gpu = option("--session"), option("--gpu")
    output = Path(option("--output-dir"))
    progress = Path(option("--progress"))
    report_path = output / "cache-report.json"
    profile = drive_cache_profile(root, gpu)
    if profile["profile_sha256"] != profile_sha256:
        raise ValueError("Cache profile changed after planning")
    setup = setup_source(upstream)  # Never bundle/patch upstream source.
    deadline = time.monotonic() + float(option("--timeout"))
    evidence = {"status": "preparing", "persistence": "unknown", "profile_sha256": profile_sha256,
                "phase": "mount"}
    save(report_path, evidence)

    def remaining(cap=None):
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Cache preparation exceeded the owned session guard")
        return min(seconds, cap) if cap is not None else seconds

    def call(args, cap=None):
        return cli.call(args, remaining(cap))

    def phase(value):
        evidence["phase"] = value
        save(report_path, evidence)
        save(progress, {"status": "cache_" + value, "log_tail": ["Drive cache: " + value]})

    def collect():
        call(["download", "--session", session, REMOTE_REPORT, str(report_path.resolve())], 20)
        result = read_json(report_path)
        if result.get("schema_version") != 1 or result.get("revision") != profile["revision"]:
            raise RuntimeError("Cache report missing or inconsistent with the planned profile")
        return result

    try:
        with tempfile.TemporaryDirectory(prefix="h3-cache-launch-") as temporary:
            folder = Path(temporary)
            # Official CLI owns OAuth. No browser, token or alternate auth logic.
            call(["drivemount", "--session", session, "/content/drive"], 180)
            phase("setup")
            setup_file = folder / "setup.py"
            setup_file.write_text(setup)
            call(["exec", "--session", session, "--timeout", str(remaining()),
                  "--env", "H3_INFERENCE_MODE=reference", "--file", str(setup_file)])
            phase("stage")
            call(["upload", "--session", session, str(SCRIPT_DIR / "model_cache.py"), "/content/model_cache.py"], 60)
            call(["upload", "--session", session, str(CACHE_MANIFEST), "/content/h3-cache-manifest.json"], 60)
            remote = folder / "remote.py"
            remote.write_text(remote_source(root))
            try:
                call(["exec", "--session", session, "--timeout", str(remaining()), "--file", str(remote)])
            except Exception:
                try:
                    evidence["model_cache"] = collect()
                except Exception:
                    pass
                raise
            result = collect()
            if result.get("status") != "ok" or len(result.get("files", [])) != 5:
                raise RuntimeError("Cache staging did not verify all five files")
            evidence["model_cache"] = result
            phase("flush")
            flush = folder / "flush.py"
            flush.write_text(flush_source())
            call(["exec", "--session", session, "--timeout", str(remaining(120)), "--file", str(flush)], 120)
            evidence.update(status="ready", persistence="flush_unmount_confirmed", phase="runner")
            save(report_path, evidence)
        runner_command = list(runner_command)
        runner_command[runner_command.index("--timeout") + 1] = str(max(1, int(remaining())))
        execute(runner_command[0], runner_command)
    except Exception:
        evidence.update(status="error")
        if evidence["phase"] != "runner":
            evidence["persistence"] = "unknown"
        save(report_path, evidence)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--colab", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--drive-cache", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("runner", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from workflow import Colab
    runner = args.runner[1:] if args.runner[:1] == ["--"] else args.runner
    try:
        launch(args.upstream, Colab(args.colab, args.auth), args.drive_cache,
               args.profile_sha256, runner)
        return 0
    except Exception:
        print("Drive cache preparation failed; persistence may be unknown; owned watchdog handles release", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
