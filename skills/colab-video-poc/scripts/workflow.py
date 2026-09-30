#!/usr/bin/env python3
"""Original bounded orchestration adapter for the separately downloaded H3 runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

UPSTREAM_URL = "https://github.com/killkli/minimax-h3-colab-skill.git"
UPSTREAM_COMMIT = "7768ebfb1627546595f37e9acfd2d353937c6cd8"
NOTEBOOK_SHA256 = "ec80a3452f5ffc4ab4f675c66e849a0f802f4faa925ee79e30fb0f16d6005b8f"
SCRIPT_DIR = Path(__file__).resolve().parent
STOP_TIMEOUT_SECONDS = 30
STOP_ATTEMPTS = 2
CLEANUP_RESERVE_SECONDS = 75  # Both stop attempts plus polling/scheduling margin.
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def scrub(text: str) -> str:
    text = ANSI.sub("", text)
    # Runtime proxy tokens and signed download URLs must not enter persistent logs.
    text = re.sub(r"https?://[^\s\"<>]+", lambda m: m[0].split("?", 1)[0] + ("?[REDACTED]" if "?" in m[0] else ""), text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:access_token|refresh_token|colab-runtime-proxy-token|HF_TOKEN)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)
    return text


def shell(args: list[str], timeout: float = 30, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode:
        raise RuntimeError(scrub(proc.stdout + proc.stderr)[-2000:] or f"Command exited {proc.returncode}")
    return proc.stdout


def positive(value: float | str, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return value


def parse_usage(text: str) -> dict[str, Any]:
    fields = {}
    patterns = {
        "balance": r"^Current balance:\s*([\d,]+(?:\.\d+)?)\s+compute units\s*$",
        "rate_per_hour": r"^Usage rate:\s*([\d,]+(?:\.\d+)?)\s*/\s*hr\s*$",
        "active_assignments": r"^Active assignments:\s*(\d+)\s*$",
    }
    for name, expression in patterns.items():
        match = re.search(expression, ANSI.sub("", text), re.MULTILINE | re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot parse live Colab usage field: {name}")
        fields[name] = float(match[1].replace(",", ""))
    fields["active_assignments"] = int(fields["active_assignments"])
    fields["checked_at"] = now()
    return fields


class Colab:
    def __init__(self, executable: str = "colab", auth: str = "oauth2"):
        resolved = shutil.which(executable)
        if not resolved:
            raise FileNotFoundError("colab is missing; install google-colab-cli and add its executable to PATH")
        self.executable, self.auth = resolved, auth

    def command(self, args: list[str]) -> list[str]:
        return [self.executable, f"--auth={self.auth}", *args]

    def call(self, args: list[str], timeout: float = 30) -> str:
        return shell(self.command(args), timeout)

    def usage(self) -> dict[str, Any]:
        return parse_usage(self.call(["usage"]))


def check_oauth(auth: str) -> None:
    if auth == "oauth2" and not (Path.home() / ".config/colab-cli/token.json").is_file():
        raise RuntimeError("First-time OAuth is required. Run colab --auth=oauth2 usage interactively through the official CLI.")


def cache_root(path: Path | None = None) -> Path:
    return (path or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "colab-video-poc").expanduser().resolve()


def verify_checkout(path: Path) -> None:
    if shell(["git", "-C", str(path), "rev-parse", "HEAD"]).strip() != UPSTREAM_COMMIT:
        raise ValueError("Cached upstream revision is different from the pinned version")
    if shell(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]).strip():
        raise ValueError("Cached upstream has modified tracked files; use a separate clean cache")
    notebook = path / "assets/MiniMax_H3_Turbo_Colab.ipynb"
    if hashlib.sha256(notebook.read_bytes()).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Pinned upstream notebook checksum does not match")
    if not (path / "scripts/runner.py").is_file():
        raise FileNotFoundError("Pinned upstream runner is missing")


def bootstrap(cache: Path | None = None) -> Path:
    root = cache_root(cache)
    # Never vendor upstream into this public skill or its package repository.
    package_root = SCRIPT_DIR.parents[2]
    if root == package_root or package_root in root.parents:
        raise ValueError("Upstream cache must be outside the public package directory")
    target = root / UPSTREAM_COMMIT
    if target.exists():
        verify_checkout(target)
        return target
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="upstream-", dir=root))
    try:
        shell(["git", "clone", "--quiet", "--no-checkout", UPSTREAM_URL, str(temporary)], 180)
        shell(["git", "-C", str(temporary), "checkout", "--quiet", "--detach", UPSTREAM_COMMIT], 30)
        verify_checkout(temporary)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def resolve_path(raw: Any, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_manifest(manifest: Path, output_dir: Path) -> list[dict[str, Any]]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    jobs = raw.get("jobs") if isinstance(raw, dict) else None
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 20:
        raise ValueError("A manifest requires 1–20 jobs")
    output_dir = output_dir.expanduser().resolve()
    result, identifiers, targets = [], set(), set()
    for index, job in enumerate(jobs, 1):
        if not isinstance(job, dict):
            raise ValueError("Each job must be an object")
        identifier = str(job.get("id", f"job-{index:03d}"))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier) or identifier in identifiers:
            raise ValueError("Job IDs must be unique letters/digits/hyphens/underscores")
        identifiers.add(identifier)
        refs = job.get("reference_images")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 9:
            raise ValueError(f"{identifier}: requires 1–9 reference images")
        images = [resolve_path(image, manifest.parent) for image in refs]
        for image in images:
            if not image.is_file() or not image.stat().st_size:
                raise ValueError(f"{identifier}: missing or empty reference image")
        prompt_path = resolve_path(job["prompt_file"], manifest.parent) if job.get("prompt_file") else None
        prompt = prompt_path.read_bytes().decode("utf-8") if prompt_path else job.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{identifier}: needs a nonempty UTF-8 prompt")
        if any(not 1 <= int(n) <= len(images) for n in re.findall(r"<Picture\s+(\d+)>", prompt)):
            raise ValueError(f"{identifier}: prompt picture index exceeds the uploaded reference count")
        duration = job.get("duration_seconds", 8)
        if isinstance(duration, bool):
            raise ValueError("Duration cannot be a boolean")
        duration = positive(duration, "duration_seconds")
        if not 4 <= duration <= 15:
            raise ValueError("duration_seconds must be between 4 and 15")
        seed = job.get("seed", random.SystemRandom().randrange(2**64))
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an integer between 0 and 2^64-1")
        target = resolve_path(job["output_path"], output_dir) if job.get("output_path") else output_dir / f"{identifier}.mp4"
        if output_dir not in target.parents or target.suffix.lower() != ".mp4":
            raise ValueError("Output files must be MP4s inside output-dir")
        if target.exists() or target in targets:
            raise ValueError("An output already exists or multiple jobs target the same file")
        targets.add(target)
        normalized = {"id": identifier, "title": str(job.get("title", identifier)), "reference_images": [str(p) for p in images],
                      "duration_seconds": duration, "seed": seed, "output_path": str(target)}
        if prompt_path:
            normalized["prompt_file"] = str(prompt_path)
        else:
            normalized["prompt"] = prompt
        if job.get("expected_dialogue") is not None:
            if not isinstance(job["expected_dialogue"], str):
                raise ValueError("expected_dialogue must be text")
            normalized["expected_dialogue"] = job["expected_dialogue"]
        result.append(normalized)
    return result


def plan(manifest: Path, output_dir: Path, gpu: str = "G4", max_minutes: float = 45, max_units: float = 8) -> dict[str, Any]:
    max_minutes, max_units = positive(max_minutes, "max_minutes"), positive(max_units, "max_units")
    jobs = resolve_manifest(manifest.resolve(), output_dir.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {path for job in jobs for path in job["reference_images"]}
    inputs.update(job["prompt_file"] for job in jobs if "prompt_file" in job)
    state = {"created_at": now(), "source_manifest": str(manifest.resolve()),
             "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
             "input_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(inputs)},
             "upstream_commit": UPSTREAM_COMMIT, "gpu": gpu,
             "max_minutes": max_minutes, "estimated_compute_unit_cap": max_units, "jobs": jobs,
             "allocates_gpu": False, "billing_scope": "Estimated runtime guard; provider settlement is not verified"}
    save(output_dir / "resolved-jobs.json", {"jobs": jobs})
    state["resolved_manifest_sha256"] = hashlib.sha256((output_dir / "resolved-jobs.json").read_bytes()).hexdigest()
    save(output_dir / "plan.json", state)
    return state


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def stream_log(pipe: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        for line in pipe:
            out.write(scrub(line))
            out.flush()


def run_batch(manifest: Path, output_dir: Path, cli: Any, upstream: Path,
              gpu: str = "G4", high_mem: bool | None = None, max_minutes: float = 45,
              max_units: float = 8, whisper_model: Path | None = None) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        permitted = {"plan.json", "resolved-jobs.json"}
        if any(p.name not in permitted for p in output_dir.iterdir()):
            raise ValueError("Use a fresh output directory; prior run artifacts are never overwritten")
    existing = read_json(output_dir / "plan.json")
    if existing:
        # Keep seeds/normalized input from the reviewed plan, but revalidate files.
        if (existing.get("gpu"), existing.get("max_minutes"), existing.get("estimated_compute_unit_cap")) != (gpu, float(max_minutes), float(max_units)):
            raise ValueError("Run limits differ from the saved plan; review a new plan in a fresh folder")
        if existing.get("source_manifest") != str(manifest.resolve()) or existing.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("Manifest changed after planning; review a new plan in a fresh folder")
        for path, checksum in existing.get("input_sha256", {}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
                raise ValueError("An input changed after planning; review a new plan in a fresh folder")
        reviewed = output_dir / "resolved-jobs.json"
        if existing.get("resolved_manifest_sha256") != hashlib.sha256(reviewed.read_bytes()).hexdigest():
            raise ValueError("Resolved jobs changed after planning; review a new plan in a fresh folder")
        spec = dict(existing)
        spec["jobs"] = resolve_manifest(reviewed.resolve(), output_dir.resolve())
    else:
        spec = plan(manifest, output_dir, gpu, max_minutes, max_units)
    before = cli.usage()
    if before["active_assignments"]:
        raise RuntimeError("An existing allocation is active; resolve ownership before starting a separately budgeted run")
    if before["balance"] <= 0:
        raise RuntimeError("No compute-unit balance is available")
    session = "h3poc-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    state: dict[str, Any] = {"status": "starting", "session": session, "started_at": now(), "usage_before": before,
                             "upstream_commit": UPSTREAM_COMMIT, "gpu": gpu, "results": [], "cleanup": {"confirmed": False}}
    save(output_dir / "result.json", state)
    allocation_attempted, proc, reader = False, None, None
    begun = time.monotonic()
    effective_limit, rate = float(spec["max_minutes"]) * 60, 0.0
    interrupted = False
    previous_termination = signal.getsignal(signal.SIGTERM)
    def handle_termination(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handle_termination)
    try:
        allocation_attempted = True
        new_args = ["new", "--session", session, "--gpu", gpu]
        if (high_mem if high_mem is not None else gpu == "A100"):
            new_args.append("--high-mem")
        cli.call(new_args, min(120, effective_limit))
        live = cli.usage()
        rate = positive(live["rate_per_hour"], "live usage rate")
        if live["active_assignments"] != 1:
            raise RuntimeError("Live account usage is not attributable to exactly one POC allocation")
        units = min(float(spec["estimated_compute_unit_cap"]), float(before["balance"]))
        effective_limit = min(effective_limit, units / rate * 3600)
        state.update(status="running", usage_live=live, effective_wall_limit_seconds=effective_limit, effective_estimated_unit_cap=units)
        remaining = effective_limit - (time.monotonic() - begun)
        if remaining <= CLEANUP_RESERVE_SECONDS + 1:
            raise RuntimeError("Insufficient remaining time inside the live-rate budget")
        save(output_dir / "result.json", state)
        # Upstream executes on our session; this adapter exclusively owns cleanup.
        command = [sys.executable, "-u", str(upstream / "scripts/runner.py"), "batch",
                   "--manifest", str((output_dir / "resolved-jobs.json").resolve()), "--session", session,
                   "--gpu", gpu, "--timeout", str(max(1, int(remaining - CLEANUP_RESERVE_SECONDS))),
                   "--progress", str((output_dir / "progress.json").resolve()), "--output-dir", str(output_dir.resolve())]
        env = os.environ.copy()
        env.update(COLAB_AUTH=cli.auth, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        env["PATH"] = str(Path(cli.executable).parent) + os.pathsep + env.get("PATH", "")
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                start_new_session=True, bufsize=1)
        reader = threading.Thread(target=stream_log, args=(proc.stdout, output_dir / "runner.log"), daemon=True)
        reader.start()
        seen = set()
        with (output_dir / "timeline.jsonl").open("w", encoding="utf-8") as timeline:
            while proc.poll() is None:
                progress = read_json(output_dir / "progress.json")
                for line in progress.get("log_tail", []):
                    safe = scrub(line)
                    if safe not in seen:
                        seen.add(safe)
                        timeline.write(json.dumps({"observed_at": now(), "line": safe}, ensure_ascii=False) + "\n")
                        timeline.flush()
                if time.monotonic() - begun >= effective_limit - CLEANUP_RESERVE_SECONDS:
                    state["error"] = "Estimated compute-unit or wall-time limit reached; no automatic retry"
                    state["guard_stop"] = True
                    break
                time.sleep(1)
        if proc.poll() is not None:
            state["runner_returncode"] = proc.returncode
            if proc.returncode:
                state["error"] = "Upstream batch failed; inspect the retained progress/log for completed jobs"
            # Download complete runtime logs before releasing the VM; evidence is optional.
            try:
                runtime_log = output_dir / "comfyui.log"
                evidence_time = min(20, effective_limit - (time.monotonic() - begun) - CLEANUP_RESERVE_SECONDS)
                if evidence_time <= 0:
                    raise RuntimeError("Skipped optional runtime evidence to preserve cleanup budget")
                cli.call(["download", "--session", session, "/content/comfyui_h3.log", str(runtime_log.resolve())], evidence_time)
                log_text = scrub(runtime_log.read_text(encoding="utf-8", errors="replace"))
                runtime_log.write_text(log_text, encoding="utf-8")
                state["comfyui_execution_seconds"] = [float(n) for n in re.findall(r"Prompt executed in ([\d.]+) seconds", log_text)]
            except Exception as exc:
                state["runtime_evidence_error"] = scrub(str(exc))
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        state["error"] = "Interrupted; stopping the owned GPU session"
    except Exception as exc:
        state["error"] = scrub(str(exc))
    finally:
        # Ignore a second Ctrl+C while releasing the sole owned allocation.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            stop_errors = []
            if allocation_attempted:
                for attempt in range(STOP_ATTEMPTS):
                    try:
                        remaining_stop_time = effective_limit - (time.monotonic() - begun)
                        # Release remains necessary even if provider latency already exceeded the guard.
                        timeout = min(STOP_TIMEOUT_SECONDS, max(1, remaining_stop_time / (STOP_ATTEMPTS - attempt))) if remaining_stop_time > 0 else STOP_TIMEOUT_SECONDS
                        cli.call(["stop", "--session", session], timeout)
                        state["cleanup"]["stop_command_succeeded"] = True
                        break
                    except Exception as exc:
                        stop_errors.append(scrub(str(exc)))
            state["allocated_wall_seconds_estimate"] = round(time.monotonic() - begun, 3)
            state["estimated_compute_units"] = round(state["allocated_wall_seconds_estimate"] * rate / 3600, 4) if rate else None
            cap = min(float(spec["estimated_compute_unit_cap"]), float(before["balance"]))
            exceeded = state["allocated_wall_seconds_estimate"] > effective_limit or (state["estimated_compute_units"] is not None and state["estimated_compute_units"] > cap)
            state["budget_guard"] = {"wall_limit_seconds": effective_limit, "estimated_unit_cap": cap,
                                     "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS, "limit_exceeded": exceeded,
                                     "estimate_scope": "Until owned stop attempts finished; an unconfirmed allocation may continue consuming units"}
            if exceeded:
                state["error"] = (state.get("error", "") + "; " if state.get("error") else "") + "Observed runtime estimate exceeded the agreed guard; release still attempted"
            if proc is not None:
                terminate(proc)
            if reader is not None:
                reader.join(timeout=5)
            if proc is not None and proc.stdout is not None:
                proc.stdout.close()
            if stop_errors:
                state["cleanup"]["stop_errors"] = stop_errors
            try:
                after = cli.usage()
                sessions_text = cli.call(["sessions"], 30)
                server_empty = "No active sessions" in sessions_text
                state["usage_after"] = after
                state["displayed_balance_delta"] = round(before["balance"] - after["balance"], 4)
                state["cleanup"].update(active_assignments=after["active_assignments"], rate_per_hour=after["rate_per_hour"],
                                        server_sessions_empty=server_empty,
                                        confirmed=server_empty and after["active_assignments"] == 0 and after["rate_per_hour"] == 0)
            except Exception as exc:
                state["cleanup"]["readback_error"] = scrub(str(exc))
            if not state["cleanup"]["confirmed"]:
                state["cleanup"]["recovery_command"] = f"colab --auth={cli.auth} stop --session {session}"
            state["billing_limitation"] = "Displayed delta and time-times-rate are observations/estimates; final settlement is unverified"
        finally:
            signal.signal(signal.SIGINT, previous)
            signal.signal(signal.SIGTERM, previous_termination)
            state["gpu_cleanup_finished_at"] = now()
            save(output_dir / "result.json", state)
    # All local QA occurs after cleanup, even if only part of the batch completed.
    for job in spec["jobs"]:
        path = Path(job["output_path"])
        if not path.is_file() or not path.stat().st_size:
            continue
        item = {"id": job["id"], "output_path": str(path), "status": "downloaded"}
        try:
            from verify_media import verify_media
            item["verification"] = verify_media(path, output_dir / "qa" / job["id"], job.get("expected_dialogue"), whisper_model,
                                                 expected_duration_seconds=job["duration_seconds"])
            qa = item["verification"]
            item["status"] = "verified" if qa.get("automated_passed") else "needs_review" if qa.get("technical_passed") else "verification_failed"
        except Exception as exc:
            item.update(status="verification_failed", error=scrub(str(exc)))
        state["results"].append(item)
    all_present = len(state["results"]) == len(spec["jobs"])
    state["status"] = "completed" if all_present and not state.get("error") and state["cleanup"]["confirmed"] else "partial" if state["results"] else "failed"
    if state["status"] == "completed" and any(x["status"] != "verified" for x in state["results"]):
        state["status"] = "needs_review"
    state["interrupted"] = interrupted
    state["finished_at"] = now()
    save(output_dir / "result.json", state)
    return state


def doctor(executable: str, auth: str) -> dict[str, Any]:
    report: dict[str, Any] = {"checked_at": now(), "dependencies": {name: shutil.which(name) for name in ["git", "ffmpeg", "ffprobe"]},
                              "python_supported": sys.version_info >= (3, 11), "allocates_gpu": False}
    try:
        cli = Colab(executable, auth)
        check_oauth(auth)
        report["colab_version"] = scrub(cli.call(["version"]))
        report["usage"] = cli.usage()
        report["ready"] = report["python_supported"] and all(report["dependencies"].values())
    except Exception as exc:
        report.update(ready=False, error=scrub(str(exc)))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colab", default="colab")
    parser.add_argument("--auth", choices=["oauth2", "adc"], default="oauth2")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor"); d.add_argument("--json", action="store_true")
    b = sub.add_parser("bootstrap"); b.add_argument("--cache-dir", type=Path)
    for name in ["plan", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--gpu", choices=["G4", "A100", "H100", "L4", "T4"], default="G4")
        p.add_argument("--max-minutes", type=float, default=45)
        p.add_argument("--max-units", type=float, default=8)
        if name == "run":
            p.add_argument("--high-mem", action=argparse.BooleanOptionalAction, default=None)
            p.add_argument("--cache-dir", type=Path)
            p.add_argument("--whisper-model", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("--video", type=Path, required=True); v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--expected-dialogue"); v.add_argument("--whisper-model", type=Path)
    v.add_argument("--expected-duration", type=float, help="requested H3 duration; check native frame alignment")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor(args.colab, args.auth)
        elif args.command == "bootstrap":
            result = {"upstream_path": str(bootstrap(args.cache_dir)), "commit": UPSTREAM_COMMIT, "allocates_gpu": False}
        elif args.command == "plan":
            result = plan(args.manifest, args.output_dir, args.gpu, args.max_minutes, args.max_units)
        elif args.command == "run":
            check_oauth(args.auth)
            if any(not shutil.which(tool) for tool in ["git", "ffmpeg", "ffprobe"]):
                raise RuntimeError("Install Git, ffmpeg and ffprobe before allocating a GPU")
            # Validate all jobs before fetching code or allocating resources.
            resolve_manifest(args.manifest.resolve(), args.output_dir.resolve())
            result = run_batch(args.manifest.resolve(), args.output_dir.resolve(), Colab(args.colab, args.auth), bootstrap(args.cache_dir),
                               args.gpu, args.high_mem, args.max_minutes, args.max_units, args.whisper_model)
        else:
            from verify_media import verify_media
            result = verify_media(args.video, args.output_dir, args.expected_dialogue, args.whisper_model,
                                  expected_duration_seconds=args.expected_duration)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if "automated_passed" in result:
            return 0 if result["automated_passed"] else 1
        return 0 if result.get("ready", True) and result.get("status", "completed") == "completed" else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": scrub(str(exc))}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
