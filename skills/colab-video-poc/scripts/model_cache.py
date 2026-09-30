"""Bounded, public MiniMax-H3 cache priming and local staging (Python >= 3.11).

Manifest JSON (sha256 is the LFS object's SHA256, not an HTTP ETag):
    {"repo_id": "Comfy-Org/MiniMax-H3", "revision": "<40 hex commit>",
     "models": [{"path": "<plan path>", "bytes": <plan size>,
                "sha256": "<64 hex LFS SHA256>"}, ...]}

All five paths/sizes from ../model-download-plan.json are required. The plan
alone is NOT a manifest: the parent must supply the commit and LFS digests.
Cache layout: CACHE/<plan path>. Supply a cache root dedicated to this commit,
e.g. /content/drive/MyDrive/ColabModelCache/MiniMax-H3/<revision>/.
Stage layout: STAGE/<plan path>. Omit stage_root for CPU-only priming.

The cache root must already exist; mounting, authentication and quota checks
belong to the parent. No token is sent to HF, and HF is imported only on miss.
The injected downloader accepts the same keyword arguments as hf_hub_download
and must return the regular file at local_dir/filename. Each miss gets one
attempt in a private local_dir on the cache filesystem. Its deterministic
SHA256-based directory lets HF resume .incomplete downloads after process death;
handled errors clean temporary files, and committed entries survive either.

Reads use bounded chunks and descriptor-relative, no-symlink traversal. Hashing
and staging share one stream; verified output is never reread just to hash it.
Existing stage files are independently verified before reuse. Per-file atomic
publication is used; a failed run may leave earlier, fully verified entries.
Cold priming needs one cache copy; cold staging additionally needs one local
copy, not a second complete cache. Repairs retain the old file until publication
and need space for that file's temporary replacement. Local free space is checked
before each stage copy; Drive quota remains explicitly unknown (parent-owned).
Byte counters describe verified files/streamed writes, not measured HTTP traffic.
Copy timings overlap the corresponding combined verify-and-copy timings.
Advisory locks serialize cooperating runs sharing either root. They require
POSIX flock support; there is no silent lock/storage fallback. The caller must
keep mounted roots stable and avoid external writers during a run.

CLI: model_cache.py --manifest manifest.json --cache-root /mounted/cache
                    [--stage-root /local/models] --report report.json
Reports contain public manifest metadata/counters/timings, never HF responses,
environment variables, tokens, download URLs, or raw exception messages.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
from typing import Any, Callable, Mapping
import uuid


REPO_ID = "Comfy-Org/MiniMax-H3"
EXPECTED_FILES = {
    "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors": 20_958_205_608,
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": 1_956_193_000,
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15_687_142_551,
    "vae/minimax_h3_audio_vae_fp32.safetensors": 605_254_808,
    "vae/minimax_h3_video_vae_fp16.safetensors": 5_207_808_496,
}
CHUNK_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
LOCK_NAME = ".model-cache.lock"


class CacheError(RuntimeError):
    """An error with a safe, fixed message suitable for logs."""


class _Invalid(CacheError):
    """A regular weight file failed integrity verification."""


@dataclass(frozen=True)
class ModelFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    revision: str
    files: tuple[ModelFile, ...]


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CacheError("Invalid relative path")
    parts = value.split("/")
    if any(p in ("", ".", "..") or ":" in p for p in parts):
        raise CacheError("Invalid relative path")
    if PurePosixPath(value).is_absolute():
        raise CacheError("Absolute model paths are forbidden")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> ModelManifest:
    """Validate the one supported profile before touching storage or HF."""
    if not isinstance(manifest, Mapping) or manifest.get("repo_id") != REPO_ID:
        raise CacheError("Only the public Comfy-Org/MiniMax-H3 profile is supported")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise CacheError("Manifest revision must be a full 40-hex commit")
    rows = manifest.get("models")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_FILES):
        raise CacheError("Manifest must contain all five planned files")
    entries = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CacheError("Invalid manifest file entry")
        path = _relative(row.get("path"))
        if path not in EXPECTED_FILES or path in entries:
            raise CacheError("Unexpected or duplicate manifest path")
        size, digest = row.get("bytes"), row.get("sha256")
        if type(size) is not int or size != EXPECTED_FILES[path] or size <= 0:
            raise CacheError("Manifest size does not match the download plan")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise CacheError("Manifest requires a 64-hex LFS SHA256 for each file")
        entries[path] = ModelFile(path, size, digest.lower())
    total = sum(f.bytes for f in entries.values())
    if "total_bytes" in manifest and (type(manifest["total_bytes"]) is not int
                                      or manifest["total_bytes"] != total):
        raise CacheError("Manifest total_bytes does not match the download plan")
    return ModelManifest(revision.lower(), tuple(entries[p] for p in EXPECTED_FILES))


def _absolute(value: os.PathLike[str] | str) -> Path:
    path = Path(value)
    if ".." in path.parts or "\x00" in os.fspath(path):
        raise CacheError("Parent traversal is forbidden")
    return Path(os.path.abspath(path))


@dataclass
class _Directory:
    path: Path
    fd: int

    def close(self) -> None:
        os.close(self.fd)


@contextmanager
def _directory(path: Path, *, create: bool = False):
    """Open each component using openat/O_NOFOLLOW, including root ancestors."""
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    raise CacheError("Required directory is absent; mount the cache first") from None
                raise CacheError("Directories must be real directories without symlinks") from None
            os.close(fd)
            fd = child
        yield _Directory(path, fd)
    finally:
        os.close(fd)


@contextmanager
def _parent(root: _Directory, relative: str, *, create: bool = False):
    parts = _relative(relative).split("/")
    fd = os.dup(root.fd)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except OSError:
                raise CacheError("Model parent directories must not be symlinks") from None
            os.close(fd)
            fd = child
        yield _Directory(root.path.joinpath(*parts[:-1]), fd), parts[-1]
    finally:
        os.close(fd)


def _info(parent: _Directory, name: str):
    try:
        result = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(result.st_mode):
        raise CacheError("Weight, lock and report entries must be regular files without symlinks")
    return result


def _fingerprint(info) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_read(parent: _Directory, name: str) -> int:
    # NONBLOCK prevents a concurrently substituted FIFO from blocking open().
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent.fd)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise CacheError("Source weights must be regular files without symlinks")
    return fd


def _stream_verify(parent: _Directory, name: str, entry: ModelFile,
                   metrics: dict[str, Any], output=None, timing=None):
    """Hash one stream; optionally write those exact bytes to a stage temp."""
    with os.fdopen(_open_read(parent, name), "rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size != entry.bytes:
            raise _Invalid("Weight size mismatch")
        digest, count = hashlib.sha256(), 0
        while chunk := source.read(CHUNK_BYTES):
            count += len(chunk)
            metrics["hashed_bytes"] += len(chunk)
            if count > entry.bytes:
                raise _Invalid("Weight grew during verification")
            digest.update(chunk)
            if output is not None:
                started = time.perf_counter()
                try:
                    output.write(chunk)
                    metrics["stage_written_bytes"] += len(chunk)
                finally:
                    timing["stage_copy"] += time.perf_counter() - started
        after = os.fstat(source.fileno())
        if (count != entry.bytes or _fingerprint(before) != _fingerprint(after)
                or digest.hexdigest() != entry.sha256):
            raise _Invalid("Weight SHA256 or stable-size verification failed")
        current = _info(parent, name)
        if current is None or _fingerprint(current) != _fingerprint(after):
            raise _Invalid("Weight changed during verification")
    if output is not None:
        started = time.perf_counter()
        try:
            output.flush()
            os.fsync(output.fileno())
            copied = os.fstat(output.fileno())
            if copied.st_size != entry.bytes:
                raise _Invalid("Staged copy size mismatch")
        finally:
            timing["stage_copy"] += time.perf_counter() - started
    return _fingerprint(after), _fingerprint(copied) if output is not None else None


def _clean_stage_temps(parent: _Directory) -> None:
    # Only helper-owned names under the held stage lock; never scan a whole root.
    for name in os.listdir(parent.fd):
        if re.fullmatch(r"\.model-cache-[0-9a-f]{32}\.tmp", name):
            _info(parent, name)
            os.unlink(name, dir_fd=parent.fd)


def _free_bytes(root: _Directory) -> int | None:
    try:
        info = os.fstatvfs(root.fd)
        return info.f_bavail * info.f_frsize
    except OSError:
        return None


def _safe_workspace(root: _Directory) -> None:
    """Reject symlinks in resumable HF metadata before giving HF a local_dir."""
    count = 0
    def visit(fd, depth):
        nonlocal count
        if depth > 16:
            raise CacheError("Download workspace exceeds the metadata depth limit")
        with os.scandir(fd) as rows:
            for row in rows:
                count += 1
                if count > 256:
                    raise CacheError("Download workspace exceeds the metadata entry limit")
                info = row.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(row.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=fd)
                    try:
                        visit(child, depth + 1)
                    finally:
                        os.close(child)
                elif not stat.S_ISREG(info.st_mode):
                    raise CacheError("Download workspace must not contain symlinks or special files")
    visit(root.fd, 0)


def _discard_download_residue(root: _Directory, digest: str) -> None:
    name = ".model-cache-download-" + digest
    try:
        info = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode):
        raise CacheError("Download workspace must be a real directory without symlinks")
    shutil.rmtree(name, dir_fd=root.fd)


@contextmanager
def _stage_temp(parent: _Directory | None):
    if parent is None:
        yield None, None
        return
    name = ".model-cache-" + uuid.uuid4().hex + ".tmp"
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=parent.fd)
    try:
        with os.fdopen(fd, "wb") as output:
            yield name, output
    finally:
        try:
            os.unlink(name, dir_fd=parent.fd)
        except FileNotFoundError:
            pass


@contextmanager
def _workspace(root: _Directory, digest: str):
    name = ".model-cache-download-" + digest
    resumed = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=root.fd)
    except FileExistsError:
        resumed = True
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root.fd)
    try:
        work = _Directory(root.path / name, fd)
        _safe_workspace(work)
        yield work, resumed
    finally:
        os.close(fd)
        shutil.rmtree(name, dir_fd=root.fd)


@contextmanager
def _lock(root: _Directory, deadline: float):
    _info(root, LOCK_NAME)
    fd = os.open(LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                 0o600, dir_fd=root.fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CacheError("Cache lock must be a regular file")
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise CacheError("Timed out waiting for the model cache lock") from None
                time.sleep(min(0.05, remaining))
            except OSError:
                raise CacheError("Cache filesystem does not support the required lock") from None
        yield
    finally:
        os.close(fd)


def _hf_download(**kwargs):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise CacheError("huggingface_hub is required only for cache misses") from None
    return hf_hub_download(**kwargs)


def _publish(source_parent: _Directory, source_name: str, target_parent: _Directory,
             target_name: str, verified: tuple[int, ...]) -> None:
    current = _info(source_parent, source_name)
    if current is None or _fingerprint(current) != verified:
        raise CacheError("Verified source changed before atomic publication")
    _info(target_parent, target_name)  # Never replace a symlink/special file.
    os.replace(source_name, target_name, src_dir_fd=source_parent.fd,
               dst_dir_fd=target_parent.fd)


def _one_file(manifest: ModelManifest, entry: ModelFile, cache: _Directory,
              stage: _Directory | None, downloader: Callable, report: dict[str, Any]) -> None:
    metrics, timing = report["counters"], report["seconds"]
    row = {"path": entry.path, "bytes": entry.bytes,
           "cache": "pending", "stage": "pending" if stage else "omitted"}
    report["files"].append(row)
    cache_relative = entry.path
    with ExitStack() as stack:
        cp, cn = stack.enter_context(_parent(cache, cache_relative, create=True))
        sp, sn = stack.enter_context(_parent(stage, entry.path, create=True)) if stage else (None, None)
        stage_valid = False
        if sp:
            _clean_stage_temps(sp)
        if sp and _info(sp, sn) is not None:
            started = time.perf_counter()
            try:
                _stream_verify(sp, sn, entry, metrics)
                stage_valid = True
                metrics["stage_hits"] += 1
                metrics["stage_verified_bytes"] += entry.bytes
                row["stage"] = "hit"
            except _Invalid:
                metrics["stage_invalid_entries"] += 1
            finally:
                timing["stage_verify"] += time.perf_counter() - started

        if sp and not stage_valid:
            available = _free_bytes(sp)
            row["stage_free_bytes_before_copy"] = available
            if available is not None and available < entry.bytes:
                raise CacheError("Insufficient local free space for an atomic staged file")

        # Cache-hit hashing and staging use the SAME read, avoiding a second
        # full Drive read. Only independently existing staged files need a hash.
        original = _info(cp, cn)
        if original is not None:
            started = time.perf_counter()
            try:
                with _stage_temp(sp if not stage_valid else None) as (tmp, output):
                    _, staged = _stream_verify(cp, cn, entry, metrics, output, timing)
                    metrics["cache_hits"] += 1
                    metrics["cache_verified_bytes"] += entry.bytes
                    row["cache"] = "hit"
                    if output is not None:
                        output.close()
                        _publish(sp, tmp, sp, sn, staged)
                        metrics["staged_files"] += 1
                        metrics["staged_bytes"] += entry.bytes
                        row["stage"] = "copied"
                _discard_download_residue(cache, entry.sha256)
                return
            except _Invalid:
                metrics["cache_invalid_entries"] += 1
            finally:
                timing["cache_verify_and_copy"] += time.perf_counter() - started

        with _workspace(cache, entry.sha256) as (work, resumed):
            if resumed:
                metrics["resume_candidates"] += 1
            local_dir = work.path / "download"
            with _directory(local_dir, create=True):
                pass
            metrics["download_attempts"] += 1
            started = time.perf_counter()
            try:
                downloaded = downloader(repo_id=REPO_ID, filename=entry.path,
                                        revision=manifest.revision, repo_type="model",
                                        local_dir=str(local_dir), token=False,
                                        endpoint="https://huggingface.co", etag_timeout=10)
            except Exception:
                raise CacheError("Hugging Face download failed; details omitted") from None
            finally:
                timing["download"] += time.perf_counter() - started
            if (not isinstance(downloaded, (str, os.PathLike))
                    or _absolute(downloaded) != local_dir / entry.path):
                raise CacheError("Downloader must return the requested file inside local_dir")
            with _parent(work, "download/" + entry.path) as (dp, dn):
                # The downloaded file is already on the cache filesystem. Hash
                # it once, tee to stage if needed, then rename directly to cache.
                started = time.perf_counter()
                try:
                    with _stage_temp(sp if not stage_valid else None) as (tmp, output):
                        verified, staged = _stream_verify(dp, dn, entry, metrics, output, timing)
                        with os.fdopen(_open_read(dp, dn), "rb") as source:
                            os.fsync(source.fileno())
                        current = _info(cp, cn)
                        reuse = False
                        if current is not None and (original is None
                                or _fingerprint(current) != _fingerprint(original)):
                            try:
                                _stream_verify(cp, cn, entry, metrics)
                                reuse = True
                            except _Invalid:
                                pass
                        if not reuse:
                            _publish(dp, dn, cp, cn, verified)
                        else:
                            row["cache_reused_after_download"] = True
                        metrics["downloads"] += 1
                        metrics["download_verified_bytes"] += entry.bytes
                        row["cache"] = "download"
                        if output is not None:
                            output.close()
                            _publish(sp, tmp, sp, sn, staged)
                            metrics["staged_files"] += 1
                            metrics["staged_bytes"] += entry.bytes
                            row["stage"] = "copied"
                finally:
                    timing["download_verify_and_copy"] += time.perf_counter() - started


def _report_destination(path: Path, cache: Path, stage: Path | None) -> None:
    # Reports may sit beside weights, but must never replace a weight or lock.
    for root in (cache, stage):
        if root and (path == root / LOCK_NAME or path == root):
            raise CacheError("Report collides with a cache root or lock")
        if root and any(path == root / p or (root / p).is_relative_to(path) for p in EXPECTED_FILES):
            raise CacheError("Report collides with a model path")
        if root and path.is_relative_to(root) and any(
                part.startswith(".model-cache-") for part in path.relative_to(root).parts):
            raise CacheError("Report collides with helper temporary storage")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    with _directory(path.parent) as parent:
        _info(parent, path.name)
        name = ".model-cache-report-" + uuid.uuid4().hex + ".tmp"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent.fd)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            _info(parent, path.name)
            os.replace(name, path.name, src_dir_fd=parent.fd, dst_dir_fd=parent.fd)
        finally:
            try:
                os.unlink(name, dir_fd=parent.fd)
            except FileNotFoundError:
                pass


def prepare_models(manifest: Mapping[str, Any], cache_root: os.PathLike[str] | str,
                   stage_root: os.PathLike[str] | str | None = None, *,
                   downloader: Callable | None = None,
                   report_path: os.PathLike[str] | str | None = None,
                   lock_timeout: float = 30.0) -> dict[str, Any]:
    """Prime persistent cache, optionally stage ALL five files, return telemetry.

    Raises CacheError on failure. If supplied, report_path receives an atomic
    success/failure report; its parent must exist. File-level partial progress
    is reusable. Lock timeout is shared across roots; network retry policy is
    owned by huggingface_hub/the parent, not an unbounded helper retry loop.
    """
    started = time.perf_counter()
    model = validate_manifest(manifest)
    cache_path = _absolute(cache_root)
    stage_path = _absolute(stage_root) if stage_root is not None else None
    destination = _absolute(report_path) if report_path is not None else None
    if type(lock_timeout) not in (int, float) or not math.isfinite(lock_timeout) or lock_timeout < 0:
        raise CacheError("Lock timeout must be a finite nonnegative number")
    if stage_path and (stage_path.is_relative_to(cache_path) or cache_path.is_relative_to(stage_path)):
        raise CacheError("Cache and stage roots must be separate, non-overlapping directories")
    if destination:
        _report_destination(destination, cache_path, stage_path)
        with _directory(destination.parent) as parent:
            _info(parent, destination.name)
    report = {
        "schema_version": 1, "status": "running", "repo_id": REPO_ID,
        "revision": model.revision, "mode": "stage" if stage_path else "prime",
        "required_bytes": sum(f.bytes for f in model.files),
        "capacity": {"drive_quota_free_bytes": None, "drive_quota_status": "unknown",
                     "local_stage_free_bytes": None,
                     "largest_atomic_file_bytes": max(f.bytes for f in model.files)},
        "network_transferred_bytes": None,
        "files": [],
        "counters": dict.fromkeys(("cache_hits", "cache_invalid_entries", "cache_verified_bytes",
                                  "download_attempts", "downloads", "download_verified_bytes",
                                  "resume_candidates", "stage_hits", "stage_verified_bytes",
                                  "stage_invalid_entries", "staged_files", "staged_bytes",
                                  "stage_written_bytes", "hashed_bytes"), 0),
        "seconds": dict.fromkeys(("lock_wait", "cache_verify_and_copy", "download",
                                  "download_verify_and_copy", "stage_verify", "stage_copy", "total"), 0.0),
    }
    try:
        with ExitStack() as stack:
            cache = stack.enter_context(_directory(cache_path))  # Never mkdir the Drive root.
            stage = stack.enter_context(_directory(stage_path, create=True)) if stage_path else None
            if stage:
                report["capacity"]["local_stage_free_bytes"] = _free_bytes(stage)
            lock_started = time.perf_counter()
            deadline = lock_started + lock_timeout
            for root in sorted((r for r in (cache, stage) if r), key=lambda r: str(r.path)):
                stack.enter_context(_lock(root, deadline))
            report["seconds"]["lock_wait"] = time.perf_counter() - lock_started
            for entry in model.files:
                _one_file(model, entry, cache, stage, downloader or _hf_download, report)
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc) if isinstance(exc, CacheError) else "Storage operation failed; details omitted"
        if isinstance(exc, CacheError):
            raise
        raise CacheError("Storage operation failed; details omitted") from None
    finally:
        report["seconds"]["total"] = time.perf_counter() - started
        if destination:
            _write_report(destination, report)
    return report


def _read_manifest(path: Path) -> Mapping[str, Any]:
    with _directory(path.parent) as parent:
        with os.fdopen(_open_read(parent, path.name), "rb") as source:
            if os.fstat(source.fileno()).st_size > MAX_MANIFEST_BYTES:
                raise CacheError("Manifest exceeds the 64 KiB limit")
            data = source.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise CacheError("Manifest exceeds the 64 KiB limit")
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise CacheError("Manifest must be valid UTF-8 JSON") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--stage-root")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        source, destination = _absolute(args.manifest), _absolute(args.report)
        if source == destination:
            raise CacheError("Report must not overwrite the manifest")
        result = prepare_models(_read_manifest(source), args.cache_root, args.stage_root,
                                report_path=destination)
    except Exception as exc:
        # Do not leak arbitrary downloader/storage exceptions in terminal logs.
        message = str(exc) if isinstance(exc, CacheError) else "Cache operation failed; details omitted"
        print(message, file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "mode": result["mode"],
                      "required_bytes": result["required_bytes"], "counters": result["counters"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
