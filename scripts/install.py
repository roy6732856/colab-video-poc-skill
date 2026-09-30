#!/usr/bin/env python3
"""Copy the original skill locally; never download code or contact Colab."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
import shutil
import sys
import tempfile
from uuid import uuid4


SKILL_NAME = "colab-video-poc"
NOTICES = ("LICENSE", "THIRD_PARTY.md")
EXCLUDED_NAMES = {
    "__pycache__", ".cache", "cache", "caches", ".git", ".DS_Store",
    ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "upstream", "models", "outputs", "qa", "logs", "inputs",
    ".env", "sessions.json",
}
EXCLUDED_PATTERNS = (
    "*.pyc", "*.pyo", ".env.*", "*.log", "*.mp4", "*.mov", "*.webm",
    "*.mkv", "*.wav", "*.mp3", "*.safetensors", "*.ckpt", "*.pt", "*.pth",
    "*.pem", "*.key", "*credentials*.json", "*oauth*.json", "*token*.json",
    "jobs*.json", ".colab-cli*", ".colab-video-poc-install-*",
    ".colab-video-poc-backups",
)


def ignored_files(_directory: str, names: list[str]) -> set[str]:
    """Exclude runtime/private artifacts at every depth of the skill tree."""
    return {
        name for name in names
        if name in EXCLUDED_NAMES
        or any(fnmatch(name, pattern) for pattern in EXCLUDED_PATTERNS)
    }


def exists(path: Path) -> bool:
    """Also recognize dangling symlinks so they cannot be overwritten."""
    return path.exists() or path.is_symlink()


def install_skill(
    target: Path,
    *,
    replace: bool = False,
    repository: Path | None = None,
) -> tuple[Path, Path | None]:
    """Install into TARGET/colab-video-poc and return destination and backup.

    Stage the complete copy before moving an existing installation. Backups
    remain in a hidden sibling directory and are never automatically deleted.
    ``repository`` permits offline tests against a temporary package fixture.
    """
    root = (repository or Path(__file__).resolve().parents[1]).resolve()
    source = root / "skills" / SKILL_NAME
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Skill source must be a regular directory")
    skill_file = source / "SKILL.md"
    if skill_file.is_symlink() or not skill_file.is_file():
        raise ValueError("Skill source is missing a regular SKILL.md")
    for name in NOTICES:
        notice = root / name
        if notice.is_symlink() or not notice.is_file():
            raise ValueError(f"Package is missing a regular {name}")

    source = source.resolve()
    target = target.expanduser().resolve()
    destination = target / SKILL_NAME
    if (target == source or target.is_relative_to(source)
            or source == destination or source.is_relative_to(destination)):
        raise ValueError("Installation target must not overlap the skill source")
    if exists(destination):
        if not replace:
            raise FileExistsError(
                f"Existing installation preserved: {destination}. "
                "Use --replace to create a backup and replace it."
            )
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("Refusing to replace a symlink or non-directory target")

    target.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix=f".{SKILL_NAME}-install-", dir=target
    ) as temporary:
        staged = Path(temporary) / SKILL_NAME
        # Preserve links during staging instead of reading external content;
        # reject links below before anything is promoted into the target.
        shutil.copytree(source, staged, symlinks=True, ignore=ignored_files)
        for path in staged.rglob("*"):
            if path.is_symlink():
                raise ValueError("Skill source contains a symlink; installation refused")
        for name in NOTICES:
            shutil.copy2(root / name, staged / name)

        # Recheck after copying in case another installation appeared meanwhile.
        if exists(destination):
            if not replace:
                raise FileExistsError(f"Existing installation preserved: {destination}")
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("Refusing to replace a symlink or non-directory target")
            backup_root = target / f".{SKILL_NAME}-backups"
            if backup_root.is_symlink():
                raise ValueError("Refusing to write backups through a symlink")
            backup_root.mkdir(exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = backup_root / f"{timestamp}-{uuid4().hex}"
            destination.rename(backup)
        try:
            if exists(destination):
                raise FileExistsError(f"Installation target appeared: {destination}")
            staged.rename(destination)
        except BaseException:
            if backup is not None and not exists(destination):
                backup.rename(destination)
            raise
    return destination, backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path.home() / ".codex" / "skills",
        help="Parent skill directory (default: ~/.codex/skills)",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Back up an existing installation before replacing it",
    )
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    try:
        destination, backup = install_skill(args.target, replace=args.replace)
    except (OSError, ValueError) as error:
        print(f"Installation failed: {error}", file=sys.stderr)
        return 1
    print(f"Installed: {destination}")
    if backup is not None:
        print(f"Backup preserved: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
