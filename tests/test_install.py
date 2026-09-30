"""Offline installer behavior using disposable package and target trees."""

from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install.py"
INSTALL = runpy.run_path(str(INSTALLER))
install_skill = INSTALL["install_skill"]


class InstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.repository = self.base / "package"
        self.source = self.repository / "skills" / "colab-video-poc"
        self.target = self.base / "target with spaces"
        self.write(self.source / "SKILL.md", "---\nname: colab-video-poc\n---\n")
        self.write(self.source / "scripts" / "workflow.py", "print('fixture')\n")
        self.write(self.source / "references" / "usage.md", "離線安裝說明\n")
        self.write(self.repository / "LICENSE", "Fixture license\n")
        self.write(self.repository / "THIRD_PARTY.md", "Fixture notices\n")

    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def install(self, **kwargs):
        return install_skill(self.target, repository=self.repository, **kwargs)

    def test_copies_skill_references_and_notices_without_links(self):
        destination, backup = self.install()
        self.assertEqual(destination, self.target / "colab-video-poc")
        self.assertIsNone(backup)
        self.assertEqual(
            (destination / "references" / "usage.md").read_text(encoding="utf-8"),
            "離線安裝說明\n",
        )
        for relative in ("SKILL.md", "scripts/workflow.py", "LICENSE", "THIRD_PARTY.md"):
            self.assertTrue((destination / relative).is_file(), relative)
            self.assertFalse((destination / relative).is_symlink(), relative)
        self.write(self.source / "scripts" / "workflow.py", "changed source\n")
        self.assertEqual(
            (destination / "scripts" / "workflow.py").read_text(), "print('fixture')\n"
        )

    def test_existing_installation_is_preserved_without_replace(self):
        destination, _ = self.install()
        self.write(destination / "local.md", "keep local changes\n")
        before = {p.relative_to(destination): p.read_bytes()
                  for p in destination.rglob("*") if p.is_file()}
        self.write(self.source / "SKILL.md", "new source\n")
        with self.assertRaises(FileExistsError):
            self.install()
        after = {p.relative_to(destination): p.read_bytes()
                 for p in destination.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.target / ".colab-video-poc-backups").exists())

    def test_replace_keeps_complete_backup_and_removes_stale_installed_files(self):
        destination, _ = self.install()
        self.write(destination / "local.md", "original local content\n")
        self.write(destination / "__pycache__" / "old.pyc", "old cache\n")
        self.write(self.source / "SKILL.md", "new skill\n")
        destination, backup = self.install(replace=True)
        self.assertIsNotNone(backup)
        self.assertEqual(backup.parent, self.target / ".colab-video-poc-backups")
        self.assertEqual((backup / "local.md").read_text(), "original local content\n")
        self.assertEqual((backup / "__pycache__" / "old.pyc").read_text(), "old cache\n")
        self.assertEqual((destination / "SKILL.md").read_text(), "new skill\n")
        self.assertFalse((destination / "local.md").exists())
        self.assertFalse((destination / "__pycache__").exists())
        _, another_backup = self.install(replace=True)
        self.assertNotEqual(backup, another_backup)
        self.assertTrue((backup / "local.md").is_file())

    def test_cache_media_and_credentials_are_excluded_at_every_depth(self):
        excluded = [
            "__pycache__/root.pyc", "scripts/__pycache__/workflow.pyc",
            "scripts/loose.pyc", "scripts/loose.pyo", ".cache/upstream.py",
            "cache/notebook.ipynb", "references/cache/data.json",
            "upstream/runner.py", "models/model.safetensors", "outputs/sample.mp4",
            "references/sample.mp4", ".env", "scripts/.env.local",
            "scripts/oauth-client.json", "sessions.json", "logs/run.log",
        ]
        for relative in excluded:
            self.write(self.source / relative, "excluded fixture\n")
        destination, _ = self.install()
        for relative in excluded:
            self.assertFalse((destination / relative).exists(), relative)
        self.assertEqual(
            sorted(str(p.relative_to(destination)) for p in destination.rglob("*")
                   if p.is_file()),
            ["LICENSE", "SKILL.md", "THIRD_PARTY.md", "references/usage.md",
             "scripts/workflow.py"],
        )

    def test_source_symlink_is_rejected_without_reading_external_content(self):
        external = self.base / "external"
        self.write(external / "private.txt", "private fixture\n")
        (self.source / "references" / "external").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.install()
        self.assertFalse((self.target / "colab-video-poc").exists())
        self.assertEqual((external / "private.txt").read_text(), "private fixture\n")
        self.assertEqual(list(self.target.iterdir()), [])

    def test_ignored_cache_symlink_is_neither_followed_nor_copied(self):
        external = self.base / "external"
        self.write(external / "runner.py", "external fixture\n")
        (self.source / ".cache").symlink_to(external, target_is_directory=True)
        destination, _ = self.install()
        self.assertFalse((destination / ".cache").exists())
        self.assertFalse((destination / "runner.py").exists())

    def test_dangling_destination_symlink_is_preserved(self):
        self.target.mkdir()
        destination = self.target / "colab-video-poc"
        destination.symlink_to(self.base / "missing", target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.install()
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.install(replace=True)
        self.assertTrue(destination.is_symlink())

    def test_destination_file_is_preserved_even_with_replace(self):
        destination = self.target / "colab-video-poc"
        self.write(destination, "keep file\n")
        with self.assertRaisesRegex(ValueError, "non-directory"):
            self.install(replace=True)
        self.assertEqual(destination.read_text(), "keep file\n")

    def test_target_overlap_is_rejected_before_source_changes(self):
        for target in (self.source.parent, self.source, self.source / "nested"):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    install_skill(target, repository=self.repository, replace=True)
        self.assertFalse((self.source / "nested").exists())
        self.assertEqual((self.source / "SKILL.md").read_text(),
                         "---\nname: colab-video-poc\n---\n")

    def test_missing_skill_or_notice_does_not_create_target(self):
        for path in (self.source / "SKILL.md", self.repository / "LICENSE",
                     self.repository / "THIRD_PARTY.md"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(ValueError):
                        self.install()
                    self.assertFalse(self.target.exists())
                finally:
                    path.write_bytes(original)

    def test_staging_failure_preserves_existing_installation(self):
        destination, _ = self.install()
        self.write(destination / "local.md", "keep on failure\n")
        with patch.object(INSTALL["shutil"], "copytree", side_effect=OSError("copy failed")):
            with self.assertRaisesRegex(OSError, "copy failed"):
                self.install(replace=True)
        self.assertEqual((destination / "local.md").read_text(), "keep on failure\n")
        self.assertEqual(sorted(p.name for p in self.target.iterdir()), ["colab-video-poc"])

    def test_promotion_failure_restores_existing_installation(self):
        destination, _ = self.install()
        self.write(destination / "local.md", "keep after rollback\n")
        original_rename = Path.rename

        def fail_promotion(path, target):
            if path.parent.name.startswith(".colab-video-poc-install-"):
                raise OSError("promotion failed")
            return original_rename(path, target)

        with patch.object(Path, "rename", fail_promotion):
            with self.assertRaisesRegex(OSError, "promotion failed"):
                self.install(replace=True)
        self.assertEqual((destination / "local.md").read_text(), "keep after rollback\n")
        self.assertFalse(any(p.name.startswith(".colab-video-poc-install-")
                             for p in self.target.iterdir()))

    def test_backup_symlink_is_rejected_before_moving_existing_skill(self):
        destination, _ = self.install()
        self.write(destination / "local.md", "keep local\n")
        external = self.base / "external-backups"
        external.mkdir()
        (self.target / ".colab-video-poc-backups").symlink_to(
            external, target_is_directory=True
        )
        with self.assertRaisesRegex(ValueError, "backups through a symlink"):
            self.install(replace=True)
        self.assertEqual((destination / "local.md").read_text(), "keep local\n")
        self.assertEqual(list(external.iterdir()), [])

    def test_cli_from_another_working_directory_uses_package_source(self):
        command = [sys.executable, "-B", str(INSTALLER), "--target", str(self.target)]
        result = subprocess.run(command, cwd=self.base, capture_output=True, text=True,
                                timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Installed:", result.stdout)
        destination = self.target / "colab-video-poc"
        self.assertEqual((destination / "SKILL.md").read_bytes(),
                         (ROOT / "skills" / "colab-video-poc" / "SKILL.md").read_bytes())
        self.assertEqual((destination / "LICENSE").read_bytes(),
                         (ROOT / "LICENSE").read_bytes())
        again = subprocess.run(command, cwd=self.base, capture_output=True, text=True,
                               timeout=20)
        self.assertEqual(again.returncode, 1)
        self.assertIn("preserved", again.stderr)
        replaced = subprocess.run(command + ["--replace"], cwd=self.base,
                                  capture_output=True, text=True, timeout=20)
        self.assertEqual(replaced.returncode, 0, replaced.stderr)
        self.assertIn("Backup preserved:", replaced.stdout)
        self.assertEqual(len(list((self.target / ".colab-video-poc-backups").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
