"""Offline cache opt-in, plan guards, launch order and failure behavior."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/colab-video-poc/scripts"
sys.path.insert(0, str(SCRIPTS))
import cache_launcher as launcher
import workflow
from test_workflow import FakeColab

ROOT = "/content/drive/MyDrive/ColabModelCache/MiniMax-H3/e5eb578a89295337b8ff433a035929ce0279e0b6"


class CacheIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.output = self.base / "outputs"
        self.output.mkdir()
        (self.base / "image.png").write_bytes(b"synthetic image")
        self.manifest = self.base / "jobs.json"
        self.manifest.write_text(json.dumps({"jobs": [{"id": "clip", "reference_images": ["image.png"],
                                                      "prompt": "<Picture 1> test", "seed": 7}]}))
        self.upstream = self.base / "upstream"
        (self.upstream / "assets").mkdir(parents=True)
        notebook = {"cells": [{"cell_type": "markdown", "source": []}] * 3 + [
            {"cell_type": "code", "source": ["SETUP_SENTINEL = True\n"]},
            {"cell_type": "code", "source": ["raise RuntimeError('must not extract download cell')\n"]}]}
        (self.upstream / "assets/MiniMax_H3_Turbo_Colab.ipynb").write_text(json.dumps(notebook))
        self.command = [sys.executable, "-u", str(self.upstream / "scripts/runner.py"), "batch",
                        "--manifest", str(self.output / "resolved-jobs.json"), "--session", "owned-session",
                        "--gpu", "G4", "--timeout", "500", "--progress", str(self.output / "progress.json"),
                        "--output-dir", str(self.output)]

    def planned(self):
        return workflow.plan(self.manifest, self.output, drive_cache=ROOT)

    def test_plan_records_root_profile_hash_and_allocates_nothing(self):
        result = self.planned()
        self.assertEqual(result["drive_cache"], workflow.drive_cache_profile(ROOT, "G4"))
        self.assertFalse(result["allocates_gpu"])
        self.assertEqual(result, json.loads((self.output / "plan.json").read_text()))
        plain = workflow.plan(self.manifest, self.base / "plain")
        self.assertNotIn("drive_cache", plain)

    def test_changed_root_profile_or_disabling_does_not_allocate(self):
        self.planned()
        for root in (None, ROOT + "-different"):
            with self.subTest(root=root):
                cli = FakeColab()
                with self.assertRaisesRegex(ValueError, "Drive cache root/profile"):
                    workflow.run_batch(self.manifest, self.output, cli, self.upstream, drive_cache=root)
                self.assertEqual(cli.calls, [])
        alternative = self.base / "profile.json"
        profile = json.loads(workflow.CACHE_MANIFEST.read_text())
        profile["models"][0]["sha256"] = "0" * 64
        alternative.write_text(json.dumps(profile))
        cli = FakeColab()
        with patch.object(workflow, "CACHE_MANIFEST", alternative):
            with self.assertRaisesRegex(ValueError, "Drive cache root/profile"):
                workflow.run_batch(self.manifest, self.output, cli, self.upstream, drive_cache=ROOT)
        self.assertEqual(cli.calls, [])

    def test_enabling_cache_after_default_plan_or_without_plan_is_rejected(self):
        for planned in (False, True):
            with self.subTest(planned=planned):
                if planned:
                    workflow.plan(self.manifest, self.output)
                cli = FakeColab()
                with self.assertRaises(ValueError):
                    workflow.run_batch(self.manifest, self.output, cli, self.upstream, drive_cache=ROOT)
                self.assertEqual(cli.calls, [])

    def test_unsupported_gpu_and_roots_rejected_before_allocation(self):
        for root, gpu in ((ROOT, "A100"), ("/content/local", "G4"),
                          ("/content/drive/MyDrive/../escape", "G4")):
            with self.subTest(root=root, gpu=gpu), self.assertRaises(ValueError):
                workflow.plan(self.manifest, self.output, gpu=gpu, drive_cache=root)

    def test_owned_guard_wraps_cache_child_and_releases_on_launch_failure(self):
        self.planned()
        cli = FakeColab()
        with patch.object(workflow.subprocess, "Popen", side_effect=OSError("child launch failed")) as popen:
            result = workflow.run_batch(self.manifest, self.output, cli, self.upstream, drive_cache=ROOT)
        command = popen.call_args.args[0]
        self.assertEqual(Path(command[2]).name, "cache_launcher.py")
        self.assertIn(ROOT, command)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["drive_cache"]["persistence"], "unknown")
        self.assertTrue(result["cleanup"]["confirmed"])
        self.assertFalse(cli.active)
        self.assertEqual(sum(c[0] == "new" for c in cli.calls), 1)

    def test_extracts_only_first_setup_cell(self):
        with patch.object(launcher, "verify_checkout") as verified:
            source = launcher.setup_source(self.upstream)
        verified.assert_called_once_with(self.upstream)
        scope = {"INFERENCE_MODE": "reference", "REF2VA_VARIANT": "fp8_scaled"}
        exec(source, scope)
        self.assertTrue(scope["SETUP_SENTINEL"])
        with self.assertRaises(AssertionError):
            exec(source, {"INFERENCE_MODE": "reference", "REF2VA_VARIANT": "bf16"})

    def remote_exec(self, mounted=True, variant="fp8_scaled"):
        root = self.base / "mounted-cache"
        comfy = self.base / "ComfyUI"
        comfy.mkdir(exist_ok=True)
        source = launcher.remote_source(str(root), stage_root=str(comfy / "models"),
                                        manifest_path=str(workflow.CACHE_MANIFEST))
        helper = types.SimpleNamespace(validate_manifest=lambda m: None)
        helper.prepare_models = unittest.mock.Mock(side_effect=lambda m, r, **kw: self.assertTrue(r.is_dir()))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {"model_cache": helper, "torch": torch}), patch("os.path.ismount", return_value=mounted):
            exec(source, {"INFERENCE_MODE": "reference", "REF2VA_VARIANT": variant})
        return helper, root, comfy

    def test_remote_mount_check_then_mkdir_then_prepare(self):
        helper, root, comfy = self.remote_exec()
        self.assertTrue(root.is_dir())
        self.assertEqual(helper.prepare_models.call_count, 1)
        self.assertEqual(helper.prepare_models.call_args.kwargs["stage_root"], str(comfy / "models"))

    def test_remote_missing_mount_or_variant_never_creates_root(self):
        for mounted, variant in ((False, "fp8_scaled"), (True, "bf16"), (True, "int8_convrot")):
            with self.subTest(mounted=mounted, variant=variant), self.assertRaises(AssertionError):
                self.remote_exec(mounted, variant)
            self.assertFalse((self.base / "mounted-cache").exists())

    def test_remote_flush_requires_mount_and_unmount_confirmation(self):
        drive = types.SimpleNamespace(flush_and_unmount=unittest.mock.Mock())
        modules = {"google.colab": types.SimpleNamespace(drive=drive)}
        for states, succeeds in (([True, False], True), ([True, True], False), ([False], False)):
            drive.flush_and_unmount.reset_mock()
            with patch.dict(sys.modules, modules), patch("os.path.ismount", side_effect=states):
                if succeeds:
                    exec(launcher.flush_source(), {})
                else:
                    with self.assertRaises(AssertionError):
                        exec(launcher.flush_source(), {})
            self.assertEqual(drive.flush_and_unmount.call_count, int(states[0]))

    def fake_launcher_cli(self, fail=None):
        calls = []
        class CLI:
            def call(_, args, timeout):
                calls.append((list(args), timeout))
                kind = args[0]
                if kind == "exec":
                    kind = Path(args[-1]).name
                if kind == fail:
                    raise RuntimeError("simulated failure")
                if args[0] == "download":
                    report = {"schema_version": 1, "status": "ok", "revision": workflow.drive_cache_profile(ROOT, "G4")["revision"],
                              "files": [{"path": str(i)} for i in range(5)], "counters": {"cache_hits": 5, "downloads": 0}}
                    Path(args[-1]).write_text(json.dumps(report))
                return "ok"
        return CLI(), calls

    def test_launcher_order_flush_before_exec_and_same_owned_session(self):
        cli, calls = self.fake_launcher_cli()
        executed = unittest.mock.Mock()
        sha = workflow.drive_cache_profile(ROOT, "G4")["profile_sha256"]
        with patch.object(launcher, "setup_source", return_value="SETUP = True"):
            launcher.launch(self.upstream, cli, ROOT, sha, self.command, execute=executed)
        phases = [Path(a[-1]).name if a[0] == "exec" else a[0] for a, _ in calls]
        self.assertEqual(phases, ["drivemount", "setup.py", "upload", "upload", "remote.py", "download", "flush.py"])
        self.assertTrue(all(a[a.index("--session") + 1] == "owned-session" for a, _ in calls))
        self.assertTrue(all(0 < timeout <= 500 for _, timeout in calls))
        executed.assert_called_once()
        resumed = executed.call_args.args[1]
        self.assertEqual(resumed[resumed.index("--session") + 1], "owned-session")
        self.assertLessEqual(float(resumed[resumed.index("--timeout") + 1]), 500)
        report = json.loads((self.output / "cache-report.json").read_text())
        self.assertEqual(report["persistence"], "flush_unmount_confirmed")
        self.assertEqual(report["model_cache"]["counters"]["downloads"], 0)

    def test_mount_stage_flush_failure_never_runs_upstream_or_falls_back(self):
        for failure in ("drivemount", "remote.py", "flush.py"):
            with self.subTest(failure=failure):
                cli, calls = self.fake_launcher_cli(failure)
                execute = unittest.mock.Mock()
                sha = workflow.drive_cache_profile(ROOT, "G4")["profile_sha256"]
                with patch.object(launcher, "setup_source", return_value="SETUP = True"), self.assertRaises(RuntimeError):
                    launcher.launch(self.upstream, cli, ROOT, sha, self.command, execute=execute)
                execute.assert_not_called()
                result = json.loads((self.output / "cache-report.json").read_text())
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["persistence"], "unknown")
                self.assertFalse(any(a[0] in ("new", "stop") for a, _ in calls))

    def test_child_profile_mutation_prevents_all_cli_calls(self):
        cli, calls = self.fake_launcher_cli()
        with self.assertRaises(ValueError):
            launcher.launch(self.upstream, cli, ROOT, "0" * 64, self.command)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
