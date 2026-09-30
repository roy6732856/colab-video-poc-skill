import importlib.util
import json
from pathlib import Path
import tempfile
import types
import sys
import subprocess
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "skills/colab-video-poc/scripts/workflow.py"
spec = importlib.util.spec_from_file_location("workflow", SCRIPT)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class FakeColab:
    executable = sys.executable
    auth = "oauth2"

    def __init__(self, new_error=False, stop_error=False, interrupted=False, preexisting=False):
        self.active = preexisting
        self.calls = []
        self.new_error, self.stop_error, self.interrupted = new_error, stop_error, interrupted
        self.interrupt_sent = False

    def usage(self):
        self.calls.append(["usage"])
        if self.active and self.interrupted and not self.interrupt_sent:
            self.interrupt_sent = True
            raise KeyboardInterrupt
        return {"balance": 80, "rate_per_hour": 9 if self.active else 0, "active_assignments": int(self.active)}

    def call(self, args, timeout=30):
        self.calls.append(args)
        if args[0] == "new":
            self.active = True
            if self.new_error:
                raise RuntimeError("Failure after allocation")
        elif args[0] == "stop":
            if self.stop_error:
                raise RuntimeError("Provider stop failed")
            self.active = False
        elif args[0] == "sessions":
            return "One active session" if self.active else "No active sessions found on server."
        elif args[0] == "download":
            Path(args[-1]).write_text("Prompt executed in 1.25 seconds\n")
        return "ok"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "reference.png"
        self.image.write_bytes(b"local test reference")
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("<Picture 1>\n今天，就讓創意動起來。\n", encoding="utf-8")
        self.manifest = self.root / "jobs.json"
        self.output = self.root / "outputs"
        self.upstream = self.root / "upstream"
        (self.upstream / "scripts").mkdir(parents=True)
        self.write_jobs(1)
        self.verifier = patch.dict(sys.modules, {"verify_media": types.SimpleNamespace(verify_media=lambda *args, **kwargs: {"technical_passed": True, "automated_passed": True})})
        self.verifier.start()

    def tearDown(self):
        self.verifier.stop()
        self.temp.cleanup()

    def write_jobs(self, count=1):
        self.manifest.write_text(json.dumps({"jobs": [{"id": f"clip-{i}", "reference_images": ["reference.png"],
            "prompt_file": "prompt.txt", "duration_seconds": 8} for i in range(count)]}))

    def write_runner(self, behavior="success"):
        if behavior == "wait":
            code = "import time\ntime.sleep(20)\n"
        else:
            code = """import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--manifest');p.add_argument('--session');p.add_argument('--gpu');p.add_argument('--timeout');p.add_argument('--progress');p.add_argument('--output-dir');a=p.parse_args()
d=json.loads(Path(a.manifest).read_text())
for i,j in enumerate(d['jobs']):
 if i and BEHAVIOR=='partial':raise SystemExit(1)
 Path(j['output_path']).write_bytes(b'completed test output')
Path(a.progress).write_text(json.dumps({'status':'completed','log_tail':['test runner completed']}))
""".replace("BEHAVIOR", repr(behavior))
        (self.upstream / "scripts/runner.py").write_text(code)

    def run_fake(self, cli, **kwargs):
        return workflow.run_batch(self.manifest, self.output, cli, self.upstream, **kwargs)

    def test_validation_precedes_allocation(self):
        self.image.unlink()
        cli = FakeColab()
        with self.assertRaises(ValueError):
            self.run_fake(cli)
        self.assertEqual(cli.calls, [])

    def test_plan_preserves_prompt_and_resolves_stable_seed(self):
        p = workflow.plan(self.manifest, self.output)
        self.write_runner()
        cli = FakeColab()
        result = self.run_fake(cli)
        resolved = json.loads((self.output / "resolved-jobs.json").read_text())["jobs"][0]
        self.assertEqual(p["jobs"][0]["seed"], resolved["seed"])
        self.assertEqual(Path(resolved["prompt_file"]).read_text(), self.prompt.read_text())
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["cleanup"]["confirmed"])
        self.assertFalse(cli.active)

    def test_changed_prompt_after_plan_does_not_allocate(self):
        workflow.plan(self.manifest, self.output)
        self.prompt.write_text("different prompt")
        cli = FakeColab()
        with self.assertRaises(ValueError):
            self.run_fake(cli)
        self.assertEqual(cli.calls, [])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        video = self.output / "clip-0.mp4";video.write_bytes(b"keep original")
        with self.assertRaises(ValueError):
            self.run_fake(FakeColab())
        self.assertEqual(video.read_bytes(), b"keep original")

    def test_changed_resolved_jobs_after_plan_does_not_allocate(self):
        workflow.plan(self.manifest, self.output)
        resolved = self.output / "resolved-jobs.json"
        data = json.loads(resolved.read_text())
        data["jobs"][0]["seed"] ^= 1
        resolved.write_text(json.dumps(data))
        cli = FakeColab()
        with self.assertRaises(ValueError):
            self.run_fake(cli)
        self.assertEqual(cli.calls, [])

    def test_rejects_nonfinite_duration_duplicate_ids_and_bad_picture(self):
        for mutation in ["nan", "duplicate", "picture"]:
            self.write_jobs(2 if mutation == "duplicate" else 1)
            data=json.loads(self.manifest.read_text())
            if mutation == "nan":data["jobs"][0]["duration_seconds"] = float("nan")
            elif mutation == "duplicate":data["jobs"][1]["id"]=data["jobs"][0]["id"]
            else:self.prompt.write_text("<Picture 2>")
            self.manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):workflow.resolve_manifest(self.manifest,self.output)

    def test_failure_after_allocation_stops_owned_session(self):
        cli = FakeColab(new_error=True)
        result = self.run_fake(cli)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["cleanup"]["confirmed"])
        self.assertTrue(any(args[0] == "stop" for args in cli.calls))

    def test_interrupt_after_allocation_stops_session(self):
        cli = FakeColab(interrupted=True)
        result = self.run_fake(cli)
        self.assertTrue(result["interrupted"])
        self.assertFalse(cli.active)

    def test_failure_of_later_clip_keeps_first_and_stops(self):
        self.write_jobs(2);self.write_runner("partial")
        cli = FakeColab();result=self.run_fake(cli)
        self.assertEqual(result["status"],"partial")
        self.assertTrue((self.output/"clip-0.mp4").is_file())
        self.assertFalse((self.output/"clip-1.mp4").exists())
        self.assertEqual(len(result["results"]),1)
        self.assertFalse(cli.active)

    def test_watchdog_stops_without_retry(self):
        self.write_runner("wait")
        ticks=iter([0,100,10000,10001]);last=[10001]
        def clock():
            try:last[0]=next(ticks)
            except StopIteration:pass
            return last[0]
        cli=FakeColab()
        with patch.object(workflow.time,"monotonic",side_effect=clock):
            result=self.run_fake(cli)
        self.assertTrue(result["guard_stop"])
        self.assertFalse(cli.active)
        self.assertEqual(sum(c[0]=="new" for c in cli.calls),1)

    def test_cleanup_failure_is_not_reported_as_success(self):
        self.write_runner()
        cli=FakeColab(stop_error=True);result=self.run_fake(cli)
        self.assertFalse(result["cleanup"]["confirmed"])
        self.assertNotEqual(result["status"],"completed")
        self.assertIn(result["session"],result["cleanup"]["recovery_command"])
        self.assertTrue((self.output/"clip-0.mp4").is_file())

    def test_cleanup_retry_stays_inside_unit_and_wall_guards(self):
        for rate in (18, 8.9):
            with self.subTest(rate=rate):
                self.output = self.root / f"retry-{rate}"
                self.write_runner("wait")
                clock = [0.0]
                original_sleep = workflow.time.sleep
                def sleep(seconds):
                    clock[0] += seconds
                    original_sleep(0.001)
                class DelayedColab(FakeColab):
                    attempts = 0
                    def usage(self):
                        data = super().usage()
                        data["rate_per_hour"] = rate if self.active else 0
                        return data
                    def call(self, args, timeout=30):
                        if args[0] == "stop":
                            self.attempts += 1
                            if self.attempts == 1:
                                clock[0] += timeout
                                raise subprocess.TimeoutExpired("colab stop", timeout)
                            clock[0] += 1
                        return super().call(args, timeout)
                with patch.object(workflow.time,"monotonic",side_effect=lambda:clock[0]), patch.object(workflow.time,"sleep",side_effect=sleep):
                    cli = DelayedColab()
                    result = self.run_fake(cli, max_minutes=20, max_units=3)
                self.assertEqual(cli.attempts,2)
                self.assertTrue(result["cleanup"]["confirmed"])
                self.assertFalse(result["budget_guard"]["limit_exceeded"])
                self.assertLessEqual(result["allocated_wall_seconds_estimate"],1200)
                self.assertLessEqual(result["estimated_compute_units"],3)

    def test_provider_latency_overrun_is_visible(self):
        clock = [0.0]
        class SlowAllocation(FakeColab):
            def call(self, args, timeout=30):
                if args[0] == "new":
                    clock[0] += 1250
                return super().call(args, timeout)
        with patch.object(workflow.time,"monotonic",side_effect=lambda:clock[0]):
            result=self.run_fake(SlowAllocation(), max_minutes=20, max_units=3)
        self.assertTrue(result["budget_guard"]["limit_exceeded"])
        self.assertTrue(result["cleanup"]["confirmed"])
        self.assertNotEqual(result["status"],"completed")

    def test_existing_allocation_is_not_stopped(self):
        cli=FakeColab(preexisting=True)
        with self.assertRaises(RuntimeError):self.run_fake(cli)
        self.assertTrue(cli.active)
        self.assertFalse(any(c[0]=="stop" for c in cli.calls))

    def test_failed_media_stage_cannot_be_reported_as_verified(self):
        self.write_runner()
        with patch.dict(sys.modules, {"verify_media": types.SimpleNamespace(verify_media=lambda *a, **kw: {"technical_passed": False, "automated_passed": False})}):
            result=self.run_fake(FakeColab())
        self.assertEqual(result["status"],"needs_review")
        self.assertEqual(result["results"][0]["status"],"verification_failed")

    def test_speech_mismatch_requires_review(self):
        self.write_runner()
        with patch.dict(sys.modules, {"verify_media": types.SimpleNamespace(verify_media=lambda *a, **kw: {"technical_passed": True, "automated_passed": False, "stages": {"asr": {"status": "mismatch", "match": False}}})}):
            result=self.run_fake(FakeColab())
        self.assertEqual(result["status"],"needs_review")

    def test_secret_redaction_keeps_only_safe_url_path(self):
        raw="https://example.com/path?colab-runtime-proxy-token=secret Authorization: Bearer hidden HF_TOKEN=value"
        cleaned=workflow.scrub(raw)
        self.assertNotIn("secret",cleaned);self.assertNotIn("hidden",cleaned);self.assertNotIn("HF_TOKEN=value",cleaned)
        self.assertIn("https://example.com/path",cleaned)


if __name__ == "__main__":unittest.main()
