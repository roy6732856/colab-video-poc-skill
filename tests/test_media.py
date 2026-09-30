"""Small real-media integration tests; no network, models or GPU required."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "colab-video-poc" / "scripts" / "verify_media.py"
SPEC = importlib.util.spec_from_file_location("verify_media", SCRIPT)
media_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(media_module)
verify_media = media_module.verify_media


class MediaPathTests(unittest.TestCase):
    def test_h3_duration_accepts_alignment_and_rejects_short_output(self):
        for requested, frames in ((4,107),(8,192),(15,362)):
            with self.subTest(requested=requested):
                stage = media_module._duration_requirement({"video_duration_seconds":frames/24},requested)
                self.assertEqual(stage["status"],"passed")
                self.assertEqual(stage["expected_aligned_frames"],frames)
        stage = media_module._duration_requirement({"video_duration_seconds":4},8)
        self.assertEqual(stage["status"],"mismatch")

    def test_source_and_output_errors_preserve_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.mp4"
            with self.assertRaises(FileNotFoundError):
                verify_media(missing, root / "out")
            self.assertFalse((root / "out").exists())
            with self.assertRaises(FileNotFoundError):
                verify_media(root, root / "out")
            source = root / "verification.json"
            source.write_bytes(b"source must remain unchanged")
            with self.assertRaises(ValueError):
                verify_media(source, root)
            with self.assertRaises(OSError):
                verify_media(source, source)
            blocked = root / "blocked"
            blocked.write_bytes(b"not a directory")
            with self.assertRaises(OSError):
                verify_media(source, blocked / "out")
            self.assertEqual(source.read_bytes(), b"source must remain unchanged")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools not installed")
class MediaIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.fixtures = {}
        for name, audio in (("signal", "sine=frequency=440:sample_rate=16000"),
                            ("silent", "anullsrc=r=16000:cl=stereo"), ("video_only", None)):
            path = cls.root / f"{name}.mp4"
            command = [shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-y",
                       "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=12"]
            if audio:
                command += ["-f", "lavfi", "-i", audio]
            # Native codecs avoid a dependency on an optional x264 build.
            command += ["-t", "1", "-c:v", "mpeg4", "-pix_fmt", "yuv420p"]
            if audio:
                command += ["-c:a", "aac", "-shortest"]
            command.append(str(path))
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise unittest.SkipTest("FFmpeg build lacks required fixture filters/codecs")
            cls.fixtures[name] = path

    def check_saved_report(self, directory, report):
        saved = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        self.assertNotIn(str(self.root), json.dumps(saved, ensure_ascii=False))

    def test_non_silent_clip_is_a_candidate_with_real_evidence(self):
        source = self.fixtures["signal"]
        output = self.root / "signal-report"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        report = verify_media(source, output)
        self.check_saved_report(output, report)
        self.assertTrue(report["technical_passed"])
        self.assertTrue(report["automated_passed"])
        self.assertEqual(report["source"]["sha256"], before)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertEqual((report["media"]["width"], report["media"]["height"]), (96, 64))
        self.assertAlmostEqual(report["media"]["fps"], 12)
        self.assertEqual(report["media"]["frame_count"], 12)
        for key in ("video_duration_seconds", "audio_duration_seconds"):
            self.assertAlmostEqual(report["media"][key], 1, delta=0.15)
        self.assertTrue(report["stages"]["audio_signal"]["non_silent"])
        self.assertGreater(report["stages"]["audio_signal"]["non_silent_duration_seconds"], 0.5)
        self.assertEqual(report["human_review"]["status"], "pending")
        self.assertEqual(report["stages"]["asr"]["status"], "skipped")
        sheet = output / report["artifacts"]["contact_sheet"]
        self.assertTrue(sheet.read_bytes().startswith(b"\xff\xd8"))
        cli = subprocess.run([sys.executable, "-B", str(SCRIPT), "--video", str(source),
                              "--output-dir", str(self.root / "cli-report"), "--ffmpeg-timeout", "30"],
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertEqual(json.loads(cli.stdout)["source"]["sha256"], before)

    def test_video_only_is_rejected(self):
        output = self.root / "video-only-report"
        report = verify_media(self.fixtures["video_only"], output)
        self.check_saved_report(output, report)
        self.assertFalse(report["technical_passed"])
        self.assertFalse(report["automated_passed"])
        self.assertEqual(report["media"]["audio_stream_count"], 0)
        self.assertFalse(report["stages"]["audio_signal"]["non_silent"])
        self.assertEqual(report["stages"]["decode"]["status"], "passed")

    def test_valid_media_with_wrong_duration_is_not_accepted(self):
        output = self.root / "wrong-duration-report"
        report = verify_media(self.fixtures["signal"], output, expected_duration_seconds=8)
        self.check_saved_report(output,report)
        self.assertTrue(report["technical_passed"])
        self.assertFalse(report["automated_passed"])
        self.assertEqual(report["stages"]["duration"]["status"],"mismatch")

    def test_silent_audio_is_not_a_pass(self):
        output = self.root / "silent-report"
        report = verify_media(self.fixtures["silent"], output)
        self.check_saved_report(output, report)
        self.assertFalse(report["technical_passed"])
        self.assertFalse(report["stages"]["audio_signal"]["non_silent"])
        self.assertEqual(report["stages"]["audio_signal"]["non_silent_duration_seconds"], 0)
        self.assertEqual(report["stages"]["decode"]["status"], "passed")
        self.assertEqual(report["media"]["audio_stream_count"], 1)

    def test_missing_checkpoint_never_invokes_whisper(self):
        original_run = subprocess.run
        commands = []

        def spy(command, **kwargs):
            commands.append(command)
            return original_run(command, **kwargs)

        with patch.object(media_module.subprocess, "run", side_effect=spy):
            report = verify_media(self.fixtures["signal"], self.root / "missing-model-report",
                                  "你好", self.root / "missing-checkpoint.pt")
        self.assertTrue(report["technical_passed"])
        self.assertFalse(report["automated_passed"])
        self.assertEqual(report["stages"]["asr"]["status"], "unavailable")
        self.assertTrue(all("--model" not in command for command in commands))

    def test_local_asr_compares_words_and_keeps_raw_text(self):
        # Simulate only Whisper, exercising real probe/decode/audio/sheet work.
        # No actual checkpoint/inference is used in this test.
        model = self.root / "explicit-checkpoint.pt"
        model.write_bytes(b"test stub, not a model")
        original_which, original_run = shutil.which, subprocess.run
        whisper_calls = []

        def which(name):
            return "test-whisper" if name == "whisper" else original_which(name)

        def run(command, **kwargs):
            if command[0] != "test-whisper":
                return original_run(command, **kwargs)
            whisper_calls.append(command)
            transcript = Path(command[1]).with_suffix(".json")
            transcript.write_text(json.dumps({"text": recognized, "language": "zh", "segments": []}),
                                  encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        for recognized, match in (("欢迎！我们来吃苹果。", True), ("我们来吃香蕉。", False)):
            with self.subTest(recognized=recognized), patch.object(media_module.shutil, "which", side_effect=which), \
                    patch.object(media_module.subprocess, "run", side_effect=run):
                output = self.root / ("asr-match" if match else "asr-mismatch")
                report = verify_media(self.fixtures["signal"], output, "歡迎，我們來吃蘋果！", model)
                self.check_saved_report(output, report)
                self.assertEqual(report["stages"]["asr"]["match"], match)
                self.assertEqual(report["automated_passed"], match)
                raw = json.loads((output / report["artifacts"]["raw_asr"]).read_text(encoding="utf-8"))
                self.assertEqual(raw["text"], recognized)
                self.assertEqual(report["human_review"]["status"], "pending")
        self.assertEqual(len(whisper_calls), 2)
        for command in whisper_calls:
            for flag, expected in (("--model", str(model.resolve())), ("--device", "cpu"),
                                   ("--fp16", "False"), ("--language", "zh")):
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertNotIn("--initial_prompt", command)
            self.assertNotIn("歡迎，我們來吃蘋果！", command)


if __name__ == "__main__":
    unittest.main()
