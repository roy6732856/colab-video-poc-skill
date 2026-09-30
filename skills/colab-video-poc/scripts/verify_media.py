#!/usr/bin/env python3
"""Local, stdlib-only media checks. FFmpeg tools must already be on PATH.

Technical success produces a candidate, never human acceptance. Optional ASR
uses an installed OpenAI Whisper CLI and an explicitly supplied checkpoint file;
model names, downloads, prompts containing the expected answer, and GPU use are
not part of this module. Models and third-party code are not bundled.
"""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave


# A deliberately limited character equivalence table, not linguistic conversion.
# Ambiguous equivalences (for example 發/髮) can hide differences in meaning;
# normalized equality is word evidence only and still needs human review.
_COMMON_VARIANTS = str.maketrans({
    "這": "这", "個": "个", "們": "们", "來": "来", "說": "说",
    "話": "话", "為": "为", "麼": "么", "嗎": "吗", "時": "时",
    "間": "间", "會": "会", "開": "开", "關": "关", "無": "无",
    "與": "与", "還": "还", "讓": "让", "從": "从", "對": "对",
    "實": "实", "現": "现", "進": "进", "過": "过", "後": "后",
    "裡": "里", "裏": "里", "臺": "台", "灣": "湾", "學": "学",
    "習": "习", "愛": "爱", "買": "买", "賣": "卖", "蘋": "苹",
    "檔": "档", "電": "电", "腦": "脑", "網": "网", "頁": "页",
    "視": "视", "頻": "频", "聲": "声", "聽": "听", "讀": "读",
    "寫": "写", "驗": "验", "證": "证", "測": "测", "試": "试",
    "轉": "转", "換": "换", "產": "产", "業": "业", "務": "务",
    "動": "动", "應": "应", "該": "该", "經": "经", "營": "营",
    "銷": "销", "錢": "钱", "點": "点", "數": "数", "據": "据",
    "優": "优", "質": "质", "體": "体", "長": "长", "種": "种",
    "選": "选", "擇": "择", "機": "机", "線": "线", "簡": "简",
    "單": "单", "雙": "双", "條": "条", "張": "张", "發": "发",
    "髮": "发", "準": "准", "備": "备", "結": "结", "夠": "够",
    "歡": "欢", "樂": "乐", "謝": "谢", "請": "请", "問": "问",
    "見": "见", "別": "别", "總": "总", "達": "达", "幫": "帮",
    "團": "团", "隊": "队", "記": "记", "錄": "录", "創": "创",
})
_TECHNICAL_STAGES = ("probe", "decode", "audio_signal", "contact_sheet")
_ARTIFACT_NAMES = ("verification.json", "contact-sheet.jpg", "raw-asr.json")


def _normalize_dialogue(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_COMMON_VARIANTS).casefold()
    return "".join(c for c in text if not c.isspace()
                   and not unicodedata.category(c).startswith("P"))


def _number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def _ratio(value) -> float | None:
    try:
        return _number(float(Fraction(str(value))))
    except (ValueError, ZeroDivisionError):
        return None


def _count(value) -> int | None:
    try:
        result = int(value)
        return result if result >= 0 else None
    except (ValueError, TypeError):
        return None


class _StageError(Exception):
    def __init__(self, reason: str, *, status: str = "failed", **details):
        self.stage = {"status": status, "reason": reason, **details}
        super().__init__(reason)


def _run(command: list[str], timeout: float) -> str:
    """Never use a shell or persist tool output containing local paths."""
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise _StageError("tool timed out", timeout_seconds=timeout) from None
    except OSError as exc:
        raise _StageError("tool could not be executed", status="unavailable",
                          error_type=type(exc).__name__) from None
    if result.returncode:
        raise _StageError("tool exited unsuccessfully", returncode=result.returncode)
    return result.stdout


def _write_json(path: Path, data: dict) -> None:
    # Replace atomically, including when the destination is a stale symlink.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".verification-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _probe(tool: str, video: Path, timeout: float) -> dict:
    raw = _run([tool, "-v", "error", "-count_frames", "-show_entries",
                "stream=index,codec_type,codec_name,width,height,avg_frame_rate,"
                "r_frame_rate,duration,nb_frames,nb_read_frames:"
                "stream_disposition=attached_pic:format=duration",
                "-of", "json", str(video)], timeout)
    try:
        data = json.loads(raw)
        streams = data["streams"]
        videos = [s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")]
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        container_duration = _number(data.get("format", {}).get("duration"))
        v = videos[0] if videos else {}
        a = audios[0] if audios else {}
        fps = _ratio(v.get("avg_frame_rate")) or _ratio(v.get("r_frame_rate"))
        counted = _count(v.get("nb_read_frames"))
        declared = _count(v.get("nb_frames"))
        frame_count = counted if counted is not None else declared
        duration = _number(v.get("duration"))
        duration_source = "stream" if duration is not None else "unavailable"
        if duration is None and fps and frame_count:
            duration, duration_source = frame_count / fps, "frame_count/average_fps"
        if duration is None and container_duration:
            duration, duration_source = container_duration, "container_fallback"
        media = {
            "width": _count(v.get("width")), "height": _count(v.get("height")),
            "fps": fps, "frame_count": frame_count,
            "frame_count_source": "ffprobe_count_frames" if counted is not None else "container",
            "declared_frame_count": declared,
            "video_duration_seconds": duration, "video_duration_source": duration_source,
            "audio_duration_seconds": _number(a.get("duration")),
            "audio_duration_source": "stream" if _number(a.get("duration")) is not None else "unavailable",
            "container_duration_seconds": container_duration,
            "video_stream_count": len(videos), "audio_stream_count": len(audios),
            "video_stream_index": v.get("index"), "audio_stream_index": a.get("index"),
        }
    except (ValueError, KeyError, TypeError, AttributeError):
        raise _StageError("ffprobe returned invalid metadata") from None
    failures = []
    if not videos:
        failures.append("no video stream")
    elif not all((media["width"], media["height"], fps, frame_count, duration)):
        failures.append("video dimensions, fps, frame count or duration unavailable/empty")
    if not audios:
        failures.append("no audio stream")
    return {"status": "failed" if failures else "passed", "issues": failures, "media": media}


def _audio_signal(wav_path: Path) -> dict:
    """Check 20 ms RMS windows at -60 dBFS, keeping channels separate.

    Detects audible signal, including music/noise; it does not establish speech,
    dialogue, loudness quality or a human-perceived acceptable level.
    """
    samples_total = energy_total = peak = active_frames = frames_total = 0
    with wave.open(str(wav_path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise _StageError("expected uncompressed signed 16-bit PCM WAV")
        rate, channels = audio.getframerate(), audio.getnchannels()
        window = max(1, round(rate * 0.02))
        threshold = 32768 * 10 ** (-60 / 20)
        while True:
            raw = audio.readframes(window)
            if not raw:
                break
            samples = array("h")
            samples.frombytes(raw)
            if sys.byteorder != "little":
                samples.byteswap()
            energy = sum(value * value for value in samples)
            frames = len(samples) // channels
            frames_total += frames
            samples_total += len(samples)
            energy_total += energy
            peak = max(peak, max(abs(value) for value in samples))
            if math.sqrt(energy / len(samples)) >= threshold:
                active_frames += frames
        rms = math.sqrt(energy_total / samples_total) if samples_total else 0
    non_silent = active_frames > 0
    return {
        "status": "passed" if non_silent else "failed", "non_silent": non_silent,
        "sample_rate": rate, "channels": channels, "duration_seconds": frames_total / rate,
        "rms_dbfs": 20 * math.log10(rms / 32768) if rms else None,
        "peak_dbfs": 20 * math.log10(peak / 32768) if peak else None,
        "threshold_dbfs": -60, "window_seconds": window / rate,
        "non_silent_duration_seconds": active_frames / rate,
        "scope": "signal in first audio stream; does not establish speech or voice quality",
    }


def _asr(tool: str | None, model: Path | None, expected: str | None,
         wav_path: Path, work: Path, output_dir: Path, timeout: float) -> dict:
    if expected is None and model is None:
        return {"status": "skipped", "reason": "ASR not requested"}
    if expected is None or model is None:
        return {"status": "unavailable", "reason": "ASR requires both dialogue and a local model file"}
    normalized_expected = _normalize_dialogue(expected)
    if not normalized_expected:
        return {"status": "unavailable", "reason": "expected dialogue has no comparable characters"}
    model = Path(model).expanduser().resolve()
    if not model.is_file():
        return {"status": "unavailable", "reason": "explicit local model file does not exist"}
    if not tool:
        return {"status": "unavailable", "reason": "Whisper CLI not found on PATH"}
    if not wav_path.is_file():
        return {"status": "unavailable", "reason": "decoded audio unavailable"}
    # An absolute, existing file can never be interpreted as a downloadable model
    # name by OpenAI Whisper. Do not feed the expected answer as an initial prompt.
    _run([tool, str(wav_path), "--model", str(model), "--device", "cpu",
          "--fp16", "False", "--language", "zh", "--task", "transcribe",
          "--output_format", "json", "--output_dir", str(work), "--verbose", "False"], timeout)
    try:
        raw = json.loads(wav_path.with_suffix(".json").read_text(encoding="utf-8"))
        text = raw["text"]
        if not isinstance(text, str):
            raise ValueError("non-text ASR result")
    except (OSError, ValueError, KeyError, TypeError):
        raise _StageError("Whisper did not produce valid transcript JSON") from None
    # Preserve only transcript fields; CLI internals or tool paths are not public artifacts.
    _write_json(output_dir / "raw-asr.json", {"text": text, "language": raw.get("language"),
                                            "segments": raw.get("segments", [])})
    normalized_actual = _normalize_dialogue(text)
    match = bool(normalized_actual) and normalized_actual == normalized_expected
    return {"status": "passed" if match else "mismatch", "match": match,
            "raw_transcript": "raw-asr.json", "normalized_expected": normalized_expected,
            "normalized_actual": normalized_actual, "comparison": "exact normalized equality",
            "normalization": "NFKC, whitespace/punctuation removal, casefold, limited Chinese variants",
            "device": "cpu", "fp16": False, "language": "zh", "initial_prompt_used": False,
            "scope": "recognized words only; no voice, accent, emotion or lip-sync acceptance"}


def _duration_requirement(media: dict, requested: float | None) -> dict:
    if requested is None:
        return {"status": "skipped", "reason": "expected H3 duration not supplied"}
    requested = float(requested)
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("expected_duration_seconds must be finite and positive")
    requested_frames = max(5, round(requested * 24))
    aligned_frames = requested_frames + (5 - requested_frames % 17) % 17
    expected = aligned_frames / 24
    observed = media.get("video_duration_seconds")
    tolerance = 2 / 24
    return {"status": "unavailable" if observed is None else "passed" if abs(observed - expected) <= tolerance else "mismatch",
            "requested_seconds": requested, "expected_aligned_frames": aligned_frames,
            "expected_aligned_seconds": expected, "observed_seconds": observed,
            "tolerance_seconds": tolerance, "scope": "H3 24 fps / 17k+5 alignment, with two-frame timestamp tolerance"}


def verify_media(video: Path, output_dir: Path, expected_dialogue: str | None = None,
                 whisper_model: Path | None = None, *, ffmpeg_timeout: float = 180,
                 expected_duration_seconds: float | None = None) -> dict:
    """Write verification.json and a contact sheet, returning the same report.

    Bad source/output paths raise OSError/ValueError. Media failures, unavailable
    tools and optional ASR failures are recorded in the report. Each external
    command (including optional Whisper) has its own timeout in seconds. Outputs
    use basenames rather than machine-specific paths. Reuse an output directory
    only when replacing these named artifacts is intended; the report names only
    artifacts successfully created during this run.
    """
    if not math.isfinite(ffmpeg_timeout) or ffmpeg_timeout <= 0:
        raise ValueError("ffmpeg_timeout must be a finite positive number")
    _duration_requirement({}, expected_duration_seconds)  # Validate before writing artifacts.
    video, output_dir = Path(video).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError("video must be an existing regular file")
    for name in _ARTIFACT_NAMES:
        destination = output_dir / name
        if destination.resolve() == video or (destination.exists() and destination.samefile(video)):
            raise ValueError("verification output would overwrite the source video")
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with video.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    tools = {name: shutil.which(name) for name in ("ffprobe", "ffmpeg", "whisper")}
    stages = {name: {"status": "skipped", "reason": "required metadata/tool unavailable"}
              for name in _TECHNICAL_STAGES}
    artifacts = {"verification": "verification.json"}
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"name": video.name, "sha256": digest.hexdigest(), "bytes": video.stat().st_size},
        "tools": {name: bool(path) for name, path in tools.items()}, "stages": stages,
        "media": {}, "artifacts": artifacts,
        "human_review": {"status": "pending", "scopes": ["voice_identity", "accent", "emotion",
                         "naturalness", "lip_sync", "visual_quality", "dialogue_meaning"]},
    }
    if tools["ffprobe"]:
        try:
            stages["probe"] = _probe(tools["ffprobe"], video, ffmpeg_timeout)
            report["media"] = stages["probe"].pop("media")
        except _StageError as exc:
            stages["probe"] = exc.stage
    else:
        stages["probe"] = {"status": "unavailable", "reason": "ffprobe not found on PATH"}
    media = report["media"]
    with tempfile.TemporaryDirectory(prefix="verify-media-") as temporary:
        work = Path(temporary)
        wav_path = work / "audio.wav"
        ffmpeg = tools["ffmpeg"]
        base = [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-xerror", "-y"] if ffmpeg else []
        if not ffmpeg:
            for name in _TECHNICAL_STAGES[1:]:
                stages[name] = {"status": "unavailable", "reason": "ffmpeg not found on PATH"}
        elif media.get("video_stream_count"):
            try:
                _run(base + ["-err_detect", "explode", "-i", str(video), "-map", "0:v",
                             "-map", "0:a?", "-f", "null", "-"], ffmpeg_timeout)
                stages["decode"] = {"status": "passed", "scope": "complete video and audio decode",
                                     "video_streams": media["video_stream_count"],
                                     "audio_streams": media["audio_stream_count"]}
            except _StageError as exc:
                stages["decode"] = exc.stage
            if media.get("audio_stream_count"):
                try:
                    _run(base + ["-i", str(video), "-map", f"0:{media['audio_stream_index']}",
                                 "-vn", "-c:a", "pcm_s16le", "-ar", "16000", str(wav_path)], ffmpeg_timeout)
                    stages["audio_signal"] = _audio_signal(wav_path)
                    media["audio_duration_seconds"] = stages["audio_signal"]["duration_seconds"]
                    media["audio_duration_source"] = "decoded_pcm"
                except _StageError as exc:
                    stages["audio_signal"] = exc.stage
                    wav_path.unlink(missing_ok=True)
                except (OSError, ValueError, EOFError, wave.Error):
                    stages["audio_signal"] = {"status": "failed", "reason": "decoded PCM WAV unreadable"}
                    wav_path.unlink(missing_ok=True)
            else:
                stages["audio_signal"] = {"status": "failed", "non_silent": False,
                                           "reason": "no audio stream"}
            duration = media.get("video_duration_seconds")
            if duration and duration > 0:
                sheet = work / "contact-sheet.jpg"
                interval = duration / 12
                filters = (f"setpts=PTS-STARTPTS,fps=fps={12 / duration:.12g}:start_time=0:round=up,"
                           "scale=320:180:force_original_aspect_ratio=decrease,"
                           "pad=320:180:(ow-iw)/2:(oh-ih)/2,"
                           "tile=4x3:nb_frames=12:padding=4:margin=4")
                try:
                    _run(base + ["-i", str(video), "-map", f"0:{media['video_stream_index']}",
                                 "-an", "-vf", filters, "-frames:v", "1", "-q:v", "3",
                                 "-update", "1", str(sheet)], ffmpeg_timeout)
                    if not sheet.is_file() or not sheet.stat().st_size:
                        raise _StageError("ffmpeg produced no contact sheet")
                    # TemporaryDirectory can be on a different filesystem.
                    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
                        staged_sheet = Path(handle.name)
                    try:
                        shutil.copyfile(sheet, staged_sheet)
                        staged_sheet.replace(output_dir / "contact-sheet.jpg")
                    finally:
                        staged_sheet.unlink(missing_ok=True)
                    artifacts["contact_sheet"] = "contact-sheet.jpg"
                    stages["contact_sheet"] = {"status": "passed", "layout": "4x3",
                                                "sample_slots": 12, "interval_seconds": interval,
                                                "scope": "frame sampling artifact; visual review pending"}
                except _StageError as exc:
                    stages["contact_sheet"] = exc.stage
        try:
            stages["asr"] = _asr(tools["whisper"], whisper_model, expected_dialogue,
                                 wav_path, work, output_dir, ffmpeg_timeout)
            if "raw_transcript" in stages["asr"]:
                artifacts["raw_asr"] = stages["asr"]["raw_transcript"]
        except _StageError as exc:
            stages["asr"] = exc.stage
    report["technical_passed"] = all(stages[name]["status"] == "passed" for name in _TECHNICAL_STAGES)
    stages["duration"] = _duration_requirement(media, expected_duration_seconds)
    report["automated_passed"] = report["technical_passed"] and stages["asr"]["status"] in ("passed", "skipped") and stages["duration"]["status"] in ("passed", "skipped")
    report["status"] = ("candidate" if report["automated_passed"] else "failed"
                        if any(stage["status"] in ("failed", "mismatch") for stage in stages.values())
                        else "incomplete")
    _write_json(output_dir / "verification.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-dialogue")
    parser.add_argument("--expected-duration", type=float, help="requested H3 duration; check native frame alignment")
    parser.add_argument("--whisper-model", type=Path, help="existing local checkpoint FILE; no model names/downloads")
    parser.add_argument("--ffmpeg-timeout", type=float, default=180, help="timeout per external command in seconds")
    args = parser.parse_args(argv)
    try:
        report = verify_media(args.video, args.output_dir, args.expected_dialogue, args.whisper_model,
                              ffmpeg_timeout=args.ffmpeg_timeout, expected_duration_seconds=args.expected_duration)
    except (OSError, ValueError) as exc:
        print(f"verification could not run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["automated_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
