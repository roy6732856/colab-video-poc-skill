"""Small synthetic weights verify the public helper without HF/network."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/colab-video-poc/scripts"))
import model_cache


class PublicModelCacheTests(unittest.TestCase):
    def test_warm_staging_hashes_once_and_never_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache, stage = root / "cache", root / "stage"
            cache.mkdir()
            data = {p: ("synthetic " + str(i)).encode() for i, p in enumerate(model_cache.EXPECTED_FILES)}
            manifest = {"repo_id": model_cache.REPO_ID, "revision": "a" * 40,
                        "models": [{"path": p, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in data.items()]}
            for p, b in data.items():
                (cache / p).parent.mkdir(parents=True, exist_ok=True)
                (cache / p).write_bytes(b)
            with patch.object(model_cache, "EXPECTED_FILES", {p: len(b) for p, b in data.items()}), patch.object(
                    model_cache, "_hf_download", side_effect=AssertionError("No HF calls allowed")):
                report = model_cache.prepare_models(manifest, cache, stage)
            self.assertEqual(report["counters"]["cache_hits"], 5)
            self.assertEqual(report["counters"]["downloads"], 0)
            self.assertEqual(report["counters"]["hashed_bytes"], sum(map(len, data.values())))
            self.assertEqual(report["counters"]["staged_files"], 5)
            for p, b in data.items():
                self.assertEqual((stage / p).read_bytes(), b)

    def test_public_profile_pinned_and_exact_sizes(self):
        source = Path(model_cache.__file__).parent.parent / "assets/cache-manifest.json"
        manifest = model_cache.validate_manifest(json.loads(source.read_text()))
        self.assertEqual(manifest.revision, "e5eb578a89295337b8ff433a035929ce0279e0b6")
        self.assertEqual(sum(f.bytes for f in manifest.files), 44_414_604_463)


if __name__ == "__main__":
    unittest.main()
