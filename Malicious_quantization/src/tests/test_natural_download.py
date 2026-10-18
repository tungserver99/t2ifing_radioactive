import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
import prepare_natural_images as data


class DownloadTests(unittest.TestCase):
    def test_verified_cache_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "image_info_test2017.zip"
            with zipfile.ZipFile(index, "w") as z:
                z.writestr("annotations/image_info_test2017.json", json.dumps({
                    "images": [{"id": i, "license": 1} for i in range(4)], "licenses": []}))
            expected = hashlib.md5(index.read_bytes()).hexdigest()
            calls = []
            def fake_download(url, path):
                calls.append(url)
                color = int(Path(path).stem) * 50
                Image.new("RGB", (32, 32), (color, 30, 80)).save(path)
            with patch.object(data, "INDEX_MD5", expected), patch.object(data, "download", fake_download):
                data.prepare(root / "images", 2, 3407)
                data.prepare(root / "images", 2, 3407)
                self.assertEqual(len(calls), 2)
                # Simulate interruption before the final pool marker was committed.
                marker = root / "images" / "dataset_provenance.json"
                original = json.loads(marker.read_text())
                marker.unlink()
                data.prepare(root / "images", 2, 3407, workers=2)
                self.assertEqual(len(calls), 2)
                self.assertEqual(json.loads(marker.read_text())["images"], original["images"])
                marker.unlink()
                (next((root / "images").glob("*.jpg"))).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    data.prepare(root / "images", 2, 3407)

    def test_index_mismatch_prevents_image_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "image_info_test2017.zip").write_bytes(b"wrong archive")
            with patch.object(data, "download") as downloader:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    data.prepare(root / "images", 2, 3407)
                downloader.assert_not_called()

    def test_transient_worker_failure_does_not_abort_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "image_info_test2017.zip"
            with zipfile.ZipFile(index, "w") as z:
                z.writestr("annotations/image_info_test2017.json", json.dumps({
                    "images": [{"id": i, "license": 1} for i in range(5)], "licenses": []}))
            expected = hashlib.md5(index.read_bytes()).hexdigest()
            attempts = {}
            def flaky_download(url, path):
                name = Path(path).name
                attempts[name] = attempts.get(name, 0) + 1
                if sum(attempts.values()) == 1:
                    raise TimeoutError("transient")
                Image.new("RGB", (32, 32), (int(Path(path).stem) * 30, 20, 70)).save(path)
            with patch.object(data, "INDEX_MD5", expected), patch.object(data, "download", flaky_download):
                data.prepare(root / "images", 3, 3407, workers=2)
            provenance = json.loads((root / "images" / "dataset_provenance.json").read_text())
            self.assertEqual(len(provenance["images"]), 3)
            self.assertEqual(sum(attempts.values()), 4)


if __name__ == "__main__":
    unittest.main()
