import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest

from wmq_fixture_utils import verify_sha256, publish_fixture, fixture_lock
from prepare_marked_fixture import fixture_complete
from evaluate_blind_watermark import detection_threshold, verify_frozen_run


def competing_writer(root, start, results, name):
    root = Path(root)
    ready = root / name
    ready.mkdir()
    (ready / "fixture_provenance.json").write_text(name)
    (ready / "model_index.json").write_text(name)
    start.wait()
    try:
        with fixture_lock(root / "model", timeout=10):
            if (root / "model").exists():
                results.put("reuse")
            else:
                publish_fixture(ready, root / "model")
                results.put("published")
    except Exception as error:
        results.put(repr(error))


class IntegrityTests(unittest.TestCase):
    def test_fixture_reuse_requires_all_loadable_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fixture_provenance.json").write_text("{}")
            (root / "model_index.json").write_text("{}")
            self.assertFalse(fixture_complete(root))
            for component, config in (("scheduler", "scheduler_config.json"),
                                      ("text_encoder", "config.json"), ("unet", "config.json"),
                                      ("vae", "config.json")):
                (root / component).mkdir()
                (root / component / config).write_text("{}")
            (root / "tokenizer").mkdir()
            (root / "tokenizer" / "tokenizer_config.json").write_text("{}")
            for component in ("text_encoder", "unet", "vae"):
                (root / component / "weights.safetensors").write_bytes(b"test")
            self.assertTrue(fixture_complete(root))

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "checkpoint"
            path.write_bytes(b"expected")
            expected = hashlib.sha256(b"expected").hexdigest()
            self.assertEqual(verify_sha256(path, expected), expected)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_sha256(path, expected)

    def test_publish_never_overwrites_existing_empty_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            ready, out = root / "ready", root / "out"
            ready.mkdir()
            (ready / "fixture_provenance.json").write_text("{}")
            out.mkdir()
            with self.assertRaises(FileExistsError):
                publish_fixture(ready, out)
            self.assertEqual(list(out.iterdir()), [])

    def test_simultaneous_preparation_serializes_and_reuses(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = mp.get_context("spawn")
            start, results = ctx.Event(), ctx.Queue()
            workers = [ctx.Process(target=competing_writer, args=(root, start, results, f"ready{i}")) for i in range(2)]
            try:
                for p in workers:
                    p.start()
                start.set()
                outputs = [results.get(timeout=45) for _ in workers]
                for p in workers:
                    p.join(timeout=10)
                    self.assertEqual(p.exitcode, 0)
                self.assertEqual(sorted(outputs), ["published", "reuse"])
                out = Path(root) / "model"
                self.assertEqual((out / "model_index.json").read_text(), (out / "fixture_provenance.json").read_text())
            finally:
                for p in workers:
                    if p.is_alive():
                        p.terminate()
                        p.join()

    def test_threshold_boundaries(self):
        self.assertEqual(detection_threshold(48, .001), 36)
        self.assertEqual(detection_threshold(4, 1 / 16, "single"), 4)
        self.assertEqual(detection_threshold(4, 2 / 16), 4)
        self.assertEqual(detection_threshold(48, .0001), 38)
        with self.assertRaisesRegex(ValueError, "Unattainable"):
            detection_threshold(4, 1 / 16)
        with self.assertRaisesRegex(ValueError, "Unattainable"):
            detection_threshold(4, .001)
        for bits, fpr in [(0, .1), (48, 0), (48, float("nan"))]:
            with self.assertRaises(ValueError):
                detection_threshold(bits, fpr)

    def test_frozen_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "quantized_vae").mkdir()
            artifact = root / "quantized_vae" / "weights.bin"
            artifact.write_bytes(b"weights")
            (root / "search.json").write_bytes(b"[]")
            hashes = {"weights.bin": hashlib.sha256(b"weights").hexdigest()}
            frozen = {"phase": "before_test_generation", "selected": {"candidate": "a"},
                      "export_sha256": hashes, "search_sha256": hashlib.sha256(b"[]").hexdigest()}
            (root / "selection_frozen.json").write_text(json.dumps(frozen))
            report = {"selected": frozen["selected"], "export_sha256": hashes}
            verify_frozen_run(root, report)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after"):
                verify_frozen_run(root, report)


if __name__ == "__main__":
    unittest.main()
