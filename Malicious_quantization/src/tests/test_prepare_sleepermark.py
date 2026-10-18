import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_sleepermark import prepare, digest, PREFLIGHT_ENV


class SleeperPreparationTests(unittest.TestCase):
    def fixture(self, root):
        unet = root/'unet_download'/'checkpoint'
        owner = root/'owner'/'weights'
        unet.mkdir(parents=True); owner.mkdir(parents=True)
        (unet/'config.json').write_text(json.dumps({'_class_name': 'UNet2DConditionModel'}))
        (owner/'decoder.pth').write_bytes(b'decoder')
        (owner/'secret.pt').write_bytes(b'secret')
        (root/'watermarkModel.py').write_text('source')
        marker = root/'provenance.json'
        marker.write_text(json.dumps({'sha256': {}}))
        return marker

    def test_verified_parent_token_skips_lock_and_resolves_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = self.fixture(root)
            (root/'.preparing').mkdir()  # A child must not touch the parent's lock.
            with patch.dict(os.environ, {PREFLIGHT_ENV: digest(marker)}):
                unet, decoder, secret, source = prepare(root)
            self.assertEqual(unet.name, 'checkpoint')
            self.assertEqual((decoder.name, secret.name, source.name),
                             ('decoder.pth', 'secret.pt', 'watermarkModel.py'))

    def test_wrong_parent_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            with patch.dict(os.environ, {PREFLIGHT_ENV: '0'*64}):
                with self.assertRaisesRegex(ValueError, 'token mismatch'):
                    prepare(root)


if __name__ == '__main__':
    unittest.main()
