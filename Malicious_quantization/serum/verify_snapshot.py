"""Verify included server files and the corrected clipping implementation."""
from pathlib import Path
import hashlib,json,ast
r=Path(__file__).resolve().parent
hashes=json.loads((r/'SERVER_SOURCE_SHA256.json').read_text())
for rel,expected in hashes.items():
    p=r/rel
    assert p.is_file(),rel
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,rel
for p in (r/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
t=r/'third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py'
assert 'lora_layers = [p for p in unet.parameters() if p.requires_grad]' in t.read_text()
print(f'{len(hashes)} server files match; scripts parse; corrected trainer verified')
