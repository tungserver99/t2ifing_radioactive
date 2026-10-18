"""Reuse the audited native-PEFT trainer; no copies of upstream code."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHARED = ROOT / 'bitmark_lora'
sys.path.insert(0, str(SHARED))
from common import freeze, read_json, write_json, sha256, image_name, runtime_root, results_root
sys.path.insert(1, str(ROOT))
