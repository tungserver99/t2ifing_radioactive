"""Single entry point for the native semantic fingerprint experiment."""
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent / 'Malicious_quantization' / 'bitmark_lora'
sys.path.insert(0, str(HARNESS))

if __name__ == '__main__':
    from run import main
    main(method='semantic_fp')
