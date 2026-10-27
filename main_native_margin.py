"""Separate entrypoint for the locked RSSF v3 campaign."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'Malicious_quantization'/'bitmark_lora'))
if __name__=='__main__':
    from semantic_native_campaign import main
    main()
