"""SERUM diffusion teacher entrypoint sharing the audited LoRA training harness."""
import signal
from run import main

if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    import sys
    if '--recover-full-export' in sys.argv[1:]:
        import argparse
        from pathlib import Path
        from recover_full_export import main as recover
        parser = argparse.ArgumentParser(description='Recover final full export without retraining, then finish evaluation')
        parser.add_argument('--recover-full-export', action='store_true', required=True)
        parser.add_argument('--resume', type=Path, required=True)
        recover(parser.parse_args().resume)
    else:
        main(method='serum')
