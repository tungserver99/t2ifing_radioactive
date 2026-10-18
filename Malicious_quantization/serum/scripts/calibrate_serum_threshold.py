"""Compatibility entry point for the frozen, reproducible scorer."""
import sys
from serum_scores import main
if __name__=='__main__':
    sys.argv=[sys.argv[0],'--images','/root/workspace/serum/data/raw/calib_clean',
              '--out','/root/workspace/serum/metrics/serum_calib_clean_scores_v2.npy','--calibrate']
    main()
