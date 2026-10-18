#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    scores = np.load(args.scores).reshape(-1)

    detected = scores >= args.threshold
    tpr = detected.mean()

    rng = np.random.default_rng(args.seed)
    boots = np.empty(args.bootstrap, dtype=np.float64)

    n = len(scores)
    for b in range(args.bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = (scores[idx] >= args.threshold).mean()

    lo, hi = np.quantile(boots, [0.025, 0.975])

    result = {
        "n": int(n),
        "threshold": float(args.threshold),
        "tpr": float(tpr),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }

    text = json.dumps(result, indent=2)
    print(text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
