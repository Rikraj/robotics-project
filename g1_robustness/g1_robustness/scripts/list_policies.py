#!/usr/bin/env python3
"""Print 'name=/path/to/model_N.pt' for the final checkpoint of every trained run (baseline_s1, dr_s1, ...)."""

import argparse
import glob
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="e.g. $ISAACLAB/logs/rsl_rl")
    args = ap.parse_args()
    out = []
    for exp in ("g1_flat_baseline", "g1_flat_dr"):
        for run in sorted(glob.glob(os.path.join(args.logs, exp, "*"))):
            m = re.search(r"_(baseline|dr)_s(\d+)$", run)
            if not m:
                continue
            ckpts = glob.glob(os.path.join(run, "model_*.pt"))
            if not ckpts:
                continue
            last = max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))
            out.append(f"{m.group(1)}_s{m.group(2)}={last}")
    if not out:
        sys.exit("no checkpoints found under " + args.logs)
    print("\n".join(out))


if __name__ == "__main__":
    main()
