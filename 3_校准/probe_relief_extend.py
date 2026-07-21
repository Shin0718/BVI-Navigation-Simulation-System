"""Probe relief parameter ranges around the calibrated solution.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "3_校准"))
import calibrate_search_v2 as cs2


def main():
    """Run the script entry point."""
    ts = json.load(open(ROOT / "calib_out_v2/theta_star.json", encoding="utf-8"))
    theta = dict(ts["theta_star"])
    obs, weights = cs2.load_obs(), cs2.load_weights()
    cands = [
        (f"extend|PROBE_RELIEF|{v}", {**theta, "PROBE_RELIEF_RATIO": v})
        for v in (0.05, 0.10, 0.15)
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=10, initializer=cs2._init) as pool:
        res = cs2.eval_candidates(pool, cands, cs2.SEEDS)
    print("θ* 背景下 PROBE_RELIEF 低端补扫：")
    for v in (0.05, 0.10, 0.15):
        m = res[f"extend|PROBE_RELIEF|{v}"]
        print(f"  PROBE_RELIEF={v}: L_unified={cs2.l_unified(m, obs, weights):.4f}")
    print(f"  对照 0.2/0.3: L_unified={ts['L_unified_star']:.4f}")


if __name__ == "__main__":
    main()
