"""Verify probe behavior under the calibrated threshold setting.

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
    theta03 = {**ts["theta_star"], "PROBE_RELIEF_RATIO": 0.3}
    obs, weights = cs2.load_obs(), cs2.load_weights()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=5, initializer=cs2._init) as pool:
        res = cs2.eval_candidates(pool, [("verify|probe0.3", theta03)], cs2.SEEDS)
    m = res["verify|probe0.3"]
    print(
        f"theta*(PROBE=0.3): L_unified={cs2.l_unified(m, obs, weights):.4f} "
        f"L_memoryTH={cs2.l_memth(m, obs):.4f}"
    )
    print(
        f"theta*(PROBE=0.2): L_unified={ts['L_unified_star']:.4f} "
        f"L_memoryTH={ts['L_memoryTH_star']:.4f}"
    )


if __name__ == "__main__":
    main()
