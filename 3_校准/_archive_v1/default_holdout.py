"""Run archived default holdout checks for the calibration workflow.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import _setup
import scenario_sensitivity as ss
import calibrate_search as cs

NEW_SEEDS = [20270101 + i for i in range(5)]

if __name__ == "__main__":
    obs, w = cs.load_obs(), cs.load_weights()
    tasks = [("default_holdout", {}, sd) for sd in NEW_SEEDS]
    res = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=5, initializer=ss._init) as pool:
        for label, m, err in pool.imap_unordered(ss._run, tasks):
            print("FAIL " + err if err else f"ok {m['_seed']}", flush=True)
            if not err:
                res.append(m)
    mm = {
        k: sum(float(r[k]) for r in res) / len(res)
        for k in ss.KEYS
        if not any(math.isnan(float(r[k])) for r in res)
    }
    out = {
        "L_unified": cs.l_unified(mm, obs, w),
        "L_memoryTH": cs.l_memth(mm, obs),
        "metrics": mm,
    }
    json.dump(out, open(ROOT / "calib_out" / "default_holdout.json", "w"), indent=2)
    print(
        f"默认参数@新种子: L_unified={out['L_unified']:.4f}, L_memoryTH={out['L_memoryTH']:.4f}"
    )
