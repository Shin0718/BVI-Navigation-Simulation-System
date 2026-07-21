"""Run archived seed holdout checks for calibrated parameters.

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

THETA = {
    "MEMORY_ACTIVE_RETRIEVAL_TH": 0.06,
    "MEMORY_ACTIVE_ABSENT_STEPS_TH": 18,
    "PROBE_RELIEF_RATIO": 0.5,
    "LANDMARK_DECAY_RATE": 0.82,
    "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK": 0.28,
    "LOOMING_BOOST_PEAK": 0.35,
    "LOOMING_BOOST_DECAY": 0.9,
}
NEW_SEEDS = [20270101 + i for i in range(5)]

if __name__ == "__main__":
    obs, w = cs.load_obs(), cs.load_weights()
    tasks = [("holdout", THETA, sd) for sd in NEW_SEEDS]
    res = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=5, initializer=ss._init) as pool:
        for label, m, err in pool.imap_unordered(ss._run, tasks):
            print("FAIL " + err if err else f"seed ok {m['_seed']}", flush=True)
            if not err:
                res.append(m)
    mm = {
        k: sum(float(r[k]) for r in res) / len(res)
        for k in ss.KEYS
        if not any(math.isnan(float(r[k])) for r in res)
    }
    out = {
        "seeds": NEW_SEEDS,
        "n_ok": len(res),
        "L_unified": cs.l_unified(mm, obs, w),
        "L_memoryTH": cs.l_memth(mm, obs),
        "metrics": mm,
    }
    json.dump(out, open(ROOT / "calib_out" / "seed_holdout.json", "w"), indent=2)
    print(f"新种子 L_unified(θ*) = {out['L_unified']:.4f}（校准种子 4.3643）")
    print(f"新种子 L_memoryTH(θ*) = {out['L_memoryTH']:.4f}（校准种子 14.4170）")
