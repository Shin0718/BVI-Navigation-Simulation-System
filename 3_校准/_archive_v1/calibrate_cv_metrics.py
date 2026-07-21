"""Prepare archived cross-validation metrics for calibration checks.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import _setup
import calibrate_search as cs

HOLDOUT = {"veh_reaction_delay_s", "noref_ep_len", "tac_stop_rate"}

OUT_CV = ROOT / "calib_out_cv"
OUT_CV.mkdir(exist_ok=True)

_orig_load_obs = cs.load_obs


def load_obs_train():
    """Handle load obs train behavior."""
    return {k: v for k, v in _orig_load_obs().items() if k not in HOLDOUT}


cs.load_obs = load_obs_train
cs.OUT = OUT_CV
cs._raw_path = OUT_CV / "calib_runs_raw.csv"
cs._raw_header_written = cs._raw_path.exists()

if __name__ == "__main__":
    print(
        f"指标级留出校准：留出 {sorted(HOLDOUT)}，训练指标 {len(load_obs_train())} 项",
        flush=True,
    )
    cs.main()
