"""Configure import paths for archived calibration scripts.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "scenario_sensitivity" not in sys.modules:
    _p = ROOT / "敏感性分析" / "02_scenario_sensitivity.py"
    _spec = importlib.util.spec_from_file_location("scenario_sensitivity", _p)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["scenario_sensitivity"] = _mod
    _spec.loader.exec_module(_mod)
