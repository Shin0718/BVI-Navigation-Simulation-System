"""Scan safety-weight values and summarize navigation outcomes.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import contextlib

import csv

import io

import math

import multiprocessing as mp

import sys

import time

from pathlib import Path


ROOT = Path(__file__).resolve().parent

if not (ROOT / "bvi_sa").exists():

    ROOT = ROOT.parent

OUT_DIR = ROOT / "sensitivity_out"

SCAN_VALUES = [0.1, 0.4, 0.7, 1.0, 1.3]

SEEDS = [20260705 + i for i in range(5)]


import importlib.util

spec = importlib.util.spec_from_file_location(
    "sm", str(ROOT / "1_敏感性分析" / "01_sensitivity_mechanism.py")
)

sm = importlib.util.module_from_spec(spec)

spec.loader.exec_module(sm)


KEYS = [
    "iw_mean",
    "iw_high_ratio",
    "stop_probe_per_100steps",
    "veh_reaction_prob",
    "veh_reaction_delay_s",
    "crossing_wait_s",
    "move_direct_rate",
    "landmark_relief_effect",
    "deprivation_mean",
    "total_sim_time_s",
    "total_steps",
    "mech_net_priority_mean",
    "mech_gate_rate",
    "mech_risk_mean",
]


_SIM = None


def _init():
    """Initialize worker-process imports and paths."""
    global _SIM

    with contextlib.redirect_stdout(io.StringIO()):

        sys.path.insert(0, str(ROOT))

        from bvi_sa import simulation as s

    s.generate_report = sm._extract

    _SIM = s


def _run(task):
    """Execute one simulation task and return metrics."""
    val, seed = task

    import random

    s = _SIM

    try:

        for n, d in sm.DEFAULTS.items():

            setattr(s, n, d)

        setattr(s, "SEEV_VALUE_SAFETY_WEIGHT", val)

        random.seed(seed)

        import numpy as np

        np.random.seed(seed % (2**32 - 1))

        with contextlib.redirect_stdout(io.StringIO()):

            m = s.run_simulation(familiarity_level=1)

        return val, seed, m, None

    except Exception as e:

        return val, seed, None, f"{type(e).__name__}: {e}"


def main():
    """Run the script entry point."""
    tasks = [(v, sd) for v in SCAN_VALUES for sd in SEEDS]

    print(f"扫描 {len(SCAN_VALUES)} 点 × {len(SEEDS)} 种子 = {len(tasks)} 次")

    t0 = time.time()

    rows = {}

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=10, initializer=_init) as pool:

        done = 0

        for val, seed, m, err in pool.imap_unordered(_run, tasks):

            done += 1

            if err:

                print(f"[{done}] SAFETY={val} seed={seed} FAILED: {err}")

                continue

            rows.setdefault(val, []).append(m)

            print(
                f"[{done}/{len(tasks)}] SAFETY={val} seed={seed} ok steps={m['total_steps']:.0f}"
            )

    print(f"完成 {(time.time()-t0)/60:.1f} 分钟")

    def mean(rs, k):
        """Handle mean behavior."""
        vs = [r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]

        return sum(vs) / len(vs) if vs else float("nan")

    out_csv = OUT_DIR / "safety_weight_sweep.csv"

    table = []

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:

        w = csv.writer(f)

        w.writerow(["SEEV_VALUE_SAFETY_WEIGHT"] + KEYS)

        for v in SCAN_VALUES:

            vals = [mean(rows.get(v, []), k) for k in KEYS]

            table.append((v, vals))

            w.writerow([v] + [f"{x:.4f}" for x in vals])

    print("CSV:", out_csv)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    for fn in ["PingFang SC", "Hiragino Sans GB", "SimHei", "Arial Unicode MS"]:

        if any(fn.lower() in t.name.lower() for t in font_manager.fontManager.ttflist):

            plt.rcParams["font.family"] = fn

            break

    plt.rcParams["axes.unicode_minus"] = False

    xs = [v for v, _ in table]

    panels = [
        ("iw_mean", "认知负荷均值"),
        ("iw_high_ratio", "高负荷占比"),
        ("stop_probe_per_100steps", "停探测/100步"),
        ("veh_reaction_prob", "车辆反应概率"),
        ("mech_gate_rate", "gate通过率"),
        ("mech_risk_mean", "风险信号均值"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=200)

    for ax, (k, cn) in zip(axes.flat, panels):

        ki = KEYS.index(k)

        ys = [vals[ki] for _, vals in table]

        ax.plot(xs, ys, "o-", color="#2166ac")

        ax.axvline(0.7, color="#999", ls="--", lw=0.8)

        ax.set_xlabel("SEEV_VALUE_SAFETY_WEIGHT")

        ax.set_title(cn, fontsize=10)

    fig.suptitle(
        "SEEV_VALUE_SAFETY_WEIGHT 全量程扫描 0.1–1.3（虚线=默认0.7；负荷联动+自适应阈值已启用）",
        fontsize=11,
    )

    fig.tight_layout()

    png = OUT_DIR / "safety_weight_sweep.png"

    fig.savefig(png, dpi=200, bbox_inches="tight")

    print("PNG:", png)


if __name__ == "__main__":

    main()
