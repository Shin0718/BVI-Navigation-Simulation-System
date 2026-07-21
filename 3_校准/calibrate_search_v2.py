"""Search calibrated parameter sets with staged grid and coordinate descent routines.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
OUT = ROOT / "calib_out_v2"
OUT.mkdir(exist_ok=True)
CAT4_PATH = ROOT / "2_参数筛选" / "candidate_sensitivity_4cat.py"

SEEDS = [20260705 + i for i in range(5)]
IMPROVE_EPS = 0.01

SEEV_UNIT = "SEEV_SAFETY_UNIT"
MEMTH = "MEMORY_ACTIVE_RETRIEVAL_TH"
MEMTH_GRID = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

ROUND1 = [
    ("MEMORY_ACTIVE_ABSENT_STEPS_TH", [7, 9, 11, 14, 18]),
    ("PROBE_RELIEF_RATIO", [0.20, 0.30, 0.40, 0.50, 0.60]),
    (SEEV_UNIT, [0.56, 0.63, 0.70, 0.77, 0.84]),
    ("LANDMARK_DECAY_RATE", [0.66, 0.74, 0.82, 0.90, 0.96]),
    ("LOOMING_BOOST_DECAY", [0.72, 0.78, 0.84, 0.90, 0.96]),
    ("LOOMING_BOOST_PEAK", [0.21, 0.28, 0.35, 0.42, 0.49]),
]
UNIT_DEFAULTS = {
    SEEV_UNIT: 0.70,
    "MEMORY_ACTIVE_ABSENT_STEPS_TH": 11,
    "PROBE_RELIEF_RATIO": 0.40,
    "LANDMARK_DECAY_RATE": 0.82,
    "LOOMING_BOOST_DECAY": 0.80,
    "LOOMING_BOOST_PEAK": 0.35,
}
INT_UNITS = {"MEMORY_ACTIVE_ABSENT_STEPS_TH"}

OBS_MAP = {
    "response_prob": "veh_reaction_prob",
    "response_delay_s": "veh_reaction_delay_s",
    "post_trigger_probe": "lm_stop_after_rate",
    "episode_length": "noref_ep_len",
    "retrieval_rate": "noref_retrieval_per100",
    "veh_time_share": "veh_share",
}
MEMTH_LOSS = {
    "retrieval_rate": ("noref_retrieval_per100", 0.6),
    "_noref_stop_rate": ("noref_stop_rate", 0.4),
}


def load_cat4():
    """Load four-category sensitivity scores."""
    spec = importlib.util.spec_from_file_location("cat4", CAT4_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cat4"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_obs():
    """Load observed calibration targets."""
    obs = {}
    with open(ROOT / "obs_values.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            obs[row["metric"]] = float(row["obs_value"])
    return obs


def load_weights():
    """Load calibration loss weights."""
    raw = {}
    with open(
        ROOT / "sensitivity_out" / "cat4_indicator_scores.csv", encoding="utf-8-sig"
    ) as f:
        for row in csv.DictReader(f):
            raw[row["metric"]] = float(row["col_sum(全参数)"])
    sub = {k: raw[k] for k in OBS_MAP if k in raw}
    tot = sum(sub.values()) or 1.0
    return {k: v / tot for k, v in sub.items()}


def rel_sq_error(sim, obs):
    """Compute relative squared error with zero-safe scaling."""
    scale = abs(obs) if abs(obs) > 1e-9 else 1.0
    return ((sim - obs) / scale) ** 2


def l_unified(metrics, obs, weights):
    """Compute the unified weighted calibration loss."""
    avail = {
        k: w
        for k, w in weights.items()
        if OBS_MAP[k] in obs and not math.isnan(metrics.get(k, float("nan")))
    }
    tw = sum(avail.values()) or 1.0
    return sum(
        (w / tw) * rel_sq_error(metrics[k], obs[OBS_MAP[k]]) for k, w in avail.items()
    )


def l_memth(metrics, obs):
    """Compute the memory-threshold calibration loss."""
    L = 0.0
    for mk, (ok, wt) in MEMTH_LOSS.items():
        v = metrics.get(mk, float("nan"))
        if ok in obs and not (isinstance(v, float) and math.isnan(v)):
            L += wt * rel_sq_error(v, obs[ok])
    return L


def expand(theta):
    """Expand grouped calibration parameters into simulation overrides."""
    ov = {}
    for k, v in theta.items():
        if k == SEEV_UNIT:
            ov["SEEV_VALUE_SAFETY_WEIGHT"] = v
            ov["SEEV_VALUE_PROGRESS_WEIGHT"] = round(1.0 - v, 4)
        else:
            ov[k] = v
    return ov


_SIM = None
_EXTRACT = None


def _init():
    """Initialize worker-process imports and paths."""
    global _SIM, _EXTRACT
    with contextlib.redirect_stdout(io.StringIO()):
        sys.path.insert(0, str(ROOT))
        from bvi_sa import simulation as sim
    cat4 = load_cat4()
    sim.generate_report = cat4._extract
    _SIM = sim
    _EXTRACT = cat4
    _init.PARAMS = cat4.PARAMS


def _run(task):
    """Execute one simulation task and return metrics."""
    label, overrides, seed = task
    import random

    sim = _SIM
    try:
        for name, (default, _c, _i, _d) in _init.PARAMS.items():
            setattr(sim, name, overrides.get(name, default))
        random.seed(seed)
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
        with contextlib.redirect_stdout(io.StringIO()):
            m = sim.run_simulation(familiarity_level=1)
        m["_seed"] = seed
        return label, m, None
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"


def mean_of(rows, k):
    """Compute the mean value of a metric across rows."""
    vs = [r.get(k) for r in rows if r is not None]
    vs = [
        v for v in vs if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return sum(vs) / len(vs) if vs else float("nan")


_raw_path = OUT / "calib_runs_raw.csv"
_raw_header_written = _raw_path.exists()
METRIC_COLS = None


def eval_candidates(pool, cand_list, seeds):
    """Evaluate candidate parameter settings across seeds."""
    global _raw_header_written, METRIC_COLS
    tasks = [(lb, expand(th), sd) for lb, th in cand_list for sd in seeds]
    per_label = {}
    with open(_raw_path, "a", newline="", encoding="utf-8-sig") as f:
        w = None
        for label, m, err in pool.imap_unordered(_run, tasks):
            if err:
                print(f"  {label} FAILED: {err}", flush=True)
                continue
            per_label.setdefault(label, []).append(m)
            if METRIC_COLS is None:
                METRIC_COLS = [k for k in m if not k.startswith("__")]
            if w is None:
                w = csv.DictWriter(
                    f, fieldnames=["label", "seed"] + METRIC_COLS, extrasaction="ignore"
                )
                if not _raw_header_written:
                    w.writeheader()
                    _raw_header_written = True
            w.writerow(
                {
                    "label": label,
                    "seed": m["_seed"],
                    **{k: m.get(k) for k in METRIC_COLS},
                }
            )
            f.flush()
    keys = set()
    for rows in per_label.values():
        for r in rows:
            keys.update(k for k in r if isinstance(r[k], (int, float)))
    return {lb: {k: mean_of(rows, k) for k in keys} for lb, rows in per_label.items()}


def main():
    """Run the script entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="基线+1点×1种子验证管线")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    obs = load_obs()
    weights = load_weights()
    print("损失权重（可观测6项，重归一化）:")
    for k, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {w:.4f}  (obs={obs.get(OBS_MAP[k])})")

    t0 = time.time()
    trace = []
    ctx = mp.get_context("spawn")

    if args.smoke:
        with ctx.Pool(processes=2, initializer=_init) as pool:
            res = eval_candidates(
                pool,
                [
                    ("smoke|default", {}),
                    ("smoke|theta", {SEEV_UNIT: 0.77, MEMTH: 0.06}),
                ],
                seeds=SEEDS[:1],
            )
        for lb, m in res.items():
            print(
                f"{lb}: L_unified={l_unified(m, obs, weights):.4f} "
                f"L_memoryTH={l_memth(m, obs):.4f} "
                f"retrieval={m.get('retrieval_rate'):.3f} "
                f"noref_stop={m.get('_noref_stop_rate'):.3f}"
            )
        print(f"smoke 完成 {(time.time()-t0)/60:.1f} 分钟")
        return

    jobs = args.jobs
    with ctx.Pool(processes=jobs, initializer=_init) as pool:
        print(f"阶段A: {MEMTH} 网格 {MEMTH_GRID}", flush=True)
        cands = [(f"memth|{v}", {MEMTH: v}) for v in MEMTH_GRID]
        res = eval_candidates(pool, cands, SEEDS)
        memth_scores = {}
        for v in MEMTH_GRID:
            m = res.get(f"memth|{v}")
            if m is None:
                continue
            memth_scores[v] = l_memth(m, obs)
            trace.append(
                {
                    "stage": "A",
                    "param": MEMTH,
                    "value": v,
                    "L_memoryTH": memth_scores[v],
                    "L_unified": l_unified(m, obs, weights),
                }
            )
            print(f"  TH={v}: L_memoryTH={memth_scores[v]:.4f}", flush=True)
        best_th = min(memth_scores, key=memth_scores.get)
        print(
            f"阶段A最优: {MEMTH}={best_th} (L_memoryTH={memth_scores[best_th]:.4f})",
            flush=True,
        )

        theta = {MEMTH: best_th}
        base_res = eval_candidates(pool, [("incumbent|start", dict(theta))], SEEDS)
        cur_L = l_unified(base_res["incumbent|start"], obs, weights)
        print(f"起点 L_unified(θ, TH={best_th}) = {cur_L:.4f}", flush=True)
        trace.append(
            {
                "stage": "B0",
                "param": "(incumbent)",
                "value": "",
                "L_memoryTH": l_memth(base_res["incumbent|start"], obs),
                "L_unified": cur_L,
            }
        )

        for rnd in (1, 2):
            moved = False
            for pname, grid in ROUND1:
                cur_v = theta.get(pname, UNIT_DEFAULTS[pname])
                pts = [v for v in grid if v != cur_v]
                if rnd == 2:
                    idx_grid = sorted(set(grid + [cur_v]))
                    i = idx_grid.index(cur_v)
                    pts = []
                    if i > 0:
                        pts.append(round((cur_v + idx_grid[i - 1]) / 2, 4))
                    if i < len(idx_grid) - 1:
                        pts.append(round((cur_v + idx_grid[i + 1]) / 2, 4))
                    if pname in INT_UNITS:
                        pts = sorted({max(2, int(round(p))) for p in pts} - {cur_v})
                if not pts:
                    continue
                print(f"[R{rnd}] {pname}: 当前={cur_v}, 试 {pts}", flush=True)
                cands = []
                for v in pts:
                    th = dict(theta)
                    th[pname] = v
                    cands.append((f"R{rnd}|{pname}|{v}", th))
                res = eval_candidates(pool, cands, SEEDS)
                best_v, best_L = cur_v, cur_L
                for v in pts:
                    m = res.get(f"R{rnd}|{pname}|{v}")
                    if m is None:
                        continue
                    L = l_unified(m, obs, weights)
                    trace.append(
                        {
                            "stage": f"B-R{rnd}",
                            "param": pname,
                            "value": v,
                            "L_memoryTH": l_memth(m, obs),
                            "L_unified": L,
                        }
                    )
                    print(f"    {pname}={v}: L={L:.4f}", flush=True)
                    if L < best_L:
                        best_v, best_L = v, L
                if (
                    best_v != cur_v
                    and (cur_L - best_L) / max(cur_L, 1e-9) >= IMPROVE_EPS
                ):
                    print(
                        f"    → 更新 {pname}: {cur_v} → {best_v} "
                        f"(L {cur_L:.4f} → {best_L:.4f})",
                        flush=True,
                    )
                    theta[pname] = best_v
                    cur_L = best_L
                    moved = True
                else:
                    print(f"    → 保留 {pname}={cur_v} (最优改善不足1%)", flush=True)
            if not moved:
                print(f"[R{rnd}] 无单元更新，提前收敛", flush=True)
                break

        final_res = eval_candidates(pool, [("theta_star", dict(theta))], SEEDS)

    m_star = final_res["theta_star"]
    unit_names = [MEMTH] + [p for p, _ in ROUND1]
    result = {
        "units": "SEEV_SAFETY_UNIT = SAFETY_WEIGHT 值，PROGRESS=1−SAFETY 联动，EXPECTANCY 固定 0.55",
        "theta_star": {
            k: theta.get(k, UNIT_DEFAULTS.get(k, None) if k != MEMTH else 0.15)
            for k in unit_names
        },
        "theta_star_expanded": expand(
            {k: theta.get(k, UNIT_DEFAULTS[k]) for k in UNIT_DEFAULTS}
            | {MEMTH: theta[MEMTH]}
        ),
        "defaults": {**UNIT_DEFAULTS, MEMTH: 0.15},
        "L_unified_star": l_unified(m_star, obs, weights),
        "L_memoryTH_star": l_memth(m_star, obs),
        "weights": weights,
        "metrics_star": {
            k: m_star.get(k) for k in list(OBS_MAP) + ["_noref_stop_rate"]
        },
        "elapsed_min": (time.time() - t0) / 60,
    }
    with open(OUT / "theta_star.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(OUT / "calib_trace.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f, fieldnames=["stage", "param", "value", "L_unified", "L_memoryTH"]
        )
        w.writeheader()
        for row in trace:
            w.writerow(row)
    print(json.dumps(result["theta_star"], indent=2))
    print(
        f"L_unified* = {result['L_unified_star']:.4f}, "
        f"L_memoryTH* = {result['L_memoryTH_star']:.4f}"
    )
    print(f"总耗时 {result['elapsed_min']:.1f} 分钟")


if __name__ == "__main__":
    main()
