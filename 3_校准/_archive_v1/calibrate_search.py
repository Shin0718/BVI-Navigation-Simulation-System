"""Run archived coordinate-descent calibration search.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import contextlib
import csv
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
sys.path.insert(0, str(ROOT))
import _setup
import scenario_sensitivity as ss

OUT = ROOT / "calib_out"
OUT.mkdir(exist_ok=True)
SEEDS = ss.SEEDS

MEMTH_GRID = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
MEMTH_WEIGHTS = {"noref_retrieval_per100": 0.6, "noref_stop_rate": 0.4}

ROUND1 = [
    ("MEMORY_ACTIVE_ABSENT_STEPS_TH", [7, 9, 11, 14, 18]),
    ("PROBE_RELIEF_RATIO", [0.20, 0.30, 0.40, 0.50, 0.60]),
    ("LANDMARK_DECAY_RATE", [0.66, 0.74, 0.82, 0.90, 0.96]),
    ("VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK", [0.28, 0.34, 0.40, 0.48, 0.56]),
    ("LOOMING_BOOST_PEAK", [0.21, 0.28, 0.35, 0.42, 0.49]),
    ("LOOMING_BOOST_DECAY", [0.72, 0.78, 0.84, 0.90, 0.96]),
]
IMPROVE_EPS = 0.01


def load_obs():
    """Load observed calibration targets."""
    obs = {}
    with open(ROOT / "obs_values.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            obs[row["metric"]] = float(row["obs_value"])
    return obs


def load_weights():
    """Load calibration loss weights."""
    w = {}
    with open(ROOT / "sensitivity_out" / "loss_weights.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            w[row["metric"]] = float(row["weight_w_k"])
    return w


def rel_sq_error(sim, obs):
    """Compute relative squared error with zero-safe scaling."""
    scale = abs(obs) if abs(obs) > 1e-9 else 1.0
    return ((sim - obs) / scale) ** 2


def l_unified(metrics, obs, weights):
    """Compute the unified weighted calibration loss."""
    avail = {
        k: w
        for k, w in weights.items()
        if k in obs and k in metrics and not math.isnan(metrics[k])
    }
    tw = sum(avail.values())
    return sum((w / tw) * rel_sq_error(metrics[k], obs[k]) for k, w in avail.items())


def l_memth(metrics, obs):
    """Compute the memory-threshold calibration loss."""
    return sum(
        w * rel_sq_error(metrics[k], obs[k])
        for k, w in MEMTH_WEIGHTS.items()
        if k in obs and not math.isnan(metrics.get(k, float("nan")))
    )


_raw_path = OUT / "calib_runs_raw.csv"
_raw_header_written = _raw_path.exists()


def eval_candidates(pool, cand_list):
    """Evaluate candidate parameter settings across seeds."""
    global _raw_header_written
    tasks = [(lb, ov, sd) for lb, ov in cand_list for sd in SEEDS]
    per_label = {}
    with open(_raw_path, "a", newline="", encoding="utf-8-sig") as f:
        w = None
        for label, m, err in pool.imap_unordered(ss._run, tasks):
            if err:
                print(f"  {label} FAILED: {err}", flush=True)
                continue
            per_label.setdefault(label, []).append(m)
            if w is None:
                cols = ["label", "seed"] + ss.KEYS
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                if not _raw_header_written:
                    w.writeheader()
                    _raw_header_written = True
            w.writerow(
                {"label": label, "seed": m["_seed"], **{k: m.get(k) for k in ss.KEYS}}
            )
            f.flush()
    return {
        lb: {k: ss.mean_of(rows, k) for k in ss.KEYS} for lb, rows in per_label.items()
    }


def main():
    """Run the script entry point."""
    obs = load_obs()
    weights = load_weights()
    jobs = max(1, (os.cpu_count() or 4) - 2)
    trace = []
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=ss._init) as pool:
        print(f"阶段A: MEMORY_ACTIVE_RETRIEVAL_TH 网格 {MEMTH_GRID}", flush=True)
        cands = [(f"memth|{v}", {"MEMORY_ACTIVE_RETRIEVAL_TH": v}) for v in MEMTH_GRID]
        res = eval_candidates(pool, cands)
        memth_scores = {}
        for v in MEMTH_GRID:
            m = res.get(f"memth|{v}")
            if m is None:
                continue
            memth_scores[v] = l_memth(m, obs)
            trace.append(
                {
                    "stage": "A",
                    "param": "MEMORY_ACTIVE_RETRIEVAL_TH",
                    "value": v,
                    "L_memoryTH": memth_scores[v],
                    "L_unified": l_unified(m, obs, weights),
                }
            )
            print(f"  TH={v}: L_memoryTH={memth_scores[v]:.4f}", flush=True)
        best_th = min(memth_scores, key=memth_scores.get)
        print(
            f"阶段A最优: MEMORY_ACTIVE_RETRIEVAL_TH={best_th} (L_memoryTH={memth_scores[best_th]:.4f})",
            flush=True,
        )

        theta = {"MEMORY_ACTIVE_RETRIEVAL_TH": best_th}
        base_res = eval_candidates(pool, [("incumbent|start", dict(theta))])
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
                cur_v = theta.get(pname, ss.DEFAULTS[pname])
                pts = [v for v in grid if v != cur_v]
                if rnd == 2:
                    idx_grid = sorted(set(grid + [cur_v]))
                    i = idx_grid.index(cur_v)
                    pts = []
                    if i > 0:
                        pts.append(round((cur_v + idx_grid[i - 1]) / 2, 4))
                    if i < len(idx_grid) - 1:
                        pts.append(round((cur_v + idx_grid[i + 1]) / 2, 4))
                    if pname == "MEMORY_ACTIVE_ABSENT_STEPS_TH":
                        pts = sorted({max(2, int(round(p))) for p in pts} - {cur_v})
                if not pts:
                    continue
                print(f"[R{rnd}] {pname}: 当前={cur_v}, 试 {pts}", flush=True)
                cands = []
                for v in pts:
                    ov = dict(theta)
                    ov[pname] = v
                    cands.append((f"R{rnd}|{pname}|{v}", ov))
                res = eval_candidates(pool, cands)
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
                        f"    → 更新 {pname}: {cur_v} → {best_v} (L {cur_L:.4f} → {best_L:.4f})",
                        flush=True,
                    )
                    theta[pname] = best_v
                    cur_L = best_L
                    moved = True
                else:
                    print(f"    → 保留 {pname}={cur_v} (最优改善不足1%)", flush=True)
            if not moved:
                print(f"[R{rnd}] 无参数更新，提前收敛", flush=True)
                break

        final_res = eval_candidates(pool, [("theta_star", dict(theta))])

    m_star = final_res["theta_star"]
    result = {
        "theta_star": {
            k: theta.get(k, ss.DEFAULTS[k])
            for k in ["MEMORY_ACTIVE_RETRIEVAL_TH"] + [p for p, _ in ROUND1]
        },
        "defaults": {
            k: ss.DEFAULTS[k]
            for k in ["MEMORY_ACTIVE_RETRIEVAL_TH"] + [p for p, _ in ROUND1]
        },
        "L_unified_star": l_unified(m_star, obs, weights),
        "L_memoryTH_star": l_memth(m_star, obs),
        "metrics_star": {k: m_star.get(k) for k in ss.KEYS},
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
        f"L_unified* = {result['L_unified_star']:.4f}, L_memoryTH* = {result['L_memoryTH_star']:.4f}"
    )
    print(f"总耗时 {result['elapsed_min']:.1f} 分钟")


if __name__ == "__main__":
    main()
