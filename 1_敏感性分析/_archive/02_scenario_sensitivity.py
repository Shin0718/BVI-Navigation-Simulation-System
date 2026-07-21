"""Run archived scenario-level sensitivity analysis experiments.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import contextlib
import csv
import io
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
OUT_DIR = ROOT / "sensitivity_out"
SEEDS = [20260705 + i for i in range(5)]
VEH_WIN = 5
LM_WIN = 3
MIN_SUBSET = 20

DEFAULTS = {
    "SEEV_VALUE_SAFETY_WEIGHT": 0.70,
    "SEEV_VALUE_PROGRESS_WEIGHT": 0.30,
    "SEEV_EXPECTANCY_RISK_WEIGHT": 0.55,
    "ACTR_RISK_CANE_GUIDANCE_RELIEF": 0.10,
    "ACTR_RISK_LANDMARK_RELIEF": 0.08,
    "MEMORY_ACTIVE_RETRIEVAL_TH": 0.15,
    "LANDMARK_DECAY_RATE": 0.82,
    "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK": 0.40,
    "LOOMING_BOOST_PEAK": 0.35,
    "LOOMING_BOOST_DECAY": 0.80,
    "PROBE_RELIEF_RATIO": 0.40,
    "MEMORY_ACTIVE_ABSENT_STEPS_TH": 11,
    "ACTR_LOAD_RESUME_THRESHOLD": 5.0,
}

PLAN = [
    ("PROBE_RELIEF_RATIO", [(0.32, 0.2), (0.48, 0.2)], {}),
    ("MEMORY_ACTIVE_ABSENT_STEPS_TH", [(10, 1 / 11), (12, 1 / 11)], {}),
    ("LANDMARK_DECAY_RATE", [(0.656, 0.2), (0.984, 0.2)], {}),
    ("VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK", [(0.32, 0.2), (0.48, 0.2)], {}),
    ("LOOMING_BOOST_PEAK", [(0.28, 0.2), (0.42, 0.2)], {}),
    ("LOOMING_BOOST_DECAY", [(0.64, 0.2), (0.96, 0.2)], {}),
    ("ACTR_RISK_CANE_GUIDANCE_RELIEF", [(0.05, 0.5), (0.20, 1.0)], {}),
    ("ACTR_RISK_LANDMARK_RELIEF", [(0.04, 0.5), (0.16, 1.0)], {}),
    (
        "MEMORY_ACTIVE_RETRIEVAL_TH",
        [(0.044, 0.2), (0.066, 0.2)],
        {"MEMORY_ACTIVE_RETRIEVAL_TH": 0.055},
    ),
]

SCEN_METRICS = [
    ("veh_share", "车辆-时长占比"),
    ("veh_iw_mean", "车辆-负荷"),
    ("veh_stop_rate", "车辆-停探测率"),
    ("veh_reaction_prob", "车辆-反应概率"),
    ("veh_reaction_delay_s", "车辆-反应延迟"),
    ("lm_share", "地标-时长占比"),
    ("lm_iw_mean", "地标-负荷"),
    ("lm_stop_rate", "地标-停探测率"),
    ("lm_relief_effect", "地标-风险缓释"),
    ("lm_stop_after_rate", "地标-触发后停探测率"),
    ("tac_share", "触觉-时长占比"),
    ("tac_iw_mean", "触觉-负荷"),
    ("tac_stop_rate", "触觉-停探测率"),
    ("tac_risk_mean", "触觉-风险"),
    ("noref_share", "无参照-时长占比"),
    ("noref_iw_mean", "无参照-负荷"),
    ("noref_stop_rate", "无参照-停探测率"),
    ("noref_ep_len", "无参照-片段步长"),
    ("noref_retrieval_per100", "无参照-记忆检索率"),
    ("cross_share", "路口-时长占比"),
    ("cross_iw_mean", "路口-负荷"),
    ("cross_stop_rate", "路口-停探测率"),
    ("cross_wait_s", "路口-等待时长"),
]
KEYS = [k for k, _ in SCEN_METRICS]
SCEN_BLOCKS = {
    "车辆逼近": [k for k in KEYS if k.startswith("veh_")],
    "地标触发": [k for k in KEYS if k.startswith("lm_")],
    "触觉引导": [k for k in KEYS if k.startswith("tac_")],
    "无参照": [k for k in KEYS if k.startswith("noref_")],
    "路口穿越": [k for k in KEYS if k.startswith("cross_")],
}


def _extract(
    sim_log,
    event_log,
    profile,
    steps,
    start_node,
    goal_node,
    current_position,
    max_steps,
    graph=None,
    initial_production_utilities=None,
    report_dir=None,
):
    """Extract analysis metrics from simulation and event logs."""
    n = len(sim_log)
    nan = float("nan")
    if n == 0:
        return {k: nan for k in KEYS}
    acts = [r.get("next_action", "") for r in sim_log]
    st = [float(r.get("sim_time", 0.0)) for r in sim_log]
    iw = [float(r.get("actr_iw_total", 0.0)) for r in sim_log]
    risk = [float(r.get("actr_risk_signal", 0.0)) for r in sim_log]
    dts = [st[0]] + [st[i] - st[i - 1] for i in range(1, n)]
    crossing = [bool(r.get("crossing_active")) for r in sim_log]

    veh = [bool(r.get("vehicle_approach")) for r in sim_log]
    v_on = [i for i in range(n) if veh[i] and (i == 0 or not veh[i - 1])]
    veh_set = set()
    for i in v_on:
        veh_set.update(range(i, min(i + VEH_WIN + 1, n)))

    lm = [r.get("matched_landmark", "none") != "none" for r in sim_log]
    lm_on = [i for i in range(n) if lm[i] and (i == 0 or not lm[i - 1])]
    lm_set = set()
    for i in lm_on:
        lm_set.update(range(i, min(i + LM_WIN + 1, n)))

    tac_set = {
        i
        for i in range(n)
        if not crossing[i]
        and (
            bool(sim_log[i].get("cane_guidance_present"))
            or sim_log[i].get("surface_type") == "tactile_guidance"
        )
    }
    gas = [int(r.get("guidance_absent_steps", 0)) for r in sim_log]
    noref_set = {i for i in range(n) if gas[i] > 0 and not crossing[i]}
    cross_set = {i for i in range(n) if crossing[i]}

    def sub(idx_set):
        """Summarize metrics for a selected scenario subset."""
        idx = sorted(idx_set)
        m = len(idx)
        if m < MIN_SUBSET:
            return {"share": m / n, "iw": nan, "stop": nan, "risk": nan}
        return {
            "share": m / n,
            "iw": sum(iw[i] for i in idx) / m,
            "stop": sum(1 for i in idx if acts[i] == "stop_and_probe") / m * 100,
            "risk": sum(risk[i] for i in idx) / m,
        }

    V, L, T, N_, C = (
        sub(veh_set),
        sub(lm_set),
        sub(tac_set),
        sub(noref_set),
        sub(cross_set),
    )

    reacted, delays = 0, []
    for i in v_on:
        for j in range(i, min(i + 3 + 1, n)):
            if acts[j] == "stop_and_probe":
                reacted += 1
                delays.append(st[j] - st[i])
                break
    v_prob = reacted / len(v_on) if v_on else nan
    v_delay = sum(delays) / len(delays) if delays else nan

    lm_stop_after, lm_relief = 0, []
    for i in lm_on:
        if any(acts[j] == "stop_and_probe" for j in range(i, min(i + LM_WIN + 1, n))):
            lm_stop_after += 1
        if i >= 1:
            after = [risk[j] for j in range(i, min(i + LM_WIN + 1, n))]
            lm_relief.append(risk[i - 1] - sum(after) / len(after))
    lm_after_rate = lm_stop_after / len(lm_on) if lm_on else nan
    lm_relief_mean = sum(lm_relief) / len(lm_relief) if lm_relief else nan

    eps, cur = [], 0
    for i in range(n):
        if gas[i] > 0:
            cur += 1
        elif cur:
            eps.append(cur)
            cur = 0
    if cur:
        eps.append(cur)
    mem = [bool(r.get("actr_memory_active", 0)) for r in sim_log]
    noref_idx = sorted(noref_set)
    noref_retr = (
        (
            sum(1 for i in noref_idx if mem[i] and (i == 0 or not mem[i - 1]))
            / len(noref_idx)
            * 100
        )
        if len(noref_idx) >= MIN_SUBSET
        else nan
    )

    cross_wait = sum(
        dts[i]
        for i in range(n)
        if crossing[i] and sim_log[i].get("crossing_subphase") == "wait"
    )

    return {
        "veh_share": V["share"],
        "veh_iw_mean": V["iw"],
        "veh_stop_rate": V["stop"],
        "veh_reaction_prob": v_prob,
        "veh_reaction_delay_s": v_delay,
        "lm_share": L["share"],
        "lm_iw_mean": L["iw"],
        "lm_stop_rate": L["stop"],
        "lm_relief_effect": lm_relief_mean,
        "lm_stop_after_rate": lm_after_rate,
        "tac_share": T["share"],
        "tac_iw_mean": T["iw"],
        "tac_stop_rate": T["stop"],
        "tac_risk_mean": T["risk"],
        "noref_share": N_["share"],
        "noref_iw_mean": N_["iw"],
        "noref_stop_rate": N_["stop"],
        "noref_ep_len": sum(eps) / len(eps) if eps else nan,
        "noref_retrieval_per100": noref_retr,
        "cross_share": C["share"],
        "cross_iw_mean": C["iw"],
        "cross_stop_rate": C["stop"],
        "cross_wait_s": cross_wait,
        "_total_steps": n,
    }


_SIM = None


def _init():
    """Initialize worker-process imports and paths."""
    global _SIM
    with contextlib.redirect_stdout(io.StringIO()):
        sys.path.insert(0, str(ROOT))
        from bvi_sa import simulation as s
    s.generate_report = _extract
    _SIM = s


def _run(task):
    """Execute one simulation task and return metrics."""
    label, overrides, seed = task
    import random

    s = _SIM
    try:
        for k, v in DEFAULTS.items():
            setattr(s, k, v)
        for k, v in overrides.items():
            setattr(s, k, v)
        random.seed(seed)
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
        with contextlib.redirect_stdout(io.StringIO()):
            m = s.run_simulation(familiarity_level=1)
        m["_seed"] = seed
        return label, m, None
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"


def mean_of(rows, k):
    """Compute the mean value of a metric across rows."""
    vs = [
        r[k] for r in rows if r and not (isinstance(r[k], float) and math.isnan(r[k]))
    ]
    return sum(vs) / len(vs) if vs else float("nan")


def main():
    """Run the script entry point."""
    jobs = max(1, (os.cpu_count() or 4) - 2)
    tasks, index = [], {}

    def add(label, overrides):
        """Append one labeled parameter override to the run plan."""
        ids = []
        for sd in SEEDS:
            tasks.append((label, overrides, sd))
        index[label] = overrides

    add("baseline", {})
    add("baseline_memth", {"MEMORY_ACTIVE_RETRIEVAL_TH": 0.055})
    for pname, pts, base_ov in PLAN:
        for v, _rel in pts:
            ov = dict(base_ov)
            ov[pname] = v
            add(f"{pname}|{v}", ov)
    print(f"计划: {len(index)} 点 × {len(SEEDS)} 种子 = {len(tasks)} 次, {jobs} 并行")

    t0 = time.time()
    results = {}
    errors = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=_init) as pool:
        done = 0
        for label, m, err in pool.imap_unordered(_run, tasks):
            done += 1
            if err:
                errors.append((label, err))
                print(f"[{done}/{len(tasks)}] {label} FAILED: {err}")
            else:
                results.setdefault(label, []).append(m)
                print(f"[{done}/{len(tasks)}] {label} ok steps={m['_total_steps']}")
    print(f"完成 {(time.time()-t0)/60:.1f} 分钟, 失败 {len(errors)}")

    base = {k: mean_of(results.get("baseline", []), k) for k in KEYS}
    base_memth = {k: mean_of(results.get("baseline_memth", []), k) for k in KEYS}

    S = {}
    for pname, pts, base_ov in PLAN:
        b = base_memth if base_ov else base
        row = {}
        for k in KEYS:
            es = []
            for v, rel in pts:
                rows = results.get(f"{pname}|{v}", [])
                yp, yb = mean_of(rows, k), b[k]
                if math.isnan(yp) or math.isnan(yb) or abs(yb) < 1e-9:
                    continue
                es.append(abs((yp - yb) / yb) / rel)
            row[k] = sum(es) / len(es) if es else float("nan")
        S[pname] = row

    params = [p for p, _, _ in PLAN]
    s_path = OUT_DIR / "scenario_S_matrix.csv"
    with open(s_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["parameter"] + KEYS)
        for p in params:
            w.writerow(
                [p]
                + [f"{S[p][k]:.4f}" if not math.isnan(S[p][k]) else "" for k in KEYS]
            )

    sum_path = OUT_DIR / "scenario_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["parameter"] + list(SCEN_BLOCKS.keys()))
        for p in params:
            row = []
            for scen, keys in SCEN_BLOCKS.items():
                vals = [S[p][k] for k in keys if not math.isnan(S[p][k])]
                row.append(f"{max(vals):.4f}" if vals else "")
            w.writerow([p] + row)

    raw_path = OUT_DIR / "scenario_runs_raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "seed"] + KEYS)
        for label, rows in results.items():
            for r in rows:
                w.writerow(
                    [label, r["_seed"]]
                    + [
                        (
                            f"{r[k]:.5f}"
                            if not (isinstance(r[k], float) and math.isnan(r[k]))
                            else "nan"
                        )
                        for k in KEYS
                    ]
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fn in ["PingFang SC", "Hiragino Sans GB", "SimHei", "Arial Unicode MS"]:
        if any(fn.lower() in t.name.lower() for t in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fn
            break
    plt.rcParams["axes.unicode_minus"] = False
    cns = [cn for _, cn in SCEN_METRICS]
    data = [[S[p][k] if not math.isnan(S[p][k]) else 0.0 for k in KEYS] for p in params]
    fig, ax = plt.subplots(
        figsize=(0.62 * len(KEYS) + 4, 0.5 * len(params) + 2.5), dpi=200
    )
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cns)))
    ax.set_xticklabels(cns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, fontsize=8)
    vmax = max((max(r) for r in data), default=1.0) or 1.0
    for i, rrow in enumerate(data):
        for j, v in enumerate(rrow):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if v > 0.6 * vmax else "#1a3a5c",
            )
    xpos = 0
    for scen, keys in SCEN_BLOCKS.items():
        if xpos:
            ax.axvline(xpos - 0.5, color="#888", lw=0.8)
        xpos += len(keys)
    ax.set_title(
        "情境敏感性矩阵（9可校准参数 × 5情境行为指标；弹性 |%Δy|/|%Δθ|）", fontsize=10
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    heat = OUT_DIR / "scenario_heatmap.png"
    fig.savefig(heat, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("输出:")
    for x in [s_path, sum_path, raw_path, heat]:
        print("  ", x)
    if errors:
        print("失败:", errors)


if __name__ == "__main__":
    main()
