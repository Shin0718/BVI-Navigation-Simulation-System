"""Run mechanism-level sensitivity analysis for calibrated simulation parameters.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import argparse
import csv
import io
import math
import multiprocessing as mp
import os
import sys
import time
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
OUT_DIR = ROOT / "sensitivity_out"
OUT_DIR.mkdir(exist_ok=True)

SEED_BASE = 20260705
VEH_REACTION_WINDOW = 3
LM_WINDOW = 3

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


def build_points():
    """Build the sensitivity sampling plan."""
    pts = [("baseline", None, None, None)]
    for p in ["ACTR_RISK_CANE_GUIDANCE_RELIEF", "ACTR_RISK_LANDMARK_RELIEF"]:
        d = DEFAULTS[p]
        pts.append((f"{p}|x0.5", p, d * 0.5, 0.5))
        pts.append((f"{p}|x2.0", p, d * 2.0, 1.0))
    for v in [0.03, 0.05, 0.08, 0.10, 0.12, 0.18]:
        pts.append(
            (
                f"MEMORY_ACTIVE_RETRIEVAL_TH|{v:.2f}",
                "MEMORY_ACTIVE_RETRIEVAL_TH",
                v,
                abs(v - 0.15) / 0.15,
            )
        )
    for p in [
        "SEEV_VALUE_SAFETY_WEIGHT",
        "SEEV_VALUE_PROGRESS_WEIGHT",
        "SEEV_EXPECTANCY_RISK_WEIGHT",
        "LOOMING_BOOST_PEAK",
        "LOOMING_BOOST_DECAY",
        "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK",
    ]:
        d = DEFAULTS[p]
        pts.append((f"{p}|up", p, d * 1.2, 0.2))
        pts.append((f"{p}|down", p, d * 0.8, 0.2))
    return pts


S1_PARAMS = [
    "SEEV_VALUE_SAFETY_WEIGHT",
    "SEEV_VALUE_PROGRESS_WEIGHT",
    "SEEV_EXPECTANCY_RISK_WEIGHT",
    "ACTR_RISK_CANE_GUIDANCE_RELIEF",
    "ACTR_RISK_LANDMARK_RELIEF",
    "LOOMING_BOOST_PEAK",
    "LOOMING_BOOST_DECAY",
    "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK",
    "MEMORY_ACTIVE_RETRIEVAL_TH",
]

MECH_KEYS = [
    ("mech_net_priority_mean", "net_priority均值"),
    ("mech_gate_rate", "gate通过率"),
    ("mech_looming_mean", "looming_boost均值"),
    ("mech_looming_duration", "looming持续步数/episode"),
    ("mech_sound_salience_mean", "声音显著性均值"),
    ("mech_veh_evidence_valid_rate", "车辆证据有效率(过闸门)"),
    ("mech_risk_mean", "ACT-R风险信号均值"),
    ("mech_memory_active_rate", "记忆激活占比"),
    ("mech_retrieval_load_mean", "检索负荷均值"),
    ("mech_retrieval_load_p90", "检索负荷90分位"),
]
OUT_KEYS = [
    ("stop_probe_per_100steps", "停探测频率"),
    ("veh_reaction_prob", "车辆反应概率"),
    ("veh_reaction_delay_s", "车辆反应延迟"),
    ("veh_miss_rate", "车辆漏报率"),
    ("veh_response_duration_steps", "车辆反应持续步数"),
    ("recovery_time_after_vehicle_s", "车辆后恢复时间"),
    ("move_direct_rate", "直行占比"),
    ("iw_mean", "认知负荷均值"),
    ("iw_high_ratio", "高负荷占比"),
    ("crossing_wait_s", "路口等待时长"),
    ("stop_probe_after_landmark_rate", "地标后停探测率"),
    ("landmark_relief_effect", "地标风险缓释量"),
    ("memory_retrieval_per_100steps", "记忆检索频率"),
    ("ref_loss_episodes_per_100steps", "参照丢失次数"),
    ("deprivation_mean", "感觉剥夺均值"),
    ("obstacle_hits_per_100steps", "障碍接触频率"),
    ("total_sim_time_s", "完成时间"),
    ("total_steps", "总步数"),
]
OUT_KEYLIST = [k for k, _ in OUT_KEYS]
MECH_KEYLIST = [k for k, _ in MECH_KEYS]
ALL_KEYS = OUT_KEYLIST + MECH_KEYLIST


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
        return {k: nan for k in ALL_KEYS}
    acts = [r.get("next_action", "") for r in sim_log]
    st = [float(r.get("sim_time", 0.0)) for r in sim_log]
    iw = [float(r.get("actr_iw_total", 0.0)) for r in sim_log]
    risk = [float(r.get("actr_risk_signal", 0.0)) for r in sim_log]
    dts = [st[0]] + [st[i] - st[i - 1] for i in range(1, n)]

    veh = [bool(r.get("vehicle_approach", False)) for r in sim_log]
    onsets = [i for i in range(n) if veh[i] and (i == 0 or not veh[i - 1])]
    reacted, delays, durations, recoveries = 0, [], [], []
    for i in onsets:
        ri = None
        for j in range(i, min(i + VEH_REACTION_WINDOW + 1, n)):
            if acts[j] == "stop_and_probe":
                ri = j
                break
        if ri is None:
            continue
        reacted += 1
        delays.append(st[ri] - st[i])
        k = ri
        while k < n and acts[k] == "stop_and_probe":
            k += 1
        durations.append(k - ri)
        while k < n and acts[k] != "move_direct":
            k += 1
        if k < n:
            recoveries.append(st[k] - st[i])
    v_prob = reacted / len(onsets) if onsets else nan
    v_delay = sum(delays) / len(delays) if delays else nan
    v_dur = sum(durations) / len(durations) if durations else nan
    v_rec = sum(recoveries) / len(recoveries) if recoveries else nan

    lm = [r.get("matched_landmark", "none") != "none" for r in sim_log]
    lm_on = [i for i in range(n) if lm[i] and (i == 0 or not lm[i - 1])]
    lm_stop, lm_relief = 0, []
    for i in lm_on:
        if any(
            acts[j] == "stop_and_probe" for j in range(i, min(i + LM_WINDOW + 1, n))
        ):
            lm_stop += 1
        if i >= 1:
            before = risk[i - 1]
            after = [risk[j] for j in range(i, min(i + LM_WINDOW + 1, n))]
            lm_relief.append(before - sum(after) / len(after))
    lm_stop_rate = lm_stop / len(lm_on) if lm_on else nan
    lm_relief_mean = sum(lm_relief) / len(lm_relief) if lm_relief else nan

    mem = [bool(r.get("actr_memory_active", 0)) for r in sim_log]
    mem_edges = sum(1 for i in range(n) if mem[i] and (i == 0 or not mem[i - 1]))

    gas = [int(r.get("guidance_absent_steps", 0)) for r in sim_log]
    ref_loss = sum(1 for i in range(n) if gas[i] > 0 and (i == 0 or gas[i - 1] == 0))

    lb = [float(r.get("looming_boost", 0.0)) for r in sim_log]
    ep_lens, cur = [], 0
    for v in lb:
        if v > 0.05:
            cur += 1
        elif cur:
            ep_lens.append(cur)
            cur = 0
    if cur:
        ep_lens.append(cur)

    raw_steps = sum(1 for r in sim_log if r.get("vehicle_approach_raw"))
    valid_steps = sum(1 for r in sim_log if r.get("vehicle_approach"))
    valid_rate = valid_steps / raw_steps if raw_steps else nan

    rl = [float(r.get("retrieval_wm_load", 0.0)) for r in sim_log]
    rl_sorted = sorted(rl)
    crossing_wait = sum(
        dts[i]
        for i in range(n)
        if sim_log[i].get("crossing_active")
        and sim_log[i].get("crossing_subphase") == "wait"
    )
    iw_sorted = sorted(iw)

    return {
        "stop_probe_per_100steps": acts.count("stop_and_probe") / n * 100,
        "veh_reaction_prob": v_prob,
        "veh_reaction_delay_s": v_delay,
        "veh_miss_rate": (1 - v_prob) if not math.isnan(v_prob) else nan,
        "veh_response_duration_steps": v_dur,
        "recovery_time_after_vehicle_s": v_rec,
        "move_direct_rate": acts.count("move_direct") / n,
        "iw_mean": sum(iw) / n,
        "iw_high_ratio": sum(1 for v in iw if v >= 6.0) / n,
        "crossing_wait_s": crossing_wait,
        "stop_probe_after_landmark_rate": lm_stop_rate,
        "landmark_relief_effect": lm_relief_mean,
        "memory_retrieval_per_100steps": mem_edges / n * 100,
        "ref_loss_episodes_per_100steps": ref_loss / n * 100,
        "deprivation_mean": sum(float(r.get("deprivation_index", 0.0)) for r in sim_log)
        / n,
        "obstacle_hits_per_100steps": sum(1 for r in sim_log if r.get("cane_obstacle"))
        / n
        * 100,
        "total_sim_time_s": st[-1],
        "total_steps": n,
        "mech_net_priority_mean": sum(
            float(r.get("net_priority", 0.0)) for r in sim_log
        )
        / n,
        "mech_gate_rate": sum(1 for r in sim_log if r.get("gate_passed")) / n,
        "mech_looming_mean": sum(lb) / n,
        "mech_looming_duration": sum(ep_lens) / len(ep_lens) if ep_lens else nan,
        "mech_sound_salience_mean": sum(
            float(r.get("sound_salience", 0.0)) for r in sim_log
        )
        / n,
        "mech_veh_evidence_valid_rate": valid_rate,
        "mech_risk_mean": sum(risk) / n,
        "mech_memory_active_rate": sum(1 for v in mem if v) / n,
        "mech_retrieval_load_mean": sum(rl) / n,
        "mech_retrieval_load_p90": rl_sorted[min(n - 1, int(math.ceil(0.9 * n)) - 1)],
        "_n_veh_onsets": len(onsets),
        "_n_lm_onsets": len(lm_on),
    }


_SIM = None


def _worker_init():
    """Initialize worker-process imports and paths."""
    global _SIM
    with contextlib.redirect_stdout(io.StringIO()):
        sys.path.insert(0, str(ROOT))
        from bvi_sa import simulation as sim
    sim.generate_report = _extract
    _SIM = sim


def _run_once(task):
    """Execute one simulation task and return metrics."""
    tid, param, value, seed = task
    import random

    sim = _SIM
    try:
        for name, d in DEFAULTS.items():
            setattr(sim, name, d)
        if param is not None:
            setattr(sim, param, value)
        random.seed(seed)
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            m = sim.run_simulation(familiarity_level=1)
        m["_wall_s"] = time.time() - t0
        m["_seed"] = seed
        return tid, m, None
    except Exception as e:
        return tid, None, f"{type(e).__name__}: {e}"


def mean_of(rows, key):
    """Compute the mean value of a metric across rows."""
    vals = [
        r[key]
        for r in rows
        if r and not (isinstance(r[key], float) and math.isnan(r[key]))
    ]
    return sum(vals) / len(vals) if vals else float("nan")


def pearson(x, y):
    """Compute the Pearson correlation coefficient."""
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in xs))
    sy = math.sqrt(sum((v - my) ** 2 for v in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def main():
    """Run the script entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        sys.path.insert(0, str(ROOT))
        from bvi_sa import simulation as sim
    bad = [
        (k, v, getattr(sim, k, None))
        for k, v in DEFAULTS.items()
        if getattr(sim, k, None) is None
        or abs(float(getattr(sim, k)) - float(v)) > 1e-9
    ]
    if bad:
        print(
            f"ℹ️ {len(bad)} 个参数代码默认值 ≠ 清单参考值（预期：代码已写回 θ*），"
            "分析按清单值显式注入，不受影响:",
            bad,
        )
    else:
        print("✓ 默认值校验通过")

    points = build_points()
    tasks, index = [], {}
    for label, param, value, _rel in points:
        ids = []
        for s in range(args.seeds):
            tid = len(tasks)
            tasks.append((tid, param, value, SEED_BASE + s))
            ids.append(tid)
        index[label] = ids
    print(
        f"计划: {len(points)} 个采样点 × {args.seeds} 种子 = {len(tasks)} 次仿真, {args.jobs} 并行"
    )

    t0 = time.time()
    results, errors = {}, []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.jobs, initializer=_worker_init) as pool:
        done = 0
        for tid, m, err in pool.imap_unordered(_run_once, tasks):
            done += 1
            if err:
                errors.append((tid, err))
                print(f"[{done}/{len(tasks)}] task{tid} FAILED: {err}")
            else:
                results[tid] = m
                print(
                    f"[{done}/{len(tasks)}] task{tid} ok steps={m['total_steps']:.0f} wall={m['_wall_s']:.0f}s"
                )
    print(f"完成: {(time.time()-t0)/60:.1f} 分钟, 失败 {len(errors)}")

    base_rows = [results[i] for i in index["baseline"] if results.get(i)]
    base = {k: mean_of(base_rows, k) for k in ALL_KEYS}

    def elasticity(label, rel, key):
        """Compute elasticity for one label and metric."""
        rows = [results[i] for i in index[label] if results.get(i)]
        yp, yb = mean_of(rows, key), base[key]
        if math.isnan(yp) or math.isnan(yb) or rel == 0:
            return float("nan")
        if abs(yb) < 1e-9:
            return float("nan")
        return abs((yp - yb) / yb) / rel

    point_meta = {
        label: (param, value, rel) for label, param, value, rel in points if param
    }
    s1_path = OUT_DIR / "S1_param_to_mech.csv"
    sx_path = OUT_DIR / "S_extended_param_to_output.csv"
    for path, keylist in [(s1_path, MECH_KEYS), (sx_path, OUT_KEYS)]:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["parameter"] + [k for k, _ in keylist])
            for p in S1_PARAMS:
                labels = [l for l in point_meta if point_meta[l][0] == p]
                row = []
                for k, _cn in keylist:
                    es = [elasticity(l, point_meta[l][2], k) for l in labels]
                    es = [e for e in es if not math.isnan(e)]
                    row.append(f"{sum(es)/len(es):.4f}" if es else "")
                w.writerow([p] + row)

    all_rows = [r for r in results.values() if r]
    s2_path = OUT_DIR / "S2_mech_to_output.csv"
    with open(s2_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mechanism_var"] + OUT_KEYLIST)
        for mk in MECH_KEYLIST:
            mv = [r[mk] for r in all_rows]
            row = []
            for ok in OUT_KEYLIST:
                r_ = pearson(mv, [r[ok] for r in all_rows])
                row.append(f"{r_:.3f}" if not math.isnan(r_) else "")
            w.writerow([mk] + row)

    sweep_keys = [
        "mech_memory_active_rate",
        "mech_retrieval_load_mean",
        "mech_retrieval_load_p90",
        "memory_retrieval_per_100steps",
        "iw_mean",
        "stop_probe_per_100steps",
        "veh_reaction_prob",
        "total_steps",
    ]
    sweep_path = OUT_DIR / "memory_th_sweep.csv"
    sweep_rows = []
    with open(sweep_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["MEMORY_ACTIVE_RETRIEVAL_TH"] + sweep_keys)
        for v in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]:
            label = (
                "baseline"
                if abs(v - 0.15) < 1e-9
                else f"MEMORY_ACTIVE_RETRIEVAL_TH|{v:.2f}"
            )
            rows = [results[i] for i in index[label] if results.get(i)]
            vals = [mean_of(rows, k) for k in sweep_keys]
            sweep_rows.append((v, vals))
            w.writerow([v] + [f"{x:.4f}" for x in vals])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        for fname in ["PingFang SC", "Hiragino Sans GB", "SimHei", "Arial Unicode MS"]:
            if any(
                fname.lower() in ft.name.lower()
                for ft in font_manager.fontManager.ttflist
            ):
                plt.rcParams["font.family"] = fname
                break
        plt.rcParams["axes.unicode_minus"] = False
        xs = [v for v, _ in sweep_rows]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=200)
        for ax, key, cn in [
            (axes[0][0], "mech_memory_active_rate", "记忆激活占比"),
            (axes[0][1], "memory_retrieval_per_100steps", "记忆检索频率/100步"),
            (axes[1][0], "iw_mean", "认知负荷均值"),
            (axes[1][1], "stop_probe_per_100steps", "停探测频率/100步"),
        ]:
            ki = sweep_keys.index(key)
            ax.plot(xs, [vals[ki] for _, vals in sweep_rows], "o-", color="#2166ac")
            ax.axvline(0.15, color="#999", ls="--", lw=0.8)
            ax.set_xlabel("MEMORY_ACTIVE_RETRIEVAL_TH")
            ax.set_title(cn, fontsize=10)
        fig.suptitle(
            "MEMORY_ACTIVE_RETRIEVAL_TH 阈值扫描（虚线=当前默认0.15）", fontsize=11
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / "memory_th_sweep.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("扫描图绘制失败:", e)

    raw_path = OUT_DIR / "mech_runs_raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            ["label", "seed"] + ALL_KEYS + ["n_veh_onsets", "n_lm_onsets", "wall_s"]
        )
        for label, ids in index.items():
            for tid in ids:
                r = results.get(tid)
                if not r:
                    w.writerow([label, "", "FAILED"])
                    continue
                w.writerow(
                    [label, r["_seed"]]
                    + [
                        (
                            f"{r[k]:.5f}"
                            if not (isinstance(r[k], float) and math.isnan(r[k]))
                            else "nan"
                        )
                        for k in ALL_KEYS
                    ]
                    + [
                        r.get("_n_veh_onsets"),
                        r.get("_n_lm_onsets"),
                        f"{r['_wall_s']:.0f}",
                    ]
                )

    print("输出:")
    for p in [
        s1_path,
        sx_path,
        s2_path,
        sweep_path,
        OUT_DIR / "memory_th_sweep.png",
        raw_path,
    ]:
        print("  ", p)
    if errors:
        print("失败任务:", errors)


if __name__ == "__main__":
    main()
