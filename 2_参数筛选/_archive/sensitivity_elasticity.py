"""Run archived elasticity analysis for candidate simulation parameters.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import argparse
import csv
import io
import json
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

PARAMS = {
    "SEEV_VALUE_SAFETY_WEIGHT": (0.70, "a.风险融合", False),
    "SEEV_VALUE_PROGRESS_WEIGHT": (0.30, "a.风险融合", False),
    "SEEV_EXPECTANCY_RISK_WEIGHT": (0.55, "a.风险融合", False),
    "ACTR_RISK_CANE_GUIDANCE_RELIEF": (0.10, "a.风险融合", False),
    "ACTR_RISK_LANDMARK_RELIEF": (0.08, "a.风险融合", False),
    "PROBE_SAFE_STREAK_RELEASE_THRESHOLD": (3, "b.探测恢复", True),
    "ACTR_LOAD_RESUME_THRESHOLD": (5.0, "b.探测恢复", False),
    "PROBE_RELIEF_RATIO": (0.40, "b.探测恢复", False),
    "PROBE_OVERLOAD_STREAK_TH": (3, "b.探测恢复", True),
    "MEMORY_ACTIVE_RETRIEVAL_TH": (0.15, "c.参照丢失", False),
    "MEMORY_ACTIVE_ABSENT_STEPS_TH": (11, "c.参照丢失", True),
    "LANDMARK_DECAY_RATE": (0.82, "c.参照丢失", False),
    "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK": (0.40, "d.车辆逼近", False),
    "LOOMING_BOOST_PEAK": (0.35, "d.车辆逼近", False),
    "LOOMING_BOOST_DECAY": (0.80, "d.车辆逼近", False),
}

PERTURB = 0.20
DEMO_PARAMS = ["SEEV_VALUE_SAFETY_WEIGHT", "PROBE_RELIEF_RATIO"]

METRICS = [
    ("stop_probe_per_100steps", "停探测频率", "stop_and_probe 次数 / 总步数 ×100"),
    ("veh_reaction_prob", "车辆逼近反应概率", "逼近事件起始后3步内出现停探测的比例"),
    (
        "veh_reaction_delay_s",
        "车辆逼近反应延迟",
        "事件起始到首次停探测的平均模拟秒（未反应事件不计）",
    ),
    ("iw_mean", "认知负荷均值", "actr_iw_total 全程均值"),
    ("iw_p95", "认知负荷峰值", "actr_iw_total 的95分位"),
    ("iw_high_ratio", "高负荷占比", "IW≥6.0 的步数占比"),
    ("crossing_wait_s", "路口等待时长", "crossing_subphase==wait 的模拟秒合计"),
    (
        "ref_loss_episodes_per_100steps",
        "参照丢失次数",
        "guidance_absent_steps 0→1 上升沿次数/100步",
    ),
    ("deprivation_mean", "感觉剥夺均值", "deprivation_index 全程均值"),
    ("obstacle_hits_per_100steps", "障碍接触频率", "cane_obstacle 步数/100步"),
    ("total_sim_time_s", "完成时间", "全程模拟秒"),
    ("total_steps", "总步数", "仿真步数"),
]
METRIC_KEYS = [m[0] for m in METRICS]

VEH_REACTION_WINDOW = 3


def _extract_metrics(
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
    """Handle extract metrics behavior."""
    n = len(sim_log)
    if n == 0:
        return {k: float("nan") for k in METRIC_KEYS}

    def col(key, default=0.0):
        """Handle col behavior."""
        return [r.get(key, default) for r in sim_log]

    actions = col("next_action", "")
    sim_times = [float(r.get("sim_time", 0.0)) for r in sim_log]
    iw = [float(r.get("actr_iw_total", 0.0)) for r in sim_log]
    dts = [sim_times[0]] + [sim_times[i] - sim_times[i - 1] for i in range(1, n)]

    stop_probe_count = sum(1 for a in actions if a == "stop_and_probe")

    veh = [bool(r.get("vehicle_approach", False)) for r in sim_log]
    onsets = [i for i in range(n) if veh[i] and (i == 0 or not veh[i - 1])]
    reacted, delays = 0, []
    for i in onsets:
        for j in range(i, min(i + VEH_REACTION_WINDOW + 1, n)):
            if actions[j] == "stop_and_probe":
                reacted += 1
                delays.append(sim_times[j] - sim_times[i])
                break
    veh_reaction_prob = reacted / len(onsets) if onsets else float("nan")
    veh_reaction_delay = sum(delays) / len(delays) if delays else float("nan")

    crossing_wait_s = sum(
        dts[i]
        for i in range(n)
        if sim_log[i].get("crossing_active")
        and sim_log[i].get("crossing_subphase") == "wait"
    )

    gas = [int(r.get("guidance_absent_steps", 0)) for r in sim_log]
    ref_loss_episodes = sum(
        1 for i in range(n) if gas[i] > 0 and (i == 0 or gas[i - 1] == 0)
    )

    obstacle_hits = sum(1 for r in sim_log if bool(r.get("cane_obstacle", False)))

    iw_sorted = sorted(iw)
    iw_p95 = iw_sorted[min(n - 1, int(math.ceil(0.95 * n)) - 1)]

    return {
        "stop_probe_per_100steps": stop_probe_count / n * 100,
        "veh_reaction_prob": veh_reaction_prob,
        "veh_reaction_delay_s": veh_reaction_delay,
        "iw_mean": sum(iw) / n,
        "iw_p95": iw_p95,
        "iw_high_ratio": sum(1 for v in iw if v >= 6.0) / n,
        "crossing_wait_s": crossing_wait_s,
        "ref_loss_episodes_per_100steps": ref_loss_episodes / n * 100,
        "deprivation_mean": sum(float(r.get("deprivation_index", 0.0)) for r in sim_log)
        / n,
        "obstacle_hits_per_100steps": obstacle_hits / n * 100,
        "total_sim_time_s": sim_times[-1],
        "total_steps": n,
        "_reached_goal": str(current_position) == str(goal_node),
        "_n_veh_onsets": len(onsets),
        "_diag_value_mean": sum(float(r.get("value", 0.0)) for r in sim_log) / n,
        "_diag_net_priority_mean": sum(
            float(r.get("net_priority", 0.0)) for r in sim_log
        )
        / n,
        "_diag_gate_rate": sum(1 for r in sim_log if r.get("gate_passed")) / n,
        "_diag_risk_mean": sum(float(r.get("actr_risk_signal", 0.0)) for r in sim_log)
        / n,
    }


DIAG_KEYS = [
    "_diag_value_mean",
    "_diag_net_priority_mean",
    "_diag_gate_rate",
    "_diag_risk_mean",
]


_SIM = None


def _worker_init():
    """Initialize worker-process imports and paths."""
    global _SIM
    import random

    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        sys.path.insert(0, str(ROOT))
        from bvi_sa import simulation as sim
    sim.generate_report = _extract_metrics
    _SIM = sim


def _run_once(task):
    """Execute one simulation task and return metrics."""
    task_id, overrides, seed = task
    global _SIM
    import random

    sim = _SIM
    try:
        for name, (default, _g, _i) in PARAMS.items():
            setattr(sim, name, overrides.get(name, default))
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32 - 1))
        except ImportError:
            pass
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = sim.run_simulation(familiarity_level=1)
        metrics["_wall_s"] = time.time() - t0
        metrics["_seed"] = seed
        return task_id, metrics, None
    except Exception as e:
        return task_id, None, f"{type(e).__name__}: {e}"


def perturbed_value(name, direction):
    """Return a perturbed value for one parameter and direction."""
    default, _group, is_int = PARAMS[name]
    if is_int:
        newv = int(default) + direction
        rel = abs(newv - default) / default
    else:
        newv = default * (1 + direction * PERTURB)
        rel = PERTURB
    return newv, rel


def build_plan(param_names, n_seeds, seed_base=20260705):
    """Build the parameter screening run plan."""
    tasks = []
    index = {}

    def add(label, overrides):
        """Append one labeled parameter override to the run plan."""
        ids = []
        for s in range(n_seeds):
            tid = len(tasks)
            tasks.append((tid, overrides, seed_base + s))
            ids.append(tid)
        index[label] = ids

    add("baseline", {})
    for name in param_names:
        for direction, tag in [(+1, "up"), (-1, "down")]:
            v, _rel = perturbed_value(name, direction)
            add(f"{name}|{tag}", {name: v})
    return tasks, index


def mean_metrics(results, ids):
    """Average metric dictionaries across selected runs."""
    out = {}
    rows = [results[i] for i in ids if results.get(i)]
    for k in METRIC_KEYS:
        vals = [
            r[k]
            for r in rows
            if r and not (isinstance(r[k], float) and math.isnan(r[k]))
        ]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out, len(rows)


def compute_elasticity(param_names, index, results, report_notes):
    """Compute sensitivity elasticities from paired perturbation runs."""
    base, n_base = mean_metrics(results, index["baseline"])
    all_rows = [r for r in results.values() if r]
    scale = {}
    for k in METRIC_KEYS:
        vals = [
            r[k] for r in all_rows if not (isinstance(r[k], float) and math.isnan(r[k]))
        ]
        m = sum(vals) / len(vals) if vals else 1.0
        var = sum((v - m) ** 2 for v in vals) / len(vals) if vals else 1.0
        scale[k] = math.sqrt(var) or (abs(m) or 1.0)

    S = {}
    for name in param_names:
        row = {}
        for k in METRIC_KEYS:
            parts = []
            for direction, tag in [(+1, "up"), (-1, "down")]:
                _v, rel = perturbed_value(name, direction)
                pert, _n = mean_metrics(results, index[f"{name}|{tag}"])
                yb, yp = base[k], pert[k]
                if (
                    isinstance(yb, float)
                    and math.isnan(yb)
                    or isinstance(yp, float)
                    and math.isnan(yp)
                ):
                    continue
                if abs(yb) < 1e-9 or abs(yb) < 0.05 * scale[k]:
                    e = abs(yp - yb) / (scale[k] or 1.0) / rel
                    report_notes.add(
                        f"指标 {k}: 基准值≈0（{yb:.4g}），改用绝对变化/经验尺度（std={scale[k]:.4g}）"
                    )
                else:
                    e = abs((yp - yb) / yb) / rel
                parts.append(e)
            row[k] = sum(parts) / len(parts) if parts else float("nan")
        S[name] = row
    return S, base


def save_outputs(
    param_names, S, base, results, index, notes, elapsed_s, n_seeds, demo=False
):
    """Handle save outputs behavior."""
    suffix = "_demo" if demo else ""
    s_path = OUT_DIR / f"S_matrix{suffix}.csv"
    with open(s_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "group"] + METRIC_KEYS)
        for name in param_names:
            w.writerow(
                [name, PARAMS[name][1]]
                + [
                    f"{S[name][k]:.4f}" if not math.isnan(S[name][k]) else ""
                    for k in METRIC_KEYS
                ]
            )

    a_raw = {}
    for k in METRIC_KEYS:
        vals = [S[n][k] for n in param_names if not math.isnan(S[n][k])]
        a_raw[k] = sum(vals) if vals else 0.0
    total = sum(a_raw.values()) or 1.0
    a_path = OUT_DIR / f"a_scores{suffix}.csv"
    with open(a_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "metric_cn", "a_raw(列和)", "a_normalized"])
        for key, cn, _desc in METRICS:
            w.writerow([key, cn, f"{a_raw[key]:.4f}", f"{a_raw[key]/total:.4f}"])

    heat_path = OUT_DIR / f"heatmap_S{suffix}.png"
    _plot_heatmap(param_names, S, heat_path)

    ident_path = OUT_DIR / f"identifiability{suffix}.csv"
    _identifiability(param_names, S, ident_path)

    raw_path = OUT_DIR / f"runs_raw{suffix}.csv"
    with open(raw_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            ["label", "seed"]
            + METRIC_KEYS
            + [k.lstrip("_") for k in DIAG_KEYS]
            + ["reached_goal", "n_veh_onsets", "wall_s"]
        )
        for label, ids in index.items():
            for tid in ids:
                r = results.get(tid)
                if not r:
                    w.writerow([label, "", "FAILED"])
                    continue
                w.writerow(
                    [label, r.get("_seed")]
                    + [
                        (
                            f"{r[k]:.4f}"
                            if not (isinstance(r[k], float) and math.isnan(r[k]))
                            else "nan"
                        )
                        for k in METRIC_KEYS
                    ]
                    + [f"{r.get(k, float('nan')):.6f}" for k in DIAG_KEYS]
                    + [
                        r.get("_reached_goal"),
                        r.get("_n_veh_onsets"),
                        f"{r.get('_wall_s',0):.1f}",
                    ]
                )

    print(
        f"输出: {s_path}\n      {a_path}\n      {heat_path}\n      {ident_path}\n      {raw_path}"
    )
    return s_path, a_path, heat_path, ident_path


def _plot_heatmap(param_names, S, out_path):
    """Handle plot heatmap behavior."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fname in ["PingFang SC", "Hiragino Sans GB", "SimHei", "Arial Unicode MS"]:
        if any(
            fname.lower() in f.name.lower() for f in font_manager.fontManager.ttflist
        ):
            plt.rcParams["font.family"] = fname
            break
    plt.rcParams["axes.unicode_minus"] = False

    metric_cns = [m[1] for m in METRICS]
    data = [
        [S[p][k] if not math.isnan(S[p][k]) else 0.0 for k in METRIC_KEYS]
        for p in param_names
    ]

    fig, ax = plt.subplots(
        figsize=(1.1 * len(METRIC_KEYS) + 3, 0.55 * len(param_names) + 2), dpi=200
    )
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(metric_cns)))
    ax.set_xticklabels(metric_cns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=8)
    vmax = max((max(row) for row in data), default=1.0) or 1.0
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v > 0.6 * vmax else "#1a3a5c",
            )
    ax.set_title(
        "参数×指标 敏感性矩阵 S（局部弹性，±20%扰动，范围为临时设定待专家确认）",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="弹性 |%Δy| / |%Δθ|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _identifiability(param_names, S, out_path):
    """Handle identifiability behavior."""

    def pearson(x, y):
        """Compute the Pearson correlation coefficient."""
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        sx = math.sqrt(sum((v - mx) ** 2 for v in x))
        sy = math.sqrt(sum((v - my) ** 2 for v in y))
        if sx == 0 or sy == 0:
            return float("nan")
        return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)

    rows_v = {
        p: [S[p][k] if not math.isnan(S[p][k]) else 0.0 for k in METRIC_KEYS]
        for p in param_names
    }
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["param_i", "param_j", "pearson_r", "abs_r>0.9"])
        for i, pi in enumerate(param_names):
            for pj in param_names[i + 1 :]:
                r = pearson(rows_v[pi], rows_v[pj])
                w.writerow(
                    [
                        pi,
                        pj,
                        f"{r:.4f}" if not math.isnan(r) else "nan",
                        "YES" if (not math.isnan(r) and abs(r) > 0.9) else "",
                    ]
                )
        w.writerow([])
        w.writerow(["param", "row_sum(全指标弹性和)", "row_max", "近零行(<0.05)"])
        for p in param_names:
            rs = sum(rows_v[p])
            rm = max(rows_v[p])
            w.writerow([p, f"{rs:.4f}", f"{rm:.4f}", "YES" if rm < 0.05 else ""])


def verify_defaults():
    """Check that runtime defaults match expected analysis baselines."""
    with contextlib.redirect_stdout(io.StringIO()):
        sys.path.insert(0, str(ROOT))
        from bvi_sa import simulation as sim
    bad = []
    for name, (default, _g, _i) in PARAMS.items():
        actual = getattr(sim, name, None)
        if actual is None or abs(float(actual) - float(default)) > 1e-9:
            bad.append((name, default, actual))
    if bad:
        print("⚠️ PARAMS 默认值与代码不一致，请先更新清单：")
        for name, d, a in bad:
            print(f"  {name}: 清单={d} 代码={a}")
        sys.exit(1)
    print("✓ 15个参数默认值与代码一致")


def main():
    """Run the script entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demo", "full"])
    ap.add_argument(
        "--seeds", type=int, default=None, help="每点种子数（demo默认2，full默认5）"
    )
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    demo = args.mode == "demo"
    n_seeds = args.seeds or (2 if demo else 5)
    param_names = DEMO_PARAMS if demo else list(PARAMS.keys())

    verify_defaults()
    tasks, index = build_plan(param_names, n_seeds)
    est_h = len(tasks) * 111 / args.jobs / 3600
    print(
        f"计划: {len(param_names)}参数 ×2方向 ×{n_seeds}种子 + 基准{n_seeds} = {len(tasks)}次仿真"
    )
    print(f"并行: {args.jobs} workers；按111s/次估算 ≈ {est_h:.1f} 小时 wall-clock")

    t0 = time.time()
    results = {}
    errors = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.jobs, initializer=_worker_init) as pool:
        done = 0
        for tid, metrics, err in pool.imap_unordered(_run_once, tasks):
            done += 1
            if err:
                errors.append((tid, err))
                print(f"[{done}/{len(tasks)}] task{tid} FAILED: {err}")
            else:
                results[tid] = metrics
                print(
                    f"[{done}/{len(tasks)}] task{tid} ok "
                    f"steps={metrics['total_steps']:.0f} wall={metrics['_wall_s']:.0f}s"
                )
    elapsed = time.time() - t0
    print(f"全部完成: {elapsed/60:.1f} 分钟, 失败 {len(errors)} 次")

    notes = set()
    S, base = compute_elasticity(param_names, index, results, notes)
    save_outputs(
        param_names, S, base, results, index, notes, elapsed, n_seeds, demo=demo
    )

    if demo:
        print("\n── demo 验证 ──")
        _worker_init()
        _, m1, _ = _run_once((0, {}, 99999))
        _, m2, _ = _run_once((1, {}, 99999))
        _, m3, _ = _run_once((2, {"SEEV_VALUE_SAFETY_WEIGHT": 0.84}, 99999))
        same = all(
            abs(m1[k] - m2[k]) < 1e-9
            for k in METRIC_KEYS + DIAG_KEYS
            if not (math.isnan(m1[k]) or math.isnan(m2[k]))
        )
        diff_internal = any(abs(m1[k] - m3[k]) > 1e-9 for k in DIAG_KEYS)
        diff_behavior = any(
            abs(m1[k] - m3[k]) > 1e-9
            for k in METRIC_KEYS
            if not (math.isnan(m1[k]) or math.isnan(m3[k]))
        )
        print(f"同seed同参数两次运行一致: {'✓' if same else '✗ 不可复现！'}")
        print(
            f"注入到达模型内部(诊断量变化): {'✓' if diff_internal else '✗ 注入无效！'}"
        )
        print(
            f"该参数引起行为指标变化: {'有' if diff_behavior else '无（行为不敏感，弹性行≈0，属正常发现）'}"
        )
        full_tasks = (len(PARAMS) * 2 + 1) * 5
        print(
            f"\n正式版预算: {full_tasks}次 × ~{elapsed/max(1,len(tasks)-len(errors))*args.jobs:.0f}s"
            f" ÷ {args.jobs}核 ≈ {full_tasks * (elapsed/max(1,len(tasks))*args.jobs) / args.jobs / 3600:.1f} 小时"
        )

    if notes:
        print("\n基准≈0处理备注:")
        for x in sorted(notes):
            print(" -", x)
    if errors:
        print("\n失败任务:")
        for tid, e in errors:
            print(f"  task{tid}: {e}")


if __name__ == "__main__":
    main()
