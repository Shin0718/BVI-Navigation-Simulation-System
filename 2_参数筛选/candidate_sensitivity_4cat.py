"""Screen candidate parameters across four mechanism categories using sensitivity metrics.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import argparse
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
OUT_DIR.mkdir(exist_ok=True)

PERTURB = 0.20
SEED_BASE = 20260705

MEMTH_NAME = "MEMORY_ACTIVE_RETRIEVAL_TH"
MEMTH_WORK_BASE = 0.055
MEMTH_UP, MEMTH_DOWN = 0.066, 0.044

WIDE = {
    "ACTR_LOAD_RESUME_THRESHOLD": [(7.5, 0.5), (2.5, 0.5)],
}

EXCLUDE_FROM_HEATMAP = {"ACTR_RISK_CANE_GUIDANCE_RELIEF"}
LM_WIN = 3
VEH_WIN_REACT = 3
VEH_WIN_SHARE = 5
MIN_SUBSET = 20

CATEGORIES = {
    "a": "风险感知与风险融合",
    "b": "探测与恢复",
    "c": "记忆检索",
    "d": "动态风险",
}
PARAMS = {
    "SEEV_VALUE_SAFETY_WEIGHT": (0.70, "a", False, "安全价值权重"),
    "SEEV_VALUE_PROGRESS_WEIGHT": (0.30, "a", False, "进展价值权重"),
    "SEEV_EXPECTANCY_RISK_WEIGHT": (0.55, "a", False, "风险预期权重"),
    "ACTR_RISK_CANE_GUIDANCE_RELIEF": (0.10, "a", False, "引导对风险的缓解系数"),
    "ACTR_RISK_LANDMARK_RELIEF": (0.08, "a", False, "地标对风险的缓解系数"),
    "PROBE_RELIEF_RATIO": (0.40, "b", False, "探测行为对风险/负荷的缓解比例"),
    "ACTR_LOAD_RESUME_THRESHOLD": (5.0, "b", False, "认知负荷恢复阈值"),
    "MEMORY_ACTIVE_RETRIEVAL_TH": (0.15, "c", False, "主动记忆检索阈值"),
    "MEMORY_ACTIVE_ABSENT_STEPS_TH": (11, "c", True, "参照缺失连续步数阈值"),
    "LANDMARK_DECAY_RATE": (0.82, "c", False, "地标记忆衰减率"),
    "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK": (
        0.40,
        "d",
        False,
        "车辆逼近有效证据显著性阈值",
    ),
    "LOOMING_BOOST_PEAK": (0.35, "d", False, "车辆逼近显著性峰值增量"),
    "LOOMING_BOOST_DECAY": (0.80, "d", False, "车辆逼近显著性每步衰减系数"),
}
DEMO_PARAMS = ["SEEV_VALUE_SAFETY_WEIGHT", "PROBE_RELIEF_RATIO"]

METRICS = [
    ("risk_mean", "Risk", "风险信号均值"),
    ("risk_relief", "Risk relief", "地标触发后风险缓释量"),
    ("probe_rate", "Probe rate", "停探测频率(/100步)"),
    ("response_prob", "Response prob", "车辆逼近反应概率"),
    ("response_delay_s", "Response delay", "车辆逼近反应延迟(s)"),
    ("post_trigger_probe", "Post-trigger probe", "地标触发后停探测率"),
    ("workload", "Workload", "认知负荷均值"),
    ("episode_length", "Episode length", "参照丢失片段步长"),
    ("retrieval_rate", "Retrieval rate", "无参照记忆检索率(/100步)"),
    ("probe_time_share", "Time share (probing)", "停探测时间占比"),
    ("veh_time_share", "Time share (vehicle)", "车辆逼近时长占比"),
]
METRIC_KEYS = [m[0] for m in METRICS]
METRIC_EN = {k: en for k, en, _ in METRICS}
METRIC_CN = {k: cn for k, _, cn in METRICS}

CAT_METRICS = {
    "a": [
        "risk_mean",
        "risk_relief",
        "probe_rate",
        "response_prob",
        "response_delay_s",
    ],
    "b": [
        "probe_rate",
        "post_trigger_probe",
        "workload",
        "episode_length",
        "probe_time_share",
    ],
    "c": [
        "retrieval_rate",
        "episode_length",
        "probe_rate",
        "post_trigger_probe",
        "workload",
    ],
    "d": [
        "response_prob",
        "response_delay_s",
        "probe_rate",
        "workload",
        "veh_time_share",
    ],
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
        return {k: nan for k in METRIC_KEYS}

    acts = [r.get("next_action", "") for r in sim_log]
    st = [float(r.get("sim_time", 0.0)) for r in sim_log]
    iw = [float(r.get("actr_iw_total", 0.0)) for r in sim_log]
    risk = [float(r.get("actr_risk_signal", 0.0)) for r in sim_log]
    dts = [st[0]] + [st[i] - st[i - 1] for i in range(1, n)]
    total_t = st[-1] if st[-1] > 0 else float(n)
    crossing = [bool(r.get("crossing_active")) for r in sim_log]

    probe_steps = [i for i in range(n) if acts[i] == "stop_and_probe"]

    veh = [bool(r.get("vehicle_approach")) for r in sim_log]
    v_on = [i for i in range(n) if veh[i] and (i == 0 or not veh[i - 1])]
    veh_set = set()
    for i in v_on:
        veh_set.update(range(i, min(i + VEH_WIN_SHARE + 1, n)))
    reacted, delays = 0, []
    for i in v_on:
        for j in range(i, min(i + VEH_WIN_REACT + 1, n)):
            if acts[j] == "stop_and_probe":
                reacted += 1
                delays.append(st[j] - st[i])
                break
    response_prob = reacted / len(v_on) if v_on else nan
    response_delay = sum(delays) / len(delays) if delays else nan

    lm = [r.get("matched_landmark", "none") != "none" for r in sim_log]
    lm_on = [i for i in range(n) if lm[i] and (i == 0 or not lm[i - 1])]
    lm_stop_after, reliefs = 0, []
    for i in lm_on:
        if any(acts[j] == "stop_and_probe" for j in range(i, min(i + LM_WIN + 1, n))):
            lm_stop_after += 1
        if i >= 1:
            after = [risk[j] for j in range(i, min(i + LM_WIN + 1, n))]
            reliefs.append(risk[i - 1] - sum(after) / len(after))
    post_trigger_probe = lm_stop_after / len(lm_on) if lm_on else nan
    risk_relief = sum(reliefs) / len(reliefs) if reliefs else nan

    gas = [int(r.get("guidance_absent_steps", 0)) for r in sim_log]
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
    noref_idx = [i for i in range(n) if gas[i] > 0 and not crossing[i]]
    retrieval_rate = (
        (
            sum(1 for i in noref_idx if mem[i] and (i == 0 or not mem[i - 1]))
            / len(noref_idx)
            * 100
        )
        if len(noref_idx) >= MIN_SUBSET
        else nan
    )
    noref_stop = (
        (
            sum(1 for i in noref_idx if acts[i] == "stop_and_probe")
            / len(noref_idx)
            * 100
        )
        if len(noref_idx) >= MIN_SUBSET
        else nan
    )

    return {
        "risk_mean": sum(risk) / n,
        "risk_relief": risk_relief,
        "probe_rate": len(probe_steps) / n * 100,
        "response_prob": response_prob,
        "response_delay_s": response_delay,
        "post_trigger_probe": post_trigger_probe,
        "workload": sum(iw) / n,
        "episode_length": sum(eps) / len(eps) if eps else nan,
        "retrieval_rate": retrieval_rate,
        "probe_time_share": sum(dts[i] for i in probe_steps) / total_t,
        "veh_time_share": len(veh_set) / n,
        "_noref_stop_rate": noref_stop,
        "_total_steps": n,
        "_n_veh_onsets": len(v_on),
        "_n_lm_onsets": len(lm_on),
        "_reached_goal": str(current_position) == str(goal_node),
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
    task_id, overrides, seed = task
    import random

    sim = _SIM
    try:
        for name, (default, _c, _i, _d) in PARAMS.items():
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
    default, _c, is_int, _d = PARAMS[name]
    if is_int:
        newv = int(default) + direction
        rel = abs(newv - default) / default
    else:
        newv = default * (1 + direction * PERTURB)
        rel = PERTURB
    return newv, rel


def build_plan(param_names, n_seeds):
    """Build the parameter screening run plan."""
    tasks, index = [], {}

    def add(label, overrides):
        """Append one labeled parameter override to the run plan."""
        ids = []
        for s in range(n_seeds):
            tid = len(tasks)
            tasks.append((tid, overrides, SEED_BASE + s))
            ids.append(tid)
        index[label] = ids

    add("baseline", {})
    for name in param_names:
        for direction, tag in [(+1, "up"), (-1, "down")]:
            v, _ = perturbed_value(name, direction)
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
    return out


def memth_labels():
    """Return run labels for memory-threshold screening."""
    return [
        ("baseline_memth", {MEMTH_NAME: MEMTH_WORK_BASE}),
        ("memth|up", {MEMTH_NAME: MEMTH_UP}),
        ("memth|down", {MEMTH_NAME: MEMTH_DOWN}),
    ]


def compute_elasticity(param_names, index, results, notes):
    """Compute sensitivity elasticities from paired perturbation runs."""
    base = mean_metrics(results, index["baseline"])
    all_rows = [r for r in results.values() if r]
    scale = {}
    for k in METRIC_KEYS:
        vals = [
            r[k] for r in all_rows if not (isinstance(r[k], float) and math.isnan(r[k]))
        ]
        m = sum(vals) / len(vals) if vals else 1.0
        var = sum((v - m) ** 2 for v in vals) / len(vals) if vals else 1.0
        scale[k] = math.sqrt(var) or (abs(m) or 1.0)

    base_memth = (
        mean_metrics(results, index["baseline_memth"])
        if "baseline_memth" in index
        else None
    )

    S = {}
    for name in param_names:
        row = {}
        if name in WIDE and f"{name}|wide0" in index:
            for k in METRIC_KEYS:
                parts = []
                for wi, (_v, rel) in enumerate(WIDE[name]):
                    lb = f"{name}|wide{wi}"
                    if lb not in index:
                        continue
                    pert = mean_metrics(results, index[lb])
                    yb, yp = base[k], pert[k]
                    if (isinstance(yb, float) and math.isnan(yb)) or (
                        isinstance(yp, float) and math.isnan(yp)
                    ):
                        continue
                    if abs(yb) < 1e-9 or abs(yb) < 0.05 * scale[k]:
                        e = abs(yp - yb) / (scale[k] or 1.0) / rel
                    else:
                        e = abs((yp - yb) / yb) / rel
                    parts.append(e)
                row[k] = sum(parts) / len(parts) if parts else float("nan")
            S[name] = row
            continue
        if name == MEMTH_NAME:
            if base_memth is None:
                notes.add(
                    f"{MEMTH_NAME}: 缺工作区间扰动 run（先跑 memth 模式），本行记 nan"
                )
                S[name] = {k: float("nan") for k in METRIC_KEYS}
                continue
            for k in METRIC_KEYS:
                parts = []
                for tag in ("up", "down"):
                    if f"memth|{tag}" not in index:
                        continue
                    pert = mean_metrics(results, index[f"memth|{tag}"])
                    yb, yp = base_memth[k], pert[k]
                    if (isinstance(yb, float) and math.isnan(yb)) or (
                        isinstance(yp, float) and math.isnan(yp)
                    ):
                        continue
                    if abs(yb) < 1e-9 or abs(yb) < 0.05 * scale[k]:
                        e = abs(yp - yb) / (scale[k] or 1.0) / PERTURB
                    else:
                        e = abs((yp - yb) / yb) / PERTURB
                    parts.append(e)
                row[k] = sum(parts) / len(parts) if parts else float("nan")
            S[name] = row
            continue
        for k in METRIC_KEYS:
            parts = []
            for direction, tag in [(+1, "up"), (-1, "down")]:
                _v, rel = perturbed_value(name, direction)
                if f"{name}|{tag}" not in index:
                    continue
                pert = mean_metrics(results, index[f"{name}|{tag}"])
                yb, yp = base[k], pert[k]
                if (isinstance(yb, float) and math.isnan(yb)) or (
                    isinstance(yp, float) and math.isnan(yp)
                ):
                    continue
                if abs(yb) < 1e-9 or abs(yb) < 0.05 * scale[k]:
                    e = abs(yp - yb) / (scale[k] or 1.0) / rel
                    notes.add(
                        f"指标 {k}: 基准≈0（{yb:.4g}），改用绝对变化/经验尺度（std={scale[k]:.4g}）"
                    )
                else:
                    e = abs((yp - yb) / yb) / rel
                parts.append(e)
            row[k] = sum(parts) / len(parts) if parts else float("nan")
        S[name] = row
    return S, base


def _setup_font():
    """Configure plotting fonts for generated figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fn in ["PingFang SC", "Hiragino Sans GB", "SimHei", "Arial Unicode MS"]:
        if any(fn.lower() in t.name.lower() for t in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fn
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _key_indicators(p, cat, S):
    """Select key indicators for a parameter category."""
    if S is None or p not in S:
        return ", ".join(METRIC_EN[k] for k in CAT_METRICS[cat])
    hits = [
        (k, S[p][k])
        for k in CAT_METRICS[cat]
        if not math.isnan(S[p][k]) and S[p][k] >= 0.1
    ]
    if not hits:
        return "— (所有关联指标弹性 < 0.1)"
    hits.sort(key=lambda x: -x[1])
    return ", ".join(f"{METRIC_EN[k]} ({v:.2f})" for k, v in hits)


def save_association_tables(param_names, S=None, suffix=""):
    """Write association tables for selected sensitivity results."""
    md_path = OUT_DIR / f"cat4_association_tables{suffix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 候选参数与行为指标关联表（Supplementary 0712）\n\n")
        for cat, cname, plist, keys in figure_groups(param_names):
            if not plist:
                continue
            ind_en = [METRIC_EN[k] for k in keys]
            f.write(f"## {cat}. {cname}\n\n")
            f.write(f"该组行为指标集：{'、'.join(ind_en)}\n\n")
            f.write(
                "|Parameter|Describe|Value|Key associated indicators (弹性≥0.1)|\n|---|---|---|---|\n"
            )
            for p in plist:
                d, _c, _i, desc = PARAMS[p]
                f.write(f"|{p}|{desc}|{d}|{_key_indicators(p, PARAMS[p][1], S)}|\n")
            if cat == "a":
                f.write(
                    "\n注：三个 SEEV 权重机制耦合（SAFETY+PROGRESS=1），"
                    "敏感性筛选与校准中作为一个组合单元处理。\n"
                )
            f.write("\n")

    plt = _setup_font()
    for cat, cname, plist, keys in figure_groups(param_names):
        if not plist:
            continue
        cols = ["Parameter", "Describe", "Value", "Key associated indicators (S≥0.1)"]
        cell = [
            [p, PARAMS[p][3], str(PARAMS[p][0]), _key_indicators(p, PARAMS[p][1], S)]
            for p in plist
        ]
        fig, ax = plt.subplots(figsize=(17, 0.6 * len(plist) + 1.6), dpi=200)
        ax.axis("off")
        tb = ax.table(
            cellText=cell,
            colLabels=cols,
            loc="center",
            cellLoc="left",
            colWidths=[0.25, 0.15, 0.04, 0.56],
        )
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 1.5)
        for j in range(len(cols)):
            tb[0, j].set_facecolor("#e8eef7")
            tb[0, j].set_text_props(weight="bold")
        ax.set_title(f"{cat}. {cname} — 候选参数与关联行为指标", fontsize=11, pad=12)
        fig.tight_layout()
        fig.savefig(
            OUT_DIR / f"cat4_association_{cat}{suffix}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


AB_HEATMAP_KEYS = [
    "risk_mean",
    "risk_relief",
    "probe_rate",
    "response_prob",
    "response_delay_s",
    "probe_time_share",
]


def figure_groups(param_names):
    """Group parameters into figure panels."""

    def pick(cats):
        """Select parameters matching the requested category set."""
        return [
            p
            for p in param_names
            if PARAMS[p][1] in cats and p not in EXCLUDE_FROM_HEATMAP
        ]

    return [
        ("a", "风险感知与风险融合 + 探测与恢复", pick(("a", "b")), AB_HEATMAP_KEYS),
        ("b", CATEGORIES["c"], pick(("c",)), CAT_METRICS["c"]),
        ("c", CATEGORIES["d"], pick(("d",)), CAT_METRICS["d"]),
    ]


def save_heatmaps(param_names, S, suffix=""):
    """Render heatmaps from the sensitivity matrix."""
    plt = _setup_font()
    for cat, cname, plist, keys in figure_groups(param_names):
        if not plist:
            continue
        data = [
            [S[p][k] if not math.isnan(S[p][k]) else 0.0 for k in keys] for p in plist
        ]
        fig, ax = plt.subplots(
            figsize=(1.6 * len(keys) + 3, 0.6 * len(plist) + 2), dpi=200
        )
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(
            [f"{METRIC_EN[k]}\n{METRIC_CN[k]}" for k in keys], fontsize=8
        )
        ax.set_yticks(range(len(plist)))
        ax.set_yticklabels(plist, fontsize=8)
        vmax = max((max(r) for r in data), default=1.0) or 1.0
        for i, row in enumerate(data):
            for j, v in enumerate(row):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 0.6 * vmax else "#1a3a5c",
                )
        wide_here = [p for p in plist if p in WIDE]
        note = "；标*参数为放大扰动（±50%）" if wide_here else ""
        ax.set_yticklabels([p + " *" if p in WIDE else p for p in plist], fontsize=8)
        ax.set_title(
            f"{cat}. {cname} — 参数×指标敏感性（弹性 |%Δy|/|%Δθ|，±20%/整数±1档{note}）",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="elasticity")
        fig.tight_layout()
        fig.savefig(
            OUT_DIR / f"cat4_heatmap_{cat}{suffix}.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig)


def save_tables(param_names, S, base, results, index, suffix="", write_raw=True):
    """Write sensitivity matrices, screening tables, and raw outputs."""
    with open(
        OUT_DIR / f"cat4_S_matrix{suffix}.csv", "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow(["parameter", "category"] + METRIC_KEYS)
        for p in param_names:
            w.writerow(
                [p, CATEGORIES[PARAMS[p][1]]]
                + [
                    f"{S[p][k]:.4f}" if not math.isnan(S[p][k]) else ""
                    for k in METRIC_KEYS
                ]
            )

    with open(
        OUT_DIR / f"cat4_indicator_scores{suffix}.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["metric", "metric_en", "metric_cn", "col_sum(全参数)", "weight_normalized"]
        )
        col = {}
        for k in METRIC_KEYS:
            vals = [S[p][k] for p in param_names if not math.isnan(S[p][k])]
            col[k] = sum(vals) if vals else 0.0
        total = sum(col.values()) or 1.0
        for k in METRIC_KEYS:
            w.writerow(
                [k, METRIC_EN[k], METRIC_CN[k], f"{col[k]:.4f}", f"{col[k]/total:.4f}"]
            )

    SEEV_GROUP = [
        "SEEV_VALUE_SAFETY_WEIGHT",
        "SEEV_VALUE_PROGRESS_WEIGHT",
        "SEEV_EXPECTANCY_RISK_WEIGHT",
    ]
    SEEV_UNIT = "SEEV_VALUE_WEIGHTS(组合)"
    ELAST_TH = 0.10
    STANDALONE = {MEMTH_NAME}

    sel_path = OUT_DIR / f"cat4_screening{suffix}.csv"
    selected = {}
    with open(sel_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "category",
                "unit(校准单元)",
                "members",
                "row_sum(类内指标)",
                "mean_elasticity(类内均值)",
                "share_in_category",
                f"selected(阀门:均值≥{ELAST_TH})",
                "note",
            ]
        )
        for cat, cname in CATEGORIES.items():
            plist = [p for p in param_names if PARAMS[p][1] == cat]
            if not plist:
                continue
            if cat == "a" and all(p in plist for p in SEEV_GROUP):
                units = [(SEEV_UNIT, SEEV_GROUP)] + [
                    (p, [p]) for p in plist if p not in SEEV_GROUP
                ]
            else:
                units = [(p, [p]) for p in plist]

            rs = {}
            for uname, members in units:
                sums = []
                for p in members:
                    vals = [
                        S[p][k] for k in CAT_METRICS[cat] if not math.isnan(S[p][k])
                    ]
                    sums.append(sum(vals) if vals else 0.0)
                rs[uname] = sum(sums) / len(sums)

            standalone = [(u, m) for u, m in units if u in STANDALONE]
            competing = [(u, m) for u, m in units if u not in STANDALONE]
            tot = sum(rs[u] for u, _ in competing) or 1.0
            n_ind = len(CAT_METRICS[cat])
            mean_e = {u: rs[u] / n_ind for u, _ in units}

            for uname, members in standalone:
                keep = mean_e[uname] >= ELAST_TH
                if keep:
                    selected.setdefault(cat, []).append(uname)
                w.writerow(
                    [
                        f"{cat}.{cname}",
                        uname,
                        "+".join(members),
                        f"{rs[uname]:.4f}",
                        f"{mean_e[uname]:.4f}",
                        "单列",
                        "YES" if keep else "",
                        (
                            (
                                f"阈值型单列：默认0.15为非触发区，弹性在工作区间中点"
                                f"{MEMTH_WORK_BASE}±20%（独立基准）下测得；"
                                f"全量程扫描确认工作区间[0.03,0.08]"
                            )
                            if keep
                            else "阈值型单列：工作区间扰动下仍低于阀门，不入选"
                        ),
                    ]
                )

            passing = [u for u, _ in competing if mean_e[u] >= ELAST_TH]
            fallback = None
            if not passing and competing:
                fallback = max(competing, key=lambda u: rs[u[0]])[0]
                if mean_e[fallback] <= 1e-9:
                    fallback = None
            ranked = sorted(competing, key=lambda u: -rs[u[0]])
            for uname, members in ranked:
                keep = uname in passing or uname == fallback
                if keep:
                    selected.setdefault(cat, []).append(uname)
                if uname == fallback:
                    note = "低于阀门但为类内最高，按'每类至少保留一个'保底入选"
                elif not keep and mean_e[uname] <= 1e-9:
                    note = "扰动下弹性为0（局部不敏感/结构性不可辨识），不入选"
                elif not keep:
                    note = f"平均弹性低于阀门{ELAST_TH}，固定为专家默认值"
                elif uname == SEEV_UNIT:
                    note = "三权重机制耦合(SAFETY+PROGRESS=1)，合并为一个校准单元"
                else:
                    note = ""
                w.writerow(
                    [
                        f"{cat}.{cname}",
                        uname,
                        "+".join(members),
                        f"{rs[uname]:.4f}",
                        f"{mean_e[uname]:.4f}",
                        f"{rs[uname]/tot:.4f}",
                        "YES" if keep else "",
                        note,
                    ]
                )

    if not write_raw:
        return selected
    with open(
        OUT_DIR / f"cat4_runs_raw{suffix}.csv", "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["label", "seed"]
            + METRIC_KEYS
            + ["total_steps", "n_veh_onsets", "n_lm_onsets", "reached_goal", "wall_s"]
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
                            f"{r[k]:.5f}"
                            if not (isinstance(r[k], float) and math.isnan(r[k]))
                            else "nan"
                        )
                        for k in METRIC_KEYS
                    ]
                    + [
                        r.get("_total_steps"),
                        r.get("_n_veh_onsets"),
                        r.get("_n_lm_onsets"),
                        r.get("_reached_goal"),
                        f"{r.get('_wall_s', 0):.1f}",
                    ]
                )
    return selected


def verify_defaults():
    """Check that runtime defaults match expected analysis baselines."""
    with contextlib.redirect_stdout(io.StringIO()):
        sys.path.insert(0, str(ROOT))
        from bvi_sa import simulation as sim
    bad = []
    for name, (default, _c, _i, _d) in PARAMS.items():
        actual = getattr(sim, name, None)
        if actual is None or abs(float(actual) - float(default)) > 1e-9:
            bad.append((name, default, actual))
    if bad:
        print(f"ℹ️ {len(bad)} 个参数的代码默认值 ≠ 清单参考值（预期：代码已写回 θ*）：")
        for name, d, a in bad:
            print(f"  {name}: 清单(校准前)={d} 代码(θ*)={a}")
        print("  分析按清单参考值显式注入，不受影响。")
    else:
        print(f"✓ {len(PARAMS)}个候选参数默认值与代码一致")


def load_raw(path):
    """Load raw sensitivity rows from disk."""
    results, index = {}, {}
    with open(path, encoding="utf-8-sig") as f:
        for tid, row in enumerate(csv.DictReader(f)):
            m = {}
            for k in METRIC_KEYS:
                v = row.get(k, "nan")
                m[k] = float("nan") if v in ("", "nan", None) else float(v)
            m["_seed"] = row.get("seed")
            m["_total_steps"] = row.get("total_steps")
            m["_n_veh_onsets"] = row.get("n_veh_onsets")
            m["_n_lm_onsets"] = row.get("n_lm_onsets")
            m["_reached_goal"] = row.get("reached_goal")
            results[tid] = m
            index.setdefault(row["label"], []).append(tid)
    return results, index


def reanalyze():
    """Recompute sensitivity outputs from existing raw runs."""
    raw = OUT_DIR / "cat4_runs_raw.csv"
    if not raw.exists():
        sys.exit(f"未找到 {raw}，请先跑 full")
    results, index = load_raw(raw)
    param_names = list(PARAMS.keys())
    notes = set()
    S, base = compute_elasticity(param_names, index, results, notes)
    save_association_tables(param_names, S)
    save_heatmaps(param_names, S)
    selected = save_tables(param_names, S, base, results, index, write_raw=False)
    print("已从 cat4_runs_raw.csv 重算分析与图表（未重跑仿真）")
    print("\n每类阀门筛选结果:")
    for cat, plist in selected.items():
        print(f"  {cat}.{CATEGORIES[cat]}: {', '.join(plist)}")
    if notes:
        print("\n备注:")
        for x in sorted(notes):
            print(" -", x)


def run_memth(jobs):
    """Run memory-threshold follow-up jobs."""
    raw = OUT_DIR / "cat4_runs_raw.csv"
    if not raw.exists():
        sys.exit("未找到 cat4_runs_raw.csv，请先跑 full")
    verify_defaults()
    labels = memth_labels()
    tasks, index = [], {}
    for label, ov in labels:
        ids = []
        for s in range(5):
            tid = len(tasks)
            tasks.append((tid, ov, SEED_BASE + s))
            ids.append(tid)
        index[label] = ids
    print(
        f"MEMORY_TH 工作区间扰动: {[lb for lb, _ in labels]} ×5种子 = {len(tasks)} 次仿真, "
        f"{jobs} 并行",
        flush=True,
    )

    results, errors = {}, []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=_worker_init) as pool:
        done = 0
        for tid, m, err in pool.imap_unordered(_run_once, tasks):
            done += 1
            if err:
                errors.append((tid, err))
                print(f"[{done}/{len(tasks)}] task{tid} FAILED: {err}", flush=True)
            else:
                results[tid] = m
                print(
                    f"[{done}/{len(tasks)}] task{tid} ok steps={m['_total_steps']} "
                    f"wall={m['_wall_s']:.0f}s",
                    flush=True,
                )
    if errors:
        print(f"⚠️ {len(errors)} 次失败，中止并入")
        sys.exit(1)

    memlabels = {lb for lb, _ in labels}
    with open(raw, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        header = next(r)
        keep_rows = [row for row in r if row and row[0] not in memlabels]
    with open(raw, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(keep_rows)
        for label, ids in index.items():
            for tid in ids:
                m = results[tid]
                w.writerow(
                    [label, m.get("_seed")]
                    + [
                        (
                            f"{m[k]:.5f}"
                            if not (isinstance(m[k], float) and math.isnan(m[k]))
                            else "nan"
                        )
                        for k in METRIC_KEYS
                    ]
                    + [
                        m.get("_total_steps"),
                        m.get("_n_veh_onsets"),
                        m.get("_n_lm_onsets"),
                        m.get("_reached_goal"),
                        f"{m.get('_wall_s', 0):.1f}",
                    ]
                )
    print(f"已并入 {len(tasks)} 行到 {raw}，开始重算分析…", flush=True)
    reanalyze()


def run_widen(jobs):
    """Run widened perturbation follow-up jobs."""
    raw = OUT_DIR / "cat4_runs_raw.csv"
    if not raw.exists():
        sys.exit("未找到 cat4_runs_raw.csv，请先跑 full")
    verify_defaults()
    labels = []
    for name, pts in WIDE.items():
        for wi, (v, _rel) in enumerate(pts):
            labels.append((f"{name}|wide{wi}", {name: v}))
    tasks, index = [], {}
    for label, ov in labels:
        ids = []
        for s in range(5):
            tid = len(tasks)
            tasks.append((tid, ov, SEED_BASE + s))
            ids.append(tid)
        index[label] = ids
    print(
        f"放大扰动重测: {[lb for lb, _ in labels]} ×5种子 = {len(tasks)} 次仿真, "
        f"{jobs} 并行",
        flush=True,
    )

    results, errors = {}, []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=_worker_init) as pool:
        done = 0
        for tid, m, err in pool.imap_unordered(_run_once, tasks):
            done += 1
            if err:
                errors.append((tid, err))
                print(f"[{done}/{len(tasks)}] task{tid} FAILED: {err}", flush=True)
            else:
                results[tid] = m
                print(
                    f"[{done}/{len(tasks)}] task{tid} ok steps={m['_total_steps']} "
                    f"wall={m['_wall_s']:.0f}s",
                    flush=True,
                )
    if errors:
        print(f"⚠️ {len(errors)} 次失败，中止并入")
        sys.exit(1)

    memlabels = {lb for lb, _ in labels}
    with open(raw, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        header = next(r)
        keep_rows = [row for row in r if row and row[0] not in memlabels]
    with open(raw, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(keep_rows)
        for label, ids in index.items():
            for tid in ids:
                m = results[tid]
                w.writerow(
                    [label, m.get("_seed")]
                    + [
                        (
                            f"{m[k]:.5f}"
                            if not (isinstance(m[k], float) and math.isnan(m[k]))
                            else "nan"
                        )
                        for k in METRIC_KEYS
                    ]
                    + [
                        m.get("_total_steps"),
                        m.get("_n_veh_onsets"),
                        m.get("_n_lm_onsets"),
                        m.get("_reached_goal"),
                        f"{m.get('_wall_s', 0):.1f}",
                    ]
                )
    print(f"已并入 {len(tasks)} 行到 {raw}，开始重算分析…", flush=True)
    reanalyze()


def main():
    """Run the script entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demo", "full", "memth", "widen", "reanalyze"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    if args.mode == "reanalyze":
        reanalyze()
        return
    if args.mode == "memth":
        run_memth(args.jobs)
        return
    if args.mode == "widen":
        run_widen(args.jobs)
        return

    demo = args.mode == "demo"
    n_seeds = args.seeds or (2 if demo else 5)
    param_names = DEMO_PARAMS if demo else list(PARAMS.keys())
    suffix = "_demo" if demo else ""

    verify_defaults()
    tasks, index = build_plan(param_names, n_seeds)
    print(
        f"计划: {len(param_names)}参数 ×2方向 ×{n_seeds}种子 + 基准{n_seeds} = {len(tasks)}次仿真, "
        f"{args.jobs} 并行",
        flush=True,
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
                print(f"[{done}/{len(tasks)}] task{tid} FAILED: {err}", flush=True)
            else:
                results[tid] = m
                print(
                    f"[{done}/{len(tasks)}] task{tid} ok steps={m['_total_steps']} "
                    f"wall={m['_wall_s']:.0f}s",
                    flush=True,
                )
    print(f"仿真完成: {(time.time()-t0)/60:.1f} 分钟, 失败 {len(errors)}", flush=True)

    notes = set()
    S, base = compute_elasticity(param_names, index, results, notes)
    save_association_tables(param_names, S, suffix)
    save_heatmaps(param_names, S, suffix)
    selected = save_tables(param_names, S, base, results, index, suffix)

    print("\n输出（sensitivity_out/）:")
    print(
        f"  cat4_association_tables{suffix}.md + cat4_association_{{a,b,c,d}}{suffix}.png"
    )
    print(f"  cat4_heatmap_{{a,b,c,d}}{suffix}.png")
    print(
        f"  cat4_S_matrix{suffix}.csv / cat4_indicator_scores{suffix}.csv / "
        f"cat4_screening{suffix}.csv / cat4_runs_raw{suffix}.csv"
    )
    if not demo:
        print("\n每类阀门筛选结果（类内 top-2）:")
        for cat, plist in selected.items():
            print(f"  {cat}.{CATEGORIES[cat]}: {', '.join(plist)}")
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
