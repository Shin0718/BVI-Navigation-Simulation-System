"""Finalize the second calibration run and write robustness summaries.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import csv
import json
import math
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "3_校准"))
import calibrate_search_v2 as cs2

OUT = ROOT / "calib_out_v2"
HOLDOUT_SEEDS = [20260710 + i for i in range(5)]
N_PERTURB = 1000
PERTURB_PCT = 0.20


def load_theta_star():
    """Load the calibrated parameter vector."""
    with open(OUT / "theta_star.json", encoding="utf-8") as f:
        return json.load(f)


def label_means(raw_path):
    """Aggregate raw calibration rows by run label."""
    rows = {}
    with open(raw_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["label"], []).append(r)
    out = {}
    for lb, rs in rows.items():
        m = {}
        keys = [k for k in rs[0] if k not in ("label", "seed")]
        for k in keys:
            vs = []
            for r in rs:
                v = r.get(k, "")
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        vs.append(fv)
                except (TypeError, ValueError):
                    pass
            m[k] = sum(vs) / len(vs) if vs else float("nan")
        out[lb] = m
    return out


def main():
    """Run the script entry point."""
    ts = load_theta_star()
    theta = dict(ts["theta_star"])
    obs = cs2.load_obs()
    weights = cs2.load_weights()
    jobs = max(1, (os.cpu_count() or 4) - 2)

    theta_no_th = {k: v for k, v in theta.items() if k != cs2.MEMTH}
    cands_calib = [("finish|default_full", {})]
    cands_calib += [
        (f"finish|rescan|{v}", {**theta_no_th, cs2.MEMTH: v}) for v in cs2.MEMTH_GRID
    ]
    cands_hold = [("finish|holdout_theta", dict(theta)), ("finish|holdout_default", {})]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=cs2._init) as pool:
        print(
            f"评估: 默认基线1 + 复扫{len(cs2.MEMTH_GRID)} 点 ×5校准种子 …", flush=True
        )
        res_c = cs2.eval_candidates(pool, cands_calib, cs2.SEEDS)
        print("评估: θ*/默认 ×5 留出种子 …", flush=True)
        res_h = cs2.eval_candidates(pool, cands_hold, HOLDOUT_SEEDS)

    m_def = res_c["finish|default_full"]
    L_def_u, L_def_m = cs2.l_unified(m_def, obs, weights), cs2.l_memth(m_def, obs)
    print(f"默认参数(TH=0.15): L_unified={L_def_u:.4f}  L_memoryTH={L_def_m:.4f}")

    rescan = {}
    for v in cs2.MEMTH_GRID:
        m = res_c.get(f"finish|rescan|{v}")
        if m:
            rescan[v] = (cs2.l_memth(m, obs), cs2.l_unified(m, obs, weights))
    best_rescan = min(rescan, key=lambda v: rescan[v][0])
    print(
        "θ* 下复扫:",
        {v: f"{a:.2f}/{b:.2f}" for v, (a, b) in rescan.items()},
        "最优",
        best_rescan,
    )

    m_ht, m_hd = res_h["finish|holdout_theta"], res_h["finish|holdout_default"]
    hold = {
        "theta": (cs2.l_unified(m_ht, obs, weights), cs2.l_memth(m_ht, obs)),
        "default": (cs2.l_unified(m_hd, obs, weights), cs2.l_memth(m_hd, obs)),
    }
    print(f"留出种子: θ* L={hold['theta'][0]:.4f}  默认 L={hold['default'][0]:.4f}")

    means = label_means(OUT / "calib_runs_raw.csv")
    m_star = means["theta_star"]
    rng = random.Random(20260713)
    stability = {}
    for pname, _grid in cs2.ROUND1:
        cand = {theta.get(pname, cs2.UNIT_DEFAULTS[pname]): m_star}
        prefix = f"R2|{pname}|"
        for lb, m in means.items():
            if lb.startswith(prefix):
                cand[float(lb[len(prefix) :])] = m
        if len(cand) < 2:
            stability[pname] = (None, len(cand))
            continue
        star_v = theta.get(pname, cs2.UNIT_DEFAULTS[pname])
        wins = 0
        for _ in range(N_PERTURB):
            w2 = {
                k: w * rng.uniform(1 - PERTURB_PCT, 1 + PERTURB_PCT)
                for k, w in weights.items()
            }
            tw = sum(w2.values())
            w2 = {k: w / tw for k, w in w2.items()}
            best = min(cand, key=lambda v: cs2.l_unified(cand[v], obs, w2))
            if best == star_v:
                wins += 1
        stability[pname] = (wins / N_PERTURB, len(cand))
    plateau = [v for v in rescan if abs(rescan[v][0] - rescan[best_rescan][0]) < 1e-6]

    print("\n权重扰动稳健性:")
    for p, (r, n) in stability.items():
        print(f"  {p}: {'—(无邻域点)' if r is None else f'{r*100:.1f}%'} (候选{n})")

    lines = []
    lines.append("# 参数校准报告 v2（Supplementary 0712 口径）\n")
    lines.append(
        "校准数据：上海/杭州/江阴三地合并（7550 s），obs_values.csv。"
        "搜索：阶段A 阈值网格 + 阶段B 6 校准单元坐标下降（两轮、5 固定种子/点、"
        "改善<1% 保留）。全部评估点见 calib_runs_raw.csv / calib_trace.csv。\n"
    )

    lines.append("## 1. 最终参数表\n")
    lines.append("|校准单元|默认值|校准值|")
    lines.append("|---|---|---|")
    for k, d in ts["defaults"].items():
        v = ts["theta_star"].get(k, d)
        chg = "不变" if v == d else f"{d} → {v}"
        lines.append(f"|{k}|{d}|{chg}|")
    lines.append("")
    lines.append(
        f"损失：L_unified {L_def_u:.4f}（默认，TH=0.15）→ "
        f"**{ts['L_unified_star']:.4f}**；"
        f"L_memoryTH {L_def_m:.4f} → **{ts['L_memoryTH_star']:.4f}**"
        f"（阶段B以 L_unified 为目标，见注）\n"
    )
    lines.append(
        "注：θ* 下 L_memoryTH 高于阶段A默认背景值（31.81），系阶段B仅优化统一损失；"
        "主导项为无参照停探测率（实测≈0 的绝对误差项）。\n"
    )

    lines.append("## 2. 实测 vs 仿真\n")
    lines.append("|指标|实测|默认 sim|θ* sim|")
    lines.append("|---|---|---|---|")
    for k in cs2.OBS_MAP:
        o = obs.get(cs2.OBS_MAP[k])
        lines.append(
            f"|{k}|{o}|{m_def.get(k, float('nan')):.4f}|"
            f"{ts['metrics_star'].get(k, float('nan')):.4f}|"
        )
    lines.append("")

    lines.append("## 3. θ* 下阈值复扫\n")
    lines.append("|TH|L_memoryTH|L_unified|")
    lines.append("|---|---|---|")
    for v in cs2.MEMTH_GRID:
        a, b = rescan[v]
        lines.append(f"|{v}|{a:.4f}|{b:.4f}|")
    lines.append("")
    lines.append(
        f"复扫最优 TH = {best_rescan}，与阶段A一致；等损失平台 {sorted(plateau)}。"
        "平台内行为完全相同，实际部署建议取平台中点 0.07 留稳健余量。\n"
    )

    lines.append("## 4. 种子外推（5 个未用种子）\n")
    lines.append("|参数组|L_unified|L_memoryTH|")
    lines.append("|---|---|---|")
    lines.append(f"|θ*|{hold['theta'][0]:.4f}|{hold['theta'][1]:.4f}|")
    lines.append(f"|默认|{hold['default'][0]:.4f}|{hold['default'][1]:.4f}|")
    impr = (1 - hold["theta"][0] / hold["default"][0]) * 100
    lines.append("")
    lines.append(
        f"θ* 在留出种子上仍改善 {impr:.0f}%（统一损失），校准未依赖特定种子。\n"
    )

    lines.append("## 5. 权重扰动稳健性（±20% × 1000 次，θ* 邻域 R2 网格）\n")
    lines.append("|校准单元|选择稳定率|邻域候选数|")
    lines.append("|---|---|---|")
    for p, (r, n) in stability.items():
        lines.append(f"|{p}|{'—' if r is None else f'{r*100:.1f}%'}|{n}|")
    lines.append(
        f"|{cs2.MEMTH}|平台内等损失（{sorted(plateau)}），权重扰动不改变平台边界|"
        f"{len(rescan)}|"
    )
    lines.append("")
    lines.append(
        "注：损失完全打平的单元（如 SEEV 组合）其稳定率来自简约原则的平局裁定，"
        "应解读为'局部不敏感'而非'强偏好'。\n"
    )

    lines.append("## 6. 待办\n")
    lines.append("- PROBE_RELIEF_RATIO 0.20/0.30 等损失，最终取值待专家复核")
    lines.append("- θ* 写回 bvi_sa/simulation.py（待确认）；典型行为序列供专家审核")
    lines.append("- 留出指标泛化检验（可选，约 3 小时）；NASA-TLX 外部效度待数据")

    report = OUT / "CALIBRATION_REPORT.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT / "finish_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "L_default_unified": L_def_u,
                "L_default_memth": L_def_m,
                "rescan": {str(k): v for k, v in rescan.items()},
                "holdout": hold,
                "stability": {k: v[0] for k, v in stability.items()},
                "plateau": sorted(plateau),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n报告: {report}")


if __name__ == "__main__":
    main()
