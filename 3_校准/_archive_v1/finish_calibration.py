"""Finalize archived calibration runs and write summary reports.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import csv
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import _setup
import scenario_sensitivity as ss
import calibrate_search as cs

OUT = ROOT / "calib_out"

THETA_CONT = {
    "MEMORY_ACTIVE_ABSENT_STEPS_TH": 18,
    "PROBE_RELIEF_RATIO": 0.50,
    "LANDMARK_DECAY_RATE": 0.82,
    "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK": 0.28,
    "LOOMING_BOOST_PEAK": 0.35,
    "LOOMING_BOOST_DECAY": 0.90,
}
MEMTH_GRID = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

M_CN = {
    "veh_share": "车辆-时长占比",
    "veh_stop_rate": "车辆-停探测率",
    "veh_reaction_prob": "车辆-反应概率",
    "veh_reaction_delay_s": "车辆-反应延迟(s)",
    "lm_share": "地标-时长占比",
    "lm_stop_rate": "地标-停探测率",
    "lm_stop_after_rate": "地标-触发后停探测率",
    "tac_share": "触觉-时长占比",
    "tac_stop_rate": "触觉-停探测率",
    "noref_share": "无参照-时长占比",
    "noref_stop_rate": "无参照-停探测率",
    "noref_ep_len": "无参照-片段步长",
    "noref_retrieval_per100": "无参照-记忆检索率",
}


def mean_metrics(rows):
    """Average metric dictionaries across selected runs."""
    out = {}
    for k in rows[0]:
        if k in ("label", "seed"):
            continue
        vs = [float(r[k]) for r in rows if r[k] not in ("", "nan")]
        out[k] = sum(vs) / len(vs) if vs else float("nan")
    return out


def main():
    """Run the script entry point."""
    obs, weights = cs.load_obs(), cs.load_weights()
    raw = list(csv.DictReader(open(OUT / "calib_runs_raw.csv", encoding="utf-8-sig")))
    by = defaultdict(list)
    for r in raw:
        by[r["label"]].append(r)

    have = {lb for lb in by if lb.startswith("final_memth|")}
    todo = [v for v in MEMTH_GRID if f"final_memth|{v}" not in have]
    if todo:
        jobs = max(1, (os.cpu_count() or 4) - 2)
        tasks = []
        for v in todo:
            ov = dict(THETA_CONT)
            ov["MEMORY_ACTIVE_RETRIEVAL_TH"] = v
            tasks += [(f"final_memth|{v}", ov, sd) for sd in ss.SEEDS]
        print(f"复扫 TH 网格: {todo} × {len(ss.SEEDS)} 种子", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=jobs, initializer=ss._init) as pool, open(
            OUT / "calib_runs_raw.csv", "a", newline="", encoding="utf-8-sig"
        ) as f:
            w = csv.DictWriter(
                f, fieldnames=["label", "seed"] + ss.KEYS, extrasaction="ignore"
            )
            for label, m, err in pool.imap_unordered(ss._run, tasks):
                if err:
                    print(f"  {label} FAILED: {err}", flush=True)
                    continue
                w.writerow(
                    {
                        "label": label,
                        "seed": m["_seed"],
                        **{k: m.get(k) for k in ss.KEYS},
                    }
                )
                f.flush()
                by[label].append(
                    {k: str(m.get(k)) for k in ss.KEYS}
                    | {"label": label, "seed": str(m["_seed"])}
                )
                print(f"  {label} ok", flush=True)

    memth_final = {}
    for v in MEMTH_GRID:
        rows = by.get(f"final_memth|{v}", [])
        if rows:
            m = mean_metrics(rows)
            memth_final[v] = (cs.l_memth(m, obs), cs.l_unified(m, obs, weights), m)
            print(
                f"θ*下 TH={v}: L_memoryTH={memth_final[v][0]:.4f}, L_unified={memth_final[v][1]:.4f}"
            )
    best_th = min(memth_final, key=lambda v: memth_final[v][0])
    print(f"θ*下最优 TH = {best_th}")

    theta_star = dict(THETA_CONT)
    theta_star["MEMORY_ACTIVE_RETRIEVAL_TH"] = best_th
    m_star = memth_final[best_th][2]
    Lu_star, Lm_star = memth_final[best_th][1], memth_final[best_th][0]

    base_rows = [
        r
        for r in csv.DictReader(
            open(
                ROOT / "sensitivity_out" / "scenario_runs_raw.csv", encoding="utf-8-sig"
            )
        )
        if r["label"] == "baseline"
    ]
    m_def = mean_metrics(base_rows)
    Lu_def, Lm_def = cs.l_unified(m_def, obs, weights), cs.l_memth(m_def, obs)

    trace = []
    for lb, rows in sorted(by.items()):
        m = mean_metrics(rows)
        trace.append(
            {
                "label": lb,
                "n_seeds": len(rows),
                "L_unified": round(cs.l_unified(m, obs, weights), 4),
                "L_memoryTH": round(cs.l_memth(m, obs), 4),
            }
        )
    with open(OUT / "calib_trace.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f, fieldnames=["label", "n_seeds", "L_unified", "L_memoryTH"]
        )
        w.writeheader()
        w.writerows(trace)

    order = [
        "MEMORY_ACTIVE_ABSENT_STEPS_TH",
        "PROBE_RELIEF_RATIO",
        "LANDMARK_DECAY_RATE",
        "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK",
        "LOOMING_BOOST_PEAK",
        "LOOMING_BOOST_DECAY",
    ]
    sweeps = {}
    for p in order:
        d = {str(THETA_CONT[p]): m_star}
        for lb, rows in by.items():
            if lb.startswith(f"R2|{p}|"):
                d[lb.split("|")[2]] = mean_metrics(rows)
        sweeps[p] = d
    chosen = {p: str(THETA_CONT[p]) for p in order}
    rng = random.Random(20260708)
    N = 1000
    stable = {p: 0 for p in order}
    th_stable = 0
    for _ in range(N):
        pw = {k: w * rng.uniform(0.8, 1.2) for k, w in weights.items()}
        for p in order:
            best = min(sweeps[p], key=lambda v: cs.l_unified(sweeps[p][v], obs, pw))
            if best == chosen[p]:
                stable[p] += 1
        w1 = min(max(rng.uniform(0.5, 0.7), 0.0), 1.0)
        w2 = 1 - w1

        def lm2(m):
            """Handle lm2 behavior."""
            return w1 * cs.rel_sq_error(
                m["noref_retrieval_per100"], obs["noref_retrieval_per100"]
            ) + w2 * cs.rel_sq_error(m["noref_stop_rate"], obs["noref_stop_rate"])

        bt = min(memth_final, key=lambda v: lm2(memth_final[v][2]))
        if bt == best_th:
            th_stable += 1

    result = {
        "theta_star": theta_star,
        "defaults": {k: ss.DEFAULTS[k] for k in theta_star},
        "L_unified": {"default": Lu_def, "star": Lu_star},
        "L_memoryTH": {"default": Lm_def, "star": Lm_star},
        "metrics_star": {k: m_star.get(k) for k in ss.KEYS},
        "metrics_default": {k: m_def.get(k) for k in ss.KEYS},
        "robustness_selection_stability": {
            **{p: stable[p] / N for p in order},
            "MEMORY_ACTIVE_RETRIEVAL_TH": th_stable / N,
        },
        "memth_regrid_at_star": {
            str(v): {"L_memoryTH": memth_final[v][0], "L_unified": memth_final[v][1]}
            for v in memth_final
        },
    }
    with open(OUT / "theta_star.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(OUT / "CALIBRATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# 参数校准报告（补充材料 1.5 第五步 / 1.7 结果一、二）\n\n")
        f.write(
            "校准数据：obs_template_0708_333（上海/杭州/江阴三地合并，总时长 7550 s），换算口径见 obs_values.csv。\n"
        )
        f.write(
            "搜索：阶段A 阈值网格 + 阶段B 坐标下降（两轮，5 固定种子/点，改善<1% 保留原值），"
            "θ* 下复扫阈值网格确认。全部评估点见 calib_runs_raw.csv / calib_trace.csv。\n\n"
        )
        f.write("## 1. 最终参数表\n\n|参数|默认值|校准值|变化|\n|---|---|---|---|\n")
        for k, v in theta_star.items():
            d = ss.DEFAULTS[k]
            f.write(f"|{k}|{d}|{v}|{'不变' if v == d else f'{d} → {v}'}|\n")
        f.write(
            f"\n损失：L_unified {Lu_def:.4f} → **{Lu_star:.4f}**；L_memoryTH {Lm_def:.4f} → **{Lm_star:.4f}**\n\n"
        )
        f.write(
            "## 2. 实测 vs 仿真指标对比\n\n|指标|实测 obs|默认参数 sim|校准后 sim*|校准后相对误差|\n|---|---|---|---|---|\n"
        )
        for k in [x for x in ss.KEYS if x in obs]:
            o, s0, s1 = obs[k], m_def.get(k), m_star.get(k)
            rel = abs(s1 - o) / abs(o) if abs(o) > 1e-9 else abs(s1 - o)
            f.write(f"|{M_CN.get(k, k)}|{o:.4f}|{s0:.4f}|{s1:.4f}|{rel*100:.1f}%|\n")
        f.write(
            "\n注：三个时长占比类指标（车辆/触觉/无参照）由已冻结的第二类环境参数决定，"
            "高层参数对其弹性近零，为损失中的固定残差；无参照-停探测率实测≈0，按绝对误差计入。\n\n"
        )
        f.write("## 3. θ* 下阈值参数复扫\n\n|TH|L_memoryTH|L_unified|\n|---|---|---|\n")
        for v in MEMTH_GRID:
            if v in memth_final:
                f.write(f"|{v}|{memth_final[v][0]:.4f}|{memth_final[v][1]:.4f}|\n")
        f.write(f"\n最优 TH = {best_th}\n\n")
        f.write(
            "注：TH=0.06/0.07/0.08 损失完全相同（平台区，显式检索停止触发），0.05 以下进入相变区、损失骤增。"
            "形式上取网格最小 0.06，但 0.06 位于悬崖边缘，建议采用平台中点 **0.07** 以留稳健余量（行为完全相同）。\n\n"
        )
        f.write(
            "## 4. 稳健性检验（权重扰动 ±20% × 1000 次，θ* 邻域局部选择稳定性）\n\n|参数|选择稳定率|\n|---|---|\n"
        )
        for p, s in result["robustness_selection_stability"].items():
            f.write(f"|{p}|{s*100:.1f}%|\n")
        f.write(
            "\n## 5. 待办\n\n- 专家审核：用 θ* 运行完整报告管线生成典型行为序列（1.7 第三类结果）\n"
            "- lm_stop_rate 实测口径待确认（'2~3s/3s'），确认后补入重算\n"
            "- L_load（NASA-TLX 分段负荷损失，λ≈0.1）待 TLX 分段数据就绪后加入\n"
        )
    print("报告已写入", OUT / "CALIBRATION_REPORT.md")


if __name__ == "__main__":
    main()
