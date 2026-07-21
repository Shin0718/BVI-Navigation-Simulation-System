"""Compute archived calibration loss values and observation templates.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
OUT = ROOT / "sensitivity_out"


METRIC_DESC = {
    "veh_share": "整段行程里，'有车辆朝自己逼近'（能听到车驶来，从听到起算约3秒内）的时间占总时长的比例。填0~1小数。",
    "veh_iw_mean": "车辆逼近期间的认知负荷（模型内部量）。实地无法直接观测，请留空，计算时自动跳过。",
    "veh_stop_rate": "车辆逼近的那些时间段里，人'停下探测'的频繁程度。停下探测=双脚停止前进且用杖点探/扫探（哪怕只停1秒）；边走边正常摆杖、只放慢不停都不算。数出次数÷这些时段总秒数×54，即'每100步几次'。",
    "veh_reaction_prob": "所有车辆逼近事件中，行人随即（约2秒内）'停下来'的比例。只算完全停步（通常伴随探测/侧耳判断）；只放慢或侧身不算——仿真侧只有'停下'这一种反应动作，口径须一致。例：共20次车辆逼近，其中7次停了下来，填0.35。",
    "veh_reaction_delay_s": "从听到车辆声音到做出反应平均隔几秒。只统计有反应的事件，单位：秒。",
    "lm_share": "行程中'刚识别到地标'的短时间窗（识别后约2秒内）占总时长的比例。地标=能帮人定位的标志物（店铺声音/气味、特殊路面、熟悉的拐角等）。",
    "lm_iw_mean": "地标识别时刻附近的认知负荷（模型内部量）。实地无法直接观测，请留空。",
    "lm_stop_rate": "识别到地标前后的短时间窗内，停下探测的频繁程度（每100步几次）。停下探测的判定同上：完全停步+用杖探测才算。",
    "lm_relief_effect": "识别到地标后紧张感下降多少（模型内部风险信号的差值）。实地难以直接量化，可留空；若口述数据里有'认出地标后是否安心'可与我们讨论换算。",
    "lm_stop_after_rate": "识别到地标之后随即（约2秒内）停下来确认方位的比例。例：共识别10次地标，其中3次随即停下确认，填0.3。",
    "tac_share": "行程中'盲杖碰着引导物（路缘石/墙/栏杆）或正走在盲道上'的时间占总时长的比例。填0~1小数。",
    "tac_iw_mean": "上述期间的认知负荷（模型内部量）。请留空。",
    "tac_stop_rate": "走盲道/沿引导物期间，停下探测的频繁程度（每100步几次）。判定同上：完全停步+用杖探测才算，正常行走摆杖不算。",
    "tac_risk_mean": "上述期间的风险信号（模型内部量）。请留空。",
    "noref_share": "行程中'完全没有参照'（不在盲道上、盲杖碰不到任何引导物、也没识别到地标）的时间占总时长的比例。",
    "noref_iw_mean": "失参照期间的认知负荷（模型内部量）。请留空。",
    "noref_stop_rate": "失参照期间停下探测的频繁程度（每100步几次）。判定同上：完全停步+用杖探测才算。",
    "noref_ep_len": "平均每次'失去参照'持续多久。数出每段失参照的持续秒数取平均，再乘1.84换算成步数填入；或直接与我们确认换算。",
    "noref_retrieval_per100": "失参照期间，人'主动回忆自己走到哪了'（停顿沉思、明显的定位思考、口述'我想想'）的次数，折算成每100步几次。",
}

UNOBSERVABLE = {
    "veh_iw_mean",
    "lm_iw_mean",
    "lm_relief_effect",
    "tac_iw_mean",
    "tac_risk_mean",
    "noref_iw_mean",
}

MEMTH_WEIGHTS = {
    "noref_retrieval_per100": 0.6,
    "noref_stop_rate": 0.4,
}


def load_weights():
    """Load calibration loss weights."""
    w = {}
    with open(OUT / "loss_weights.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            w[row["metric"]] = {
                "w": float(row["weight_w_k"]),
                "cn": row["metric_cn"],
                "scen": row["scenario"],
            }
    return w


def load_sim_baseline(sim_csv):
    """Load baseline simulation statistics for calibration loss."""
    rows = []
    with open(sim_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["label"] == "baseline":
                rows.append(row)
    if not rows:
        raise SystemExit("sim csv 中未找到 baseline 行")
    keys = [k for k in rows[0] if k not in ("label", "seed")]
    stat = {}
    for k in keys:
        vs = [float(r[k]) for r in rows if r[k] not in ("", "nan")]
        if not vs:
            stat[k] = (float("nan"), float("nan"))
            continue
        m = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs)) if len(vs) > 1 else 0.0
        stat[k] = (m, sd)
    return stat, len(rows)


def rel_sq_error(sim, obs):
    """Compute relative squared error with zero-safe scaling."""
    scale = abs(obs) if abs(obs) > 1e-9 else 1.0
    return ((sim - obs) / scale) ** 2


def make_template(weights, sim_stat):
    """Create an observation template from weighted metrics."""
    path = ROOT / "obs_template.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "loss",
                "metric",
                "metric_cn",
                "白话说明(记录什么、怎么数、什么单位)",
                "scenario",
                "weight",
                "sim_baseline_mean",
                "sim_baseline_std",
                "obs_value(待填)",
                "obs_source(填数据来源便于溯源)",
            ]
        )
        for k, info in weights.items():
            if k in UNOBSERVABLE:
                continue
            loss_tag = "L_unified"
            wt_tag = f"{info['w']:.4f}"
            if k in MEMTH_WEIGHTS:
                loss_tag = "L_unified + L_memoryTH"
                wt_tag = f"{info['w']:.4f} / {MEMTH_WEIGHTS[k]:.1f}"
            m, sd = sim_stat.get(k, (float("nan"), float("nan")))
            w.writerow(
                [
                    loss_tag,
                    k,
                    info["cn"],
                    METRIC_DESC.get(k, ""),
                    info["scen"],
                    wt_tag,
                    f"{m:.4f}",
                    f"{sd:.4f}",
                    "",
                    "",
                ]
            )
    print(f"模板已生成: {path}")
    print(
        "填写说明：obs_value 与 sim 同单位（各 rate 为 /100步 或 比例，见 metric_cn）；"
        "无法观测的指标留空，计算时自动跳过并重归一化。"
    )


def compute(weights, sim_stat, obs_path):
    """Compute calibration loss from observations and simulation statistics."""
    obs = {}
    with open(obs_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = row.get("obs_value(待填)") or row.get("obs_value") or ""
            if v.strip():
                obs[row["metric"]] = float(v)

    print("=" * 62)
    print("L_unified（统一加权损失，7 连续参数用）")
    print("=" * 62)
    avail = {
        k: w
        for k, w in weights.items()
        if k in obs and not math.isnan(sim_stat.get(k, (float("nan"),))[0])
    }
    missing = [k for k in weights if k not in avail]
    total_w = sum(w["w"] for w in avail.values())
    L = 0.0
    print(f"{'指标':<26}{'sim':>10}{'obs':>10}{'D':>10}{'w(重归一)':>10}{'贡献':>10}")
    for k, info in sorted(avail.items(), key=lambda x: -x[1]["w"]):
        sim = sim_stat[k][0]
        d = rel_sq_error(sim, obs[k])
        wn = info["w"] / total_w
        L += wn * d
        print(
            f"{info['cn']:<26}{sim:>10.4f}{obs[k]:>10.4f}{d:>10.4f}{wn:>10.4f}{wn*d:>10.4f}"
        )
    print(f"\nL_unified(θ_default) = {L:.6f}")
    print(
        f"可用指标 {len(avail)}/{len(weights)}；缺实测跳过并重归一化: {len(missing)} 项"
    )
    if missing:
        print("  缺失:", ", ".join(weights[k]["cn"] for k in missing))

    print()
    print("=" * 62)
    print("L_memoryTH（阈值参数专属损失）")
    print("=" * 62)
    Lm, used = 0.0, 0
    for k, wt in MEMTH_WEIGHTS.items():
        if k not in obs:
            print(f"  {k}: 缺实测，跳过")
            continue
        sim = sim_stat[k][0]
        d = rel_sq_error(sim, obs[k])
        Lm += wt * d
        used += 1
        print(f"  {k}: sim={sim:.4f} obs={obs[k]:.4f} D={d:.4f} w={wt} 贡献={wt*d:.4f}")
    if used == len(MEMTH_WEIGHTS):
        print(f"\nL_memoryTH(θ_default) = {Lm:.6f}")
    else:
        print("\nL_memoryTH: 实测不全，未输出总值")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--obs", type=str, default=None)
    ap.add_argument("--sim-csv", type=str, default=str(OUT / "scenario_runs_raw.csv"))
    args = ap.parse_args()

    weights = load_weights()
    sim_stat, n = load_sim_baseline(args.sim_csv)
    print(f"仿真基线: {n} 个种子（来自 {args.sim_csv}）")

    if args.make_template:
        make_template(weights, sim_stat)
    elif args.obs:
        compute(weights, sim_stat, args.obs)
    else:
        print("用 --make-template 生成实测填写模板，或 --obs obs_values.csv 计算损失")
