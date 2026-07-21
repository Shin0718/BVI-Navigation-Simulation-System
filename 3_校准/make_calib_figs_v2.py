"""Create publication figures for the second calibration workflow.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "3_校准"))
import calibrate_search_v2 as cs2
from finish_calibration_v2 import label_means

OUT = ROOT / "calib_out_v2"
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fn in ["PingFang SC", "Hiragino Sans GB", "SimHei", "Arial Unicode MS"]:
    if any(fn.lower() in t.name.lower() for t in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fn
        break
plt.rcParams["axes.unicode_minus"] = False

ts = json.load(open(OUT / "theta_star.json", encoding="utf-8"))
fin = json.load(open(OUT / "finish_summary.json", encoding="utf-8"))
obs = cs2.load_obs()
weights = cs2.load_weights()
means = label_means(OUT / "calib_runs_raw.csv")
trace = list(csv.DictReader(open(OUT / "calib_trace.csv", encoding="utf-8-sig")))

CN = {
    "response_prob": "车辆-反应概率",
    "response_delay_s": "车辆-反应延迟",
    "post_trigger_probe": "地标-触发后停探测率",
    "episode_length": "无参照-片段步长",
    "retrieval_rate": "无参照-记忆检索率",
    "veh_time_share": "车辆-时长占比",
}

m_def, m_star = means["finish|default_full"], means["theta_star"]
rows = []
for k in cs2.OBS_MAP:
    o = obs[cs2.OBS_MAP[k]]
    if abs(o) < 1e-9:
        continue
    e0 = abs(m_def[k] - o) / abs(o) * 100
    e1 = abs(m_star[k] - o) / abs(o) * 100
    rows.append((CN[k], e0, e1, weights.get(k, 0)))
rows.sort(key=lambda r: -r[3])
fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 2), dpi=200)
ys = range(len(rows))
for y, (name, e0, e1, w) in zip(ys, rows):
    better = e1 <= e0
    ax.plot([e0, e1], [y, y], color="#bbb", lw=2, zorder=1)
    ax.scatter(
        [e0], [y], color="#9db8d9", s=60, zorder=2, label="默认" if y == 0 else ""
    )
    ax.scatter(
        [e1],
        [y],
        color="#1a5fa8" if better else "#c0392b",
        s=70,
        zorder=3,
        label="θ*" if y == 0 else "",
    )
ax.set_yticks(list(ys))
ax.set_yticklabels([f"{r[0]}  (w={r[3]:.2f})" for r in rows], fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("相对误差 |sim-obs|/|obs| (%)，对数轴")
ax.set_title("各行为指标实测贴合度（默认 → θ*）")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(FIGS / "figV2_dumbbell.png", bbox_inches="tight")
plt.close(fig)

ths = sorted(float(k) for k in fin["rescan"])
lm = [fin["rescan"][str(t)][0] for t in ths]
lu = [fin["rescan"][str(t)][1] for t in ths]
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
ax.plot(ths, lm, "o-", color="#1a5fa8", label="L_memoryTH")
ax.plot(ths, lu, "s--", color="#e67e22", label="L_unified")
ax.set_yscale("log")
plateau = fin["plateau"]
ax.axvspan(
    min(plateau),
    max(plateau),
    color="#1a5fa8",
    alpha=0.10,
    label=f"等损失平台 [{min(plateau)}, {max(plateau)}]",
)
star = ts["theta_star"]["MEMORY_ACTIVE_RETRIEVAL_TH"]
ax.axvline(star, color="#888", ls=":", lw=1)
ax.set_xlabel("MEMORY_ACTIVE_RETRIEVAL_TH")
ax.set_ylabel("损失（对数轴）")
ax.set_title("阈值参数在 θ* 下的网格复扫")
ax.legend(fontsize=9)
ax.grid(ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(FIGS / "figV2_threshold.png", bbox_inches="tight")
plt.close(fig)

steps, labels, updated = [], [], []
cur = None
for r in trace:
    if r["stage"] == "B0":
        cur = float(r["L_unified"])
        steps.append(cur)
        labels.append("起点")
        updated.append(True)
for rnd in ("B-R1", "B-R2"):
    for pname, _g in cs2.ROUND1:
        cands = [
            float(r["L_unified"])
            for r in trace
            if r["stage"] == rnd and r["param"] == pname
        ]
        if not cands:
            continue
        best = min(cands)
        moved = best < cur and (cur - best) / max(cur, 1e-9) >= cs2.IMPROVE_EPS
        if moved:
            cur = best
        steps.append(cur)
        labels.append(
            f"{'R1' if rnd == 'B-R1' else 'R2'}\n{pname.replace('_', chr(10)) if False else pname}"
        )
        updated.append(moved)
fig, ax = plt.subplots(figsize=(11, 4.5), dpi=200)
xs = range(len(steps))
ax.plot(xs, steps, "-", color="#bbb", lw=1.5, zorder=1)
for x, y, u in zip(xs, steps, updated):
    ax.scatter(
        [x],
        [y],
        s=70,
        zorder=2,
        facecolor="#1a5fa8" if u else "white",
        edgecolor="#1a5fa8",
    )
short = [
    (
        l
        if l == "起点"
        else l.split("\n")[0]
        + "\n"
        + l.split("\n")[1]
        .replace("MEMORY_ACTIVE_", "MEM_")
        .replace("VEHICLE_APPROACH_", "")
        .replace("SALIENCE_GATE_SIDEWALK", "GATE")
        .replace("LOOMING_BOOST_", "LOOM_")
        .replace("LANDMARK_DECAY_RATE", "LM_DECAY")
        .replace("PROBE_RELIEF_RATIO", "PROBE_RELIEF")
        .replace("ABSENT_STEPS_TH", "ABSENT_TH")
        .replace("SEEV_SAFETY_UNIT", "SEEV")
    )
    for l in labels
]
ax.set_xticks(list(xs))
ax.set_xticklabels(short, fontsize=7)
ax.set_ylabel("L_unified")
ax.set_title("坐标下降收敛轨迹（实心=更新，空心=简约原则保留；R2 全保留，提前收敛）")
ax.grid(axis="y", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(FIGS / "figV2_trace.png", bbox_inches="tight")
plt.close(fig)

incumbent = {}
cur_L = next(float(r["L_unified"]) for r in trace if r["stage"] == "B0")
cur_v = {p: cs2.UNIT_DEFAULTS[p] for p, _ in cs2.ROUND1}
for pname, _g in cs2.ROUND1:
    incumbent[pname] = (cur_v[pname], cur_L)
    cands = [
        (float(r["value"]), float(r["L_unified"]))
        for r in trace
        if r["stage"] == "B-R1" and r["param"] == pname
    ]
    if cands:
        bv, bl = min(cands, key=lambda x: x[1])
        if bl < cur_L and (cur_L - bl) / max(cur_L, 1e-9) >= cs2.IMPROVE_EPS:
            cur_v[pname], cur_L = bv, bl

units = [p for p, _ in cs2.ROUND1]
PANEL_NAME = {"SEEV_SAFETY_UNIT": "SEEV_VALUE_WEIGHTS"}
fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=200)
for idx, (ax, pname) in enumerate(zip(axes.flat, units)):
    pts = [
        (float(r["value"]), float(r["L_unified"]))
        for r in trace
        if r["stage"] == "B-R1" and r["param"] == pname
    ]
    pts.append(incumbent[pname])
    if pname == "PROBE_RELIEF_RATIO":
        pts = [(v, l) for v, l in pts if v > 0.30]
        for lb, m in means.items():
            if lb.startswith("extend|PROBE_RELIEF|"):
                pts.append(
                    (float(lb.rsplit("|", 1)[1]), cs2.l_unified(m, obs, weights))
                )
        if "theta_star" in means:
            pts.append((0.2, cs2.l_unified(means["theta_star"], obs, weights)))
        if "verify|probe0.3" in means:
            pts.append((0.3, cs2.l_unified(means["verify|probe0.3"], obs, weights)))
    pts = sorted(set(pts))
    if pts:
        ax.plot([v for v, _ in pts], [l for _, l in pts], "o-", color="#1a5fa8", ms=5)
    final_v = ts["theta_star"].get(pname, ts["defaults"][pname])
    ax.axvline(final_v, color="#e67e22", ls=":", lw=1.2)
    y_final = min(
        (l for v, l in pts if abs(v - final_v) < 1e-9), default=min(l for _, l in pts)
    )
    ax.scatter(
        [final_v],
        [y_final],
        marker="o",
        s=90,
        facecolor="none",
        edgecolor="#e67e22",
        lw=2,
        zorder=3,
    )
    ax.set_title(f"({chr(97 + idx)}) {PANEL_NAME.get(pname, pname)}", fontsize=9)
    ax.grid(ls=":", alpha=0.5)
fig.suptitle(
    "各校准单元一维损失剖面（第一轮网格；纵轴条件于扫描时点的参数组；橙=最终值）",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(FIGS / "figV2_profiles.png", bbox_inches="tight")
plt.close(fig)

print("四图已生成:", FIGS)
