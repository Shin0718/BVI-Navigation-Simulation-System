"""Create validation figures from calibration and holdout outputs.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import csv
import json
import math
import random
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
obs = cs2.load_obs()
weights = cs2.load_weights()
means = label_means(OUT / "calib_runs_raw.csv")
PANEL_NAME = {"SEEV_SAFETY_UNIT": "SEEV_VALUE_WEIGHTS"}


def robustness_fig(pct, fname):
    """Handle robustness fig behavior."""
    N = 1000
    rng = random.Random(20260713)
    theta = dict(ts["theta_star"])
    win = {}
    for pname, _g in cs2.ROUND1:
        cand = {theta.get(pname, cs2.UNIT_DEFAULTS[pname]): means["theta_star"]}
        prefix = f"R2|{pname}|"
        for lb, m in means.items():
            if lb.startswith(prefix):
                cand[float(lb[len(prefix) :])] = m
        win[pname] = {v: 0 for v in cand}
        for _ in range(N):
            w2 = {k: w * rng.uniform(1 - pct, 1 + pct) for k, w in weights.items()}
            tw = sum(w2.values())
            w2 = {k: w / tw for k, w in w2.items()}
            best = min(cand, key=lambda v: cs2.l_unified(cand[v], obs, w2))
            win[pname][best] += 1
    stab = {p: win[p][theta.get(p, cs2.UNIT_DEFAULTS[p])] / N for p, _ in cs2.ROUND1}

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), dpi=200)
    for idx, (ax, (pname, _g)) in enumerate(zip(axes.flat, cs2.ROUND1)):
        vals = sorted(win[pname])
        shares = [win[pname][v] / N * 100 for v in vals]
        star_v = theta.get(pname, cs2.UNIT_DEFAULTS[pname])
        colors = ["#1a5fa8" if abs(v - star_v) < 1e-9 else "#b8cbe3" for v in vals]
        bars = ax.bar([str(v) for v in vals], shares, color=colors, width=0.55)
        for b, s in zip(bars, shares):
            if s > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    s + 1.5,
                    f"{s:.0f}%",
                    ha="center",
                    fontsize=8,
                )
        ax.set_ylim(0, 115)
        ax.set_ylabel("被选为最优的比例 (%)", fontsize=8)
        ax.set_title(f"({chr(97+idx)}) {PANEL_NAME.get(pname, pname)}", fontsize=9)
        ax.tick_params(labelsize=8)
    all100 = all(abs(v - 1.0) < 1e-9 for v in stab.values())
    note = (
        "全部单元选择稳定率 100%"
        if all100
        else "稳定率: "
        + ", ".join(
            f"{PANEL_NAME.get(p, p).split('_')[0]}={v*100:.0f}%"
            for p, v in stab.items()
        )
    )
    fig.suptitle(
        f"参数估计分布：指标权重 ±{int(pct*100)}% 均匀扰动 ×1000 次下，"
        f"θ* 邻域各候选值被选为最优的频率\n（深色 = θ* 取值；{note}）",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGS / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"±{int(pct*100)}% 稳定率:", {p: round(v, 3) for p, v in stab.items()})


robustness_fig(0.20, "figV2_robustness.png")
robustness_fig(0.50, "figV2_robustness_50.png")

tlx = json.load(open(OUT / "tlx_external_validity.json", encoding="utf-8"))
comp = tlx["normalized_comparison"]
segs = [s for s in tlx["tlx_segments"] if s["familiar"]]
model = tlx["model_familiar"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)

names = ["平均负荷", "峰值负荷*", "高负荷比例", "路口-非路口\n差异", "陌生-熟悉\n差异"]
keys = [
    "mean",
    "peak",
    "high_ratio",
    "junction_minus_non",
    "unfamiliar_minus_familiar_mean",
]
x = range(len(keys))
w = 0.34
ax1.bar(
    [i - w / 2 for i in x],
    [comp[k]["tlx"] for k in keys],
    w,
    label="NASA-TLX（实测）",
    color="#e67e22",
    alpha=0.85,
)
ax1.bar(
    [i + w / 2 for i in x],
    [comp[k]["model"] for k in keys],
    w,
    label="模型 θ*（仿真）",
    color="#1a5fa8",
    alpha=0.85,
)
ax1.set_xticks(list(x))
ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel("量程归一值（TLX/100，IW/10）")
ax1.set_title("(a) 四项负荷指标对比", fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(axis="y", ls=":", alpha=0.5)
ax1.text(
    0.02,
    -0.20,
    "* 峰值受粒度影响：模型为逐步瞬时量，TLX 为整段回溯均评，系统性压平段内峰值",
    transform=ax1.transAxes,
    fontsize=7,
    color="#666",
)

model_u = tlx["model_unfamiliar"]
all_segs = tlx["tlx_segments"]
xs, ys, cs_, mk = [], [], [], []
for s in all_segs:
    is_j = "路口" in s["type"]
    m = model if s["familiar"] else model_u
    xs.append(m["iw_junction_mean"] if is_j else m["iw_nonjunction_mean"])
    ys.append(s["composite"])
    cs_.append("#c0392b" if is_j else "#1a5fa8")
    mk.append("o" if s["familiar"] else "^")
rngj = random.Random(7)
xj = [v + rngj.uniform(-0.07, 0.07) for v in xs]
for x_, y_, c_, m_ in zip(xj, ys, cs_, mk):
    ax2.scatter([x_], [y_], c=c_, marker=m_, s=55, alpha=0.8)


def spearman(a, b):
    """Handle spearman behavior."""

    def rank(v):
        """Handle rank behavior."""
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k2 in range(i, j + 1):
                r[order[k2]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


rho_all = spearman(xs, ys)
fam_idx = [i for i, s in enumerate(all_segs) if s["familiar"]]
rho_fam = spearman([xs[i] for i in fam_idx], [ys[i] for i in fam_idx])
for lbl, c in [("路口段", "#c0392b"), ("非路口段", "#1a5fa8")]:
    ax2.scatter([], [], c=c, label=lbl)
ax2.scatter([], [], c="#666", marker="o", label="熟悉")
ax2.scatter([], [], c="#666", marker="^", label="陌生")
ax2.set_xlabel("模型对应情境负荷均值（actr_iw_total）")
ax2.set_ylabel("TLX 分段综合负荷（0–100）")
ax2.set_title(
    f"(b) 分段级一致性（三城全部 {len(all_segs)} 段，Spearman ρ = {rho_all:.2f}；"
    f"熟悉段 ρ = {rho_fam:.2f}）",
    fontsize=10,
)
ax2.legend(fontsize=8)
ax2.grid(ls=":", alpha=0.5)

fig.suptitle(
    "外部效度检验：模型认知负荷输出 vs 实地 NASA-TLX（沪杭澄三城；校准未使用负荷数据）",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(FIGS / "figV2_tlx.png", bbox_inches="tight")
plt.close(fig)


fig, ax3 = plt.subplots(figsize=(7, 5.2), dpi=200)
for x_, y_, c_, m_ in zip(xj, ys, cs_, mk):
    ax3.scatter([x_], [y_], c=c_, marker=m_, s=55, alpha=0.8)
ax3.scatter([], [], c="#c0392b", label="Crossing segments")
ax3.scatter([], [], c="#1a5fa8", label="Non-crossing segments")
ax3.scatter([], [], c="#666", marker="o", label="Familiar route")
ax3.scatter([], [], c="#666", marker="^", label="Unfamiliar route")
ax3.set_xlabel("Model workload in matched condition (actr_iw_total)", fontsize=10)
ax3.set_ylabel("NASA-TLX segment composite workload (0-100)", fontsize=10)
ax3.set_title(
    f"Segment-level consistency between model workload and NASA-TLX\n"
    f"(all {len(all_segs)} segments across three cities, Spearman ρ = {rho_all:.2f}; "
    f"familiar only ρ = {rho_fam:.2f})",
    fontsize=10,
)
ax3.legend(fontsize=8, loc="upper left")
ax3.grid(ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(FIGS / "figV2_tlx_scatter_en.png", bbox_inches="tight")
plt.close(fig)

print(f"Spearman: all={rho_all:.3f}  familiar={rho_fam:.3f}")
print("两图已生成:", FIGS / "figV2_robustness.png", FIGS / "figV2_tlx.png")
