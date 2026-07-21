"""Create archived calibration figures for the first workflow.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if not (ROOT / "bvi_sa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import _setup
import calibrate_search as cs

FIGS = Path("/Users/ellawu/UCI/research/项目2/参数校准/figs")
FIGS.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": [
            "PingFang SC",
            "Hiragino Sans GB",
            "Arial Unicode MS",
            "STHeiti",
        ],
        "axes.unicode_minus": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#b8b7b0",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "font.size": 9,
    }
)
BLUE, YELLOW, INK, MUT = "#2a78d6", "#eda100", "#0b0b0b", "#52514e"

obs = cs.load_obs()
weights = cs.load_weights()
star = json.load(open(ROOT / "calib_out" / "theta_star.json", encoding="utf-8"))
m_def, m_star = star["metrics_default"], star["metrics_star"]

raw = list(
    csv.DictReader(
        open(ROOT / "calib_out" / "calib_runs_raw.csv", encoding="utf-8-sig")
    )
)
by = defaultdict(list)
for r in raw:
    by[r["label"]].append(r)


def mm(rows):
    """Compute label-level mean metrics from raw rows."""
    out = {}
    for k in rows[0]:
        if k in ("label", "seed"):
            continue
        vs = [float(r[k]) for r in rows if r[k] not in ("", "nan")]
        out[k] = sum(vs) / len(vs) if vs else float("nan")
    return out


def L(label):
    """Return rows matching a calibration label."""
    return cs.l_unified(mm(by[label]), obs, weights)


M_CN = {
    "veh_share": "车辆-时长占比※",
    "veh_stop_rate": "车辆-停探测率",
    "veh_reaction_prob": "车辆-反应概率",
    "veh_reaction_delay_s": "车辆-反应延迟",
    "lm_share": "地标-时长占比",
    "lm_stop_after_rate": "地标-触发后停探测率",
    "tac_share": "触觉-时长占比※",
    "tac_stop_rate": "触觉-停探测率",
    "noref_share": "无参照-时长占比※",
    "noref_stop_rate": "无参照-停探测率＊",
    "noref_ep_len": "无参照-片段步长",
    "noref_retrieval_per100": "无参照-记忆检索率",
}
ORDER = [
    "veh_stop_rate",
    "veh_reaction_prob",
    "veh_reaction_delay_s",
    "lm_share",
    "lm_stop_after_rate",
    "tac_stop_rate",
    "noref_ep_len",
    "noref_retrieval_per100",
    "noref_stop_rate",
    "veh_share",
    "tac_share",
    "noref_share",
]


def relerr(sim, o):
    """Compute signed relative error against an observation."""
    return abs(sim - o) / abs(o) * 100 if abs(o) > 1e-9 else abs(sim - o) * 100


fig, ax = plt.subplots(figsize=(7.2, 4.6))
ys = range(len(ORDER) - 1, -1, -1)
for y, k in zip(ys, ORDER):
    e0, e1 = relerr(m_def[k], obs[k]), relerr(m_star[k], obs[k])
    ax.plot([e0, e1], [y, y], color="#c9c8c1", lw=1.4, zorder=1)
    ax.scatter([e0], [y], s=42, color=YELLOW, zorder=3)
    ax.scatter([e1], [y], s=46, color=BLUE, zorder=4)
    ax.annotate(
        f"{e1:.1f}%",
        (e1, y),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=7.2,
        color="#104281",
    )
ax.set_yticks(list(ys), [M_CN[k] for k in ORDER])
ax.set_xscale("log")
ax.set_xlabel("相对误差 |sim - obs| / |obs|（%，对数轴）")
ax.scatter([], [], s=42, color=YELLOW, label="校准前（默认参数）")
ax.scatter([], [], s=46, color=BLUE, label="校准后（θ*）")
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("图 C1  各行为指标的实测贴合度：校准前 → 校准后", loc="left", fontsize=10)
ax.text(
    0.0,
    -0.16,
    "※ 时长占比类指标由已冻结的第二类环境参数决定（结构性残差）；＊ 实测≈0，按绝对误差×100 显示。",
    transform=ax.transAxes,
    fontsize=7.2,
    color=MUT,
)
fig.savefig(FIGS / "figC1_dumbbell.png")
plt.close(fig)

grid = sorted((float(k), v) for k, v in star["memth_regrid_at_star"].items())
xs = [g[0] for g in grid]
lm = [g[1]["L_memoryTH"] for g in grid]
lu = [g[1]["L_unified"] for g in grid]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.axvspan(0.06, 0.08, color="#cde2fb", alpha=0.5, zorder=0)
ax.plot(xs, lm, "-o", color=BLUE, lw=2, ms=5, label="L_memoryTH（专属损失）")
ax.plot(xs, lu, "-o", color=YELLOW, lw=2, ms=5, label="L_unified（统一损失）")
for x, v in zip(xs, lm):
    ax.annotate(
        f"{v:.0f}",
        (x, v),
        textcoords="offset points",
        xytext=(0, -13),
        ha="center",
        fontsize=7.2,
        color="#104281",
    )
ax.set_yscale("log")
ax.set_xlabel("MEMORY_ACTIVE_RETRIEVAL_TH")
ax.set_ylabel("损失（对数轴）")
ax.annotate(
    "相变区：显式检索过频",
    (0.04, 216.7),
    xytext=(0.037, 700),
    fontsize=7.5,
    color=MUT,
    arrowprops=dict(arrowstyle="-", color=MUT, lw=0.7),
)
ax.annotate(
    "等损失平台\n（θ* 取 0.06，建议 0.07）",
    (0.07, 14.4),
    xytext=(0.063, 1.6),
    fontsize=7.5,
    color=MUT,
)
ax.set_ylim(1, 2000)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("图 C2  阈值参数在 θ* 下的网格复扫", loc="left", fontsize=10)
fig.savefig(FIGS / "figC2_threshold.png")
plt.close(fig)

steps = [
    ("起点\n(TH=0.06)", 17.0101, True),
    ("ABSENT_STEPS\n11→18", 6.9168, True),
    ("PROBE_RELIEF\n0.40→0.50", 4.6073, True),
    ("LANDMARK_DECAY\n保留 0.82", 4.6073, False),
    ("VEH_GATE\n0.40→0.28", 4.5468, True),
    ("LOOMING_PEAK\n保留 0.35", 4.5468, False),
    ("LOOMING_DECAY\n0.80→0.90", 4.3643, True),
    ("第二轮\n无更新，收敛", 4.3643, False),
]
fig, ax = plt.subplots(figsize=(9.0, 3.8))
xs = list(range(len(steps)))
ax.step(xs, [s[1] for s in steps], where="post", color=BLUE, lw=2, zorder=2)
for x, (lb, v, changed) in zip(xs, steps):
    ax.scatter(
        [x],
        [v],
        s=40,
        color=BLUE if changed else "#9ec5f4",
        edgecolor=BLUE,
        lw=1,
        zorder=3,
    )
    ax.annotate(
        f"{v:.2f}",
        (x, v),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=7.5,
        color="#104281",
    )
ax.set_xticks(xs, [s[0] for s in steps], fontsize=6.4)
ax.set_ylabel("L_unified")
ax.set_ylim(3, 19)
ax.set_title(
    "图 C3  坐标下降收敛轨迹（实心=参数更新，空心=按简约原则保留）",
    loc="left",
    fontsize=10,
)
fig.savefig(FIGS / "figC3_trace.png")
plt.close(fig)

PROFILES = [
    (
        "MEMORY_ACTIVE_ABSENT_STEPS_TH",
        18,
        [
            (7, L("R1|MEMORY_ACTIVE_ABSENT_STEPS_TH|7")),
            (9, L("R1|MEMORY_ACTIVE_ABSENT_STEPS_TH|9")),
            (11, L("incumbent|start")),
            (14, L("R1|MEMORY_ACTIVE_ABSENT_STEPS_TH|14")),
            (18, L("R1|MEMORY_ACTIVE_ABSENT_STEPS_TH|18")),
        ],
    ),
    (
        "PROBE_RELIEF_RATIO",
        0.5,
        [
            (0.2, L("R1|PROBE_RELIEF_RATIO|0.2")),
            (0.3, L("R1|PROBE_RELIEF_RATIO|0.3")),
            (0.4, L("R1|MEMORY_ACTIVE_ABSENT_STEPS_TH|18")),
            (0.5, L("R1|PROBE_RELIEF_RATIO|0.5")),
            (0.6, L("R1|PROBE_RELIEF_RATIO|0.6")),
        ],
    ),
    (
        "LANDMARK_DECAY_RATE",
        0.82,
        [
            (0.66, L("R1|LANDMARK_DECAY_RATE|0.66")),
            (0.74, L("R1|LANDMARK_DECAY_RATE|0.74")),
            (0.82, L("R1|PROBE_RELIEF_RATIO|0.5")),
            (0.90, L("R1|LANDMARK_DECAY_RATE|0.9")),
            (0.96, L("R1|LANDMARK_DECAY_RATE|0.96")),
        ],
    ),
    (
        "VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK",
        0.28,
        [
            (0.28, L("R1|VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK|0.28")),
            (0.34, L("R1|VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK|0.34")),
            (0.40, L("R1|PROBE_RELIEF_RATIO|0.5")),
            (0.48, L("R1|VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK|0.48")),
            (0.56, L("R1|VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK|0.56")),
        ],
    ),
    (
        "LOOMING_BOOST_PEAK",
        0.35,
        [
            (0.21, L("R1|LOOMING_BOOST_PEAK|0.21")),
            (0.28, L("R1|LOOMING_BOOST_PEAK|0.28")),
            (0.35, L("R1|VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK|0.28")),
            (0.42, L("R1|LOOMING_BOOST_PEAK|0.42")),
            (0.49, L("R1|LOOMING_BOOST_PEAK|0.49")),
        ],
    ),
    (
        "LOOMING_BOOST_DECAY",
        0.90,
        [
            (0.72, L("R1|LOOMING_BOOST_DECAY|0.72")),
            (0.78, L("R1|LOOMING_BOOST_DECAY|0.78")),
            (0.84, L("R1|LOOMING_BOOST_DECAY|0.84")),
            (0.90, L("R1|LOOMING_BOOST_DECAY|0.9")),
            (0.96, L("R1|LOOMING_BOOST_DECAY|0.96")),
        ],
    ),
]
fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0))
for ax, (pname, chosen, pts) in zip(axes.flat, PROFILES):
    xs = [p[0] for p in pts]
    ls = [p[1] for p in pts]
    ax.plot(xs, ls, "-o", color=BLUE, lw=1.8, ms=4.5)
    cx = [p for p in pts if p[0] == chosen][0]
    ax.scatter(
        [cx[0]], [cx[1]], s=90, facecolor="none", edgecolor=INK, lw=1.4, zorder=5
    )
    ax.set_title(pname, fontsize=7.8)
    ax.tick_params(labelsize=7)
    ax.set_ylabel("L_unified", fontsize=7)
fig.suptitle(
    "图 C4  各参数一维损失剖面（第一轮网格；黑圈=入选值；纵轴条件于扫描时点的参数组）",
    x=0.02,
    ha="left",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGS / "figC4_profiles.png")
plt.close(fig)

print("figs →", FIGS)
