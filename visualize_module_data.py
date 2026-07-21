"""Render module-level ACT-R workload timelines from simulation CSV outputs.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import argparse

import csv

import glob

import os

from typing import Dict, List


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


MODULE_KEYS = ["auditory", "tactile", "manual", "central", "memory"]

MODULE_LABELS = {
    "auditory": "Auditory",
    "tactile": "Tactile",
    "manual": "Manual",
    "central": "Central",
    "memory": "Memory",
}

COLORS = {
    "auditory": "#42A5F5",
    "tactile": "#66BB6A",
    "manual": "#FFA726",
    "central": "#AB47BC",
    "memory": "#8D6E63",
}


def _rolling_mean(values: List[float], window: int = 50) -> List[float]:
    """Compute a trailing rolling average for a numeric sequence."""
    if not values:

        return []

    out = []

    total = 0.0

    queue = []

    for value in values:

        queue.append(value)

        total += value

        if len(queue) > window:

            total -= queue.pop(0)

        out.append(total / len(queue))

    return out


def _load_module_csv(csv_path: str) -> Dict[str, List[float]]:
    """Load module-level simulation metrics from a CSV file."""
    data: Dict[str, List[float]] = {
        "step": [],
    }

    for key in MODULE_KEYS:

        data[f"iw_{key}"] = []

        data[f"a_{key}"] = []

        data[f"e_{key}"] = []

        data[f"dt_{key}"] = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            data["step"].append(float(row["step"]))

            for key in MODULE_KEYS:

                data[f"iw_{key}"].append(float(row[f"actr_iw_{key}"]))

                data[f"a_{key}"].append(float(row[f"actr_{key}_active"]))

                data[f"e_{key}"].append(float(row[f"actr_{key}_error"]))

                data[f"dt_{key}"].append(float(row[f"actr_dt_{key}"]) * 1000.0)

    return data


def render_module_figure(csv_path: str, output_path: str) -> str:
    """Render a multi-panel module workload figure from CSV data."""
    data = _load_module_csv(csv_path)

    steps = data["step"]

    if not steps:

        raise ValueError("CSV has no data rows.")

    iw_stacks = [data[f"iw_{key}"] for key in MODULE_KEYS]

    iw_total = [sum(values) for values in zip(*iw_stacks)]

    iw_mean = sum(iw_total) / len(iw_total)

    iw_total_roll = _rolling_mean(iw_total, window=50)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.5, 1.5, 1.5]},
    )

    fig.patch.set_facecolor("#FAFAFA")

    ax = axes[0]

    ax.set_facecolor("#F6F6F6")

    ax.stackplot(
        steps,
        *iw_stacks,
        labels=[MODULE_LABELS[key] for key in MODULE_KEYS],
        colors=[COLORS[key] for key in MODULE_KEYS],
        alpha=0.78,
    )

    ax.plot(
        steps, iw_total, color="#263238", linewidth=1.0, alpha=0.55, label="IW total"
    )

    ax.plot(
        steps,
        iw_total_roll,
        color="#D32F2F",
        linewidth=2.0,
        alpha=0.95,
        label="IW average (roll-50)",
    )

    ax.axhline(
        iw_mean,
        color="#FF7043",
        linestyle="--",
        linewidth=1.1,
        alpha=0.95,
        label=f"IW mean {iw_mean:.2f}",
    )

    ax.axhline(
        6.0,
        color="#D32F2F",
        linestyle=":",
        linewidth=0.9,
        alpha=0.75,
        label="threshold 6.0",
    )

    ax.set_ylabel("IW")

    ax.set_title(
        f"ACT-R Module Contributions ({len(steps)} steps) | IW mean = {iw_mean:.2f}"
    )

    ax.legend(loc="upper right", ncol=3, fontsize=7.2, framealpha=0.85)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    ax = axes[1]

    ax.set_facecolor("#F6F6F6")

    for key in MODULE_KEYS:

        ax.plot(
            steps,
            _rolling_mean(data[f"a_{key}"], window=50),
            color=COLORS[key],
            linewidth=1.0,
            alpha=0.9,
            label=MODULE_LABELS[key],
        )

    ax.set_ylabel("Activation (roll-50)")

    ax.set_ylim(0.0, 1.05)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    ax = axes[2]

    ax.set_facecolor("#F6F6F6")

    for key in MODULE_KEYS:

        ax.plot(
            steps,
            _rolling_mean(data[f"e_{key}"], window=50),
            color=COLORS[key],
            linewidth=1.0,
            alpha=0.9,
            label=MODULE_LABELS[key],
        )

    ax.set_ylabel("Error_weight (roll-50)")

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    ax = axes[3]

    ax.set_facecolor("#F6F6F6")

    for key in MODULE_KEYS:

        ax.plot(
            steps,
            _rolling_mean(data[f"dt_{key}"], window=50),
            color=COLORS[key],
            linewidth=1.0,
            alpha=0.9,
            label=MODULE_LABELS[key],
        )

    ax.set_ylabel("Dwell_Time ms (roll-50)")

    ax.set_xlabel("Step")

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout(h_pad=0.35)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    plt.close(fig)

    return output_path


def _find_latest_module_csv() -> str:
    """Return the newest module-level simulation CSV in the reports directory."""
    cwd = os.getcwd()

    patterns = [
        os.path.join(cwd, "reports", "sim_module_data_*.csv"),
        os.path.join(cwd, "sim_module_data_*.csv"),
    ]

    candidates: List[str] = []

    for pattern in patterns:

        candidates.extend(glob.glob(pattern))

    if not candidates:

        raise FileNotFoundError(
            "未找到 sim_module_data_*.csv。请传入 csv_path，或先运行一次模拟生成模块CSV。"
        )

    candidates.sort(key=os.path.getmtime)

    return os.path.abspath(candidates[-1])


def main():
    """Run the script entry point."""
    parser = argparse.ArgumentParser(description="Visualize sim_module_data CSV.")

    parser.add_argument(
        "csv_path", nargs="?", default=None, help="Path to sim_module_data_*.csv"
    )

    parser.add_argument("--out", dest="out_path", default=None, help="Output PNG path")

    args = parser.parse_args()

    csv_path = (
        os.path.abspath(args.csv_path) if args.csv_path else _find_latest_module_csv()
    )

    if args.out_path:

        out_path = os.path.abspath(args.out_path)

    else:

        base_name = os.path.splitext(os.path.basename(csv_path))[0]

        out_path = os.path.join(os.path.dirname(csv_path), f"{base_name}_viz.png")

    result = render_module_figure(csv_path, out_path)

    print(f"[使用CSV] {csv_path}")

    print(f"[模块可视化已生成] {result}")


if __name__ == "__main__":

    main()
