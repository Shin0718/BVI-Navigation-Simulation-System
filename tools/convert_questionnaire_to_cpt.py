"""Convert transposed expert questionnaire responses into smoothed CPT parameters.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import argparse
import csv
import json
from pathlib import Path


RISK_STATES = ("low", "medium", "high")

LEVEL_TO_RISK = {
    "safe": "low",
    "uneven": "medium",
    "regular": "medium",
    "complex": "high",
}

TARGET_VARIABLES = (
    "ambient_noise",
    "distance_feedback",
    "vehicle_approach",
    "cane_hit",
    "surface_change",
    "surface_risky",
    "reference_available",
    "on_tactile_guidance",
    "at_intersection",
    "crossing_active",
)

PROTECTIVE_VARIABLES = {
    "reference_available",
    "on_tactile_guidance",
}


def _to_float(value):
    """Convert a questionnaire cell to a float when possible."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _likert_to_prob(score, reverse=False):
    """Map a Likert score to a probability value."""
    p = (score - 1.0) / 4.0
    if reverse:
        p = 1.0 - p
    if p < 0:
        return 0.0
    if p > 1:
        return 1.0
    return p


def _beta_smooth(p, n, alpha=1.0, beta=1.0):
    """Apply beta smoothing to a probability estimate."""
    return (n * p + alpha) / (n + alpha + beta)


def _enforce_monotonic(values, decreasing=False):
    """Enforce monotonic ordering across risk-level values."""
    if decreasing:
        fixed = list(values)
        if fixed[1] > fixed[0]:
            fixed[1] = fixed[0]
        if fixed[2] > fixed[1]:
            fixed[2] = fixed[1]
        return fixed
    fixed = list(values)
    if fixed[1] < fixed[0]:
        fixed[1] = fixed[0]
    if fixed[2] < fixed[1]:
        fixed[2] = fixed[1]
    return fixed


def load_transposed_questionnaire_csv(csv_path):
    """Load questionnaire responses from a transposed CSV export."""
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            rows = None
    if rows is None:
        raise last_error

    if not rows or len(rows[0]) < 2:
        raise ValueError("CSV 结构不符合问卷导出格式：至少需要 id 列和一个受试者列")

    row_map = {}
    for row in rows:
        if not row:
            continue
        key = str(row[0]).strip()
        if not key:
            continue
        row_map[key] = row[1:]
    return row_map


def build_cpt_from_questionnaire(
    row_map, reverse_protective=True, enforce_monotonic=True
):
    """Build CPT entries from questionnaire response rows."""
    cpt = {}
    diagnostics = {}

    for var in TARGET_VARIABLES:
        state_to_probs = {state: [] for state in RISK_STATES}

        for suffix, risk_state in LEVEL_TO_RISK.items():
            row_key = f"{var}_{suffix}"
            if row_key not in row_map:
                continue

            reverse = reverse_protective and (var in PROTECTIVE_VARIABLES)
            for cell in row_map[row_key]:
                score = _to_float(cell)
                if score is None:
                    continue
                p = _likert_to_prob(score, reverse=reverse)
                state_to_probs[risk_state].append(p)

        state_means = {}
        state_counts = {}
        for state in RISK_STATES:
            vals = state_to_probs[state]
            if vals:
                mean_p = sum(vals) / len(vals)
                smoothed = _beta_smooth(mean_p, len(vals), alpha=1.0, beta=1.0)
                state_means[state] = smoothed
                state_counts[state] = len(vals)
            else:
                state_means[state] = 0.5
                state_counts[state] = 0

        values = [state_means["low"], state_means["medium"], state_means["high"]]
        is_protective = var in PROTECTIVE_VARIABLES
        if enforce_monotonic:
            values = _enforce_monotonic(values, decreasing=is_protective)

        cpt[var] = {
            "low": round(values[0], 3),
            "medium": round(values[1], 3),
            "high": round(values[2], 3),
        }
        diagnostics[var] = {
            "counts": state_counts,
            "protective": is_protective,
        }

    return cpt, diagnostics


def main():
    """Run the script entry point."""
    parser = argparse.ArgumentParser(
        description="将转置问卷CSV转换为 inference.EXPERT_CPT 格式"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入问卷CSV路径（形如第一列是字段名，后续列是受试者）",
    )
    parser.add_argument(
        "--output", default="cpt_from_questionnaire.json", help="输出CPT JSON路径"
    )
    parser.add_argument(
        "--no-reverse-protective",
        action="store_true",
        help="不对保护性变量(reference_available/on_tactile_guidance)做反向映射",
    )
    parser.add_argument(
        "--no-monotonic",
        action="store_true",
        help="不强制单调约束（风险正向变量 low<=medium<=high，保护性变量相反）",
    )
    args = parser.parse_args()

    row_map = load_transposed_questionnaire_csv(args.input)
    cpt, diagnostics = build_cpt_from_questionnaire(
        row_map,
        reverse_protective=not args.no_reverse_protective,
        enforce_monotonic=not args.no_monotonic,
    )

    output_path = Path(args.output)
    payload = {
        "EXPERT_CPT": cpt,
        "meta": {
            "input": str(Path(args.input)),
            "reverse_protective": not args.no_reverse_protective,
            "enforce_monotonic": not args.no_monotonic,
            "diagnostics": diagnostics,
        },
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] CPT 已输出到: {output_path}")


if __name__ == "__main__":
    main()
