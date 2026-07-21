# 2. 参数筛选（四类候选参数敏感性 + 阀门筛选）

管线第二步（对应 Supplementary 0712 · 1.3）：13 候选参数分四类做弹性分析，
按类别阀门（类内平均弹性 ≥ 0.1，每类保底一个）筛出 7 个校准单元，
并产出第 3 步用的指标权重。

```bash
python 2_参数筛选/candidate_sensitivity_4cat.py full       # 正式跑（135 次仿真，约 2 小时）
python 2_参数筛选/candidate_sensitivity_4cat.py memth      # 补：MEMORY_TH 工作区间扰动（15 次）
python 2_参数筛选/candidate_sensitivity_4cat.py widen      # 补：死区参数放大扰动核查（10 次）
python 2_参数筛选/candidate_sensitivity_4cat.py reanalyze  # 秒级重算分析与图（不重跑仿真）
python 2_参数筛选/scan_safety.py                           # 辅助：SEEV 安全权重全量程扫描
python 2_参数筛选/scan_decay.py                            # 辅助：LOOMING_DECAY 全量程扫描
```

**产物**（→ `../sensitivity_out/`）：`cat4_screening.csv`（**筛选结果，以此为准**）、
`cat4_S_matrix.csv`、`cat4_indicator_scores.csv`（→ 第 3 步损失权重）、
`cat4_heatmap_{a,b,c}.png`（a=风险+探测合并 6×6 / b=记忆 / c=动态）、
`cat4_association_{a,b,c}.png`、`cat4_runs_raw.csv`（160 次原始明细）。

**本轮结果**：7 校准单元 = SEEV 组合、PROBE_RELIEF_RATIO、MEMORY_TH(阈值型单列)、
MEMORY_ABSENT_STEPS_TH、LANDMARK_DECAY_RATE、LOOMING_DECAY、LOOMING_PEAK。

`_archive/`：旧版全局弹性初筛（sensitivity_elasticity.py），已被本脚本替代。
