# 3. 校准（搜索 + 收尾检验 + 出图）

管线第三步：用实测数据（`../obs_values.csv`，三地 7550s）校准 7 个单元。
损失 = 加权相对平方误差；权重取第 2 步 `cat4_indicator_scores.csv` 在可观测
6 项内重归一化。论文表述见 `../../参数校准/校准论文稿_0713.md`。

```bash
python 3_校准/calibrate_search_v2.py       # ① 主校准：阶段A阈值网格 + 阶段B坐标下降（约 1.5 小时）
python 3_校准/finish_calibration_v2.py     # ② 收尾：默认基线 + 复扫 + 种子外推 + 权重稳健性 + 报告（约 15 分钟）
python 3_校准/make_calib_figs_v2.py        # ③ 四张校准图（秒级，需先跑 ②）
python 3_校准/probe_relief_extend.py       # 附：PROBE_RELIEF 低端补扫（平台 [0.05,0.30] 的出处）
python 3_校准/verify_probe03.py            # 附：0.3 在 θ* 背景等损失的验证
```

**产物**（→ `../calib_out_v2/`）：`theta_star.json`（最终参数）、`CALIBRATION_REPORT.md`、
`calib_runs_raw.csv` / `calib_trace.csv`、`finish_summary.json`、`figs/figV2_*.png`（四图）。

**本轮结果**：L 14.56 → 3.74（-74.3%）；θ* = MEMORY_TH 0.06（部署 0.07）、ABSENT 14、
PROBE_RELIEF 0.30、LOOMING_PEAK 0.42，其余维持默认；θ* 已写回 `../bvi_sa/simulation.py`。

`_archive_v1/`：0708 版校准全套（基于旧情境指标口径，结果在 `../calib_out/`，仅作历史参照）。
