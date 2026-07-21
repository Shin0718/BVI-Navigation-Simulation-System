# 机制中介敏感性分析报告（mechanism-mediated SA）

**依据**：导师《修改.docx》四项方案
**运行规模**：23 采样点 × 5 种子 = 115 次仿真，全部成功，0 失败（32.1 分钟，10 核并行）
**种子集**：20260705–20260709；埋点仅新增统计字段（vehicle_approach_raw / sound_salience / retrieval_wm_load），不改变模型行为
**日期**：2026-07-06

## 一、断点定位总表（核心交付）

两段式判定：**段1** = 参数→中间机制变量弹性（S1）；**段2** = 机制变量→行为输出（S_extended 行为弹性 + S2 相关）。

| 参数 | 段1（参数→机制） | 段2（机制→行为） | 断点结论 |
|---|---|---|---|
| SEEV_VALUE_SAFETY_WEIGHT | ✅ 活：gate通过率弹性 **1.84**，net_priority 0.18 | ❌ 断：行为全零 | 传导断在 gate→行为；gate 信号仅进 bookkeeping |
| SEEV_VALUE_PROGRESS_WEIGHT | ✅ 活：net_priority 0.47，gate 0.26 | ❌ 断 | 同上 |
| SEEV_EXPECTANCY_RISK_WEIGHT | ✅ 活：gate 1.58 | ❌ 断 | 同上 |
| ACTR_RISK_CANE_GUIDANCE_RELIEF | ⚠️ 基本断在段1：×0.5/×2 扰动下 risk_mean 弹性仅 0.011 | （事件条件指标有微效应：地标后停探测率 0.14） | 缓释只在引导物在场瞬间起作用，占总时长比例过小，被总体风险信号稀释 |
| ACTR_RISK_LANDMARK_RELIEF | ⚠️ 基本断在段1：risk_mean 弹性 0.002 | （landmark_relief_effect 0.14，定向指标可测） | 同上；若校准，损失函数须用事件条件指标而非全程均值 |
| LOOMING_BOOST_PEAK | ✅ 活：looming 0.71，证据有效率 0.36 | ✅ 通：反应延迟 1.01、反应持续 0.30 | 两段均通，可校准 |
| LOOMING_BOOST_DECAY | ✅ 段1极活：looming均值 **8.79**、持续步数 **12.36**、声显著性 0.68 | ❌ 断：行为全零，gate/证据有效率弹性均为 0 | 衰减改变 looming 轨迹但从不跨越任何决策阈值（峰值步已决定过闸与否，衰减只影响事后尾巴） |
| VEHICLE_APPROACH_SALIENCE_GATE | ✅ 活：证据有效率 0.45，gate 0.47 | ✅ 通：反应延迟 1.71、地标后停探测 0.42 | 两段均通，可校准 |
| MEMORY_ACTIVE_RETRIEVAL_TH | ✅ 活（扫描范围内）：记忆激活率弹性 1.66、risk 0.50 | ✅ 通：地标后停探测 4.57、停探测 1.10、高负荷占比 1.19 | 可校准，但仅在工作区间内（见二） |

## 二、MEMORY_ACTIVE_RETRIEVAL_TH 阈值扫描（0.03–0.18）

| TH | 记忆激活占比 | 检索频率/100步 | 负荷均值 | 停探测/100步 |
|---|---|---|---|---|
| 0.03 | 0.825 | 3.48 | 6.77 | 30.9 |
| 0.05 | 0.706 | 5.03 | 6.07 | 27.3 |
| **0.08** | **0.161** | **1.21** | **3.13** | **8.7** |
| 0.10–0.18 | 0.166（完全平坦） | 1.01 | 3.17 | 8.5 |

- **悬崖在 0.05→0.08 之间**：retrieval_wm_load 的 90 分位仅 0.05，绝大多数负荷值落在 [0.03, 0.08)；TH≥0.08 后阈值高于负荷分布，激活只剩 guidance_absent 步数通道兜底（占比恒 0.166）。
- **老师的诊断获数据证实**：默认 0.15 在死区，±20%（0.12–0.18）整段平坦，所以此前局部弹性为 0。
- **校准建议**：该参数取值域应设为 **[0.03, 0.08]**；0.08 以上无梯度。注意 TH 从 0.08 降到 0.03 时行为剧变（停探测 ×3.6、负荷 ×2.2），实际是"记忆检索常开/常关"的开关，专家先验应回答"BVI 行走中显式回忆的合理频率"来定位取值。

## 三、RELIEF 参数放大扰动结果（×0.5 / ×2）

扰动放大到 4 倍量程后，两参数对**全程风险均值**的弹性仍 ≤0.011——低敏感的原因不是"扰动太小"，而是**缓释项只在地标/引导物在场的少数步生效，被全程均值稀释**。但对事件条件指标有可测的定向效应：
- LANDMARK_RELIEF → landmark_relief_effect（地标后风险下降量）弹性 0.138
- CANE_RELIEF → 地标后停探测率 0.140

**结论**：若这两个参数保留在校准中，损失函数必须包含事件条件指标（地标后风险缓释、地标后停探测率）；只用全程统计量则它们不可辨识。

## 四、更新后的参数处置建议

| 处置 | 参数 |
|---|---|
| 直接校准（两段通） | PROBE_RELIEF_RATIO、MEMORY_ACTIVE_ABSENT_STEPS_TH、LANDMARK_DECAY_RATE、VEHICLE_APPROACH_SALIENCE_GATE、LOOMING_BOOST_PEAK |
| 限定取值域校准 | MEMORY_ACTIVE_RETRIEVAL_TH ∈ [0.03, 0.08] |
| 用事件条件指标才可校准 | ACTR_RISK_CANE_GUIDANCE_RELIEF、ACTR_RISK_LANDMARK_RELIEF |
| 固定默认值（结构性不可辨识，断点已定位） | SEEV 三权重（断在 gate→行为）、LOOMING_BOOST_DECAY（断在轨迹→阈值） |
| 暂退出校准（导师指示） | PROBE_SAFE_STREAK_RELEASE_THRESHOLD、PROBE_OVERLOAD_STREAK_TH、ACTR_LOAD_RESUME_THRESHOLD |

## 四补、SEEV 权重终审：负荷联动 + 全量程扫描（2026-07-06 补充）

按导师方案实施了注意门控→负荷联动（未过 gate 的线索通道激活×0.5；过 gate 的危险线索 central +0.20；
风险缓释挂 gate）并启用分位数自适应阈值（滚动 80 分位，gate 通过率锚定 ~21%）。在此基础上对
SEEV_VALUE_SAFETY_WEIGHT 做合法域全量程扫描（0.1/0.4/0.7/1.0/1.3 × 5 种子，25 次）：

- 内部机制单调响应：net_priority 均值 0.053→0.073（+37%）、风险信号 0.1163→0.1189（缓释门控链路工作正常）；
- 行为输出全部平坦：负荷均值变化 0.2%（非单调）、车辆反应概率五点完全相同（0.3376）、总步数五点完全相同（4227.6）。

**终审结论**：SEEV 价值权重结构性不可辨识——biased competition 阈值锚定注意总量后，
权重仅重排注意分配且重排集中于行为惰性步，行为弹性上限为噪声量级。三个 SEEV 权重
固定为专家默认值（0.70/0.30/0.55）；负荷联动机制保留（有独立开关），作为论文中
"注意分配与注意总量解耦"的机制性发现报告。
数据：`safety_weight_sweep.csv` / `safety_weight_sweep.png`。

## 四补2、LOOMING_BOOST_DECAY 恢复联动 + 扫描（2026-07-06 补充）

按导师机制表（LOOMING_BOOST_DECAY → recovery_time_after_vehicle）实施恢复联动：
looming_boost 未衰减到 LOOMING_RESUME_THRESHOLD(0.10) 以下时，写入 ACT-R tick_signal
危险证据位（"惊魂未定"视作威胁仍在），保护性停探测不解除。开关 LOOMING_RESUME_GATE_ENABLED。
（实施说明：原 probe 释放代码块只写日志不影响行为——这同时解释了 PROBE_SAFE_STREAK
参数此前的零弹性；行为级挂钩因此下在 tick_signal。）

扫描 0.64–0.96 × 5 种子（25 次）结果：
- **参数已激活**，0.72 以上剂量响应清晰：负荷均值 3.04→2.93、高负荷占比 0.159→0.145、
  停探测 8.52→8.23、路口等待 204.6→196.5、车辆反应概率 0.338→0.306（0.96 点弹性约 0.2–0.46）；
- **0.64–0.72 段平坦**：衰减快时警觉尾巴在车辆事件持续期内即降至阈下，无附加效应
  → 有效校准取值域约 **[0.72, 0.96]**；
- **方向说明（需导师知悉）**：警觉时间越长（decay 越大），停探测频率与负荷反而略降——
  持续危险证据使代理维持"警戒行走"而非反复触发新的停探测循环。若导师期望的语义
  严格为"只推迟恢复、不影响触发"，需在 ACT-R tick_signal 增加独立字段区分两条通路（改动稍大）。

**处置更新**：LOOMING_BOOST_DECAY 由"固定默认值"移入"限定取值域校准"（[0.72, 0.96]）。
数据：`decay_sweep.csv` / `decay_sweep.png`。

## 四补3、情境敏感性分析（scenario-specific SA，2026-07-06 补充）

对 9 个候选校准参数，将行为输出按五类导航情境后验切分（车辆逼近=事件起5步窗口、
地标触发=匹配起3步窗口、触觉引导=引导物/盲道非路口步、无参照=失参照非路口步、
路口穿越=crossing 阶段），每情境 4–5 个行为指标，共 23 个情境指标。
扰动：常规参数 ±20%；RELIEF 两参数 ×0.5/×2；MEMORY_ACTIVE_RETRIEVAL_TH 在
工作区间中点 0.055 ±20%（默认 0.15 处于死区，配独立基准）。
运行：20 采样点 × 5 种子 = 100 次，全部成功（28.1 分钟）。

### 参数×情境汇总（块内最大弹性）

| 参数 | 车辆逼近 | 地标触发 | 触觉引导 | 无参照 | 路口穿越 | 主责情境 |
|---|---|---|---|---|---|---|
| PROBE_RELIEF_RATIO | 1.25 | 1.25 | 1.01 | **2.87** | 0.35 | 无参照（跨情境响应） |
| MEMORY_ACTIVE_ABSENT_STEPS_TH | 3.72 | 4.26 | 3.56 | **5.81** | 0.72 | 无参照（全域强） |
| LANDMARK_DECAY_RATE | 1.43 | **1.79** | 0.61 | 0.54 | 0.44 | 地标触发 |
| VEHICLE_APPROACH_SALIENCE_GATE | 0.25 | **0.88** | 0.31 | 0.31 | 0.33 | 地标（跨情境耦合，见注3） |
| LOOMING_BOOST_PEAK | **0.92** | 0.40 | 0.07 | 0.23 | 0.15 | 车辆逼近 |
| LOOMING_BOOST_DECAY | **1.38** | 0.84 | 0.53 | 0.58 | 0.14 | 车辆逼近（恢复联动生效） |
| ACTR_RISK_CANE_GUIDANCE_RELIEF | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 无（判定不可辨识） |
| ACTR_RISK_LANDMARK_RELIEF | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 地标（弱） |
| MEMORY_ACTIVE_RETRIEVAL_TH | 2.89 | 20.26 | **62.67** | 3.84 | 0.80 | 触觉/地标（开关型，见注2） |

### 主要结论

1. **参数-情境职责分工与机制预期一致**：LOOMING 双参数主责车辆逼近（恢复联动使 DECAY
   在车辆-反应延迟上弹性 1.38）；LANDMARK_DECAY 主责地标；MEMORY_ABSENT_STEPS 与
   PROBE_RELIEF 主责无参照；为按"情境×指标"两级设定校准损失权重提供了直接依据。
2. **MEMORY_ACTIVE_RETRIEVAL_TH 为开关型参数**：在悬崖中点扰动时弹性无界（62.67 出现于
   触觉-停探测率——盲道上的停探测几乎纯由内部检索机制驱动，是该参数最干净的观察窗口）。
   建议不与连续参数混入同一损失做梯度校准，采用两步法：先由专家/实测（盲道段停顿频率，
   实地有观测）确定悬崖侧，再侧内微调。热力图对其做色阶封顶（vmax=4，超出格子以*标注实值）。
3. **VEHICLE_GATE 跨情境耦合**：其在地标情境的弹性（0.88）高于车辆情境（0.25），
   路段声音闸门变化会改变地标附近的停探测行为，校准中与 LANDMARK_DECAY 可能存在交互。
4. **路口穿越整列弱**（最大 0.80）：路口行为由信号灯状态机与路口参数（不在校准集）主导，
   校准损失中路口指标对该参数集信息量低，建议降权。
5. **情境信息量排序**：无参照 > 地标触发 ≈ 车辆逼近 > 触觉引导 > 路口穿越。
6. **处置更新**：ACTR_RISK_CANE_GUIDANCE_RELIEF 在 ×0.5/×2 大扰动 + 情境专属指标下仍无
   可测效应，由"事件条件指标校准"改判**固定默认值**；ACTR_RISK_LANDMARK_RELIEF 仅在
   地标-风险缓释上弱可测（0.12），保留但权重预期很低。

### 最终参数处置清单（截至 2026-07-06）

| 处置 | 参数 |
|---|---|
| 直接校准（5） | PROBE_RELIEF_RATIO、MEMORY_ACTIVE_ABSENT_STEPS_TH、LANDMARK_DECAY_RATE、VEHICLE_APPROACH_SALIENCE_GATE、LOOMING_BOOST_PEAK |
| 限定取值域校准（2） | MEMORY_ACTIVE_RETRIEVAL_TH ∈ [0.03, 0.08]（两步法）、LOOMING_BOOST_DECAY ∈ [0.72, 0.96] |
| 固定默认值（5） | SEEV 三权重（全量程扫描铁证）、ACTR_RISK_CANE_GUIDANCE_RELIEF、ACTR_RISK_LANDMARK_RELIEF（唯一敏感指标为内部量，实地不可观测） |
| 暂退出校准（导师指示，3） | PROBE_SAFE_STREAK_RELEASE_THRESHOLD、PROBE_OVERLOAD_STREAK_TH、ACTR_LOAD_RESUME_THRESHOLD |

### 模型版本说明

本报告第一~三节数据基于原始模型结构；四补起启用了三组新机制（均有独立开关，config.py 有注释）：
①注意门控→负荷联动 + 分位数自适应阈值（导师方案，四补）；②警觉消退→恢复行走联动（四补2）。
情境分析（四补3）在最新版本上运行。**建议正式校准前在最终版本上复跑一次基础弹性矩阵
（155 次，约 45 分钟）以对齐 aₖ 权重的数字口径。**

## 五、交付物

| 文件 | 内容 |
|---|---|
| `S1_param_to_mech.csv` | 9 参数 × 10 机制变量弹性（段1） |
| `S_extended_param_to_output.csv` | 9 参数 × 18 行为指标弹性（含事件条件新指标） |
| `S2_mech_to_output.csv` | 10 机制变量 × 18 行为指标跨运行 Pearson 相关（段2） |
| `memory_th_sweep.csv` / `.png` | MEMORY_TH 阈值扫描数据与四联图 |
| `safety_weight_sweep.csv` / `.png` | SEEV_SAFETY 全量程扫描（四补） |
| `decay_sweep.csv` / `.png` | LOOMING_DECAY 恢复联动扫描（四补2） |
| `scenario_S_matrix.csv` | 9 参数 × 23 情境指标弹性矩阵（四补3） |
| `scenario_summary.csv` | 参数 × 5 情境汇总（块内最大弹性） |
| `scenario_heatmap.png` | 情境热力图（展示版：7 参数 × 四情境，路口列略去，vmax=4 封顶；路口数据保留于 CSV） |
| `scenario_runs_raw.csv` | 情境分析 100 次原始明细 |
| `mech_runs_raw.csv` | 机制分析 115 次原始明细（含机制变量、种子） |
| `sensitivity_mechanism.py` / `scenario_sensitivity.py` / `scan_safety.py` / `scan_decay.py` | 分析脚本 |
