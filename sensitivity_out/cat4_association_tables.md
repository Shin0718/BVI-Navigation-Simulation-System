# 候选参数与行为指标关联表（Supplementary 0712）

## a. 风险感知与风险融合 + 探测与恢复

该组行为指标集：Risk、Risk relief、Probe rate、Response prob、Response delay、Time share (probing)

|Parameter|Describe|Value|Key associated indicators (弹性≥0.1)|
|---|---|---|---|
|SEEV_VALUE_SAFETY_WEIGHT|安全价值权重|0.7|Response delay (0.48), Risk relief (0.12)|
|SEEV_VALUE_PROGRESS_WEIGHT|进展价值权重|0.3|Response delay (0.48), Response prob (0.14)|
|SEEV_EXPECTANCY_RISK_WEIGHT|风险预期权重|0.55|Response delay (0.48), Response prob (0.14)|
|ACTR_RISK_LANDMARK_RELIEF|地标对风险的缓解系数|0.08|Risk relief (0.13)|
|PROBE_RELIEF_RATIO|探测行为对风险/负荷的缓解比例|0.4|Post-trigger probe (1.25), Time share (probing) (0.93), Probe rate (0.71)|
|ACTR_LOAD_RESUME_THRESHOLD|认知负荷恢复阈值|5.0|— (所有关联指标弹性 < 0.1)|

注：三个 SEEV 权重机制耦合（SAFETY+PROGRESS=1），敏感性筛选与校准中作为一个组合单元处理。

## b. 记忆检索

该组行为指标集：Retrieval rate、Episode length、Probe rate、Post-trigger probe、Workload

|Parameter|Describe|Value|Key associated indicators (弹性≥0.1)|
|---|---|---|---|
|MEMORY_ACTIVE_RETRIEVAL_TH|主动记忆检索阈值|0.15|Post-trigger probe (14.40), Probe rate (6.03), Workload (2.57), Retrieval rate (1.28), Episode length (0.37)|
|MEMORY_ACTIVE_ABSENT_STEPS_TH|参照缺失连续步数阈值|11|Retrieval rate (5.81), Post-trigger probe (4.26), Workload (0.44), Probe rate (0.36), Episode length (0.33)|
|LANDMARK_DECAY_RATE|地标记忆衰减率|0.82|Post-trigger probe (1.79), Retrieval rate (0.54), Probe rate (0.12)|

## c. 动态风险

该组行为指标集：Response prob、Response delay、Probe rate、Workload、Time share (vehicle)

|Parameter|Describe|Value|Key associated indicators (弹性≥0.1)|
|---|---|---|---|
|VEHICLE_APPROACH_SALIENCE_GATE_SIDEWALK|车辆逼近有效证据显著性阈值|0.4|Time share (vehicle) (0.15), Response prob (0.14)|
|LOOMING_BOOST_PEAK|车辆逼近显著性峰值增量|0.35|Response delay (0.92), Response prob (0.69), Probe rate (0.12)|
|LOOMING_BOOST_DECAY|车辆逼近显著性每步衰减系数|0.8|Response delay (1.38), Response prob (0.37), Time share (vehicle) (0.23)|

