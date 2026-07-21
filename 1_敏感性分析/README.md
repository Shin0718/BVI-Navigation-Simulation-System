# 1. 敏感性分析（机制层）

管线第一步：机制中介分析 + MEMORY_ACTIVE_RETRIEVAL_TH 全量程阈值扫描，
为第 2 步提供"工作区间 [0.03, 0.08]"等机制证据。

```bash
python 1_敏感性分析/01_sensitivity_mechanism.py
```

**产物**（→ `../sensitivity_out/`）：`memory_th_sweep.csv/.png`（阈值扫描）、
`S1_param_to_mech.csv` / `S2_mech_to_output.csv`（参数→机制→行为两段分析）、`MECHANISM_REPORT.md`。

注意：`2_参数筛选/scan_*.py` 通过 importlib 复用本脚本骨架，勿改文件名。
代码默认值已写回 θ*（2026-07-13），启动校验为提示制，分析按清单参考值显式注入。

`_archive/`：旧版情境敏感性分析（02，五情境切分），已被 cat4 主线替代。
