# Clear_ActR — BVI 导航认知模型：参数校准工程

模型本体在 `bvi_sa/`（ACT-R + SEEV 双层认知仿真）。围绕它的参数校准工作按流程拆成三个文件夹，
外加共享的输出目录。整条链路的方法学见 `../参数校准/校准论文_0708.md`。

## 运行核心模型

```bash
# 单次仿真
python main.py --familiarity 0
python main.py --familiarity 1

# 蒙特卡洛批量仿真（独立出行，不做 pretrain）
python -m bvi_sa.main --profile 2 --mc-runs 30 --seed-start 42
```

## 目录结构

| 目录 | 作用 | 详见 |
|---|---|---|
| `bvi_sa/` | 核心仿真模型（被所有脚本导入，勿移动） | — |
| `参数筛选/` | 决定 15 个高层参数中哪些值得校准（局部弹性初筛 + 全量程扫描） | `参数筛选/README.md` |
| `敏感性分析/` | 机制中介诊断 + 情境敏感性 → 产出损失权重 | `敏感性分析/README.md` |
| `损失/` | 用实测数据校准参数、稳健性/泛化检验、出图 | `损失/README.md` |
| `sensitivity_out/` | 上两步的产物（弹性矩阵、损失权重、热力图等） | — |
| `calib_out/` `calib_out_cv/` | 校准与留出验证的产物 | — |
| `obs_values.csv` | 实测观测值（仿真同口径），损失计算的输入 | — |

## 端到端运行顺序

1. `参数筛选/` → 确定可校准参数集合（产物写入 `sensitivity_out/`）
2. `敏感性分析/01 → 02` → 产出 `sensitivity_out/loss_weights.csv`
3. `损失/` → 读 `loss_weights.csv` + `obs_values.csv`，搜索最优参数 θ*，产物写入 `calib_out/`

## 重要约定

- 所有脚本运行时通过 `setattr(bvi_sa.simulation, ...)` 注入参数，**不改动模型源文件**。
- 脚本移入子文件夹后，各自向上一级定位项目根（含 `bvi_sa/`），
  因此**输出目录始终落在项目根下**（`sensitivity_out/`、`calib_out/`），从任意位置运行都一致。
- `损失/` 内脚本依赖 `敏感性分析/02_scenario_sensitivity.py`，通过 `损失/_setup.py`
  以可导入名 `scenario_sensitivity` 注册（因文件名以数字开头无法直接 import）。
