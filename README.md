# Qlib Factor Testing Guide

本项目用于在 `uv + Python 3.12` 环境下，基于 `qlib` 进行因子测试与快速回测验证。

## 1. qlib 是否可以合成因子？

可以。`qlib` 支持通过表达式把基础字段和算子组合成新因子，例如：

- 动量类：`Ref($close, 1)/Ref($close, 5)-1`
- 波动类：`Std($close, 20)/$close`
- 相关性类：`Corr($close, Log($volume+1), 20)`
- 多因子线性合成：`0.6*(Ref($close,1)/$close) + 0.4*(Std($close,20)/$close)`

你也可以像本项目一样，在自定义 Handler 里追加因子（见 `src/project_qlib/factors/custom_alpha158.py`）。

## 2. 环境准备

```bash
cd /Volumes/Workspace/agentic-alpha
uv sync
```

## 3. 数据准备

```bash
uv run python /Volumes/Workspace/agentic-alpha/scripts/prepare_data.py
```

默认数据目录：

- `/Volumes/Workspace/agentic-alpha/data/qlib/cn_data`

数据来源：

- [investment_data releases](https://github.com/chenditc/investment_data/releases/latest)

网络失败时，脚本会自动回退代理：

- `http://127.0.0.1:7890`

## 4. 运行官方基线（Alpha158）

```bash
uv run python /Volumes/Workspace/agentic-alpha/scripts/run_official.py
```

输出：

- `/Volumes/Workspace/agentic-alpha/outputs/official_result.json`
- `/Volumes/Workspace/agentic-alpha/outputs/logs/official_lightgbm.log`

说明：优先 LightGBM，失败自动降级到 XGBoost。

## 5. 运行自定义因子流程

```bash
uv run python /Volumes/Workspace/agentic-alpha/scripts/run_custom_factor.py
```

当前自定义因子：

- `CSTM_MOM_5 = Ref($close, 1)/Ref($close, 5)-1`

输出：

- `/Volumes/Workspace/agentic-alpha/outputs/custom_factor_result.json`
- `/Volumes/Workspace/agentic-alpha/outputs/logs/custom_factor_lightgbm.log`

脚本会检查 `CSTM_MOM_5` 是否真实进入训练数据。

## 6. 一键验收

```bash
uv run python /Volumes/Workspace/agentic-alpha/scripts/verify_all.py
```

输出：

- `/Volumes/Workspace/agentic-alpha/outputs/verification_report.json`
- `/Volumes/Workspace/agentic-alpha/outputs/verification_report.md`

当 `overall_success=true` 时，表示环境、数据、官方流程、自定义因子流程全部通过。

## 7. 如何新增并测试你自己的因子

1. 在 `src/project_qlib/factors/custom_alpha158.py` 中追加表达式和名称。
2. 运行：

```bash
uv run python /Volumes/Workspace/agentic-alpha/scripts/run_custom_factor.py
```

3. 查看结果：
- `outputs/custom_factor_result.json`（是否成功、选用模型、因子检查）
- `mlruns/` 中 `IC/RankIC/IR/年化超额` 等指标

## 8. 指标解读建议

- `IC / RankIC`：预测排序能力（越大越好）
- `ICIR / RankICIR`：稳定性（均值/波动）
- `excess_return_with_cost`：考虑交易成本后的真实可交易性

建议先看 `with_cost` 指标，再看 `IC/RankIC`，避免只看样本内收益。

## 9. 项目关键文件

- 自定义因子：`/Volumes/Workspace/agentic-alpha/src/project_qlib/factors/custom_alpha158.py`
- 运行时配置：`/Volumes/Workspace/agentic-alpha/src/project_qlib/runtime.py`
- 官方配置：`/Volumes/Workspace/agentic-alpha/configs/workflow_official_lightgbm_quick.yaml`
- 自定义配置：`/Volumes/Workspace/agentic-alpha/configs/workflow_custom_factor_lightgbm_quick.yaml`
- 入口脚本：`/Volumes/Workspace/agentic-alpha/scripts/`

## 10. 如何查看 `mlruns` 数据

`qlib` 的训练与回测结果会落到：

- `/Volumes/Workspace/agentic-alpha/mlruns`

目录结构（简化）：

- `mlruns/1/<run_id>/metrics/`：指标（IC、Rank IC、IR、年化超额等）
- `mlruns/1/<run_id>/params/`：本次运行参数（模型、数据集、时间段、handler）
- `mlruns/1/<run_id>/artifacts/`：产物（`pred.pkl`、`label.pkl`、`params.pkl`）
- `mlruns/1/<run_id>/meta.yaml`：运行元数据（开始/结束时间、状态）
- `mlruns/1/<run_id>/tags/`：运行标签

### 10.1 用命令行快速看

查看 run 列表：

```bash
find /Volumes/Workspace/agentic-alpha/mlruns/1 -maxdepth 1 -type d
```

查看某个 run 的关键指标：

```bash
cat /Volumes/Workspace/agentic-alpha/mlruns/1/<run_id>/metrics/IC
cat /Volumes/Workspace/agentic-alpha/mlruns/1/<run_id>/metrics/Rank\\ IC
cat /Volumes/Workspace/agentic-alpha/mlruns/1/<run_id>/metrics/1day.excess_return_with_cost.information_ratio
```

说明：`metrics` 文件每行格式为 `timestamp value step`，通常最后一行是最终值。

查看本次 run 用了哪个 handler（是否包含自定义因子）：

```bash
cat /Volumes/Workspace/agentic-alpha/mlruns/1/<run_id>/params/dataset.kwargs.handler.class
cat /Volumes/Workspace/agentic-alpha/mlruns/1/<run_id>/params/dataset.kwargs.handler.module_path
```

### 10.2 用 Python 汇总多个 run（推荐）

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

root = Path('/Volumes/Workspace/agentic-alpha/mlruns/1')
for run in root.iterdir():
    if not run.is_dir() or not (run / 'meta.yaml').exists():
        continue
    meta = yaml.safe_load((run / 'meta.yaml').read_text())
    metric_file = run / 'metrics' / 'IC'
    ic = None
    if metric_file.exists():
        line = metric_file.read_text().strip().splitlines()[-1]
        ic = float(line.split()[1])
    handler = (run / 'params' / 'dataset.kwargs.handler.class')
    handler_name = handler.read_text().strip() if handler.exists() else 'N/A'
    print(run.name, handler_name, ic, meta.get('status'))
PY
```

### 10.3 用 UI 可视化查看

推荐方式（已处理 macOS 兼容问题）：

```bash
/Volumes/Workspace/agentic-alpha/scripts/mlflow_ui.sh 5000
```

等价原始命令：

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run mlflow ui --backend-store-uri file:///Volumes/Workspace/agentic-alpha/mlruns --host 127.0.0.1 --port 5000
```

然后浏览器打开：

- `http://127.0.0.1:5000`

可以对比不同 run 的参数、指标、产物与时间线。

### 10.4 常见问题（你遇到的这个）

如果直接执行 `uv run mlflow ui ...` 后页面打不开，日志里出现：

- `objc ... initialize ... when fork() was called`
- `Worker was sent SIGKILL`

这是 macOS 下 Gunicorn `fork` 与 Objective-C 运行时的兼容问题，不是 mlruns 数据损坏。  
解决方式就是在启动前设置：

- `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`

建议直接使用本项目脚本 `scripts/mlflow_ui.sh`，避免重复踩坑。
