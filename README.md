# Qlib Factor Research (Skill-Driven)

本项目使用 `uv + Python 3.12 + Qlib` 进行因子研究，执行脚本统一维护在 `.agents/skills` 下。

## Skill 入口

- 环境与数据：`$qlib-env-data-prep`
- 单因子挖掘：`$qlib-single-factor-mining`
- 多因子组合回测：`$qlib-multi-factor-backtest`

## 1) 环境与数据准备

```bash
uv sync
uv run python .agents/skills/qlib-env-data-prep/scripts/prepare_data.py
uv run python .agents/skills/qlib-env-data-prep/scripts/download_financial_data.py --phase 1
uv run python .agents/skills/qlib-env-data-prep/scripts/download_financial_data.py --phase 2
uv run python .agents/skills/qlib-env-data-prep/scripts/check_data.py
uv run python .agents/skills/qlib-env-data-prep/scripts/verify_all.py
```

## 2) 单因子挖掘（SFA）

```bash
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_ic.py --market csi1000
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_new_factors.py --market csi1000 --backfill
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_new_factor_batch.py --market csi1000 --backfill
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_ortho_factor_batch.py --market csi1000 --backfill
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_composite_factors.py --market csi1000 --backfill
```

## 3) 多因子组合回测（MFA）

```bash
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_phase2_comparison.py
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_topn_comparison.py
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_optimization.py --phase 1 --quick
```

可视化：

```bash
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/visualize_factors.py ranking --top 20
```

## 4) 核心工作流与留痕

核心工作流已统一维护在 `AGENTS.md`，按其中流程执行 SFA（单因子）与 MFA（多因子）两条子工作流：
- SFA：`$qlib-env-data-prep` -> `$qlib-single-factor-mining`
- MFA：`$qlib-env-data-prep` -> `$qlib-multi-factor-backtest`

实验记录模板维护在 skill 中：
- 单因子模板：`.agents/skills/qlib-single-factor-mining/assets/templates/single_factor_experiment_record.md`
- 多因子模板：`.agents/skills/qlib-multi-factor-backtest/assets/templates/multi_factor_experiment_record.md`

常用因子库查询命令：

```bash
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py summary --market csi1000
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py list --market csi1000 --top 20
uv run python .agents/skills/qlib-single-factor-mining/scripts/import_factors_to_db.py
```

## 5) 结果与证据

- 实验输出：`outputs/`
- 运行日志：`mlruns/`
- 单因子文档：`docs/workflows/single-factor/`
- 多因子文档：`docs/workflows/multi-factor/`
- 历史 HEA 文档：`docs/heas/`
- 因子库：`data/factor_library.db`
