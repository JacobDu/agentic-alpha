# Qlib Factor Research (Skill-Driven)

本项目使用 `uv + Python 3.12 + Qlib` 进行因子研究。
核心编排在 `AGENTS.md`，执行细节在各 skill。

## Skills

- 环境与数据：`$qlib-env-data-prep`
- 单因子挖掘（SFA）：`$qlib-single-factor-mining`
- 多因子组合回测（MFA）：`$qlib-multi-factor-backtest`

## 核心工作流（R-G-E-D）

- `Retrieve -> Generate -> Evaluate -> Distill`
- SFA 路由：`$qlib-env-data-prep -> $qlib-single-factor-mining`
- MFA 路由：`$qlib-env-data-prep -> $qlib-multi-factor-backtest`

## 快速开始

```bash
uv sync
uv run python .agents/skills/qlib-env-data-prep/scripts/prepare_data.py
uv run python .agents/skills/qlib-env-data-prep/scripts/check_data.py
uv run python .agents/skills/qlib-env-data-prep/scripts/verify_all.py
```

## 常用命令

### SFA

```bash
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_ic.py --market csi1000
uv run python .agents/skills/qlib-single-factor-mining/scripts/sfa_record_cli.py list --top 20
```

### MFA

```bash
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_topn_comparison.py
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/mfa_record_cli.py list --top 20
```

### 因子库与工作流查询

```bash
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py summary --market csi1000
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py runs list --type sfa --top 20
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py similarity show --market csi1000 --top 20
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py replace history --market csi1000 --top 20
```

## 脚本治理

- 可复用脚本仅保留在 `.agents/skills/*/scripts/`
- 一次性脚本仅允许在 `./scripts/`，完成后删除
- 可用治理脚本：

```bash
uv run python .agents/skills/qlib-env-data-prep/scripts/audit_skill_scripts.py
uv run python .agents/skills/qlib-env-data-prep/scripts/cleanup_temp_scripts.py --apply
```

## 结果与证据

- 实验输出：`outputs/`
- 运行日志：`mlruns/`
- SFA 文档：`docs/workflows/single-factor/`
- MFA 文档：`docs/workflows/multi-factor/`
- 历史文档：`docs/heas/`
- 因子库：`data/factor_library.db`
