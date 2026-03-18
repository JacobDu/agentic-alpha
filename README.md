# Qlib Factor Research (Skill-Driven)

本项目使用 `uv + Python 3.12 + Qlib` 进行因子研究。
核心编排在 `AGENTS.md`，执行细节在各 skill。

## Skills

- 环境与数据：`$qlib-env-data-prep`
- 单因子挖掘（SFA）：`$qlib-single-factor-mining`
- 多因子组合回测（MFA）：`$qlib-multi-factor-backtest`
- 研报因子研究：`$factor-research-review`

## 核心工作流（R-G-E-V-D）

```
Research → Retrieve → Generate → Evaluate → Validate → Distill
```

- **Research**：从研报/论文提取因子创意 → `$factor-research-review`
- **SFA 路由**：`$qlib-env-data-prep` → `$qlib-single-factor-mining`（含 Validate 深度验证）
- **MFA 路由**：`$qlib-env-data-prep` → `$qlib-multi-factor-backtest`（日频 + 周频双策略）

## 快速开始

```bash
uv sync
uv run python .agents/skills/qlib-env-data-prep/scripts/prepare_data.py
uv run python .agents/skills/qlib-env-data-prep/scripts/check_data.py
uv run python .agents/skills/qlib-env-data-prep/scripts/verify_all.py
```

## 常用命令

### SFA（单因子挖掘）

```bash
# 基础 IC 显著性测试（Evaluate 阶段）
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_ic.py --market csi1000

# IC 衰减分析（Validate 阶段）—— 多 horizon IC 评估因子持久性
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_ic_decay.py --market csi1000 --top-n 50 --backfill

# 因子稳定性分析（Validate 阶段）—— rolling IC 检测因子退化
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_stability.py --market csi1000 --top-n 50 --window 60

# 因子相关性分析（Validate 阶段）—— 截面 Spearman 相关矩阵
uv run python .agents/skills/qlib-single-factor-mining/scripts/analyze_factor_correlation.py --market csi1000 --top-n 30

# 批量测试新因子
uv run python .agents/skills/qlib-single-factor-mining/scripts/test_new_factor_batch.py --market csi1000

# 查看 SFA 记录
uv run python .agents/skills/qlib-single-factor-mining/scripts/sfa_record_cli.py list --top 20
```

### MFA（多因子组合回测）

```bash
# 日频 TopkDropout 回测（默认策略）
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_topn_comparison.py

# 周频调仓回测（5d label + 每5日调仓，适合周级持股）
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/train_weekly_rebal.py --market csi1000 --topk 20 --rebal-days 5

# 周频参数扫描（topk × rebal_days × label_horizon 全网格）
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/train_weekly_rebal.py --market csi1000 --sweep

# 查看 MFA 记录
uv run python .agents/skills/qlib-multi-factor-backtest/scripts/mfa_record_cli.py list --top 20
```

### Validate 阶段详解

Validate 是 Evaluate 之后、Distill 之前的深度验证环节，包含 4 项检查：

| 检查项 | 脚本 | 输出 | 门槛 |
|--------|------|------|------|
| IC 衰减 | `test_factor_ic_decay.py` | `factor_ic_decay` 表 | 必须有 1d+5d 记录 |
| 周适配 | （同上）| `weekly_suitable` 标记 | `5d_ICIR/1d_ICIR >= 0.5` |
| 稳定性 | `test_factor_stability.py` | `stability_score` | `>= 0.3`（否则 Drop） |
| 近期有效 | （同上）| `ic_recent_vs_full` | `>= 0.5`（否则不入池） |

### 因子库与工作流查询

```bash
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py summary --market csi1000
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py runs list --type sfa --top 20
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py similarity show --market csi1000 --top 20
uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py replace history --market csi1000 --top 20
```

### 因子创意管理

```bash
# 因子创意文档位于 docs/factor_ideas/ 目录
# - backlog.md：待测试的因子假设队列
# - literature_notes.md：研报/论文阅读笔记
# - rejected_ideas.md：已淘汰的因子想法

# 使用 $factor-research-review skill 从研报中提取因子构造思路
# 输出因子假设卡片（YAML），可批量传递给 test_new_factor_batch.py
```

## 日频 vs 周频策略对比

| 维度 | 日频（TopkDropout） | 周频（Weekly Rebalance） |
|------|---------------------|--------------------------|
| Label | 1d forward return | 5d forward return |
| 调仓频率 | 每日 | 每 5 个交易日 |
| 换手控制 | `hold_thresh` 参数 | 自然低换手 |
| 因子要求 | 全因子池 | 仅 `weekly_suitable=True` |
| SOTA 参数 | topk=20, n_drop=2, hold=80 | 待验证 |
| 适用场景 | 高频信号开发 | 实盘周级持仓 |

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
- 因子创意：`docs/factor_ideas/`
- 历史文档：`docs/heas/`
- 因子库：`data/factor_library.db`
