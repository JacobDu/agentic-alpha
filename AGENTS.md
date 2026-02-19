# AGENTS.md

本文件只维护三类信息：
1. 核心工作流编排
2. 硬门槛决策规则
3. 脚本治理规则

## 🎯 项目目标

通过 Agent 持续研究循环，沉淀可复用的高质量因子资产库。
默认市场：`csi1000`。

## 核心工作流

### 单因子挖掘（SFA）
1. 环境与数据门禁：调用 `$qlib-env-data-prep`
2. 单因子实验与统计：调用 `$qlib-single-factor-mining`
3. 结果留痕：写入 `docs/workflows/single-factor/` 与 `data/factor_library.db`

### 多因子组合（MFA）
1. 环境与数据门禁：调用 `$qlib-env-data-prep`
2. 多因子训练与含成本回测：调用 `$qlib-multi-factor-backtest`
3. 结果留痕：写入 `docs/workflows/multi-factor/` 与 `data/factor_library.db`

## 判定矩阵（硬门槛）

| Decision | 必要条件 | 说明 |
|----------|----------|------|
| Promote | `fdr_p < 0.01` 且 `|rank_icir| >= 0.10` 且证据完整 | 进入候选保留池 |
| Iterate | 未达 Promote 但存在可修复空间 | 必填 `failure_mode` 与 `next_action` |
| Drop | 多轮不稳定或显著退化且无修复路径 | 停止继续投入 |

补充规则：
1. 默认允许反向因子，方向以统计显著性和稳定性为准。
2. `FDR < 0.01` 为硬门槛，不得弱化。

## 记录与归档

1. 单因子记录：`docs/workflows/single-factor/SFA-YYYY-MM-DD-XX.md`
2. 多因子记录：`docs/workflows/multi-factor/MFA-YYYY-MM-DD-XX.md`
3. 索引：各目录下 `INDEX.md`
4. 数据库：`data/factor_library.db`

模板来源：
- `.agents/skills/qlib-single-factor-mining/assets/templates/single_factor_experiment_record.md`
- `.agents/skills/qlib-multi-factor-backtest/assets/templates/multi_factor_experiment_record.md`

## 脚本治理

1. Skill 内 `scripts/` 只允许保留“可复用、与单次研究无关”的脚本。
2. 单次研究临时脚本必须放在项目根目录 `./scripts/`。
3. 临时脚本使用完成后必须删除，不得沉淀到 `.agents/skills/*/scripts/`。
4. 若临时脚本产出了证据，保留 `outputs/`、`docs/`、DB 记录即可，不保留脚本本体。
5. 若某临时脚本被复用 2 次及以上，再评估是否升格进对应 skill。

## Agent 行为约束

1. 不得编造数据、指标、run_id。
2. 任何结论必须可追溯到真实文件或数据库记录。
3. 若环境异常，先修复流程可用性，再继续因子研究。
