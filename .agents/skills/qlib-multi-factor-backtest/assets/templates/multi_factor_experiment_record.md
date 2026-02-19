# MFA-YYYY-MM-DD-XX

## round_meta
- round_id: MFA-YYYY-MM-DD-XX
- round_type: mfa
- date: YYYY-MM-DD
- owner: <agent/user>
- market: csi1000

## retrieve
- pool_source_query: <factor_db_cli query>
- stable_pool_query: <factor_db_cli query>
- memory_refs:
  - AGENTS.md#推荐方向
  - AGENTS.md#禁止方向

## generate
- hypothesis: <组合策略假设>
- market_logic: <组合为何有效>
- expected_direction: <超额/IR/回撤方向预期>
- model_families:
  - linear: <lasso/linear-weight/...>
  - nonlinear: <lgb/xgb/...>
- factor_pool_notes: <筛选规则>

## evaluate
- script: .agents/skills/qlib-multi-factor-backtest/scripts/<script>.py
- start: YYYY-MM-DD
- end: YYYY-MM-DD
- run_id: <mlflow_run_id or N/A>
- output_paths:
  - outputs/...
- executed: yes/no
- not_run_reason: <若未执行必填>
- excess_return_with_cost: <float>
- ir_with_cost: <float>
- max_drawdown: <float>
- stress_test_notes: <成本/参数压力测试结果>
- mfa_result: pass/fail/not_run

## distill_decision
- decision: Promote/Iterate/Drop
- decision_basis: <必须引用多因子指标 + 门控结果>
- failure_mode: <若 Iterate/Drop 必填>
- next_action: <下一轮动作>

## distill_evidence
- db_query: <factor_db_cli 命令或结果键>
- run_id: <mlflow run id>
- outputs:
  - outputs/...
- doc: docs/workflows/multi-factor/MFA-YYYY-MM-DD-XX.md
