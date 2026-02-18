# MFA-YYYY-MM-DD-XX

## round_meta
- round_id: MFA-YYYY-MM-DD-XX
- round_type: mfa
- date: YYYY-MM-DD
- owner: <agent/user>
- market: csi1000

## portfolio_hypothesis
- hypothesis: <组合策略假设>
- market_logic: <组合为何有效>
- expected_direction: <超额/IR/回撤方向预期>

## factor_pool_source
- source_query: <factor_db_cli 查询或固定清单>
- n_factors: <int>
- factor_pool_notes: <筛选规则>

## preflight_gate
- parse_ok: pass/fail
- complexity_level: low/medium/high
- redundancy_flag: low/medium/high
- data_availability: pass/fail
- gate_notes: <异常与修复建议>

## experiment_config
- script: .agents/skills/qlib-multi-factor-backtest/scripts/<script>.py
- start: YYYY-MM-DD
- end: YYYY-MM-DD
- run_id: <mlflow_run_id or N/A>
- output_paths:
  - outputs/...

## multi_factor_metrics
- executed: yes/no
- not_run_reason: <若未执行必填>
- excess_return_with_cost: <float>
- ir_with_cost: <float>
- max_drawdown: <float>
- mfa_result: pass/fail/not_run

## decision
- decision: Promote/Iterate/Drop
- decision_basis: <必须引用多因子指标 + 门控结果>
- failure_mode: <若 Iterate/Drop 必填>
- next_action: <下一轮动作>

## evidence_links
- db_query: <factor_db_cli 命令或结果键>
- run_id: <mlflow run id>
- outputs:
  - outputs/...
