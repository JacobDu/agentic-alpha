# SFA-YYYY-MM-DD-XX

## round_meta
- round_id: SFA-YYYY-MM-DD-XX
- round_type: sfa
- date: YYYY-MM-DD
- owner: <agent/user>
- market: csi1000

## hypothesis
- hypothesis: <一句话假设>
- market_logic: <为何可能有效>
- expected_direction: <正/负/绝对值增强>

## factor_expression_list
- factor_1: <name> = <expression>

## preflight_gate
- parse_ok: pass/fail
- complexity_level: low/medium/high
- redundancy_flag: low/medium/high
- data_availability: pass/fail
- gate_notes: <异常与修复建议>

## experiment_config
- script: .agents/skills/qlib-single-factor-mining/scripts/<script>.py
- start: YYYY-MM-DD
- end: YYYY-MM-DD
- run_id: <mlflow_run_id or N/A>
- output_paths:
  - outputs/...

## single_factor_metrics
- rank_ic_mean: <float>
- rank_ic_t: <float>
- fdr_p: <float>
- rank_icir: <float>
- n_days: <int>
- sfa_result: pass/fail

## decision
- decision: Promote/Iterate/Drop
- decision_basis: <必须引用单因子指标 + 门控结果>
- failure_mode: <若 Iterate/Drop 必填>
- next_action: <下一轮动作>

## evidence_links
- db_query: <factor_db_cli 命令或结果键>
- run_id: <mlflow run id>
- outputs:
  - outputs/...
