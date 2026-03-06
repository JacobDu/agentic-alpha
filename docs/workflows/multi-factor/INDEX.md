# Multi-Factor Workflow Index

| round_id | date | hypothesis_short | market | ir_with_cost | mfa_result | decision | doc |
|----------|------|------------------|--------|--------------|------------|----------|-----|
| MFA-V5-2026-03-05 | 2026-03-05 | 数据刷新(2026-03-05 release)+SOTA验证,OOS延伸至2026-03-04 | csi1000 | 1.589 | 年化+21.75%, IR=1.589, DD=-8.76%, 累计+25.60% | Promote | docs/workflows/multi-factor/MFA-V5-2026-03-05.md |
| MFA-V6-2026-03-06 | N/A | N/A | N/A | 1.847 | Promote: tk20_d2_h80 IR=1.847 Ret=+33.28% DD=-11.66% Turn=1.53% | Promote | N/A |

维护规则：
1. 每新增一轮 MFA 记录，需同步更新本索引。
2. evidence 至少包含 doc/output_path/db_query/run_id 之一。
3. decision 仅允许 Promote / Iterate / Drop。
