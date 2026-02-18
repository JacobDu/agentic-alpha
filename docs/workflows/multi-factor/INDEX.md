# Multi-Factor Workflow Index

| round_id | date | hypothesis_short | market | ir_with_cost | mfa_result | decision | doc |
|----------|------|------------------|--------|--------------|------------|----------|-----|

维护规则：
1. 每新增一轮 MFA 记录，需同步更新本索引。
2. evidence 至少包含 doc/output_path/db_query/run_id 之一。
3. decision 仅允许 Promote / Iterate / Drop。
