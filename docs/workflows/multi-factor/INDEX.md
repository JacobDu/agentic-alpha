# Multi-Factor Workflow Index

| round_id | date | hypothesis_short | market | ir_with_cost | mfa_result | decision | doc |
|----------|------|------------------|--------|--------------|------------|----------|-----|
| MFA-2026-02-19-01 | 2026-02-19 | hold_thresh完整扫描(1d/2d/LGB/XGB，topk=30) | csi1000 | LGB+1d+hold=20: IR=1.564; XGB+1d+hold=10: IR=0.743 | hold=20最优(双峰效应)，XGB hold=10低回撤备选 | Promote | [MFA-2026-02-19-01.md](MFA-2026-02-19-01.md) |
| MFA-2026-02-18-01 | 2026-02-18 | SFA-07新因子（8方向）融入Top30/50多因子组合 | csi1000 | -0.189（新Top30）vs 1.564（旧Top30） | 新因子未带来增量，旧Top30最优 | Iterate | [MFA-2026-02-18-01.md](MFA-2026-02-18-01.md) |

维护规则：
1. 每新增一轮 MFA 记录，需同步更新本索引。
2. evidence 至少包含 doc/output_path/db_query/run_id 之一。
3. decision 仅允许 Promote / Iterate / Drop。
4. 记录结构遵循 R-G-E-D：Retrieve/Generate/Evaluate/Distill。
5. 每轮至少给出线性+非线性方案对照与含成本指标。
