# HEA Index

本文件用于汇总每轮 HEA 结果，按 `hea_round` 时间倒序维护。

| hea_round | date | hypothesis_short | market | layer_a_result | layer_b_result | decision | evidence |
|-----------|------|------------------|--------|----------------|----------------|----------|----------|
| HEA-2026-02-18-06 | 2026-02-18 | 5维度系统优化(调仓/TopK/标签/损失/集成)：**top30_drop5_hold20最优**(IR=+1.564, Ret=+17.23%)，hold_thresh是第一驱动因素 | csi1000 | pass (IC=0.035) | pass (IR=+1.564) | Promote | outputs/optimization_results.json |
| HEA-2026-02-18-05 | 2026-02-18 | TopN因子LightGBM组合：**TopN30最优**(IR=-0.47 vs baseline-0.83改善44%)，30因子为最优点，>50因子噪声退化(倒U形) | csi1000 | pass (RankIC优于baseline) | fail (neg excess) | Iterate | outputs/topn_comparison.json |
| HEA-2026-02-18-04 | 2026-02-18 | Alpha158+30HEA因子LightGBM训练回测，IC持平但组合超额改善46%(含成本-11.86%→-6.37%)，仍为负超额需优化调仓频率 | csi1000 | pass (IC=0.037) | fail (neg excess) | Iterate | outputs/logs/phase2_*.log |
| HEA-2026-02-18-03 | 2026-02-18 | 正交因子组合20个，19/20显著(95%)，COMP_TRIPLE_TQ_CV_GAP登顶(ICIR=-0.469)，较单因子最优提升19% | csi1000 | pass (19/20 sig) | not_run | Promote | outputs/composite_factor_results.csv |
| HEA-2026-02-18-02 | 2026-02-18 | 正交设计11类33因子，22/33显著(p<0.01)，ORTH_GAP_FREQ_20进入Top4(ICIR=-0.364)，ORTH vs EXIST avg\|ρ\|=0.294 | csi1000 | pass (22/33 sig) | not_run | Promote | outputs/ortho_factor_batch_results.csv |
| HEA-2026-02-18-01 | 2026-02-18 | 量价+估值9类41因子批量筛选，32/41显著(p<0.01)，NEW_TURN_QUANTILE_120登顶#1(ICIR=-0.394) | csi1000 | pass (32/41 sig) | not_run | Promote | outputs/new_factor_batch_results.csv |

维护规则：
1. 每新增一轮 `docs/heas/HEA-YYYY-MM-DD-XX.md`，必须同步在本表新增一行。
2. 新轮次中 `evidence` 若引用脚本，统一使用 `.agents/skills/<skill>/scripts/<script>` 路径。
3. 历史单轮 HEA 文档允许保留旧路径，不强制回改。
4. `evidence` 至少包含一个可追溯标识：`run_id`、`outputs/...` 路径或 `factor_db_cli` 查询命令。
5. `decision` 仅允许 `Promote` / `Iterate` / `Drop`。
