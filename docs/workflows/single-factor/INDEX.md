# Single-Factor Workflow Index

| round_id | date | hypothesis_short | market | rank_icir | sfa_result | decision | doc |
|----------|------|------------------|--------|-----------|------------|----------|-----|
| HEA-2026-02-18-06 | 2026-02-18 | 通过系统优化调仓频率(hold_thresh)、TopK/n_drop组合、标签期限、损失函数、模型集成五个维度，可将To... | csi1000 | 0.1888 | pass (IC/RankIC均显著为正) | Promote | docs/heas/HEA-2026-02-18-06.md |
| HEA-2026-02-18-05 | 2026-02-18 | 从因子库中按 |RankICIR| 排名选取 Top-N 因子（N=20/30/50/80），仅用这些因子训练 Light... | csi1000 | N/A | pass (所有模型 IC > 0.02) | Iterate | docs/heas/HEA-2026-02-18-05.md |
| HEA-2026-02-18-04 | 2026-02-18 | 将 HEA 三轮筛选的 Top-30 因子（20 single + 10 composite）加入 Alpha158 基线... | csi1000 | N/A | pass (IC > 0.03, p significant) | Iterate | docs/heas/HEA-2026-02-18-04.md |
| HEA-2026-02-18-03 | 2026-02-18 | 将正交的高效因子进行组合（加法/乘法/比率/z-score），可产生比单因子更强的预测信号，因为组合因子捕捉了多维度信息的... | csi1000 | N/A | pass (19/20 显著) | Promote | docs/heas/HEA-2026-02-18-03.md |
| HEA-2026-02-18-02 | 2026-02-18 | 通过正交设计（避开已有 volume CV / range / price-vol correlation / extre... | csi1000 | N/A | pass (22/33 显著, 11 个类别中 9 个有显著因子) | Promote | docs/heas/HEA-2026-02-18-02.md |
| SFA-2026-02-18-07 | 2026-02-18 | 基于338因子库空白分析，8个新方向（OFI/PIDE/换手制度/隔夜非对称/日内结构/估值加速/波动制度/量价健康）批量测试40因子，33/40通过，top因子PIDE_HIGHVOL_REVERSAL_10 ICIR=0.322。 | csi1000 | 0.3224 | pass (33/40 p<0.01, 8方向全部有效) | Promote | docs/workflows/single-factor/SFA-2026-02-18-07.md |
| HEA-2026-02-18-01 | 2026-02-18 | 量价+估值字段中存在大量未被开发的因子组合，特别是估值波动率、估值×换手率交互、多尺度均值回归、微观结构、换手率体制变化等领域。 | csi1000 | N/A | pass (32/41 显著, FDR 校正后仍≥30 个通过) | Promote | docs/heas/HEA-2026-02-18-01.md |

维护规则：
1. 每新增一轮 SFA 记录，需同步更新本索引。
2. evidence 至少包含 doc/output_path/db_query/run_id 之一。
3. decision 仅允许 Promote / Iterate / Drop。
4. 记录结构遵循 R-G-E-D：Retrieve/Generate/Evaluate/Distill。
5. 正交预算固定为 max|rho| <= 0.50。
