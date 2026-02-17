# HEA Index

本文件用于汇总每轮 HEA 结果，按 `hea_round` 时间倒序维护。

| hea_round | date | hypothesis_short | market | layer_a_result | layer_b_result | decision | evidence |
|-----------|------|------------------|--------|----------------|----------------|----------|----------|

维护规则：
1. 每新增一轮 `docs/heas/HEA-YYYY-MM-DD-XX.md`，必须同步在本表新增一行。
2. `evidence` 至少包含一个可追溯标识：`run_id`、`outputs/...` 路径或 `factor_db_cli` 查询命令。
3. `decision` 仅允许 `Promote` / `Iterate` / `Drop`。
