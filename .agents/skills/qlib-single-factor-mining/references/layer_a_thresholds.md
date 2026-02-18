# Layer A 阈值

## Promote 门槛

- `fdr_p < 0.01`
- `|rank_icir| >= 0.10`
- 结论必须映射到 DB/outputs 证据。

## 决策规则

- `Promote`：硬门槛全部通过。
- `Iterate`：未通过但存在明确修复路径。
- `Drop`：信号不稳定或无清晰修复路径。
