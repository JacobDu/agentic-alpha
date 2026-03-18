---
name: factor-research-review
description: 负责从量化研报、学术论文中提取因子构造思路，转化为可测试的 Qlib 因子表达式，并生成结构化的因子假设卡片。用于因子创意来源拓展、文献回顾和外部信号引入。
---

# 因子研报研究（Factor Research Review）

本 skill 负责从外部研究资料中系统性提取因子构造思路，并转化为 SFA 流程的输入。

## 核心流程

### 1) Source（来源识别）
1. 识别研报 / 论文中的因子构造逻辑。
2. 分类来源类型：
   - **券商金工研报**：国内量化因子类研报
   - **学术论文**：JFE/RFS/JF 等顶刊 anomaly 研究
   - **开源社区**：WorldQuant Alpha101、聚宽因子库等

### 2) Extract（信息提取）
从研报中提取结构化信息：
1. **因子名称**与分类（动量/反转/波动率/流动性/微观结构等）
2. **构造逻辑**：文字描述 + 数学公式
3. **市场假设**：因子背后的经济学逻辑（行为金融/风险补偿/信息不对称等）
4. **适用范围**：市场/频率/市值范围
5. **历史表现**：研报中报告的 IC/ICIR/收益等（仅作参考，不作为决策依据）

### 3) Translate（转化为 Qlib 表达式）
1. 将高级描述转化为 Qlib Expression Engine 语法
2. 可用算子：`Mean`, `Std`, `Corr`, `Rank`, `Ref`, `Min`, `Max`, `Sum`, `Abs`, `Log`, `Power`, `Sign`, `If`, `Greater`, `Less`
3. 可用字段：`$open`, `$close`, `$high`, `$low`, `$volume`, `$amount`, `$vwap`, `$change`, `$factor`
4. 确保表达式可解析、无除零风险

### 4) Hypothesize（生成因子假设卡片）
输出结构化的因子假设卡片，作为 SFA Generate 阶段的输入：

```yaml
factor_hypothesis:
  name: "CSTM_XXX"
  expression: "..."
  category: "momentum|volatility|liquidity|..."
  source: "研报名称/论文引用"
  market_logic: "因子背后的经济学假设"
  expected_direction: "positive|negative"  # 预期 IC 方向
  expected_horizon: "1d|5d|20d"  # 预期最佳持有期
  priority: "high|medium|low"
  risks: "潜在失效风险"
```

## 因子分类体系

### 价量类（Price-Volume）
| 大类 | 子类 | 典型因子 | 经济学逻辑 |
|------|------|----------|------------|
| 动量 | 短期反转 | 1-5日反转 | 流动性溢价/过度反应 |
| 动量 | 中期动量 | 20-60日动量 | 趋势跟随/信息扩散 |
| 波动率 | 特质波动 | 残差波动率 | 博彩偏好/波动率异象 |
| 波动率 | 波动率变化 | 波动率扩张/收缩 | 风险预期修正 |
| 流动性 | 成交量异动 | 量比/换手率偏离 | 知情交易/注意力 |
| 流动性 | 量价关系 | 量价相关/Amihud | 信息不对称 |
| 微观结构 | 日内模式 | 高低价位置/缺口 | 交易行为/情绪 |

### 基本面类（Fundamental，需额外数据）
| 大类 | 子类 | 典型因子 |
|------|------|----------|
| 估值 | PE/PB/PS | 需 PIT 数据 |
| 成长 | ROE变化/营收增速 | 需财务数据 |
| 质量 | 应计/现金流 | 需财务数据 |

## 常见研报因子转化示例

### 1. 聪明钱因子（Smart Money）
- **逻辑**：大单/高金额交易蕴含更多信息
- **Qlib 表达式**：
  ```
  Mean(($close/Ref($close,1)-1) * $amount, 10) / (Mean($amount, 10) + 1e-8)
  ```
- **预期方向**：正（高聪明钱得分→预期上涨）

### 2. 波动率压缩因子
- **逻辑**：波动率收窄后通常伴随方向性突破
- **Qlib 表达式**：
  ```
  Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-8)
  ```
- **预期方向**：负（低波动压缩比→即将突破）

### 3. 量价背离因子
- **逻辑**：价格上涨但成交量萎缩，趋势可能反转
- **Qlib 表达式**：
  ```
  Corr($close/Ref($close,1)-1, $volume/Ref($volume,1)-1, 10)
  ```
- **预期方向**：负（低量价相关→反转信号）

## 研报解读要点

### 阅读研报时关注：
1. **样本外表现**：区分 IS vs OOS 结果
2. **过拟合风险**：因子个数/参数搜索空间
3. **交易成本**：是否考虑了滑点和冲击成本
4. **市值偏差**：因子是否仅在小市值有效
5. **时变性**：因子有效性是否随时间衰减
6. **CSI1000 适用性**：研报可能基于全A/CSI500，需评估在 CSI1000 上的有效性

### 不要盲目复制：
1. 研报的回测环境可能与当前框架不同
2. 因子在不同市值/行业/市场阶段表现差异大
3. 必须通过本框架 SFA 流程验证后才能 Promote

## 因子创意工作台

在 `docs/factor_ideas/` 目录维护因子创意池：
- `backlog.md`：待测试的因子假设列表
- `literature_notes.md`：研报/论文阅读笔记
- `rejected_ideas.md`：已测试但无效的想法（避免重复）

## 输出

1. 因子假设卡片（YAML 格式）→ 供 SFA Generate 使用
2. 研报阅读笔记 → `docs/factor_ideas/literature_notes.md`
3. 因子创意 backlog → `docs/factor_ideas/backlog.md`

## 临时脚本边界

1. 本 skill 不包含运行脚本，仅产出因子假设和文档。
2. 因子测试由 `qlib-single-factor-mining` skill 执行。
3. 因子假设卡片可批量传递给 `test_new_factor_batch.py` 进行快筛。
