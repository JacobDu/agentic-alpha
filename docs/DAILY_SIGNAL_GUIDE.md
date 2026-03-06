# Daily Signal 使用指南

使用 `scripts/daily_signal.py` 基于 SOTA 策略（MFA-V6）每日生成 CSI1000 组合仓位信号。

## 策略概要

| 项目 | 配置 |
|------|------|
| 模型 | XGBoost + LightGBM 均值集成 |
| 特征 | Alpha158 + DB 因子 Top30（max_per_cat=5） |
| 策略 | TopkDropout（topk=20, n_drop=2, hold_thresh=80） |
| 重训频率 | 每 3 个月自动触发 |
| 交易成本 | 买入 5bp / 卖出 15bp / 最低 5 元 |
| 默认初始资金 | 100 万元 |

## 前置条件

1. **Python 环境**：项目虚拟环境已安装依赖（`uv sync`）。
2. **Qlib 数据**：`data/qlib/cn_data/` 下已有行情数据。首次可通过 `uv run python src/project_qlib/data.py --force` 下载。
3. **因子库**：`data/factor_library.db` 中已有 csi1000 因子检验结果（用于 TopN 选因子）。

## 快速开始

```bash
# 激活环境
source .venv/bin/activate

# 首次运行：训练模型 + 生成首日信号（耗时约 10~30 分钟）
uv run python scripts/daily_signal.py --init

# 之后每个交易日运行：增量更新数据 + 生成信号
uv run python scripts/daily_signal.py
```

## 命令参考

### 首次初始化

```bash
uv run python scripts/daily_signal.py --init
```

执行流程：
1. 调用 baostock 增量更新行情数据
2. 训练 XGBoost 和 LightGBM 模型
3. 对全市场打分并生成首日买入信号
4. 初始化持仓和交易记录

### 每日运行

```bash
uv run python scripts/daily_signal.py
```

执行流程：
1. 增量更新数据
2. 检查是否满 3 个月需要重训（满则自动重训）
3. 用 Ensemble 模型给全市场打分
4. 基于 TopkDropout 逻辑生成买/卖/持有信号
5. 更新持仓并记录交易日志

如果当日信号已存在，默认跳过。加 `--force` 可强制重新生成：

```bash
uv run python scripts/daily_signal.py --force
```

### 强制重训模型

```bash
uv run python scripts/daily_signal.py --retrain
```

不等 3 个月周期，立即重新训练并保存模型。

### 查看当前状态

```bash
uv run python scripts/daily_signal.py --status
```

输出内容：
- 模型训练时间和区间
- 是否需要重训
- 当前持仓明细（股票、入场日、入场价、股数、成本）
- 历史信号天数和交易记录统计

### 常用选项

| 选项 | 说明 |
|------|------|
| `--init` | 首次初始化（训练 + 首日信号） |
| `--retrain` | 强制重训模型 |
| `--status` | 查看当前状态 |
| `--force` | 强制重新生成今日信号 |
| `--skip-data` | 跳过 baostock 数据更新（使用本地现有数据） |
| `--capital N` | 设置初始资金（默认 1,000,000） |
| `--profile NAME` | 独立实例名称，支持并行运行多套策略 |

## 输出文件

所有输出保存在 `outputs/dryrun/`（使用 `--profile` 时为 `outputs/dryrun-{name}/`）：

```
outputs/dryrun/
├── models/
│   ├── xgb_latest.pkl       # XGBoost 模型
│   └── lgb_latest.pkl       # LightGBM 模型
├── signals/
│   ├── 2026-03-05.json       # 每日信号（含买/卖/持有明细）
│   └── ...
├── portfolio.json            # 当前持仓状态
├── trade_log.csv             # 累计交易记录
└── state.json                # 系统状态（上次训练时间等）
```

### 信号文件格式（signals/YYYY-MM-DD.json）

```json
{
  "date": "2026-03-05",
  "model_date": "2026-03-04",
  "total_scored": 998,
  "portfolio_size_before": 20,
  "portfolio_size_after": 20,
  "capital": 1000000,
  "target_weight": 0.05,
  "target_amount_per_stock": 50000.00,
  "buy": [
    {
      "instrument": "SZ300123",
      "score": 0.012345,
      "rank": 5,
      "price": 25.60,
      "target_amount": 50000.00,
      "shares": 1300,
      "actual_amount": 33280.00,
      "estimated_cost": "0.05%"
    }
  ],
  "sell": [
    {
      "instrument": "SH600456",
      "score": -0.001234,
      "rank": 75,
      "price": 18.30,
      "shares": 1800,
      "estimated_amount": 32940.00,
      "reason": "排名跌出 hold_thresh",
      "estimated_cost": "0.15%"
    }
  ],
  "hold": [
    {
      "instrument": "SZ000789",
      "score": 0.008765,
      "rank": 12,
      "price": 42.10,
      "shares": 700,
      "market_value": 29470.00
    }
  ]
}
```

### 持仓文件格式（portfolio.json）

```json
{
  "holdings": {
    "SZ300123": {
      "entry_date": "2026-03-05",
      "entry_score": 0.012345,
      "entry_rank": 5,
      "shares": 1300,
      "entry_price": 25.60,
      "target_amount": 50000.00
    }
  },
  "cash": 1000000,
  "last_update": "2026-03-05"
}
```

## Profile 多实例

使用 `--profile` 可同时运行多个独立策略实例，每个实例有自己的持仓和信号目录：

```bash
# 默认实例
uv run python scripts/daily_signal.py --init

# 另一个实例，使用 50 万资金
uv run python scripts/daily_signal.py --init --profile conservative --capital 500000

# 查看各实例状态
uv run python scripts/daily_signal.py --status
uv run python scripts/daily_signal.py --status --profile conservative
```

## TopkDropout 策略逻辑

每日信号生成遵循以下逻辑：

1. **全市场打分**：Ensemble 模型对 ~1000 只 CSI1000 成分股打分
2. **排名筛选**：取 Top-80（hold_thresh）为候选池
3. **卖出判定**：当前持仓中排名跌出 Top-80 的股票标记为卖出，每天最多卖 2 只（n_drop）
4. **买入补位**：卖出后持仓不足 20（topk）只时，从 Top-20 中补入新股票
5. **仓位等权**：每只股票目标金额 = 总资金 / 持仓数，按 100 股整手取整

## 典型日常工作流

```
每个交易日开盘前:
  1. uv run python scripts/daily_signal.py
  2. 查看终端输出的买卖信号摘要
  3. 参照信号手动下单（Dry-Run 模式不会自动交易）

每周末:
  uv run python scripts/daily_signal.py --status    # 检查持仓状态

季度（或需要时）:
  uv run python scripts/daily_signal.py --retrain    # 强制重训模型
```

> **注意**：此工具为 Dry-Run（模拟）模式，仅输出交易建议，不会连接券商或自动下单。所有信号需人工核验后手动执行。
