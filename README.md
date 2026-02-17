# Qlib Factor Testing Guide

本项目用于在 `uv + Python 3.12` 环境下，基于 `qlib` 进行因子测试与快速回测验证。

## 1. 环境准备

```bash
cd <PROJECT_ROOT>
uv sync
```

## 2. 数据准备

```bash
uv run python scripts/prepare_data.py
```

默认数据目录：`data/qlib/cn_data`

数据来源：[investment_data releases](https://github.com/chenditc/investment_data/releases/latest)

网络失败时，脚本会自动回退代理：`http://127.0.0.1:7890`

## 3. 运行官方基线（Alpha158）

```bash
uv run python scripts/run_official.py
```

说明：优先 LightGBM，失败自动降级到 XGBoost。

## 4. 运行自定义因子流程

```bash
uv run python scripts/run_custom_factor.py
```

## 5. 一键验收

```bash
uv run python scripts/verify_all.py
```

当 `overall_success=true` 时，表示环境、数据、官方流程、自定义因子流程全部通过。

## 6. 因子库（SQLite）

项目使用 SQLite 数据库 `data/factor_library.db` 持久化所有因子的定义和测试结果。

```bash
uv run python scripts/factor_db_cli.py summary                     # 总览
uv run python scripts/factor_db_cli.py list --market csi1000        # 因子列表
uv run python scripts/factor_db_cli.py show CSTM_VOL_CV_10          # 单因子详情
uv run python scripts/factor_db_cli.py markets                      # 各市场统计
```

因子状态流转：`Candidate → Tested → Accepted / Rejected / Baseline`

## 7. 因子可视化分析工具

`scripts/visualize_factors.py` 提供因子效果的可视化分析。

### 7.1 快速入门：因子综合报告

输入因子名称即可生成包含 5 张图表的综合报告，自动在浏览器打开：

```bash
uv run python scripts/visualize_factors.py CSTM_MAX_RET_20
uv run python scripts/visualize_factors.py CSTM_AMT_CV_20 --groups 10 --market csi1000
```

### 7.2 单独图表命令

也可以单独生成某一类图表：

```bash
# 因子排名（按 |ICIR| 排序的 Top-N 因子横向柱状图）
uv run python scripts/visualize_factors.py ranking --top 20

# IC 时间序列（某因子的每日 Rank IC + 滚动均值 + 累计 IC）
uv run python scripts/visualize_factors.py ic_ts CSTM_MAX_RET_20

# IC 分布直方图（某因子的 IC 分布 + KDE 拟合 + 统计指标）
uv run python scripts/visualize_factors.py ic_dist CSTM_MAX_RET_20

# 累计 IC 对比（多个因子的累计 IC 曲线叠加）
uv run python scripts/visualize_factors.py cum_ic CSTM_MAX_RET_20 CSTM_AMT_CV_20 CSTM_VOL_CV_10

# 分层回测（按因子值分 N 组，展示各组累计收益）
uv run python scripts/visualize_factors.py quantile CSTM_MAX_RET_20 --groups 5

# 类别分析（各因子类别的平均/最大 |ICIR| 对比）
uv run python scripts/visualize_factors.py category

# 因子相关性热力图（Top-N 因子的 Spearman 相关矩阵）
uv run python scripts/visualize_factors.py corr --top 15

# 综合仪表盘（4 面板：排名 + 类别 + 状态分布 + 来源对比）
uv run python scripts/visualize_factors.py dashboard
```

所有单独图表命令默认在浏览器打开。添加 `--save` 则仅保存 PNG 文件不打开浏览器：

```bash
uv run python scripts/visualize_factors.py ranking --top 20 --save
```

### 7.3 图表含义与解读方法

#### IC 时间序列图 (`ic_ts`)

- **上半部分**：每日 Rank IC 柱状图（绿色正值 / 红色负值）+ 60天滚动均值线（蓝色）
- **下半部分**：累计 IC 曲线
- **如何解读**：
  - 滚动均值线持续在零轴上方/下方 → 因子有稳定的方向性预测能力
  - 累计 IC 曲线斜率稳定 → 因子信号持续有效（不是某一段暴涨贡献）
  - 左上角统计框：Mean IC（均值越远离0越好）、ICIR（大于0.1视为有信号）

#### IC 分布图 (`ic_dist`)

- 日度 IC 的频率直方图 + KDE 核密度拟合 + 均值线
- **如何解读**：
  - 分布中心偏离 0 → 因子有预测能力
  - t-stat 绝对值 > 2，p-value < 0.05 → 统计显著
  - Skew 偏度：正偏表示偶有极强正信号，负偏反之
  - Kurtosis 峰度：高峰度说明有肥尾（极端 IC 天数多）

#### 分层回测图 (`quantile`)

将股票按因子值从低到高分为 N 组（默认 5 组），每天等权构建组合，展示各组累计收益。

- **上半部分**：Q1（低分组）到 Q5（高分组）的累计收益曲线
- **下半部分**：多空收益曲线（Q5 - Q1）
- **如何解读**：
  - 各组曲线呈明显扇形展开 → 因子有良好的单调性选股能力
  - Q1 和 Q5 之间差距越大 → 因子区分度越强
  - 多空收益持续上升 → 因子信号稳定可交易
  - 如果 Q5（高值）收益低于 Q1（低值），说明因子方向为负（CSI1000 常见的反转/均值回归特征）
  - 左上角统计框：L/S Cum（多空累计）、L/S Ann（年化）、Daily Spread（日均收益差）
  - **注意**：因子方向在实际建模中由模型自动学习，此图主要看单调性而非绝对方向

#### 因子排名图 (`ranking`)

- Top-N 因子按 |Rank ICIR| 降序排列的横向柱状图
- 红色 = 自定义因子，蓝色 = Alpha158 基线因子
- **如何解读**：|ICIR| > 0.1 表示有统计显著的预测信号

#### 累计 IC 对比图 (`cum_ic`)

- 多个因子的累计 Rank IC 曲线叠加对比
- **如何解读**：斜率越陡 → 因子信号越强；走势不同的因子 → 低相关，适合组合使用

#### 类别分析图 (`category`)

- 左图：各因子类别的平均 / 最大 |ICIR|
- 右图：各类别中显著因子的数量
- **如何解读**：识别哪些类别（如波动率、动量、流动性）整体信号最强

#### 相关性热力图 (`corr`)

- Top-N 因子之间的 Spearman 相关系数矩阵
- **如何解读**：
  - 深红/深蓝 → 高相关（冗余，选其一即可）
  - 接近 0 → 低相关（互补，适合组合使用）

#### 综合仪表盘 (`dashboard`)

四面板全局概览：因子排名 + 类别表现 + 状态饼图 + Alpha158 vs Custom 对比。适合快速了解因子库全貌。

## 8. 指标解读建议

| 指标 | 含义 | 参考标准 |
|------|------|----------|
| IC / Rank IC | 因子值与未来收益的相关性 | 绝对值 > 0.02 有参考价值 |
| ICIR / Rank ICIR | IC 均值 / IC 标准差，衡量稳定性 | 绝对值 > 0.1 有信号 |
| t-stat | IC 均值的 t 检验统计量 | 绝对值 > 2 为显著 |
| p-value (FDR) | 经 BH 多重检验校正后的 p 值 | < 0.01 视为通过 |
| excess_return_with_cost | 扣除交易成本后的超额收益 | > 0 且稳定 |

建议先看 `with_cost` 指标，再看 `IC/RankIC`，避免只看样本内收益。

## 9. 项目关键文件

| 路径 | 说明 |
|------|------|
| `src/project_qlib/factors/` | 因子定义（Alpha158 + 自定义） |
| `src/project_qlib/factor_db.py` | 因子库核心模块（4 张表） |
| `src/project_qlib/runtime.py` | Qlib 初始化 & 运行时配置 |
| `scripts/visualize_factors.py` | 因子可视化分析工具 |
| `scripts/test_factor_ic.py` | 统一 IC 测试脚本 |
| `scripts/factor_db_cli.py` | 因子库 CLI 查询工具 |
| `scripts/import_factors_to_db.py` | 因子批量导入 |
| `configs/` | Qlib 工作流 YAML 配置 |
| `data/factor_library.db` | SQLite 因子库 |
| `outputs/charts/` | 可视化图表输出目录 |

## 10. 如何查看 mlruns 数据

```bash
scripts/mlflow_ui.sh 5000
# 浏览器打开 http://127.0.0.1:5000
```
