#!/usr/bin/env python3
"""CSI1000 SOTA 策略每日信号生成器（Dry-Run）

数据源: baostock → Qlib 本地数据
模型: XGBoost + LightGBM 均值集成 (Rolling 3m)
策略: TopkDropout (topk=30, n_drop=5, hold_thresh=60)
特征: Alpha158 + DB 因子 Top30 (max_per_cat=5)

用法:
    # 首次运行（训练模型 + 生成信号）
    uv run python scripts/daily_signal.py --init

    # 每日运行（更新数据 + 生成信号）
    uv run python scripts/daily_signal.py

    # 强制重训模型
    uv run python scripts/daily_signal.py --retrain

    # 查看当前持仓
    uv run python scripts/daily_signal.py --status

输出:
    outputs/dryrun/models/           模型文件 (xgb_*.pkl, lgb_*.pkl)
    outputs/dryrun/portfolio.json    当前持仓状态
    outputs/dryrun/signals/          每日信号 (YYYY-MM-DD.json)
    outputs/dryrun/trade_log.csv     交易记录
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 路径 ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DRYRUN_DIR = ROOT / "outputs" / "dryrun"
MODEL_DIR = DRYRUN_DIR / "models"
SIGNAL_DIR = DRYRUN_DIR / "signals"
PORTFOLIO_FILE = DRYRUN_DIR / "portfolio.json"
TRADE_LOG_FILE = DRYRUN_DIR / "trade_log.csv"
STATE_FILE = DRYRUN_DIR / "state.json"

for d in [DRYRUN_DIR, MODEL_DIR, SIGNAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 策略参数 ─────────────────────────────────────────────────────────
MARKET = "csi1000"
BENCHMARK = "SH000852"
TOPK = 30
N_DROP = 5
HOLD_THRESH = 60
TOPN_FACTORS = 30
MAX_PER_CAT = 5
RETRAIN_FREQ_MONTHS = 3
TRAIN_START = "2018-01-01"

# 交易成本
OPEN_COST = 0.0005   # 买入 5bp
CLOSE_COST = 0.0015  # 卖出 15bp
MIN_COST = 5

# XGBoost 参数
XGB_PARAMS = dict(
    objective="reg:squarederror",
    max_depth=8, eta=0.05,
    colsample_bytree=0.8879, subsample=0.8789,
    alpha=205.6999, reg_lambda=580.9768,
    nthread=8,
)

# LightGBM 参数
LGB_PARAMS = dict(
    loss="mse",
    colsample_bytree=0.8879, learning_rate=0.05,
    subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
    max_depth=8, num_leaves=128, num_threads=8,
    n_estimators=1000, early_stopping_rounds=50,
)


# ══════════════════════════════════════════════════════════════════════
#  1. 数据更新 (baostock → Qlib)
# ══════════════════════════════════════════════════════════════════════

def update_data_from_baostock():
    """调用已有的 baostock 数据下载脚本，增量更新 Qlib 数据。"""
    print("=" * 60)
    print("[数据更新] 通过 baostock 增量更新行情数据...")
    script = ROOT / ".agents/skills/qlib-env-data-prep/scripts/download_financial_data.py"
    if not script.exists():
        print(f"  ⚠ 数据更新脚本不存在: {script}")
        print("  跳过数据更新，使用本地已有数据")
        return False

    import subprocess
    result = subprocess.run(
        [sys.executable, str(script), "--phase", "1"],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  ⚠ 数据更新失败: {result.stderr[-500:]}")
        return False
    print("  ✓ 数据更新完成")
    return True


def init_qlib():
    """初始化 Qlib。"""
    import qlib
    try:
        qlib.init(provider_uri=str(ROOT / "data/qlib/cn_data"), region="cn")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
#  2. 模型训练与管理
# ══════════════════════════════════════════════════════════════════════

def create_dataset(train, valid, test):
    """创建 Alpha158 + TopN 因子数据集。"""
    from qlib.data.dataset import DatasetH
    from project_qlib.factors.topn_db import DBAlpha158PlusTopN

    class Handler(DBAlpha158PlusTopN):
        TOPN = TOPN_FACTORS
        MARKET = "csi1000"
        MAX_PER_CAT = MAX_PER_CAT

    label_expr = ["Ref($close, -2) / Ref($close, -1) - 1"]
    label_name = ["LABEL0"]

    handler = Handler(
        instruments=MARKET,
        start_time=train[0], end_time=test[1],
        fit_start_time=train[0], fit_end_time=train[1],
        label=(label_expr, label_name),
    )
    return DatasetH(
        handler=handler,
        segments={"train": train, "valid": valid, "test": test},
    )


def train_xgb(ds):
    """训练 XGBoost 模型。"""
    from qlib.contrib.model.xgboost import XGBModel
    model = XGBModel(**XGB_PARAMS)
    model.fit(ds, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=0)
    return model


def train_lgb(ds):
    """训练 LightGBM 模型。"""
    from qlib.contrib.model.gbdt import LGBModel
    model = LGBModel(**LGB_PARAMS)
    model.fit(ds)
    return model


def save_model(model, name: str):
    """持久化模型到磁盘。"""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  模型已保存: {path.name}")


def load_model(name: str):
    """从磁盘加载模型。"""
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def needs_retrain(state: dict) -> bool:
    """检查是否需要重训模型（距上次训练 >= 3个月）。"""
    last = state.get("last_retrain")
    if last is None:
        return True
    from dateutil.relativedelta import relativedelta
    last_dt = datetime.strptime(last, "%Y-%m-%d")
    next_retrain = last_dt + relativedelta(months=RETRAIN_FREQ_MONTHS)
    return datetime.now() >= next_retrain


def train_and_save_models(force=False):
    """训练 XGB+LGB 模型并保存。

    训练集: TRAIN_START ~ (当前日期 - 1年 - 1天)
    验证集: (当前日期 - 1年) ~ (当前日期 - 1天)
    测试集: 当前日期 ~ 未来（仅用于特征计算范围）
    """
    from dateutil.relativedelta import relativedelta

    state = load_state()
    if not force and not needs_retrain(state):
        print(f"  模型无需重训（上次训练: {state['last_retrain']}）")
        return

    today = datetime.now()
    valid_start = today - relativedelta(years=1)
    train_end = valid_start - relativedelta(days=1)
    valid_end = today - relativedelta(days=1)

    train = (TRAIN_START, train_end.strftime("%Y-%m-%d"))
    valid = (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d"))
    # test segment covers recent data for feature computation
    test = (today.strftime("%Y-%m-%d"), (today + relativedelta(days=30)).strftime("%Y-%m-%d"))

    print(f"\n[模型训练] Ensemble (XGB + LGB)")
    print(f"  训练集: {train[0]} ~ {train[1]}")
    print(f"  验证集: {valid[0]} ~ {valid[1]}")
    t0 = time.time()

    print("  训练 XGBoost...")
    ds = create_dataset(train, valid, test)
    xgb_model = train_xgb(ds)
    save_model(xgb_model, "xgb_latest")

    print("  训练 LightGBM...")
    lgb_model = train_lgb(ds)
    save_model(lgb_model, "lgb_latest")

    del ds
    gc.collect()

    state["last_retrain"] = today.strftime("%Y-%m-%d")
    state["train_range"] = f"{train[0]} ~ {train[1]}"
    state["valid_range"] = f"{valid[0]} ~ {valid[1]}"
    save_state(state)
    print(f"  ✓ 模型训练完成 [{time.time()-t0:.0f}s]")


# ══════════════════════════════════════════════════════════════════════
#  3. 每日预测与信号生成
# ══════════════════════════════════════════════════════════════════════

def get_today_scores() -> pd.Series:
    """用 Ensemble 模型对所有 CSI1000 成分股打分。

    返回 Series，index = (date, instrument), value = score
    """
    from dateutil.relativedelta import relativedelta

    xgb_model = load_model("xgb_latest")
    lgb_model = load_model("lgb_latest")

    if xgb_model is None or lgb_model is None:
        raise RuntimeError("模型文件不存在，请先运行 --init 或 --retrain")

    state = load_state()
    today = datetime.now()
    # 需要足够的历史数据来计算技术指标（Alpha158 需要约 240 个交易日）
    feat_start = today - relativedelta(years=2)
    feat_end = today + relativedelta(days=5)

    train_range = state.get("train_range", f"{TRAIN_START} ~ 2024-12-31")
    train_start, train_end = train_range.split(" ~ ")

    ds = create_dataset(
        train=(train_start, train_end),
        valid=(train_end, today.strftime("%Y-%m-%d")),
        test=(today.strftime("%Y-%m-%d"), feat_end.strftime("%Y-%m-%d")),
    )

    # XGB 预测
    p_xgb = xgb_model.predict(ds)
    if isinstance(p_xgb, pd.DataFrame):
        p_xgb = p_xgb.iloc[:, 0]

    # LGB 预测
    p_lgb = lgb_model.predict(ds)
    if isinstance(p_lgb, pd.DataFrame):
        p_lgb = p_lgb.iloc[:, 0]

    # Ensemble: 简单平均
    idx = p_xgb.index.intersection(p_lgb.index)
    scores = (p_xgb.loc[idx] + p_lgb.loc[idx]) / 2
    scores.name = "score"

    del ds, xgb_model, lgb_model
    gc.collect()

    return scores


def generate_signals(scores: pd.Series, trade_date: str) -> dict:
    """基于 TopkDropout 逻辑生成买卖信号。

    参数:
        scores: 全市场股票预测分数
        trade_date: 交易日期 YYYY-MM-DD

    返回:
        信号字典，包含 buy/sell/hold 列表
    """
    portfolio = load_portfolio()
    current_holdings = set(portfolio.get("holdings", {}).keys())

    # 获取当日所有股票分数并排名
    # scores 的 index 是 MultiIndex(date, instrument)
    # 取最近一个交易日的数据
    if isinstance(scores.index, pd.MultiIndex):
        dates = scores.index.get_level_values(0).unique()
        # 取 <= trade_date 的最近一天
        valid_dates = [d for d in dates if str(d)[:10] <= trade_date]
        if not valid_dates:
            print(f"  ⚠ 无法找到 {trade_date} 或之前的预测数据")
            return {"date": trade_date, "buy": [], "sell": [], "hold": list(current_holdings)}
        latest_date = max(valid_dates)
        day_scores = scores.xs(latest_date, level=0)
    else:
        day_scores = scores

    day_scores = day_scores.dropna().sort_values(ascending=False)

    # ── TopkDropout 逻辑 ────────────────────────────────────────────
    top_instruments = set(day_scores.index[:HOLD_THRESH])  # 排名在 hold_thresh 内的股票
    top_k = set(day_scores.index[:TOPK])  # 排名在 topk 内的股票

    # 需要卖出的: 当前持仓中排名跌出 hold_thresh 的
    to_sell_candidates = current_holdings - top_instruments
    # 限制每天最多卖出 n_drop 只
    to_sell = set(list(to_sell_candidates)[:N_DROP])

    # 卖出后的持仓
    after_sell = current_holdings - to_sell

    # 需要补充到 topk 的持仓数
    n_to_buy = TOPK - len(after_sell)

    # 从 top_k 中选择不在现有持仓中的
    buy_candidates = [s for s in day_scores.index if s in top_k and s not in after_sell]
    to_buy = buy_candidates[:max(0, n_to_buy)]

    # 继续持有的
    to_hold = list(after_sell)

    # 构建信号
    signal = {
        "date": trade_date,
        "model_date": str(latest_date)[:10] if isinstance(scores.index, pd.MultiIndex) else trade_date,
        "total_scored": len(day_scores),
        "buy": [],
        "sell": [],
        "hold": [],
        "portfolio_size_before": len(current_holdings),
        "portfolio_size_after": len(after_sell) + len(to_buy),
    }

    # 买入信号（附带分数和排名）
    for inst in to_buy:
        rank = list(day_scores.index).index(inst) + 1
        signal["buy"].append({
            "instrument": inst,
            "score": round(float(day_scores[inst]), 6),
            "rank": rank,
            "estimated_cost": f"{OPEN_COST * 100:.2f}%",
        })

    # 卖出信号
    for inst in to_sell:
        rank = list(day_scores.index).index(inst) + 1 if inst in day_scores.index else -1
        signal["sell"].append({
            "instrument": inst,
            "score": round(float(day_scores.get(inst, 0)), 6),
            "rank": rank,
            "reason": "排名跌出 hold_thresh" if rank > HOLD_THRESH else "清退",
            "estimated_cost": f"{CLOSE_COST * 100:.2f}%",
        })

    # 持有信号
    for inst in sorted(to_hold):
        rank = list(day_scores.index).index(inst) + 1 if inst in day_scores.index else -1
        signal["hold"].append({
            "instrument": inst,
            "score": round(float(day_scores.get(inst, 0)), 6),
            "rank": rank,
        })

    return signal


# ══════════════════════════════════════════════════════════════════════
#  4. 组合状态管理
# ══════════════════════════════════════════════════════════════════════

def load_portfolio() -> dict:
    """加载当前持仓。"""
    if PORTFOLIO_FILE.exists():
        return json.load(open(PORTFOLIO_FILE))
    return {"holdings": {}, "cash": 1e8, "last_update": None}


def save_portfolio(portfolio: dict):
    """保存持仓状态。"""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)


def load_state() -> dict:
    """加载系统状态。"""
    if STATE_FILE.exists():
        return json.load(open(STATE_FILE))
    return {}


def save_state(state: dict):
    """保存系统状态。"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def update_portfolio(signal: dict):
    """根据信号更新持仓。"""
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", {})

    # 执行卖出
    for s in signal["sell"]:
        inst = s["instrument"]
        if inst in holdings:
            del holdings[inst]

    # 执行买入
    for b in signal["buy"]:
        inst = b["instrument"]
        holdings[inst] = {
            "entry_date": signal["date"],
            "entry_score": b["score"],
            "entry_rank": b["rank"],
        }

    portfolio["holdings"] = holdings
    portfolio["last_update"] = signal["date"]
    save_portfolio(portfolio)


def save_signal(signal: dict):
    """保存每日信号到文件。"""
    path = SIGNAL_DIR / f"{signal['date']}.json"
    with open(path, "w") as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)
    print(f"  信号已保存: {path.name}")


def append_trade_log(signal: dict):
    """追加交易记录到 CSV。"""
    rows = []
    for b in signal["buy"]:
        rows.append({
            "date": signal["date"],
            "action": "BUY",
            "instrument": b["instrument"],
            "score": b["score"],
            "rank": b["rank"],
        })
    for s in signal["sell"]:
        rows.append({
            "date": signal["date"],
            "action": "SELL",
            "instrument": s["instrument"],
            "score": s["score"],
            "rank": s["rank"],
            "reason": s.get("reason", ""),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    header = not TRADE_LOG_FILE.exists()
    df.to_csv(TRADE_LOG_FILE, mode="a", header=header, index=False)


# ══════════════════════════════════════════════════════════════════════
#  5. 主流程
# ══════════════════════════════════════════════════════════════════════

def print_signal_summary(signal: dict):
    """打印信号摘要。"""
    print(f"\n{'='*60}")
    print(f"📅 交易日: {signal['date']}  (模型数据日: {signal.get('model_date', '?')})")
    print(f"📊 打分股票数: {signal['total_scored']}")
    print(f"📁 组合: {signal['portfolio_size_before']} → {signal['portfolio_size_after']} 只")
    print(f"{'='*60}")

    if signal["buy"]:
        print(f"\n🟢 买入 ({len(signal['buy'])} 只):")
        print(f"  {'股票':<12} {'分数':>10} {'排名':>6} {'成本':>8}")
        print(f"  {'-'*40}")
        for b in signal["buy"]:
            print(f"  {b['instrument']:<12} {b['score']:>10.6f} {b['rank']:>6d} {b['estimated_cost']:>8}")

    if signal["sell"]:
        print(f"\n🔴 卖出 ({len(signal['sell'])} 只):")
        print(f"  {'股票':<12} {'分数':>10} {'排名':>6} {'原因':<20}")
        print(f"  {'-'*52}")
        for s in signal["sell"]:
            print(f"  {s['instrument']:<12} {s['score']:>10.6f} {s['rank']:>6d} {s.get('reason',''):<20}")

    if signal["hold"]:
        print(f"\n⚪ 持有 ({len(signal['hold'])} 只):")
        top5 = sorted(signal["hold"], key=lambda x: x["rank"])[:5]
        for h in top5:
            print(f"  {h['instrument']:<12} rank={h['rank']:>4d}  score={h['score']:.6f}")
        if len(signal["hold"]) > 5:
            print(f"  ... 及其他 {len(signal['hold'])-5} 只")

    if not signal["buy"] and not signal["sell"]:
        print("\n  ℹ 今日无交易信号")

    print()


def cmd_init(args):
    """初始化：更新数据 + 训练模型 + 生成首日信号。"""
    print("🚀 初始化 CSI1000 SOTA 策略 Dry-Run")
    print(f"   策略: Ensemble(XGB+LGB) + TopkDropout(tk={TOPK}, drop={N_DROP}, hold={HOLD_THRESH})")
    print(f"   特征: Alpha158 + DB因子Top{TOPN_FACTORS} (mpc={MAX_PER_CAT})")

    # 1. 更新数据
    update_data_from_baostock()

    # 2. 初始化 Qlib
    init_qlib()

    # 3. 训练模型
    train_and_save_models(force=True)

    # 4. 生成首日信号
    print("\n[首日信号生成]")
    scores = get_today_scores()
    today = datetime.now().strftime("%Y-%m-%d")
    signal = generate_signals(scores, today)
    save_signal(signal)
    update_portfolio(signal)
    append_trade_log(signal)
    print_signal_summary(signal)

    print("✅ 初始化完成! 后续每日运行: uv run python scripts/daily_signal.py")


def cmd_daily(args):
    """每日运行：更新数据 + 检查是否需要重训 + 生成信号。"""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"📅 每日信号生成: {today}")

    # 检查信号是否已存在
    signal_file = SIGNAL_DIR / f"{today}.json"
    if signal_file.exists() and not args.force:
        print(f"  ⚠ 今日信号已生成: {signal_file.name}")
        signal = json.load(open(signal_file))
        print_signal_summary(signal)
        return

    # 1. 更新数据
    update_data_from_baostock()

    # 2. 初始化 Qlib
    init_qlib()

    # 3. 检查是否需要重训
    state = load_state()
    if needs_retrain(state):
        print("\n[模型重训] 距上次训练已满 3 个月")
        train_and_save_models(force=True)
    else:
        print(f"  模型状态: 上次训练 {state.get('last_retrain', '未知')}")

    # 4. 生成信号
    print("\n[信号生成]")
    scores = get_today_scores()
    signal = generate_signals(scores, today)
    save_signal(signal)
    update_portfolio(signal)
    append_trade_log(signal)
    print_signal_summary(signal)


def cmd_retrain(args):
    """强制重训模型。"""
    print("🔄 强制重训模型")
    update_data_from_baostock()
    init_qlib()
    train_and_save_models(force=True)


def cmd_status(args):
    """查看当前状态。"""
    print("📊 CSI1000 SOTA 策略状态")
    print(f"{'='*60}")

    # 系统状态
    state = load_state()
    print(f"\n  上次训练: {state.get('last_retrain', '未训练')}")
    print(f"  训练区间: {state.get('train_range', '--')}")
    print(f"  验证区间: {state.get('valid_range', '--')}")

    retrain_needed = needs_retrain(state)
    print(f"  需要重训: {'是 ⚠' if retrain_needed else '否 ✓'}")

    # 模型文件
    xgb_exists = (MODEL_DIR / "xgb_latest.pkl").exists()
    lgb_exists = (MODEL_DIR / "lgb_latest.pkl").exists()
    print(f"\n  XGB 模型: {'✓ 已保存' if xgb_exists else '✗ 不存在'}")
    print(f"  LGB 模型: {'✓ 已保存' if lgb_exists else '✗ 不存在'}")

    # 持仓
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", {})
    print(f"\n  持仓数量: {len(holdings)} 只")
    print(f"  最后更新: {portfolio.get('last_update', '--')}")

    if holdings:
        print(f"\n  持仓列表:")
        for inst, info in sorted(holdings.items()):
            entry = info.get("entry_date", "?")
            print(f"    {inst:<12} 入场={entry}  rank={info.get('entry_rank', '?')}")

    # 信号历史
    signals = sorted(SIGNAL_DIR.glob("*.json"))
    print(f"\n  历史信号: {len(signals)} 天")
    if signals:
        latest = signals[-1]
        print(f"  最新信号: {latest.stem}")

    # 交易记录
    if TRADE_LOG_FILE.exists():
        df = pd.read_csv(TRADE_LOG_FILE)
        n_buy = len(df[df["action"] == "BUY"])
        n_sell = len(df[df["action"] == "SELL"])
        print(f"\n  交易记录: {len(df)} 笔 (买入 {n_buy}, 卖出 {n_sell})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="CSI1000 SOTA 策略每日信号生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # 默认：每日运行（无子命令时）
    parser.add_argument("--init", action="store_true", help="首次初始化（训练模型 + 生成信号）")
    parser.add_argument("--retrain", action="store_true", help="强制重训模型")
    parser.add_argument("--status", action="store_true", help="查看当前状态")
    parser.add_argument("--force", action="store_true", help="强制重新生成今日信号")

    args = parser.parse_args()

    if args.init:
        cmd_init(args)
    elif args.retrain:
        cmd_retrain(args)
    elif args.status:
        cmd_status(args)
    else:
        cmd_daily(args)


if __name__ == "__main__":
    main()
