#!/usr/bin/env python3
"""CSI1000 SOTA 策略每日信号生成器（Dry-Run）

数据源: baostock → Qlib 本地数据
模型: XGBoost + LightGBM 均值集成 (Rolling 3m)
策略: TopkDropout (topk=20, n_drop=2, hold_thresh=80)
特征: Alpha158 + DB 因子 Top30 (max_per_cat=5)

SOTA 基准: MFA-V6 (2026-03-06)
  OOS IR=1.847, Ret=+33.28%, DD=-11.66%, Turn=1.53%

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


def set_profile(profile_name: str | None):
    """切换 profile，每个 profile 有独立的输出目录。

    profile=None 时使用默认目录 outputs/dryrun/；
    否则使用 outputs/dryrun-{profile_name}/。
    模型可以共享（如果参数相同），信号和持仓各自独立。
    """
    global DRYRUN_DIR, MODEL_DIR, SIGNAL_DIR, PORTFOLIO_FILE, TRADE_LOG_FILE, STATE_FILE
    if profile_name:
        DRYRUN_DIR = ROOT / "outputs" / f"dryrun-{profile_name}"
    else:
        DRYRUN_DIR = ROOT / "outputs" / "dryrun"
    MODEL_DIR = DRYRUN_DIR / "models"
    SIGNAL_DIR = DRYRUN_DIR / "signals"
    PORTFOLIO_FILE = DRYRUN_DIR / "portfolio.json"
    TRADE_LOG_FILE = DRYRUN_DIR / "trade_log.csv"
    STATE_FILE = DRYRUN_DIR / "state.json"
    for d in [DRYRUN_DIR, MODEL_DIR, SIGNAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# 默认初始化目录
for d in [DRYRUN_DIR, MODEL_DIR, SIGNAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def clear_qlib_cache():
    """清理 Qlib 内存缓存，释放特征计算中间结果。"""
    try:
        from qlib.data.cache import H
        H.clear()
    except Exception:
        pass
    gc.collect()

# ── 策略参数 ─────────────────────────────────────────────────────────
MARKET = "csi1000"
BENCHMARK = "SH000852"
TOPK = 20
N_DROP = 2
HOLD_THRESH = 80
TOPN_FACTORS = 30
MAX_PER_CAT = 5
RETRAIN_FREQ_MONTHS = 3
TRAIN_START = "2018-01-01"

# 资金与交易成本
INITIAL_CAPITAL = 1_000_000  # 初始资金 100 万元
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
    print(f"[数据更新] 通过 baostock 增量更新 {MARKET} 行情数据...")
    script = ROOT / ".agents/skills/qlib-env-data-prep/scripts/download_financial_data.py"
    if not script.exists():
        print(f"  ⚠ 数据更新脚本不存在: {script}")
        print("  跳过数据更新，使用本地已有数据")
        return False

    import subprocess
    result = subprocess.run(
        [sys.executable, str(script), "--phase", "1", "--market", MARKET],
        cwd=str(ROOT),
        timeout=3600,
    )
    if result.returncode != 0:
        print("  ⚠ 数据更新失败")
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


def _get_last_complete_calendar_date() -> str:
    """读取 Qlib 日历文件，返回最后一个数据完整的交易日。

    跳过今天（可能部分股票尚无数据），返回昨天或更早的日期。
    """
    cal_file = ROOT / "data/qlib/cn_data/calendars/day.txt"
    today_str = datetime.now().strftime("%Y-%m-%d")
    if cal_file.exists():
        dates = [l.strip() for l in cal_file.read_text().strip().split("\n") if l.strip()]
        # 返回 < today 的最后一个日期，确保数据完整
        safe_dates = [d for d in dates if d < today_str]
        if safe_dates:
            return safe_dates[-1]
        if dates:
            return dates[-1]
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


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
    # test segment: 仅用于占位，不超过 valid_end
    test = (valid_end.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d"))

    print(f"\n[模型训练] Ensemble (XGB + LGB)")
    print(f"  训练集: {train[0]} ~ {train[1]}")
    print(f"  验证集: {valid[0]} ~ {valid[1]}")
    t0 = time.time()

    print("  训练 XGBoost...")
    ds = create_dataset(train, valid, test)
    xgb_model = train_xgb(ds)
    save_model(xgb_model, "xgb_latest")
    del xgb_model, ds
    clear_qlib_cache()

    print("  训练 LightGBM...")
    ds = create_dataset(train, valid, test)
    lgb_model = train_lgb(ds)
    save_model(lgb_model, "lgb_latest")
    del lgb_model, ds
    clear_qlib_cache()

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

    逐模型加载预测以减少峰值内存：先 XGB 预测并释放，再 LGB 预测并释放。
    返回 Series，index = (date, instrument), value = score
    """

    state = load_state()
    today = datetime.now()
    data_end = _get_last_complete_calendar_date()
    test_start = today - timedelta(days=30)

    train_range = state.get("train_range", f"{TRAIN_START} ~ 2024-12-31")
    train_start, train_end = train_range.split(" ~ ")

    seg_args = dict(
        train=(train_start, train_end),
        valid=(train_end, test_start.strftime("%Y-%m-%d")),
        test=(test_start.strftime("%Y-%m-%d"), data_end),
    )

    # XGB 预测
    xgb_model = load_model("xgb_latest")
    if xgb_model is None:
        raise RuntimeError("XGB 模型文件不存在，请先运行 --init 或 --retrain")
    ds = create_dataset(**seg_args)
    p_xgb = xgb_model.predict(ds)
    if isinstance(p_xgb, pd.DataFrame):
        p_xgb = p_xgb.iloc[:, 0]
    del xgb_model, ds
    clear_qlib_cache()

    # LGB 预测
    lgb_model = load_model("lgb_latest")
    if lgb_model is None:
        raise RuntimeError("LGB 模型文件不存在，请先运行 --init 或 --retrain")
    ds = create_dataset(**seg_args)
    p_lgb = lgb_model.predict(ds)
    if isinstance(p_lgb, pd.DataFrame):
        p_lgb = p_lgb.iloc[:, 0]
    del lgb_model, ds
    clear_qlib_cache()

    # Ensemble: 简单平均（DataFrame 对齐，安全处理 index 不一致）
    combined = pd.DataFrame({"xgb": p_xgb, "lgb": p_lgb}).dropna()
    scores = combined.mean(axis=1)
    scores.name = "score"

    del p_xgb, p_lgb, combined
    gc.collect()

    return scores


def get_latest_close_prices(instruments: list[str], trade_date: str) -> dict[str, float]:
    """从 Qlib 获取最新收盘价。"""
    from qlib.data import D
    try:
        end = trade_date
        start = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        df = D.features(instruments, ["$close"], start_time=start, end_time=end)
        if df.empty:
            return {}
        # 取每只股票最新一天的收盘价
        prices = {}
        for inst in instruments:
            try:
                sub = df.xs(inst, level="instrument")
                if not sub.empty:
                    prices[inst] = float(sub.iloc[-1, 0])
            except (KeyError, IndexError):
                continue
        return prices
    except Exception as e:
        print(f"  ⚠ 获取收盘价失败: {e}")
        return {}


def generate_signals(scores: pd.Series, trade_date: str) -> dict:
    """基于 TopkDropout 逻辑生成买卖信号。

    参数:
        scores: 全市场股票预测分数
        trade_date: 交易日期 YYYY-MM-DD

    返回:
        信号字典，包含 buy/sell/hold 列表（含目标金额和股数）
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
            return {
                "date": trade_date, "model_date": "N/A",
                "total_scored": 0,
                "buy": [], "sell": [], "hold": list(current_holdings),
                "portfolio_size_before": len(current_holdings),
                "portfolio_size_after": len(current_holdings),
                "warning": f"数据未覆盖 {trade_date}，请先更新数据",
            }
        latest_date = max(valid_dates)
        day_scores = scores.xs(latest_date, level=0)
    else:
        day_scores = scores

    day_scores = day_scores.dropna().sort_values(ascending=False)
    actual_date = str(latest_date)[:10] if isinstance(scores.index, pd.MultiIndex) else trade_date

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

    # ── 获取收盘价并计算仓位 ────────────────────────────────────────
    capital = portfolio.get("cash", INITIAL_CAPITAL)
    all_instruments = list(set(to_buy) | set(to_sell) | set(to_hold))
    prices = get_latest_close_prices(all_instruments, trade_date) if all_instruments else {}
    total_positions = len(after_sell) + len(to_buy)
    target_weight = 1.0 / total_positions if total_positions > 0 else 0
    target_amount = capital * target_weight  # 每只股票的目标金额

    # 构建信号
    signal = {
        "date": trade_date,
        "model_date": actual_date,
        "total_scored": len(day_scores),
        "buy": [],
        "sell": [],
        "hold": [],
        "portfolio_size_before": len(current_holdings),
        "portfolio_size_after": total_positions,
        "capital": capital,
        "target_weight": round(target_weight, 4),
        "target_amount_per_stock": round(target_amount, 2),
    }

    # 买入信号（附带分数、排名、目标金额和股数）
    for inst in to_buy:
        rank = list(day_scores.index).index(inst) + 1
        price = prices.get(inst)
        shares = 0
        if price and price > 0:
            shares = int(target_amount / price / 100) * 100  # 按手取整（100股/手）
        signal["buy"].append({
            "instrument": inst,
            "score": round(float(day_scores[inst]), 6),
            "rank": rank,
            "price": round(price, 2) if price else None,
            "target_amount": round(target_amount, 2),
            "shares": shares,
            "actual_amount": round(shares * price, 2) if price and shares else 0,
            "estimated_cost": f"{OPEN_COST * 100:.2f}%",
        })

    # 卖出信号
    for inst in to_sell:
        rank = list(day_scores.index).index(inst) + 1 if inst in day_scores.index else -1
        price = prices.get(inst)
        # 卖出数量 = 组合中该股票的当前持仓量
        held_info = portfolio.get("holdings", {}).get(inst, {})
        held_shares = held_info.get("shares", 0)
        signal["sell"].append({
            "instrument": inst,
            "score": round(float(day_scores.get(inst, 0)), 6),
            "rank": rank,
            "price": round(price, 2) if price else None,
            "shares": held_shares,
            "estimated_amount": round(held_shares * price, 2) if price and held_shares else 0,
            "reason": "排名跌出 hold_thresh" if rank > HOLD_THRESH else "清退",
            "estimated_cost": f"{CLOSE_COST * 100:.2f}%",
        })

    # 持有信号
    for inst in sorted(to_hold):
        rank = list(day_scores.index).index(inst) + 1 if inst in day_scores.index else -1
        price = prices.get(inst)
        held_info = portfolio.get("holdings", {}).get(inst, {})
        held_shares = held_info.get("shares", 0)
        signal["hold"].append({
            "instrument": inst,
            "score": round(float(day_scores.get(inst, 0)), 6),
            "rank": rank,
            "price": round(price, 2) if price else None,
            "shares": held_shares,
            "market_value": round(held_shares * price, 2) if price and held_shares else 0,
        })

    return signal


# ══════════════════════════════════════════════════════════════════════
#  4. 组合状态管理
# ══════════════════════════════════════════════════════════════════════

def load_portfolio() -> dict:
    """加载当前持仓。"""
    if PORTFOLIO_FILE.exists():
        return json.load(open(PORTFOLIO_FILE))
    return {"holdings": {}, "cash": INITIAL_CAPITAL, "last_update": None}


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
            "shares": b.get("shares", 0),
            "entry_price": b.get("price"),
            "target_amount": b.get("target_amount", 0),
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
            "price": b.get("price"),
            "shares": b.get("shares", 0),
            "amount": b.get("actual_amount", 0),
        })
    for s in signal["sell"]:
        rows.append({
            "date": signal["date"],
            "action": "SELL",
            "instrument": s["instrument"],
            "score": s["score"],
            "rank": s["rank"],
            "price": s.get("price"),
            "shares": s.get("shares", 0),
            "amount": s.get("estimated_amount", 0),
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

    # 资金信息
    if "capital" in signal:
        print(f"💰 总资金: {signal['capital']:,.0f}  每只目标: {signal.get('target_amount_per_stock', 0):,.0f}元")

    if signal["buy"]:
        print(f"\n🟢 买入 ({len(signal['buy'])} 只):")
        print(f"  {'股票':<12} {'现价':>8} {'股数':>8} {'金额':>12} {'排名':>6} {'分数':>10}")
        print(f"  {'-'*62}")
        for b in signal["buy"]:
            price_str = f"{b['price']:.2f}" if b.get('price') else '  N/A'
            shares_str = f"{b['shares']:>6d}" if b.get('shares') else '   N/A'
            amount_str = f"{b.get('actual_amount', 0):>10,.0f}" if b.get('actual_amount') else '       N/A'
            print(f"  {b['instrument']:<12} {price_str:>8} {shares_str:>8} {amount_str:>12} {b['rank']:>6d} {b['score']:>10.6f}")

    if signal["sell"]:
        print(f"\n🔴 卖出 ({len(signal['sell'])} 只):")
        print(f"  {'股票':<12} {'现价':>8} {'股数':>8} {'金额':>12} {'排名':>6} {'原因':<16}")
        print(f"  {'-'*68}")
        for s in signal["sell"]:
            price_str = f"{s['price']:.2f}" if s.get('price') else '  N/A'
            shares_str = f"{s['shares']:>6d}" if s.get('shares') else '   N/A'
            amount_str = f"{s.get('estimated_amount', 0):>10,.0f}" if s.get('estimated_amount') else '       N/A'
            print(f"  {s['instrument']:<12} {price_str:>8} {shares_str:>8} {amount_str:>12} {s['rank']:>6d} {s.get('reason',''):<16}")

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
    print("🚀 初始化 CSI1000 SOTA 策略 Dry-Run (MFA-V6)")
    print(f"   策略: Ensemble(XGB+LGB) + TopkDropout(tk={TOPK}, drop={N_DROP}, hold={HOLD_THRESH})")
    print(f"   特征: Alpha158 + DB因子Top{TOPN_FACTORS} (mpc={MAX_PER_CAT})")

    # 1. 更新数据
    if not args.skip_data:
        update_data_from_baostock()
    else:
        print("  [跳过数据更新]")

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
    if not args.skip_data:
        update_data_from_baostock()
    else:
        print("  [跳过数据更新]")

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
    if not args.skip_data:
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
        total_cost = 0
        print(f"\n  {'股票':<12} {'入场日':>12} {'入场价':>8} {'股数':>8} {'成本':>12} {'排名':>6}")
        print(f"  {'-'*64}")
        for inst, info in sorted(holdings.items()):
            entry = info.get("entry_date", "?")
            price = info.get("entry_price")
            shares = info.get("shares", 0)
            cost = round(shares * price, 2) if price and shares else 0
            total_cost += cost
            price_str = f"{price:.2f}" if price else "N/A"
            print(f"  {inst:<12} {entry:>12} {price_str:>8} {shares:>8d} {cost:>12,.0f} {info.get('entry_rank', '?'):>6}")
        print(f"  {'-'*64}")
        print(f"  {'合计':<12} {'':>12} {'':>8} {'':>8} {total_cost:>12,.0f}")
        print(f"  初始资金: {portfolio.get('cash', INITIAL_CAPITAL):,.0f}元")

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
    parser.add_argument("--skip-data", action="store_true", help="跳过 baostock 数据更新")
    parser.add_argument("--capital", type=float, default=None,
                        help=f"初始资金（默认 {INITIAL_CAPITAL:,.0f} 元）")
    parser.add_argument("--profile", type=str, default=None,
                        help="策略配置名称，支持同时运行多个独立实例（如: test1, conservative）")

    args = parser.parse_args()

    # 设置 profile
    if args.profile:
        set_profile(args.profile)
        print(f"📁 Profile: {args.profile} (目录: outputs/dryrun-{args.profile}/)") 

    if args.init:
        if args.capital:
            # 更新初始资金
            portfolio = load_portfolio()
            portfolio["cash"] = args.capital
            save_portfolio(portfolio)
        cmd_init(args)
    elif args.retrain:
        cmd_retrain(args)
    elif args.status:
        cmd_status(args)
    else:
        cmd_daily(args)


if __name__ == "__main__":
    main()
