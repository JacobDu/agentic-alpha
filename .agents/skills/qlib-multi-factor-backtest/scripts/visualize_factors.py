"""Factor visualization & analysis tool.

Default usage: specify a factor name to generate a comprehensive report page.

    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/visualize_factors.py CSTM_MAX_RET_20
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/visualize_factors.py CSTM_AMT_CV_20 --groups 10

Individual chart subcommands:
  - ranking:   Top-N factors bar chart by |ICIR|
  - ic_ts:     Daily IC time series with rolling mean for a factor
  - ic_dist:   IC distribution histogram for a factor
  - cum_ic:    Cumulative IC curve over time
  - category:  Average |ICIR| grouped by factor category
  - corr:      Correlation heatmap of top-N factor signals
  - quantile:  Factor quantile stratification backtest (分层回测)
  - dashboard: Multi-panel overview combining key charts

All single-chart commands open in browser by default. Use --save to only save.

Usage:
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/visualize_factors.py CSTM_MAX_RET_20          # full report
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/visualize_factors.py ranking --top 20         # single chart
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/visualize_factors.py quantile CSTM_MAX_RET_20 --groups 5
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from project_qlib.factor_db import FactorDB
from project_qlib.runtime import init_qlib

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"


def _open_in_browser(png_path: str | None = None, fig: plt.Figure | None = None,
                     title: str = "Factor Chart") -> None:
    """Open a chart in the browser by wrapping it in a minimal HTML page."""
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    elif png_path:
        with open(png_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    else:
        return

    _open_report_in_browser([(img_b64, title)], title=title)


def _open_report_in_browser(
    images: list[tuple[str, str]],
    title: str = "Factor Report",
) -> None:
    """Open multiple charts in a single HTML report page.

    Args:
        images: list of (base64_png, section_title) tuples.
        title: page title.
    """
    sections = []
    for img_b64, sec_title in images:
        sections.append(f"""
        <div class="chart-section">
          <h2>{sec_title}</h2>
          <img src="data:image/png;base64,{img_b64}" />
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px 40px;
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  h1 {{ text-align: center; color: #f0f0f0; margin-bottom: 30px;
        border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
  .chart-section {{ margin-bottom: 40px; }}
  .chart-section h2 {{ color: #3498db; font-size: 1.1em; margin-bottom: 10px; }}
  img {{ max-width: 95vw; height: auto; border-radius: 8px;
         box-shadow: 0 4px 20px rgba(0,0,0,0.5); display: block; margin: 0 auto; }}
</style></head>
<body>
<h1>{title}</h1>
{"".join(sections)}
</body></html>"""

    html_path = OUTPUT_DIR / f"_viewer.html"
    html_path.write_text(html, encoding="utf-8")

    if platform.system() == "Darwin":
        subprocess.run(["open", str(html_path)], check=False)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", str(html_path)], check=False)
    else:
        webbrowser.open(html_path.as_uri())


def _finish_chart(fig: plt.Figure, save_path: str | None, default_name: str,
                  show: bool = True, title: str = "Factor Chart") -> str:
    """Save figure to file, optionally open in browser, return path."""
    if not save_path:
        save_path = str(OUTPUT_DIR / default_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    if show:
        _open_in_browser(fig=fig, title=title)
    plt.close(fig)
    return save_path


def _get_factor_expression(factor_name: str) -> str | None:
    """Get factor expression from factor library."""
    db = FactorDB()
    f = db.get_factor(factor_name)
    db.close()
    if f and f.get("expression"):
        return f["expression"]
    return None


def _compute_daily_ic(
    factor_name: str,
    market: str = "csi1000",
    start: str = "2019-01-01",
    end: str = "2026-02-13",
) -> pd.DataFrame:
    """Compute daily cross-sectional IC and Rank IC for a factor."""
    expr = _get_factor_expression(factor_name)
    if not expr:
        raise ValueError(f"No expression found for factor '{factor_name}' in factor library.")

    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market)
    stock_list = D.list_instruments(instruments, start_time=start, end_time=end, as_list=True)

    data = D.features(stock_list, [LABEL_EXPR, expr], start_time=start, end_time=end)
    data.columns = ["label", "factor"]

    combined = data.dropna()
    min_stocks = 30 if market in ("csi1000", "csi300") else 50
    dates = combined.index.get_level_values(1)

    def _daily_stats(g):
        if len(g) < min_stocks:
            return pd.Series({"ic": np.nan, "rank_ic": np.nan, "n_stocks": len(g)})
        ic = g["factor"].corr(g["label"])
        ric = g["factor"].rank().corr(g["label"].rank())
        return pd.Series({"ic": ic, "rank_ic": ric, "n_stocks": len(g)})

    daily = combined.groupby(dates).apply(_daily_stats)
    daily.index.name = "date"
    daily = daily.dropna(subset=["rank_ic"])
    return daily


def _compute_multi_factor_daily_ic(
    factor_names: list[str],
    market: str = "csi1000",
    start: str = "2019-01-01",
    end: str = "2026-02-13",
    batch_size: int = 10,
) -> dict[str, pd.Series]:
    """Compute daily Rank IC for multiple factors. Returns {name: daily_rank_ic_series}."""
    db = FactorDB()
    exprs = {}
    for name in factor_names:
        f = db.get_factor(name)
        if f and f.get("expression"):
            exprs[name] = f["expression"]
    db.close()

    if not exprs:
        raise ValueError("No factors with expressions found.")

    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market)
    stock_list = D.list_instruments(instruments, start_time=start, end_time=end, as_list=True)
    min_stocks = 30 if market in ("csi1000", "csi300") else 50

    result: dict[str, pd.Series] = {}
    names = list(exprs.keys())

    for batch_start in range(0, len(names), batch_size):
        batch_names = names[batch_start:batch_start + batch_size]
        batch_exprs = [LABEL_EXPR] + [exprs[n] for n in batch_names]
        print(f"  Loading batch {batch_start // batch_size + 1}: {len(batch_names)} factors...")

        data = D.features(stock_list, batch_exprs, start_time=start, end_time=end)
        fields = [f"f{i}" for i in range(len(batch_exprs))]
        data.columns = fields

        label = data["f0"]
        dates = data.index.get_level_values(1)

        for j, name in enumerate(batch_names):
            factor = data[f"f{j+1}"]
            combined = pd.DataFrame({"factor": factor, "label": label}).dropna()
            d = combined.index.get_level_values(1)

            def _ric(g):
                return np.nan if len(g) < min_stocks else g["factor"].rank().corr(g["label"].rank())

            daily_ric = combined.groupby(d).apply(_ric).dropna()
            daily_ric.index.name = "date"
            result[name] = daily_ric

        gc.collect()

    return result


# ==========================================================================
#  Chart: Ranking bar chart
# ==========================================================================

def chart_ranking(
    market: str = "csi1000",
    top: int = 20,
    source: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Bar chart of top-N factors by |ICIR|."""
    db = FactorDB()
    df = db.list_factors(market=market, source=source, significant_only=True, limit=top)
    db.close()

    if df.empty:
        print("No significant factors found.")
        return ""

    # Reverse for horizontal bar chart (top at top)
    df = df.iloc[::-1].reset_index(drop=True)

    # Color by source
    colors = []
    for _, row in df.iterrows():
        if row.get("source") == "Custom":
            colors.append("#e74c3c")  # red for custom
        else:
            colors.append("#3498db")  # blue for Alpha158

    fig, ax = plt.subplots(figsize=(10, max(6, top * 0.35)))
    icir_vals = df["rank_icir"].abs().values
    labels = df["name"].values

    bars = ax.barh(range(len(df)), icir_vals, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|Rank ICIR|", fontsize=11)
    ax.set_title(f"Top {len(df)} Factors by |Rank ICIR| ({market.upper()})", fontsize=13, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, icir_vals):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7, color="#333")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor="#e74c3c", label="Custom"),
                    Patch(facecolor="#3498db", label="Alpha158")]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    return _finish_chart(fig, save_path, f"ranking_{market}_top{top}.png",
                         show=show, title=f"Top {len(df)} Factors — {market.upper()}")


# ==========================================================================
#  Chart: IC time series
# ==========================================================================

def chart_ic_ts(
    factor_name: str,
    market: str = "csi1000",
    rolling_window: int = 60,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Daily IC time series with rolling mean."""
    daily = _compute_daily_ic(factor_name, market=market)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    dates = pd.to_datetime(daily.index)

    # Top panel: daily Rank IC + rolling mean
    ax1.bar(dates, daily["rank_ic"], alpha=0.3, color="#3498db", width=1, label="Daily Rank IC")
    rolling_ic = daily["rank_ic"].rolling(rolling_window).mean()
    ax1.plot(dates, rolling_ic, color="#e74c3c", linewidth=1.5,
             label=f"{rolling_window}d Rolling Mean")
    ax1.axhline(0, color="black", linewidth=0.5)

    mean_ic = daily["rank_ic"].mean()
    ax1.axhline(mean_ic, color="#2ecc71", linewidth=1, linestyle="--",
                label=f"Overall Mean = {mean_ic:+.4f}")

    ax1.set_ylabel("Rank IC", fontsize=11)
    ax1.set_title(f"{factor_name} — Daily Rank IC ({market.upper()})", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom panel: cumulative IC
    cum_ic = daily["rank_ic"].cumsum()
    ax2.plot(dates, cum_ic, color="#8e44ad", linewidth=1.2)
    ax2.fill_between(dates, 0, cum_ic, alpha=0.15, color="#8e44ad")
    ax2.set_ylabel("Cumulative IC", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(alpha=0.3)

    # Annotations
    icir = mean_ic / daily["rank_ic"].std() if daily["rank_ic"].std() > 0 else 0
    t_stat = mean_ic / (daily["rank_ic"].std() / np.sqrt(len(daily))) if daily["rank_ic"].std() > 0 else 0
    annotation = f"RankIC={mean_ic:+.4f}  ICIR={icir:+.3f}  t={t_stat:+.1f}  N={len(daily)}"
    ax1.text(0.02, 0.95, annotation, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    return _finish_chart(fig, save_path, f"ic_ts_{factor_name}_{market}.png",
                         show=show, title=f"{factor_name} IC Time Series")


# ==========================================================================
#  Chart: IC distribution
# ==========================================================================

def chart_ic_dist(
    factor_name: str,
    market: str = "csi1000",
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Histogram + density plot of daily Rank IC."""
    daily = _compute_daily_ic(factor_name, market=market)

    fig, ax = plt.subplots(figsize=(10, 6))

    ic_vals = daily["rank_ic"].dropna().values
    ax.hist(ic_vals, bins=50, density=True, alpha=0.6, color="#3498db", edgecolor="white")

    # KDE curve
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ic_vals)
    x = np.linspace(ic_vals.min() - 0.02, ic_vals.max() + 0.02, 200)
    ax.plot(x, kde(x), color="#e74c3c", linewidth=2, label="KDE")

    # Mean line
    mean = ic_vals.mean()
    ax.axvline(mean, color="#2ecc71", linewidth=2, linestyle="--", label=f"Mean = {mean:+.4f}")
    ax.axvline(0, color="black", linewidth=0.5)

    # Stats annotation
    std = ic_vals.std()
    icir = mean / std if std > 0 else 0
    skew = stats.skew(ic_vals)
    kurt = stats.kurtosis(ic_vals)
    stats_text = (f"Mean = {mean:+.4f}\nStd = {std:.4f}\n"
                  f"ICIR = {icir:+.3f}\nSkew = {skew:+.2f}\nKurt = {kurt:.2f}\n"
                  f"N = {len(ic_vals)}")
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Daily Rank IC", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{factor_name} — Rank IC Distribution ({market.upper()})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return _finish_chart(fig, save_path, f"ic_dist_{factor_name}_{market}.png",
                         show=show, title=f"{factor_name} IC Distribution")


# ==========================================================================
#  Chart: Cumulative IC comparison
# ==========================================================================

def chart_cum_ic(
    factor_names: list[str],
    market: str = "csi1000",
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Cumulative IC curves for multiple factors on one chart."""
    print(f"Computing daily IC for {len(factor_names)} factors...")
    daily_ics = _compute_multi_factor_daily_ic(factor_names, market=market)

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(daily_ics))))
    for i, (name, series) in enumerate(daily_ics.items()):
        cum = series.cumsum()
        ax.plot(pd.to_datetime(cum.index), cum.values,
                linewidth=1.3, label=name, color=colors[i % len(colors)])

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Rank IC", fontsize=11)
    ax.set_title(f"Cumulative Rank IC Comparison ({market.upper()})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    names_str = "_".join(factor_names[:3])
    return _finish_chart(fig, save_path, f"cum_ic_{names_str}_{market}.png",
                         show=show, title=f"Cumulative Rank IC — {market.upper()}")


# ==========================================================================
#  Chart: Category performance
# ==========================================================================

def chart_category(
    market: str = "csi1000",
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Grouped bar chart: average |ICIR| and count by factor category."""
    db = FactorDB()
    df = db.list_factors(market=market, significant_only=True)
    db.close()

    if df.empty or "rank_icir" not in df.columns:
        print("No data available.")
        return ""

    df["abs_icir"] = df["rank_icir"].abs()
    cat_stats = df.groupby("category").agg(
        mean_abs_icir=("abs_icir", "mean"),
        max_abs_icir=("abs_icir", "max"),
        count=("name", "count"),
    ).sort_values("mean_abs_icir", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(cat_stats) * 0.4)),
                                     gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: Mean & Max |ICIR| by category
    y = range(len(cat_stats))
    ax1.barh(y, cat_stats["max_abs_icir"], alpha=0.3, color="#e74c3c", label="Max |ICIR|")
    ax1.barh(y, cat_stats["mean_abs_icir"], alpha=0.8, color="#3498db", label="Mean |ICIR|")
    ax1.set_yticks(y)
    ax1.set_yticklabels(cat_stats.index, fontsize=9)
    ax1.set_xlabel("|Rank ICIR|", fontsize=11)
    ax1.set_title(f"Factor Category Performance ({market.upper()})", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # Right panel: Count per category
    ax2.barh(y, cat_stats["count"], color="#2ecc71", alpha=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("# Significant Factors", fontsize=11)
    ax2.set_title("Count", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)

    for i, cnt in enumerate(cat_stats["count"]):
        ax2.text(cnt + 0.3, i, str(cnt), va="center", fontsize=8)

    plt.tight_layout()
    return _finish_chart(fig, save_path, f"category_{market}.png",
                         show=show, title=f"Category Performance — {market.upper()}")


# ==========================================================================
#  Chart: Correlation heatmap
# ==========================================================================

def chart_corr(
    market: str = "csi1000",
    top: int = 15,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Correlation heatmap of top-N factor daily Rank IC."""
    db = FactorDB()
    df = db.list_factors(market=market, significant_only=True, limit=top)
    db.close()

    if df.empty:
        print("No factors found.")
        return ""

    factor_names = df["name"].tolist()
    print(f"Computing daily IC for {len(factor_names)} factors for correlation...")
    daily_ics = _compute_multi_factor_daily_ic(factor_names, market=market)

    # Build correlation matrix
    ic_df = pd.DataFrame(daily_ics)
    corr = ic_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(max(8, top * 0.6), max(8, top * 0.6)))

    # Custom colormap
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman Correlation")

    # Labels
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)

    # Annotate values
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    ax.set_title(f"Top {len(factor_names)} Factor IC Correlation ({market.upper()})",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    return _finish_chart(fig, save_path, f"corr_{market}_top{top}.png",
                         show=show, title=f"Factor IC Correlation — {market.upper()}")


# ==========================================================================
#  Chart: Dashboard (multi-panel overview)
# ==========================================================================

def chart_dashboard(
    market: str = "csi1000",
    top: int = 10,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Multi-panel dashboard: ranking + category + top-5 cumulative IC."""
    db = FactorDB()
    df = db.list_factors(market=market, significant_only=True, limit=top)
    all_df = db.list_factors(market=market, significant_only=True)
    counts = db.count_by_status()
    db.close()

    if df.empty:
        print("No data available for dashboard.")
        return ""

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Factor Analysis Dashboard — {market.upper()}", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Top-N ranking (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    plot_df = df.iloc[::-1].reset_index(drop=True)
    colors_bar = ["#e74c3c" if r.get("source") == "Custom" else "#3498db" for _, r in plot_df.iterrows()]
    icir_vals = plot_df["rank_icir"].abs().values
    ax1.barh(range(len(plot_df)), icir_vals, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(plot_df)))
    ax1.set_yticklabels(plot_df["name"].values, fontsize=7)
    ax1.set_xlabel("|Rank ICIR|")
    ax1.set_title(f"Top {len(df)} Factors by |ICIR|")
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(ax1.patches, icir_vals):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=6)

    # Panel 2: Category breakdown (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    if "rank_icir" in all_df.columns:
        all_df["abs_icir"] = all_df["rank_icir"].abs()
        cat_stats = all_df.groupby("category").agg(
            mean_icir=("abs_icir", "mean"),
            count=("name", "count"),
        ).sort_values("mean_icir", ascending=True)

        y = range(len(cat_stats))
        ax2.barh(y, cat_stats["mean_icir"], color="#27ae60", alpha=0.7)
        ax2.set_yticks(y)
        ax2.set_yticklabels(cat_stats.index, fontsize=7)
        ax2.set_xlabel("Mean |ICIR|")
        ax2.set_title("Category Avg |ICIR| (significant only)")
        ax2.grid(axis="x", alpha=0.3)
        for i, (icir_v, cnt) in enumerate(zip(cat_stats["mean_icir"], cat_stats["count"])):
            ax2.text(icir_v + 0.002, i, f"{icir_v:.3f} (n={cnt})", va="center", fontsize=6)

    # Panel 3: Status pie chart (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    labels = list(counts.keys())
    sizes = list(counts.values())
    pie_colors = {"Accepted": "#2ecc71", "Baseline": "#3498db", "Candidate": "#f39c12",
                  "Rejected": "#e74c3c", "Dropped": "#95a5a6"}
    c = [pie_colors.get(l, "#bdc3c7") for l in labels]
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=c, autopct='%1.0f%%',
                                        startangle=90, textprops={"fontsize": 9})
    ax3.set_title("Factor Status Distribution")

    # Panel 4: Source comparison (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    if "rank_icir" in all_df.columns and "source" in all_df.columns:
        src_stats = all_df.groupby("source").agg(
            mean_icir=("abs_icir", "mean"),
            median_icir=("abs_icir", "median"),
            count=("name", "count"),
        )
        x = range(len(src_stats))
        width = 0.35
        ax4.bar([i - width/2 for i in x], src_stats["mean_icir"], width, label="Mean |ICIR|", color="#3498db")
        ax4.bar([i + width/2 for i in x], src_stats["median_icir"], width, label="Median |ICIR|", color="#e74c3c")
        ax4.set_xticks(x)
        ax4.set_xticklabels(src_stats.index, fontsize=9)
        ax4.set_ylabel("|ICIR|")
        ax4.set_title("Alpha158 vs Custom: |ICIR| Comparison")
        ax4.legend(fontsize=8)
        ax4.grid(axis="y", alpha=0.3)
        for i, cnt in enumerate(src_stats["count"]):
            ax4.text(i, max(src_stats["mean_icir"].iloc[i], src_stats["median_icir"].iloc[i]) + 0.005,
                     f"n={cnt}", ha="center", fontsize=8, color="#666")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _finish_chart(fig, save_path, f"dashboard_{market}.png",
                         show=show, title=f"Factor Dashboard — {market.upper()}")


# ==========================================================================
#  Chart: Quantile stratification (分层回测)
# ==========================================================================

def chart_quantile(
    factor_name: str,
    market: str = "csi1000",
    n_groups: int = 5,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Quantile stratification chart: sort stocks by factor value into N groups,
    plot cumulative return for each group over time.

    This reveals whether the factor creates meaningful spread between
    top and bottom quantile portfolios.
    """
    expr = _get_factor_expression(factor_name)
    if not expr:
        print(f"Factor {factor_name} not found in library.")
        return ""

    init_qlib()
    import qlib
    from qlib.data import D

    # Determine instrument set
    instruments = D.instruments(market)

    # Build fields: factor value + next-day return (label)
    fields = [expr, LABEL_EXPR]
    names = ["factor", "label"]

    print(f"Loading data for {factor_name} on {market} ({start} ~ {end})...")
    df = D.features(instruments, fields, start_time=start, end_time=end)
    df.columns = names
    df = df.dropna()

    if df.empty:
        print("No data available.")
        return ""

    # For each date, assign quantile group
    print(f"Computing {n_groups}-group quantile stratification...")
    dates = df.index.get_level_values("datetime").unique().sort_values()

    # Pre-compute group returns per date
    group_returns = {g: [] for g in range(1, n_groups + 1)}
    valid_dates = []

    for dt in dates:
        day_df = df.xs(dt, level="datetime")
        if len(day_df) < n_groups * 5:  # need reasonable number of stocks
            continue

        # Assign quantile group (1=lowest factor value, N=highest)
        try:
            day_df = day_df.copy()
            day_df["group"] = pd.qcut(day_df["factor"], q=n_groups, labels=False, duplicates="drop") + 1
        except ValueError:
            continue

        actual_groups = day_df["group"].nunique()
        if actual_groups < n_groups:
            continue

        valid_dates.append(dt)
        for g in range(1, n_groups + 1):
            mean_ret = day_df.loc[day_df["group"] == g, "label"].mean()
            group_returns[g].append(mean_ret)

    if not valid_dates:
        print("Insufficient data for quantile analysis.")
        return ""

    print(f"Valid trading days: {len(valid_dates)}")

    # Compute cumulative returns
    cum_returns = {}
    for g in range(1, n_groups + 1):
        ret_series = pd.Series(group_returns[g], index=valid_dates)
        cum_returns[g] = (1 + ret_series).cumprod() - 1  # cumulative return

    # Long-short: top group minus bottom group
    ls_ret = pd.Series(
        [group_returns[n_groups][i] - group_returns[1][i] for i in range(len(valid_dates))],
        index=valid_dates,
    )
    cum_ls = (1 + ls_ret).cumprod() - 1

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1],
                                    sharex=True)

    # Color palette: blue(low) -> red(high)
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(i / (n_groups - 1)) for i in range(n_groups)]

    # Upper panel: all group cumulative returns
    for g in range(1, n_groups + 1):
        label = f"Q{g}" if g not in (1, n_groups) else (f"Q{g} (Low)" if g == 1 else f"Q{g} (High)")
        lw = 2.0 if g in (1, n_groups) else 1.0
        ax1.plot(cum_returns[g].index, cum_returns[g].values * 100,
                 label=label, color=colors[g - 1], linewidth=lw)

    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax1.set_title(f"{factor_name} — {n_groups}-Group Quantile Stratification ({market.upper()})",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(alpha=0.3)

    # Lower panel: long-short cumulative return
    ax2.fill_between(cum_ls.index, 0, cum_ls.values * 100,
                     where=cum_ls.values >= 0, alpha=0.4, color="#2ecc71", label="L/S > 0")
    ax2.fill_between(cum_ls.index, 0, cum_ls.values * 100,
                     where=cum_ls.values < 0, alpha=0.4, color="#e74c3c", label="L/S < 0")
    ax2.plot(cum_ls.index, cum_ls.values * 100, color="black", linewidth=1.2)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Long-Short Cum Return (%)", fontsize=11)
    ax2.set_title(f"Q{n_groups} − Q1 (Long High, Short Low)", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Add statistics text box
    final_ls = cum_ls.iloc[-1] * 100 if len(cum_ls) > 0 else 0
    ann_ret = ((1 + cum_ls.iloc[-1]) ** (252 / len(cum_ls)) - 1) * 100 if len(cum_ls) > 252 else final_ls
    spread_daily = ls_ret.mean() * 100
    stats_text = (f"L/S Cum Return: {final_ls:.1f}%\n"
                  f"L/S Ann Return: {ann_ret:.1f}%\n"
                  f"Daily Spread: {spread_daily:.3f}%\n"
                  f"Trading Days: {len(valid_dates)}")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    return _finish_chart(fig, save_path, f"quantile_{factor_name}_{market}_g{n_groups}.png",
                         show=show, title=f"Quantile Stratification — {factor_name}")


# ==========================================================================
#  Report: all charts for a single factor
# ==========================================================================

def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def chart_report(
    factor_name: str,
    market: str = "csi1000",
    n_groups: int = 5,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
) -> str:
    """Generate a comprehensive HTML report for a single factor.

    Includes: IC time series, IC distribution, quantile stratification,
    cumulative IC comparison with top peers, and ranking context.
    All charts are rendered into one HTML page and opened in the browser.
    """
    expr = _get_factor_expression(factor_name)
    if not expr:
        print(f"Factor {factor_name} not found in library.")
        return ""

    print(f"=== Generating report for {factor_name} on {market} ===\n")
    images: list[tuple[str, str]] = []

    # --- 1. IC Time Series ---
    print("[1/5] IC Time Series...")
    ic_series = None
    try:
        init_qlib()
        ic_df = _compute_daily_ic(factor_name, market=market)
        if ic_df is not None and len(ic_df) > 0:
            ic_series = ic_df["rank_ic"].dropna()
        if ic_series is not None and len(ic_series) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1]})
            rolling = 60
            ic_roll = ic_series.rolling(rolling).mean()

            colors_bar = ["#2ecc71" if v > 0 else "#e74c3c" for v in ic_series.values]
            ax1.bar(ic_series.index, ic_series.values, color=colors_bar, alpha=0.4, width=1)
            ax1.plot(ic_roll.index, ic_roll.values, color="#3498db", linewidth=1.5,
                     label=f"{rolling}d rolling mean")
            ax1.axhline(0, color="black", linewidth=0.5)
            ax1.set_ylabel("Daily Rank IC", fontsize=11)
            ax1.set_title(f"{factor_name} — Daily Rank IC ({market.upper()})",
                          fontsize=13, fontweight="bold")
            ax1.legend(fontsize=9)
            ax1.grid(alpha=0.3)

            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            icir = mean_ic / std_ic if std_ic > 0 else 0
            stats_text = (f"Mean IC: {mean_ic:.4f}\nStd: {std_ic:.4f}\n"
                          f"ICIR: {icir:.3f}\nN: {len(ic_series)}")
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

            cum_ic = ic_series.cumsum()
            ax2.plot(cum_ic.index, cum_ic.values, color="#9b59b6", linewidth=1.2)
            ax2.fill_between(cum_ic.index, 0, cum_ic.values, alpha=0.15, color="#9b59b6")
            ax2.axhline(0, color="black", linewidth=0.5)
            ax2.set_ylabel("Cumulative IC", fontsize=11)
            ax2.set_xlabel("Date", fontsize=11)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            images.append((_fig_to_b64(fig), "IC Time Series"))
    except Exception as e:
        print(f"  Warning: IC time series failed: {e}")

    # --- 2. IC Distribution ---
    print("[2/5] IC Distribution...")
    try:
        if ic_series is not None and len(ic_series) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ic_vals = ic_series.values
            n_bins = min(80, max(30, len(ic_vals) // 20))
            ax.hist(ic_vals, bins=n_bins, density=True, alpha=0.6,
                    color="#3498db", edgecolor="white", linewidth=0.5)

            x_kde = np.linspace(ic_vals.min(), ic_vals.max(), 300)
            kde = stats.gaussian_kde(ic_vals)
            ax.plot(x_kde, kde(x_kde), color="#e74c3c", linewidth=2, label="KDE")
            ax.axvline(ic_vals.mean(), color="#2ecc71", linewidth=2, linestyle="--",
                       label=f"Mean={ic_vals.mean():.4f}")
            ax.axvline(0, color="black", linewidth=0.5)

            t_stat, p_val = stats.ttest_1samp(ic_vals, 0)
            skew_val = stats.skew(ic_vals)
            kurt_val = stats.kurtosis(ic_vals)
            stats_text = (f"Mean: {ic_vals.mean():.4f}\nStd: {ic_vals.std():.4f}\n"
                          f"t-stat: {t_stat:.2f}\np-value: {p_val:.2e}\n"
                          f"Skew: {skew_val:.3f}\nKurtosis: {kurt_val:.3f}")
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

            ax.set_xlabel("Daily Rank IC", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"{factor_name} — IC Distribution ({market.upper()})",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            images.append((_fig_to_b64(fig), "IC Distribution"))
    except Exception as e:
        print(f"  Warning: IC distribution failed: {e}")

    # --- 3. Quantile Stratification ---
    print("[3/5] Quantile Stratification...")
    try:
        from qlib.data import D
        instruments = D.instruments(market)
        fields = [expr, LABEL_EXPR]
        df = D.features(instruments, fields, start_time=start, end_time=end)
        df.columns = ["factor", "label"]
        df = df.dropna()

        if not df.empty:
            dates = df.index.get_level_values("datetime").unique().sort_values()
            group_returns = {g: [] for g in range(1, n_groups + 1)}
            valid_dates = []

            for dt in dates:
                day_df = df.xs(dt, level="datetime")
                if len(day_df) < n_groups * 5:
                    continue
                try:
                    day_df = day_df.copy()
                    day_df["group"] = pd.qcut(day_df["factor"], q=n_groups,
                                              labels=False, duplicates="drop") + 1
                except ValueError:
                    continue
                if day_df["group"].nunique() < n_groups:
                    continue
                valid_dates.append(dt)
                for g in range(1, n_groups + 1):
                    group_returns[g].append(day_df.loc[day_df["group"] == g, "label"].mean())

            if valid_dates:
                cum_returns = {}
                for g in range(1, n_groups + 1):
                    ret_series = pd.Series(group_returns[g], index=valid_dates)
                    cum_returns[g] = (1 + ret_series).cumprod() - 1

                ls_ret = pd.Series(
                    [group_returns[n_groups][i] - group_returns[1][i]
                     for i in range(len(valid_dates))],
                    index=valid_dates,
                )
                cum_ls = (1 + ls_ret).cumprod() - 1

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                                height_ratios=[3, 1], sharex=True)
                cmap = plt.cm.RdYlBu_r
                colors_q = [cmap(i / (n_groups - 1)) for i in range(n_groups)]
                for g in range(1, n_groups + 1):
                    label = f"Q{g}" if g not in (1, n_groups) else (
                        f"Q{g} (Low)" if g == 1 else f"Q{g} (High)")
                    lw = 2.0 if g in (1, n_groups) else 1.0
                    ax1.plot(cum_returns[g].index, cum_returns[g].values * 100,
                             label=label, color=colors_q[g-1], linewidth=lw)
                ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
                ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
                ax1.set_title(f"{factor_name} — {n_groups}-Group Quantile ({market.upper()})",
                              fontsize=13, fontweight="bold")
                ax1.legend(fontsize=9, loc="best")
                ax1.grid(alpha=0.3)

                final_ls = cum_ls.iloc[-1] * 100 if len(cum_ls) > 0 else 0
                ann_ret = ((1 + cum_ls.iloc[-1]) ** (252/len(cum_ls)) - 1) * 100 if len(cum_ls) > 252 else final_ls
                stats_text = (f"L/S Cum: {final_ls:.1f}%\n"
                              f"L/S Ann: {ann_ret:.1f}%\n"
                              f"Days: {len(valid_dates)}")
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                         verticalalignment="top",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

                ax2.fill_between(cum_ls.index, 0, cum_ls.values * 100,
                                 where=cum_ls.values >= 0, alpha=0.4, color="#2ecc71")
                ax2.fill_between(cum_ls.index, 0, cum_ls.values * 100,
                                 where=cum_ls.values < 0, alpha=0.4, color="#e74c3c")
                ax2.plot(cum_ls.index, cum_ls.values * 100, color="black", linewidth=1.2)
                ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
                ax2.set_xlabel("Date", fontsize=11)
                ax2.set_ylabel("L/S Cum Ret (%)", fontsize=11)
                ax2.grid(alpha=0.3)

                plt.tight_layout()
                images.append((_fig_to_b64(fig), "Quantile Stratification (分层回测)"))
    except Exception as e:
        print(f"  Warning: Quantile stratification failed: {e}")

    # --- 4. Cumulative IC vs Peers ---
    print("[4/5] Cumulative IC vs Top Peers...")
    try:
        db = FactorDB()
        top_df = db.list_factors(market=market, significant_only=True, limit=5)
        db.close()
        peer_names = top_df["name"].tolist()
        if factor_name not in peer_names:
            peer_names = [factor_name] + peer_names[:4]

        multi_ic = _compute_multi_factor_daily_ic(peer_names, market=market)
        if multi_ic:
            fig, ax = plt.subplots(figsize=(14, 6))
            colors_line = plt.cm.Set2(np.linspace(0, 1, len(multi_ic)))
            for i, (name, ic_s) in enumerate(multi_ic.items()):
                cum = ic_s.cumsum()
                lw = 2.5 if name == factor_name else 1.0
                ls = "-" if name == factor_name else "--"
                ax.plot(cum.index, cum.values, linewidth=lw, linestyle=ls,
                        label=name, color=colors_line[i])
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Cumulative Rank IC", fontsize=11)
            ax.set_title(f"Cumulative IC: {factor_name} vs Top Peers ({market.upper()})",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            images.append((_fig_to_b64(fig), "Cumulative IC vs Top Peers"))
    except Exception as e:
        print(f"  Warning: Cumulative IC comparison failed: {e}")

    # --- 5. Ranking Context ---
    print("[5/5] Factor Ranking Context...")
    try:
        db = FactorDB()
        all_df = db.list_factors(market=market, significant_only=True, limit=20)
        db.close()
        if not all_df.empty and "rank_icir" in all_df.columns:
            # Highlight the target factor
            plot_df = all_df.iloc[::-1].reset_index(drop=True)
            bar_colors = []
            for _, r in plot_df.iterrows():
                if r["name"] == factor_name:
                    bar_colors.append("#f1c40f")  # gold highlight
                elif r.get("source") == "Custom":
                    bar_colors.append("#e74c3c")
                else:
                    bar_colors.append("#3498db")

            fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.3)))
            icir_vals = plot_df["rank_icir"].abs().values
            ax.barh(range(len(plot_df)), icir_vals, color=bar_colors,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(plot_df)))
            ax.set_yticklabels(plot_df["name"].values, fontsize=7)
            ax.set_xlabel("|Rank ICIR|", fontsize=11)
            ax.set_title(f"Top-20 Ranking ({market.upper()}) — {factor_name} highlighted",
                         fontsize=13, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            for bar, val in zip(ax.patches, icir_vals):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=6)
            plt.tight_layout()
            images.append((_fig_to_b64(fig), "Factor Ranking Context"))
    except Exception as e:
        print(f"  Warning: Ranking context failed: {e}")

    # --- Open report ---
    if not images:
        print("No charts generated.")
        return ""

    print(f"\nOpening report with {len(images)} charts...")
    _open_report_in_browser(images, title=f"Factor Report — {factor_name} ({market.upper()})")
    return str(OUTPUT_DIR / "_viewer.html")


# ==========================================================================
#  CLI
# ==========================================================================

def main() -> int:
    # Known subcommands for explicit chart type selection
    SUBCOMMANDS = {"ranking", "ic_ts", "ic_dist", "cum_ic", "category", "corr",
                   "dashboard", "quantile"}

    # If first positional arg is not a subcommand, treat as factor name (report mode)
    if len(sys.argv) > 1 and sys.argv[1] not in SUBCOMMANDS and not sys.argv[1].startswith("-"):
        return _main_report()

    return _main_subcommand()


def _main_report() -> int:
    """Report mode: generate comprehensive report for a single factor."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive factor report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("name", help="Factor name")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--groups", type=int, default=5, help="Quantile groups")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")

    args = parser.parse_args()
    chart_report(args.name, market=args.market, n_groups=args.groups,
                 start=args.start, end=args.end)
    return 0


def _main_subcommand() -> int:
    """Subcommand mode: generate a specific chart type."""
    parser = argparse.ArgumentParser(
        description="Factor visualization & analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Shared parent parser for common flags
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--save", action="store_true",
                        help="Only save to file, do not open in browser")

    sub = parser.add_subparsers(dest="command", required=True)

    # ranking
    p_rank = sub.add_parser("ranking", parents=[common], help="Top-N factors bar chart by |ICIR|")
    p_rank.add_argument("--market", default="csi1000")
    p_rank.add_argument("--top", type=int, default=20)
    p_rank.add_argument("--source", help="Filter by source (Alpha158/Custom)")
    p_rank.add_argument("-o", "--output", help="Output file path")

    # ic_ts
    p_icts = sub.add_parser("ic_ts", parents=[common], help="Daily IC time series + rolling mean")
    p_icts.add_argument("name", help="Factor name")
    p_icts.add_argument("--market", default="csi1000")
    p_icts.add_argument("--rolling", type=int, default=60, help="Rolling window (days)")
    p_icts.add_argument("-o", "--output", help="Output file path")

    # ic_dist
    p_dist = sub.add_parser("ic_dist", parents=[common], help="IC distribution histogram")
    p_dist.add_argument("name", help="Factor name")
    p_dist.add_argument("--market", default="csi1000")
    p_dist.add_argument("-o", "--output", help="Output file path")

    # cum_ic
    p_cum = sub.add_parser("cum_ic", parents=[common], help="Cumulative IC comparison for multiple factors")
    p_cum.add_argument("names", nargs="+", help="Factor names (space-separated)")
    p_cum.add_argument("--market", default="csi1000")
    p_cum.add_argument("-o", "--output", help="Output file path")

    # category
    p_cat = sub.add_parser("category", parents=[common], help="Category performance chart")
    p_cat.add_argument("--market", default="csi1000")
    p_cat.add_argument("-o", "--output", help="Output file path")

    # corr
    p_corr = sub.add_parser("corr", parents=[common], help="Factor IC correlation heatmap")
    p_corr.add_argument("--market", default="csi1000")
    p_corr.add_argument("--top", type=int, default=15)
    p_corr.add_argument("-o", "--output", help="Output file path")

    # dashboard
    p_dash = sub.add_parser("dashboard", parents=[common], help="Multi-panel overview dashboard")
    p_dash.add_argument("--market", default="csi1000")
    p_dash.add_argument("--top", type=int, default=10)
    p_dash.add_argument("-o", "--output", help="Output file path")

    # quantile
    p_q = sub.add_parser("quantile", parents=[common], help="Quantile stratification backtest (分层回测)")
    p_q.add_argument("name", help="Factor name")
    p_q.add_argument("--market", default="csi1000")
    p_q.add_argument("--groups", type=int, default=5, help="Number of quantile groups")
    p_q.add_argument("--start", default="2020-01-01")
    p_q.add_argument("--end", default="2024-12-31")
    p_q.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args()
    show = not args.save

    if args.command == "ranking":
        chart_ranking(market=args.market, top=args.top, source=args.source,
                      save_path=args.output, show=show)
    elif args.command == "ic_ts":
        chart_ic_ts(args.name, market=args.market, rolling_window=args.rolling,
                    save_path=args.output, show=show)
    elif args.command == "ic_dist":
        chart_ic_dist(args.name, market=args.market, save_path=args.output, show=show)
    elif args.command == "cum_ic":
        chart_cum_ic(args.names, market=args.market, save_path=args.output, show=show)
    elif args.command == "category":
        chart_category(market=args.market, save_path=args.output, show=show)
    elif args.command == "corr":
        chart_corr(market=args.market, top=args.top, save_path=args.output, show=show)
    elif args.command == "dashboard":
        chart_dashboard(market=args.market, top=args.top, save_path=args.output, show=show)
    elif args.command == "quantile":
        chart_quantile(args.name, market=args.market, n_groups=args.groups,
                       start=args.start, end=args.end, save_path=args.output, show=show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
