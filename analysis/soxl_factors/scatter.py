"""恐怖指数・失業率（横軸）と SOXL（縦軸）の日次散布図。1 日 = 1 点、色 = 年。

実行: cd analysis && python -m soxl_factors.scatter --cache data_cache_latest --out output/scatter
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import cycle  # noqa: E402
from .data import load  # noqa: E402

# 単一色相の連続ランプ（明→暗 = 古→新）
SEQ = LinearSegmentedColormap.from_list("seq", ["#9ec5f4", "#5598e7", "#256abf", "#104281", "#0d366b"])
HIGHLIGHT = {"2020-03-18": "2020/3/18", "2020-04-17": "2020/4/17", "2022-10-14": "2022/10/14",
             "2025-04-04": "2025/4/4", "2025-04-08": "2025/4/8"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"


def _font():
    for name in ("Noto Sans CJK JP", "IPAexGothic", "WenQuanYi Zen Hei"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False  # CJK フォントに U+2212 が無いため
            return


def frame(px: pd.DataFrame, ohlcv: dict, fred: dict) -> pd.DataFrame:
    c = ohlcv["SOXL"]["Close"].dropna()
    cal = c.index
    u = cycle._on_calendar(fred["UNRATE"], cycle.FRED_EXTRA["UNRATE"], cal)
    urise = cycle._on_calendar(fred["UNRATE"] - fred["UNRATE"].rolling(12).min(), cycle.FRED_EXTRA["UNRATE"], cal)
    return pd.DataFrame({
        "soxl": c, "fwd_1y": c.shift(-252) / c,
        "VIX（S&P500の恐怖指数）": px["VIX"].reindex(cal),
        "VXN（ナスダック100の恐怖指数）": px["VXN"].reindex(cal),
        "VIX ÷ VIX3M（1超え＝目先の恐怖が上回る）": (px["VIX"] / px["VIX3M"]).reindex(cal),
        "失業率（%）": u,
        "失業率 - 過去12か月の最低（pt）": urise,
    })


def plot(D: pd.DataFrame, ycol: str, ylabel: str, path: Path, title: str, ref_line: float | None = None):
    xs = [c for c in D.columns if c not in ("soxl", "fwd_1y")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    years = D.index.year + D.index.dayofyear / 366
    sc = None
    for ax, xc in zip(axes, xs):
        d = D[[xc, ycol]].dropna()
        yr = years[D.index.isin(d.index)]
        sc = ax.scatter(d[xc], d[ycol], c=yr, cmap=SEQ, s=6, alpha=0.6, linewidths=0, rasterized=True)
        for k, dt in enumerate(HIGHLIGHT, 1):
            t = pd.Timestamp(dt)
            if t in d.index:
                ax.scatter(d.at[t, xc], d.at[t, ycol], s=150, facecolor="#e8590c", edgecolor="white", linewidth=1.5, zorder=5)
                ax.annotate(str(k), (d.at[t, xc], d.at[t, ycol]), ha="center", va="center", fontsize=7.5,
                            color="white", fontweight="bold", zorder=6)
        last = d.index[-1]
        ax.scatter(d.at[last, xc], d.at[last, ycol], s=110, marker="*", facecolor="#c92a2a", edgecolor="white",
                   linewidth=1, zorder=6)
        ax.annotate(f"{'直近' if ycol == 'soxl' else '1年後が確定した最後の日'} {last:%Y/%m/%d}", (d.at[last, xc], d.at[last, ycol]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color="#c92a2a", zorder=6)
        if ref_line is not None:
            ax.axhline(ref_line, color=MUTED, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel(xc, color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[-1].axis("off")
    cb = fig.colorbar(sc, ax=axes[-1], fraction=0.5, location="left")
    cb.solids.set_alpha(1)
    cb.set_label("年（薄い＝古い → 濃い＝新しい）", color=INK)
    legend = "\n".join(f"{k}: {lab}" for k, lab in enumerate(HIGHLIGHT.values(), 1))
    axes[-1].text(0.5, 0.5, "オレンジ丸の番号（指定の日付）\n" + legend + "\n\n赤い星 = 直近（右図は 1 年後が確定した最後の日）\n1 点 = 1 営業日",
                  transform=axes[-1].transAxes, fontsize=11, color=INK, va="center")
    fig.suptitle(title, fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/scatter"))
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _font()
    px, ohlcv = load(a.cache)
    fred = cycle.load_fred_extra(a.cache / "fred_monthly")
    D = frame(px, ohlcv, fred)
    D.to_csv(a.out / "scatter_data.csv")
    plot(D, "soxl", "SOXL 終値（ドル、対数目盛、分割・配当調整後）", a.out / "fig_scatter_price.png",
         "SOXL の価格 vs 恐怖指数・失業率（2010〜、1 点 = 1 日）")
    plot(D, "fwd_1y", "その日に買った SOXL の 1 年後の倍率（対数目盛）", a.out / "fig_scatter_fwd1y.png",
         "その日に買っていたら 1 年後に何倍？ vs 恐怖指数・失業率（1.0 = 損益なし）", ref_line=1.0)
    print(f"図: {a.out}")


if __name__ == "__main__":
    main()
