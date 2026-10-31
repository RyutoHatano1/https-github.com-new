"""恐怖指数・失業率・テクニカル（横軸）と SOXL（縦軸）の日次散布図。1 日 = 1 点、色 = 年。

実行: cd analysis && python -m soxl_factors.scatter --cache data_cache_latest --out output/scatter [--extended]
--extended: 2010-03 以前を ^SOX から合成した 3 倍系列で補い 1996 年から描く（2000 年・2008 年を含む）。
縦軸は 3 種類: 価格 / その日に買った 1 年後の倍率 / その後 1 年の最大下落（0% = その日が 1 年の底）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import cycle, features  # noqa: E402
from .data import load  # noqa: E402
from .report import md_table  # noqa: E402
from .scale_in import build_prices  # noqa: E402

# 単一色相の連続ランプ（明→暗 = 古→新）
SEQ = LinearSegmentedColormap.from_list("seq", ["#9ec5f4", "#5598e7", "#256abf", "#104281", "#0d366b"])
HIGHLIGHT = {"2000-10-12": "2000/10/12", "2008-10-10": "2008/10/10", "2008-11-20": "2008/11/20",
             "2020-03-18": "2020/3/18", "2020-04-17": "2020/4/17", "2022-10-14": "2022/10/14",
             "2025-04-04": "2025/4/4", "2025-04-08": "2025/4/8"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"

X_COLS = {
    "vix": "VIX（S&P500の恐怖指数）",
    "vxn": "VXN（ナスダック100の恐怖指数）",
    "vix_term": "VIX ÷ VIX3M（1超え＝目先の恐怖が上回る）",
    "unrate_rise": "失業率 - 過去12か月の最低（pt）",
    "rsi14": "RSI(14)（半導体指数 ^SOX）",
    "dist_sma50": "50日移動平均線からの距離（^SOX、%）",
    "dist_sma200": "200日移動平均線からの距離（^SOX、%）",
}
Y_COLS = {
    "price": ("SOXL 終値（対数目盛）", "log"),
    "fwd_1y": ("その日に買った SOXL の 1 年後の倍率（対数目盛）", "log"),
    "further_dd": ("その後 1 年で SOXL がさらに下げた幅（%、0 = その日が底）", "linear"),
}
BUCKETS = {
    "vix": [0, 15, 20, 25, 30, 40, 200],
    "unrate_rise": [-1, 0.25, 0.5, 1, 20],
    "rsi14": [0, 25, 30, 40, 50, 60, 70, 100],
    "dist_sma50": [-100, -20, -10, -5, 0, 5, 10, 100],
    "dist_sma200": [-100, -30, -20, -10, 0, 10, 20, 100],
}


def _font():
    for name in ("Noto Sans CJK JP", "IPAexGothic", "WenQuanYi Zen Hei"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False  # CJK フォントに U+2212 が無いため
            return


def frame(px: pd.DataFrame, ohlcv: dict, fred: dict, extended: bool) -> pd.DataFrame:
    if extended:
        c = build_prices(px)["soxl"]
        c = c.loc[c.index[252]:]  # 52 週の指標が定義できてから
    else:
        c = ohlcv["SOXL"]["Close"].dropna()
    cal = c.index
    sox = px["SOX"].dropna()
    urise = cycle._on_calendar(fred["UNRATE"] - fred["UNRATE"].rolling(12).min(), cycle.FRED_EXTRA["UNRATE"], cal)
    fwd_min = c[::-1].rolling(252, min_periods=1).min()[::-1].shift(-1)  # 翌日〜1 年後の最安値
    # 相対的な底圏: その日の価格が前後 3 か月（±63 営業日）の最安値から +20% 以内（SOXL 基準）
    lo = c.rolling(127, center=True).min()
    return pd.DataFrame({
        "price": c, "fwd_1y": c.shift(-252) / c,
        "further_dd": (fwd_min / c - 1).clip(upper=0).where(c.shift(-252).notna()),
        "bottom_zone": (c <= lo * 1.2).astype(float).where(lo.notna()),
        "vix": px["VIX"].reindex(cal), "vxn": px["VXN"].reindex(cal),
        "vix_term": (px["VIX"] / px["VIX3M"]).reindex(cal),
        "unrate_rise": urise,
        "rsi14": features._rsi(sox, 14).reindex(cal),
        "dist_sma50": (sox / sox.rolling(50).mean() - 1).reindex(cal) * 100,
        "dist_sma200": (sox / sox.rolling(200).mean() - 1).reindex(cal) * 100,
    })


def plot(D: pd.DataFrame, ycol: str, path: Path, title: str, synth_until=None):
    ylabel, yscale = Y_COLS[ycol]
    scale = 100 if ycol == "further_dd" else 1
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    axes = axes.ravel()
    years = D.index.year + D.index.dayofyear / 366
    sc = None
    for ax, (xc, xlabel) in zip(axes, X_COLS.items()):
        d = D[[xc, ycol]].dropna()
        yr = years[D.index.isin(d.index)]
        sc = ax.scatter(d[xc], d[ycol] * scale, c=yr, cmap=SEQ, s=5, alpha=0.55, linewidths=0, rasterized=True)
        for k, dt in enumerate(HIGHLIGHT, 1):
            t = pd.Timestamp(dt)
            if t in d.index:
                yv = d.at[t, ycol] * scale
                ax.scatter(d.at[t, xc], yv, s=150, facecolor="#e8590c", edgecolor="white", linewidth=1.5, zorder=5)
                ax.annotate(str(k), (d.at[t, xc], yv), ha="center", va="center", fontsize=7.5, color="white",
                            fontweight="bold", zorder=6)
        if ycol == "price":
            last = d.index[-1]
            ax.scatter(d.at[last, xc], d.at[last, ycol], s=120, marker="*", facecolor="#c92a2a", edgecolor="white", zorder=6)
            ax.annotate(f"直近 {last:%Y/%m/%d}", (d.at[last, xc], d.at[last, ycol]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color="#c92a2a", zorder=6)
        if ycol == "fwd_1y":
            ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[-1].axis("off")
    cb = fig.colorbar(sc, ax=axes[-1], fraction=0.5, location="left")
    cb.solids.set_alpha(1)
    cb.set_label("年（薄い＝古い → 濃い＝新しい）", color=INK)
    legend = "\n".join(f"{k}: {lab}" for k, lab in enumerate(HIGHLIGHT.values(), 1))
    note = f"\n\n{synth_until:%Y/%m} 以前の SOXL は\n^SOX からの合成値" if synth_until is not None else ""
    axes[-1].text(0.45, 0.5, "オレンジ丸の番号\n" + legend + "\n\n1 点 = 1 営業日" + note,
                  transform=axes[-1].transAxes, fontsize=10.5, color=INK, va="center")
    fig.suptitle(title, fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def bucket_tables(D: pd.DataFrame, eras: dict) -> pd.DataFrame:
    """横軸の区間ごとの成績。日次の点は強く連続しているので、独立した事例数の目安として「該当月数」も示す。"""
    rows = []
    for xc, edges in BUCKETS.items():
        for era, (a, b) in eras.items():
            d = D.loc[a:b, [xc, "fwd_1y", "further_dd", "bottom_zone"]].dropna(subset=[xc, "fwd_1y"])
            for iv, g in d.groupby(pd.cut(d[xc], edges), observed=True):
                rows.append(dict(indicator=X_COLS[xc], era=era, range=str(iv), days=len(g),
                                 months=g.index.to_period("M").nunique(),
                                 fwd1y_median=g.fwd_1y.median(), fwd1y_plus=(g.fwd_1y > 1).mean(),
                                 fwd1y_worst=g.fwd_1y.min(), further_dd_median=g.further_dd.median(),
                                 near_bottom_1y=(g.further_dd > -0.10).mean(), bottom_zone_3m=g.bottom_zone.mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/scatter"))
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--extended", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _font()
    px, ohlcv = load(a.cache)
    fred = cycle.load_fred_extra(a.cache / "fred_monthly")
    D = frame(px, ohlcv, fred, a.extended)
    tag = "extended" if a.extended else "actual"
    D.to_csv(a.out / f"scatter_data_{tag}.csv")
    synth = px["SOXL"].first_valid_index() if a.extended else None
    span = f"{D.index[0]:%Y}〜" + ("、2010 年以前は合成 SOXL" if a.extended else "")
    plot(D, "price", a.out / f"fig_{tag}_price.png", f"SOXL の価格（{span}、1 点 = 1 日）", synth)
    plot(D, "fwd_1y", a.out / f"fig_{tag}_fwd1y.png", f"その日に買ったら 1 年後に何倍？（{span}、1.0 = 損益なし）", synth)
    plot(D, "further_dd", a.out / f"fig_{tag}_further_dd.png",
         f"その日は底だった？ その後 1 年でさらに下げた幅（{span}、0% = その日が 1 年の底）", synth)
    eras = ({"1996-2009（合成）": ("1996", "2010-03-10"), "2010-（実SOXL）": ("2010-03-11", "2100")} if a.extended
            else {"2010-": ("2010", "2100")})
    T = bucket_tables(D, eras)
    T.to_csv(a.out / f"buckets_{tag}.csv", index=False)
    (a.out / f"buckets_{tag}.md").write_text(md_table(T), encoding="utf-8")
    print(f"図: {a.out}")


if __name__ == "__main__":
    main()
