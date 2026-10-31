"""2010 年以降の実 SOXL で、RSI(14) と VXN の組み合わせが「前後 3 か月の底圏」をどれだけ捉えるか。

実行: cd analysis && python -m soxl_factors.scatter --cache data_cache_latest --extended && python -m soxl_factors.vxn_rsi
（scatter が出力する scatter_data_extended.csv を使う）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .report import md_table  # noqa: E402
from .scatter import GRID, INK, MUTED, _font  # noqa: E402

VXN_BINS, VXN_LAB = [0, 20, 25, 30, 40, 200], ["<20", "20-25", "25-30", "30-40", "40+"]
RSI_BINS, RSI_LAB = [0, 30, 35, 40, 50, 100], ["<30", "30-35", "35-40", "40-50", "50+"]


def prepare(path: Path) -> pd.DataFrame:
    D = pd.read_csv(path, index_col=0, parse_dates=True).loc["2010-03-11":]
    c = D.price
    D["fwd_6m"] = c.shift(-126) / c
    lo = c.rolling(127, center=True).min()
    D["swing_low"] = (c == lo).astype(float).where(lo.notna())  # 前後 3 か月の最安値の日
    return D


def grids(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    vb = pd.cut(d.vxn, VXN_BINS, labels=VXN_LAB)
    rb = pd.cut(d.rsi14, RSI_BINS, labels=RSI_LAB)
    g = d.groupby([rb, vb], observed=False)
    return {
        "底圏だった割合": g.bottom_zone.mean().unstack(),
        "該当した月数": g.apply(lambda x: x.index.to_period("M").nunique()).unstack(),
        "6か月後の倍率（中央値）": g.fwd_6m.median().unstack(),
        "1年後プラスの割合": g.fwd_1y.apply(lambda s: (s.dropna() > 1).mean() if s.notna().any() else np.nan).unstack(),
    }


def figure(D: pd.DataFrame, pz: pd.DataFrame, months: pd.DataFrame, path: Path):
    d = D.dropna(subset=["vxn", "rsi14", "bottom_zone"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    nz, z = d[d.bottom_zone == 0], d[d.bottom_zone == 1]
    ax.scatter(nz.vxn, nz.rsi14, s=5, c="#c9c8c3", alpha=0.5, linewidths=0, label="底圏ではない日", rasterized=True)
    ax.scatter(z.vxn, z.rsi14, s=7, c="#2a78d6", alpha=0.7, linewidths=0, label="前後3か月の底圏の日（最安値+20%以内）", rasterized=True)
    sw = D[D.swing_low == 1]
    ax.scatter(sw.vxn, sw.rsi14, s=160, marker="*", c="#e8590c", edgecolor="white", linewidth=1, zorder=5,
               label=f"前後3か月の最安値の日（{len(sw)}回）")
    for t, r in sw.iterrows():
        ax.annotate(f"{t:%Y/%m}", (r.vxn, r.rsi14), xytext=(5, -9), textcoords="offset points", fontsize=7, color=INK)
    cur = D.dropna(subset=["vxn", "rsi14"])
    ax.scatter(cur.vxn.iloc[-1], cur.rsi14.iloc[-1], s=140, marker="X", c="#c92a2a", edgecolor="white", zorder=6,
               label=f"直近 {cur.index[-1]:%Y/%m/%d}")
    for y in (30, 35, 40):
        ax.axhline(y, color=MUTED, lw=0.7, ls="--")
    ax.axvline(30, color=MUTED, lw=0.7, ls="--")
    ax.set_xlabel("VXN（ナスダック100の恐怖指数）", color=INK)
    ax.set_ylabel("RSI(14)（半導体指数 ^SOX）", color=INK)
    ax.set_title("VXN × RSI（2010〜、1点＝1日）", color=INK)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, color=GRID, lw=0.6)
    ax = axes[1]
    cmap = LinearSegmentedColormap.from_list("s", ["#f3f2ee", "#86b6ef", "#256abf", "#0d366b"])
    rows = list(pz.index)[::-1]
    im = ax.imshow(pz.loc[rows].to_numpy(float), cmap=cmap, vmin=0, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(pz.columns)), pz.columns)
    ax.set_yticks(range(len(rows)), rows)
    for i, r in enumerate(rows):
        for j, cn in enumerate(pz.columns):
            v = pz.loc[r, cn]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}\n({months.loc[r, cn]}か月)", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.4 else INK)
    ax.set_xlabel("VXN", color=INK)
    ax.set_ylabel("RSI(14)", color=INK)
    ax.set_title(f"底圏だった日の割合（全日平均 {d.bottom_zone.mean():.0%}）\n括弧内は該当した月の数", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("output/scatter/scatter_data_extended.csv"))
    ap.add_argument("--out", type=Path, default=Path("output/scatter"))
    a = ap.parse_args()
    _font()
    D = prepare(a.data)
    d = D.dropna(subset=["vxn", "rsi14", "bottom_zone"])
    G = grids(d)
    sw = D[D.swing_low == 1]
    L = ["# VXN × RSI（2010 年〜、実 SOXL）\n", "## 前後 3 か月の最安値の日\n",
         md_table(sw[["price", "rsi14", "vxn", "vix", "dist_sma50", "fwd_6m", "fwd_1y"]], index=True), ""]
    for name, g in G.items():
        L += [f"## {name}\n", md_table(g, index=True, floatfmt=".2f"), ""]
    for era, (s, e) in {"2010-2017": ("2010", "2017-12-31"), "2018-": ("2018", "2100")}.items():
        L += [f"## 底圏だった割合（{era}）\n", md_table(grids(d.loc[s:e])["底圏だった割合"], index=True, floatfmt=".2f"), ""]
    rows = []
    for name, m in {"RSI<35": d.rsi14 < 35, "VXN>=30": d.vxn >= 30, "RSI<35 & VXN>=30": (d.rsi14 < 35) & (d.vxn >= 30),
                    "RSI<35 & VXN<30": (d.rsi14 < 35) & (d.vxn < 30), "全日": d.rsi14.notna()}.items():
        g = d[m]
        rows.append(dict(条件=name, 日数=len(g), 月数=g.index.to_period("M").nunique(), 底圏の割合=g.bottom_zone.mean(),
                         六か月後中央値=g.fwd_6m.median(), 一年後プラス=(g.fwd_1y.dropna() > 1).mean(),
                         一年後最悪=g.fwd_1y.min(), 捉えた最安値の日=f"{sw.index.isin(g.index).sum()}/{len(sw)}"))
    L += ["## 条件別\n", md_table(pd.DataFrame(rows)), "", "![vxn_rsi](fig_vxn_rsi_2010.png)"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "vxn_rsi_2010.md").write_text("\n".join(L), encoding="utf-8")
    figure(D, G["底圏だった割合"], G["該当した月数"], a.out / "fig_vxn_rsi_2010.png")
    print(f"レポート: {a.out / 'vxn_rsi_2010.md'}")


if __name__ == "__main__":
    main()
