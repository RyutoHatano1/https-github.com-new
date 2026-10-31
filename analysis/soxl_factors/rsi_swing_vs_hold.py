"""「RSI<40 で買い RSI>=75 で売る（売却代金は次の合図で全額再投資＝複利）」と「RSI<40 で買って持ち続ける」の比較。

実行: cd analysis && python -m soxl_factors.rsi_swing_vs_hold --cache data_cache_latest
判断は引け、売買は翌営業日の引け。待機中の現金は短期金利。手数料・税・為替は含めない。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import backtest, features  # noqa: E402
from .data import load  # noqa: E402
from .exit_rules import never, rsi_exit, simulate  # noqa: E402
from .report import md_table  # noqa: E402
from .scale_in import build_prices  # noqa: E402
from .scatter import GRID, INK, MUTED, _font  # noqa: E402


def run_pair(P: pd.DataFrame, rsi: pd.Series, start, sell_level: float = 75):
    m = P.index >= pd.Timestamp(start)
    pr, sx, rs, rf = P.soxl[m], P.sox[m], rsi[m], P.rf[m]
    entry = rs < 40
    out = {}
    for name, rule in ((f"RSI40で買い RSI{sell_level:g}で売る（複利）", rsi_exit(sell_level)), ("RSI40で買って持ち続ける", never())):
        r, ex, tr = simulate(pr, sx, rs, entry, rf, rule)
        out[name] = (r, ex, tr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("output/rsi_swing_vs_hold"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _font()
    px, _ = load(a.cache)
    P = build_prices(px)
    rsi = features._rsi(px["SOX"].dropna(), 14).reindex(P.index)

    rows = []
    res = run_pair(P, rsi, "2010-03-11")
    for name, (r, ex, tr) in res.items():
        eq = (1 + r).cumprod()
        rows.append(dict(方法=name, 最終倍率=eq.iloc[-1], **backtest.metrics(r, P.rf.reindex(r.index)),
                         投資している日の割合=ex.mean(), 売買回数=len(tr)))
    S = pd.DataFrame(rows)

    yrs = []
    for y in range(2010, 2025):
        rr = run_pair(P, rsi, f"{y}-01-01" if y > 2010 else "2010-03-11")
        mult = {k: float((1 + v[0]).cumprod().iloc[-1]) for k, v in rr.items()}
        k1, k2 = list(mult)
        yrs.append(dict(開始年=y, **{f"{k}_倍率": v for k, v in mult.items()}, 勝ち=("売買" if mult[k1] > mult[k2] else "持ち続け")))
    Y = pd.DataFrame(yrs)

    # 売る水準の感度（2010 年開始）
    sens = []
    for lv in (65, 70, 75, 80, 85):
        rr = run_pair(P, rsi, "2010-03-11", lv)
        k = list(rr)[0]
        r = rr[k][0]
        sens.append(dict(売るRSI=lv, 最終倍率=float((1 + r).cumprod().iloc[-1]), **backtest.metrics(r, P.rf.reindex(r.index))))
    Z = pd.DataFrame(sens)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (name, (r, ex, tr)), col in zip(res.items(), ("#2a78d6", "#e8590c")):
        ax.semilogy(r.index, (1 + r).cumprod(), lw=1.3, color=col, label=name)
    ax.set_ylabel("1 を投資した場合の資産（対数目盛）", color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    ax.legend()
    ax.set_title("2010年3月〜（実 SOXL）", color=INK)
    fig.tight_layout()
    fig.savefig(a.out / "fig_equity.png", dpi=130)
    plt.close(fig)

    S.to_csv(a.out / "summary.csv", index=False)
    Y.to_csv(a.out / "by_start_year.csv", index=False)
    Z.to_csv(a.out / "sell_level.csv", index=False)
    L = ["# RSI40 で買い RSI75 で売る（複利） vs RSI40 で買って持ち続ける\n",
         "- 判断は引け、売買は翌営業日の引け。待機中の現金は短期金利。手数料・税は含めない\n",
         "## 2010 年 3 月から\n", md_table(S), "", "## 開始年ごと（終点は直近）\n", md_table(Y), "",
         "## 売る RSI の水準（2010 年開始）\n", md_table(Z), "", "![equity](fig_equity.png)"]
    (a.out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
