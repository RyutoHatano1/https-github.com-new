"""RSI40 で買った後の売り方「RSI≥75」「+50% 利益確定」「どちらか早い方」の比較（複利・2010 年〜の実 SOXL）。

実行: cd analysis && python -m soxl_factors.exit_compare --cache data_cache_latest
判断は引け、売買は翌営業日の引け。待機中の現金は短期金利。税は日本の 20.315% を売却益ごとに概算（損失繰越なし）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import backtest, features
from .data import load
from .exit_rules import never, rsi_exit, simulate, target
from .report import md_table
from .scale_in import build_prices

TAX = 0.20315


def either(a, b):
    return lambda i, c: max(a(i, c), b(i, c))


def rules() -> dict:
    R = {"持ち続ける": never()}
    for lv in (70, 75, 80):
        R[f"RSI{lv}で売る"] = rsi_exit(lv)
    for g in (0.3, 0.5, 0.7, 1.0):
        R[f"+{int(g * 100)}%で売る"] = target(g)
    R["RSI75 か +50% の早い方"] = either(rsi_exit(75), target(0.5))
    return R


def after_tax(trades, final_px: float) -> float:
    net = 1.0
    for t in trades:
        g = (t["exits"][-1][1] if t["exits"] else final_px) / t["entry_px"]
        net *= g - (TAX * (g - 1) if (g > 1 and t["exits"]) else 0)
    # 保有中の玉は最後に売ったとして課税
    if trades and not trades[-1]["exits"]:
        g = final_px / trades[-1]["entry_px"]
        net = net / g * (g - (TAX * (g - 1) if g > 1 else 0))
    return net


def run(P, rsi, start, rule):
    m = P.index >= pd.Timestamp(start)
    pr = P.soxl[m]
    r, ex, tr = simulate(pr, P.sox[m], rsi[m], rsi[m] < 40, P.rf[m], rule)
    return r, ex, tr, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("output/exit_compare"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    px, _ = load(a.cache)
    P = build_prices(px)
    rsi = features._rsi(px["SOX"].dropna(), 14).reindex(P.index)
    rows, yrs = [], []
    for name, rule in rules().items():
        r, ex, tr, pr = run(P, rsi, "2010-03-11", rule)
        rets = [(t["exits"][-1][1] / t["entry_px"] - 1) for t in tr if t["exits"]]
        days = [t["exits"][-1][0] - t["entry_i"] for t in tr if t["exits"]]
        rows.append(dict(売り方=name, 最終倍率=float((1 + r).cumprod().iloc[-1]), 税引後倍率=after_tax(tr, float(pr.iloc[-1])),
                         **{k: v for k, v in backtest.metrics(r, P.rf.reindex(r.index)).items() if k in ("CAGR", "sharpe", "maxDD")},
                         投資期間=ex.mean(), 売却回数=len(rets), 勝率=sum(x > 0 for x in rets) / max(len(rets), 1),
                         一回の利益_中央値=pd.Series(rets).median(), 保有日数_中央値=pd.Series(days).median()))
        row = dict(売り方=name)
        for y in range(2010, 2025):
            r, *_ = run(P, rsi, f"{y}-01-01" if y > 2010 else "2010-03-11", rule)
            row[str(y)] = float((1 + r).cumprod().iloc[-1])
        yrs.append(row)
    S, Y = pd.DataFrame(rows), pd.DataFrame(yrs).set_index("売り方")
    # 開始年ごとに、RSI75 と +50% と 早い方 のどれが最大か
    focus = ["RSI75で売る", "+50%で売る", "RSI75 か +50% の早い方"]
    wins = Y.loc[focus].idxmax().value_counts()
    S.to_csv(a.out / "summary.csv", index=False)
    Y.to_csv(a.out / "by_start_year.csv")
    L = ["# 売り方の比較: RSI75 / +50% / 早い方（RSI40 で買い、複利、2010 年〜）\n",
         "- 判断は引け、売買は翌営業日の引け。税引後は 20.315% を売却益ごとに概算（損失繰越なし）\n",
         md_table(S), "", "## 開始年ごとの最終倍率（終点は直近）\n", md_table(Y.T.reset_index().rename(columns={"index": "開始年"})), "",
         "## 3 つの売り方で最大だった開始年の数\n", md_table(wins.rename("年数").reset_index())]
    (a.out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
