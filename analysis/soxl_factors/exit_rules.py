"""売り方（出口）の比較。買いは「^SOX RSI(14)<40 かつ 失業率上昇<0.4」の翌日に一括。

実行: cd analysis && python -m soxl_factors.exit_rules --cache data_cache_latest
- 判断は引け、売買は翌営業日の引け。ノーポジ中は短期金利。手数料は含めない（回数が少ないため）。
- 売った後は、次の買いシグナル（売却後に新たに出たもの）まで待つ。
- 比較期間: 2010-17 / 2018-（参考に 1996-2009 は合成 SOXL）。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import backtest, cycle, features
from .data import load
from .report import md_table
from .scale_in import build_prices

ERAS = {"1996-2009(合成)": ("1996-01-01", "2010-03-10"), "2010-17": ("2010-03-11", "2017-12-31"),
        "2018-": ("2018-01-01", "2100-01-01")}

# 出口ルール: (i, ctx) -> 売る比率（0〜1、保有分に対する割合）
Exit = Callable[[int, dict], float]


def rsi_exit(level: float) -> Exit:
    return lambda i, c: 1.0 if c["rsi"][i] >= level else 0.0


def rsi_two_step(l1: float, l2: float) -> Exit:
    def f(i, c):
        if c["rsi"][i] >= l2:
            return 1.0
        if c["rsi"][i] >= l1 and not c["state"].get("half"):
            c["state"]["half"] = True
            return 0.5
        return 0.0
    return f


def hold_days(n: int) -> Exit:
    return lambda i, c: 1.0 if i - c["entry_i"] >= n else 0.0


def trailing(stop: float) -> Exit:
    return lambda i, c: 1.0 if c["px"][i] <= c["peak"] * (1 - stop) else 0.0


def target(gain: float) -> Exit:
    return lambda i, c: 1.0 if c["px"][i] >= c["entry_px"] * (1 + gain) else 0.0


def ma_break(n: int) -> Exit:
    """エントリー後に一度 n 日線を上回り、その後下回ったら売る。"""
    def f(i, c):
        above = c["sox"][i] > c[f"sma{n}"][i]
        if above:
            c["state"][f"was_above{n}"] = True
            return 0.0
        return 1.0 if c["state"].get(f"was_above{n}") else 0.0
    return f


def rsi_then_trail(level: float, stop: float) -> Exit:
    """RSI が level に達した後は、高値から stop 下がったら売る（利益を伸ばす）。"""
    def f(i, c):
        if c["rsi"][i] >= level:
            c["state"]["armed"] = True
        if c["state"].get("armed") and c["px"][i] <= c["peak"] * (1 - stop):
            return 1.0
        return 0.0
    return f


def never() -> Exit:
    return lambda i, c: 0.0


def exits() -> dict[str, Exit]:
    E: dict[str, Exit] = {"売らない（持ち続ける）": never()}
    for lv in (60, 65, 70, 75, 80):
        E[f"RSI {lv}以上で全部売る"] = rsi_exit(lv)
    E["RSI 70で半分・80で残り"] = rsi_two_step(70, 80)
    for n, lab in ((63, "3か月"), (126, "6か月"), (252, "1年")):
        E[f"{lab}持って売る"] = hold_days(n)
    for s in (0.2, 0.3, 0.4):
        E[f"高値から{int(s * 100)}%下げたら売る"] = trailing(s)
    for g in (0.5, 1.0):
        E[f"+{int(g * 100)}%で利益確定"] = target(g)
    E["SOX 50日線割れで売る"] = ma_break(50)
    E["SOX 200日線割れで売る"] = ma_break(200)
    for s in (0.15, 0.25):
        E[f"RSI 70到達後、高値から{int(s * 100)}%下げたら売る"] = rsi_then_trail(70, s)
    return E


def simulate(price: pd.Series, sox: pd.Series, rsi: pd.Series, entry: pd.Series, rf: pd.Series, rule: Exit):
    px, rs, en = price.to_numpy(), rsi.to_numpy(), entry.fillna(False).to_numpy(bool)
    ctx = {"px": px, "rsi": rs, "sox": sox.to_numpy(), "sma50": sox.rolling(50).mean().to_numpy(),
           "sma200": sox.rolling(200).mean().to_numpy()}
    rfv = rf.to_numpy()
    n = len(px)
    pos = 0.0  # 保有比率（資産に対する SOXL の割合ではなく、建玉の残り比率）
    ret = np.zeros(n)
    expo = np.zeros(n)
    trades = []
    pend_buy = pend_sell = 0.0
    w = 0.0  # 資産に占める SOXL の比率
    for i in range(1, n):
        # 前日引けの判断を本日引けで執行（当日のリターンは前日のポジションで計算）
        r_asset = px[i] / px[i - 1] - 1
        ret[i] = w * r_asset + (1 - w) * rfv[i]
        w = w * (1 + r_asset) / (1 + ret[i]) if (1 + ret[i]) != 0 else 0.0
        if pend_sell > 0 and pos > 0:
            sold = pos * pend_sell
            w *= (1 - pend_sell)
            trades[-1]["exits"].append((i, px[i], sold))
            pos -= sold
            if pos <= 1e-9:
                pos, w = 0.0, 0.0
                trades[-1]["exit_i"] = i
        if pend_buy and pos == 0:
            pos, w = 1.0, 1.0
            ctx.update(entry_i=i, entry_px=px[i], peak=px[i], state={})
            trades.append(dict(entry_i=i, entry_px=px[i], exits=[]))
        pend_buy, pend_sell = False, 0.0
        expo[i] = w
        if pos > 0:
            ctx["peak"] = max(ctx["peak"], px[i])
            pend_sell = rule(i, ctx)
        elif en[i] and (not trades or i > trades[-1].get("exit_i", -1)):
            pend_buy = True
    return pd.Series(ret, index=price.index), pd.Series(expo, index=price.index), trades


def trade_table(trades: list[dict], price: pd.Series) -> pd.DataFrame:
    px = price.to_numpy()
    rows = []
    for t in trades:
        if not t["exits"]:
            continue
        sold = sum(f for _, _, f in t["exits"])
        avg_exit = sum(p * f for _, p, f in t["exits"]) / sold
        last_i = t["exits"][-1][0]
        after = px[last_i:last_i + 64]
        rows.append(dict(entry=price.index[t["entry_i"]], exit=price.index[last_i], days=last_i - t["entry_i"],
                         ret=avg_exit / t["entry_px"] - 1, closed=t.get("exit_i") is not None,
                         after_3m_max=after.max() / avg_exit - 1 if len(after) else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("output/exit_rules"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    px, _ = load(a.cache)
    P = build_prices(px).loc["1996":]
    price, sox = P["soxl"], P["sox"]
    rsi = features._rsi(px["SOX"].dropna(), 14).reindex(price.index)
    fred = cycle.load_fred_extra(a.cache / "fred_monthly")
    u = fred["UNRATE"]
    urise = cycle._on_calendar(u - u.rolling(12).min(), cycle.FRED_EXTRA["UNRATE"], price.index)
    entry = (rsi < 40) & (urise < 0.4)
    rf = P["rf"]
    rows, trade_rows = [], []
    for era, (s, e) in ERAS.items():
        m = (price.index >= s) & (price.index <= e)
        pr, sx, rs, en, rr = price[m], sox[m], rsi[m], entry[m], rf[m]
        bh = pr.pct_change().fillna(0.0)
        rows.append(dict(期間=era, 売り方="（参考）期間の最初から買い持ち", **backtest.metrics(bh, rr), exposure=1.0))
        for name, rule in exits().items():
            r, ex, tr = simulate(pr, sx, rs, en, rr, rule)
            T = trade_table(tr, pr)
            m_ = backtest.metrics(r, rr)
            rows.append(dict(期間=era, 売り方=name, **m_, exposure=ex.mean(), 取引数=len(tr),
                             勝率=(T.ret > 0).mean() if len(T) else np.nan,
                             一回の損益_中央値=T.ret.median() if len(T) else np.nan,
                             保有日数_中央値=T.days.median() if len(T) else np.nan,
                             売った後3か月の最高値_中央値=T.after_3m_max.median() if len(T) else np.nan))
            if len(T):
                T.insert(0, "売り方", name)
                T.insert(0, "期間", era)
                trade_rows.append(T)
    R = pd.DataFrame(rows)
    R.to_csv(a.out / "exit_rules.csv", index=False)
    pd.concat(trade_rows).to_csv(a.out / "trades.csv", index=False)
    cols = ["期間", "売り方", "CAGR", "maxDD", "sharpe", "exposure", "取引数", "勝率", "一回の損益_中央値", "保有日数_中央値",
            "売った後3か月の最高値_中央値"]
    L = ["# 売り方の比較（買い: ^SOX RSI14<40 かつ 失業率上昇<0.4 の翌日に一括）\n",
         "- 判断は引け、売買は翌営業日の引け。ノーポジ中は短期金利。手数料・税は含めない",
         "- 売った後3か月の最高値 = 売値から 3 か月以内に SOXL がどこまで上がったか（大きいほど売り急ぎ）\n",
         md_table(R[[c for c in cols if c in R]])]
    (a.out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
