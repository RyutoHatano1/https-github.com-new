"""RSI(14)<40 をきっかけにした購入パターン（一括・時間分割・ナンピン・RSI 段階・反発確認など）の平均購入単価の比較。

実行: cd analysis && python -m soxl_factors.buy_patterns --cache data_cache_latest
- 局面: ^SOX の RSI(14) が 40 を割った日（RSI が 50 以上に戻るまで次の局面は数えない）。2010 年以降の実 SOXL。
- 判断は引け、購入は翌営業日の引け。どのルールも期限（既定 63 営業日）までに予算を使い切り、残りは期限の日に買う。
- 比較の物差し: 平均購入単価 ÷ 一括（シグナル翌日）の購入価格。1 未満なら一括より安く買えた。
- ルールの選択は 2010-17 の局面で行い、2018- の局面で検証する。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import features
from .data import load
from .report import md_table

SPLIT = pd.Timestamp("2018-01-01")


@dataclass
class Episode:
    start: pd.Timestamp
    price: np.ndarray   # シグナル日 s から s+W+1 までの SOXL 終値（index 0 = シグナル日）
    rsi: np.ndarray


# ルール: (Episode, W) -> [(購入日オフセット, 予算比率)]。オフセットは 1..W（翌日〜期限）
Rule = Callable[[Episode, int], list[tuple[int, float]]]


def _fill(buys: list[tuple[int, float]], W: int) -> list[tuple[int, float]]:
    spent = sum(f for _, f in buys)
    if spent < 1 - 1e-9:
        buys = buys + [(W, 1 - spent)]
    return buys


def lump() -> Rule:
    return lambda e, W: [(1, 1.0)]


def time_split(n: int, k: int) -> Rule:
    return lambda e, W: _fill([(min(1 + i * k, W), 1 / n) for i in range(n)], W)


def nanpin(n: int, drop: float, weights: str = "equal") -> Rule:
    """初回はシグナル翌日。以降、終値が前回の購入価格から drop 以上下がったら翌日に買い増し。"""
    w = np.ones(n) if weights == "equal" else 2.0 ** np.arange(n)
    w = w / w.sum()

    def rule(e: Episode, W: int):
        buys, last, k = [(1, w[0])], e.price[1], 1
        for d in range(1, W):
            if k >= n:
                break
            if e.price[d] <= last * (1 - drop) and d + 1 <= W:
                buys.append((d + 1, w[k]))
                last, k = e.price[d + 1], k + 1
        return _fill(buys, W)
    return rule


def rsi_ladder(levels: tuple[float, ...]) -> Rule:
    """RSI が各水準を初めて下回った翌日に 1/len ずつ買う（最初の水準 40 はシグナル日）。"""
    def rule(e: Episode, W: int):
        buys = []
        for lv in levels:
            hit = np.where(e.rsi[: W] < lv)[0]
            if len(hit):
                buys.append((min(hit[0] + 1, W), 1 / len(levels)))
        return _fill(buys, W)
    return rule


def confirm(first: float, cross: float = 40) -> Rule:
    """first をシグナル翌日に買い、残りは RSI が cross を上抜けた翌日に買う（反発確認）。"""
    def rule(e: Episode, W: int):
        buys = [(1, first)] if first > 0 else []
        up = np.where((e.rsi[1:W] >= cross) & (e.rsi[: W - 1] < cross))[0]
        if len(up):
            buys.append((up[0] + 2, 1 - first))
        return _fill(buys, W)
    return rule


def daily_while_low(n: int, thr: float = 40) -> Rule:
    """RSI<thr の日ごとに翌日 1/n ずつ買う（最大 n 回）。"""
    def rule(e: Episode, W: int):
        days = np.where(e.rsi[:W] < thr)[0][:n]
        return _fill([(d + 1, 1 / n) for d in days], W)
    return rule


def low_rebound(first: float, bounce: float) -> Rule:
    """first をシグナル翌日に買い、残りは「シグナル後の最安値から bounce 以上戻した」翌日に買う。"""
    def rule(e: Episode, W: int):
        buys = [(1, first)] if first > 0 else []
        lo = e.price[0]
        for d in range(1, W):
            lo = min(lo, e.price[d])
            if e.price[d] >= lo * (1 + bounce) and lo < e.price[0]:
                buys.append((d + 1, 1 - first))
                break
        return _fill(buys, W)
    return rule


def rules() -> dict[str, Rule]:
    R: dict[str, Rule] = {"一括（翌日）": lump()}
    for n in (2, 3, 4, 5, 6):
        for k in (5, 10, 21):
            if 1 + (n - 1) * k <= 63:
                R[f"時間分割 {n}回・{k}日おき"] = time_split(n, k)
    for n in (2, 3, 4, 5):
        for drop in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            R[f"ナンピン均等 {n}回・{int(drop * 100)}%下落ごと"] = nanpin(n, drop)
            R[f"ナンピン倍々 {n}回・{int(drop * 100)}%下落ごと"] = nanpin(n, drop, "double")
    for lv in ((40, 35), (40, 35, 30), (40, 35, 30, 25), (40, 30), (40, 30, 25), (40, 30, 20)):
        R["RSI段階 " + "/".join(map(str, lv))] = rsi_ladder(lv)
    for f in (0.0, 0.25, 0.5):
        R[f"反発確認 初回{int(f * 100)}%＋RSI40上抜けで残り"] = confirm(f)
        for b in (0.10, 0.20):
            R[f"安値から{int(b * 100)}%反発 初回{int(f * 100)}%＋残り"] = low_rebound(f, b)
    for n in (3, 5, 10):
        R[f"RSI<40の日ごと {n}回"] = daily_while_low(n)
    return R


def episodes(price: pd.Series, rsi: pd.Series, W: int, enter: float = 40, reset: float = 50) -> list[Episode]:
    p, r, idx = price.to_numpy(), rsi.to_numpy(), price.index
    eps, armed = [], True
    for i in range(len(p)):
        if np.isnan(r[i]):
            continue
        if r[i] >= reset:
            armed = True
        elif armed and r[i] < enter:
            armed = False
            if i + W + 1 < len(p):
                eps.append(Episode(idx[i], p[i:i + W + 2], r[i:i + W + 2]))
    return eps


def evaluate(eps: list[Episode], R: dict[str, Rule], W: int) -> pd.DataFrame:
    rows = []
    for e in eps:
        lump_px = e.price[1]
        win_min = e.price[1:W + 1].min()
        for name, rule in R.items():
            buys = rule(e, W)
            assert abs(sum(f for _, f in buys) - 1) < 1e-6, name
            shares = sum(f / e.price[d] for d, f in buys)
            avg = 1 / shares
            rows.append(dict(rule=name, start=e.start, cost_vs_lump=avg / lump_px, cost_vs_low=avg / win_min,
                             n_buys=len(buys), last_buy_day=max(d for d, _ in buys)))
    return pd.DataFrame(rows)


def summarize(E: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        lr = np.log(g.cost_vs_lump)
        return pd.Series(dict(局面数=len(g), 単価比_幾何平均=np.exp(lr.mean()), 単価比_中央値=g.cost_vs_lump.median(),
                              一括より安く買えた割合=(g.cost_vs_lump < 0.999).mean(), 最悪=g.cost_vs_lump.max(),
                              最良=g.cost_vs_lump.min(), 期間内最安値比_中央値=g.cost_vs_low.median()))
    out = {}
    for era, m in {"2010-17": E.start < SPLIT, "2018-": E.start >= SPLIT, "全期間": E.start.notna()}.items():
        out[era] = E[m].groupby("rule").apply(agg)
    T = pd.concat(out, axis=1)
    T.columns = [f"{b}_{a}" for a, b in T.columns]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("output/buy_patterns"))
    ap.add_argument("--window", type=int, default=63)
    ap.add_argument("--reset", type=float, default=50, help="RSI がこの値以上に戻ったら次の局面を数える（40 なら下抜けごとに 1 局面）")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    px, ohlcv = load(a.cache)
    soxl = ohlcv["SOXL"]["Close"].dropna()
    rsi = features._rsi(px["SOX"].dropna(), 14).reindex(soxl.index)
    R = rules()
    res = {}
    for W in sorted({a.window, 126}):
        eps = episodes(soxl, rsi, W, reset=a.reset)
        E = evaluate(eps, R, W)
        E.to_csv(a.out / f"episodes_W{W}.csv", index=False)
        T = summarize(E)
        T.to_csv(a.out / f"summary_W{W}.csv")
        res[W] = (eps, E, T)

    W = a.window
    eps, E, T = res[W]
    key_a, key_b, key_all = "単価比_幾何平均_2010-17", "単価比_幾何平均_2018-", "単価比_幾何平均_全期間"
    T["順位_2010-17"] = T[key_a].rank()
    T["順位_2018-"] = T[key_b].rank()
    cols = [key_a, key_b, key_all, "一括より安く買えた割合_全期間", "最悪_全期間", "期間内最安値比_中央値_全期間", "順位_2010-17", "順位_2018-"]
    rank_corr = T[key_a].corr(T[key_b], method="spearman")
    best_a = T.sort_values(key_a).head(10)
    L = ["# RSI(14)<40 をきっかけにした購入パターンの比較（2010〜、実 SOXL）\n",
         f"- 局面数 {len(eps)}（2010-17: {sum(e.start < SPLIT for e in eps)}、2018-: {sum(e.start >= SPLIT for e in eps)}）、ルール数 {len(T)}",
         f"- 期限 {W} 営業日で全額を投入。単価比 = 平均購入単価 ÷ 一括（翌日）の価格。1 未満なら一括より安い",
         f"- 2010-17 と 2018- のルール順位の相関: {rank_corr:.2f}\n",
         "## 2010-17 で単価が低かった上位 10 ルール（2018- は検証）\n", md_table(best_a[cols], index=True), "",
         "## 全ルール（全期間の幾何平均の低い順）\n", md_table(T.sort_values(key_all)[cols], index=True), ""]
    W2 = 126
    T2 = res[W2][2]
    L += [f"## 参考: 期限 {W2} 営業日（6 か月）\n",
          md_table(T2.sort_values(key_all)[[key_a, key_b, key_all, "一括より安く買えた割合_全期間", "最悪_全期間"]].head(15), index=True)]
    (a.out / "report.md").write_text("\n".join(L), encoding="utf-8")
    T.to_csv(a.out / f"summary_W{W}.csv")
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
