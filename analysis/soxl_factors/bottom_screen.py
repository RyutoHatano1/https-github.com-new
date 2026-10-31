"""2010 年以降の実 SOXL で「前後 3 か月の底圏（最安値 +20% 以内）」とそれ以外を分ける指標の総当たり。

実行: cd analysis && python -m soxl_factors.bottom_screen --cache data_cache_latest
1. 全日で各指標の AUC（期間を 2010-17 / 2018- に分けて安定性も見る）
2. RSI(14)<40 の日に限って、底圏とそれ以外をさらに分ける指標（上乗せ効果）
3. 上乗せ条件の閾値は 2010-17 だけで決め、2018- で検証
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import cycle, features, reversal
from .data import load
from .report import md_table

ERA_A, ERA_B = ("2010-03-11", "2017-12-31"), ("2018-01-01", "2100-01-01")


def build(px: pd.DataFrame, ohlcv: dict, cache: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    c = ohlcv["SOXL"]["Close"].dropna()
    lo = c.rolling(127, center=True).min()
    zone = (c <= lo * 1.2).astype(float).where(lo.notna())
    X1, _ = features.build_features(px, ohlcv)
    X1 = X1.drop(columns=[k for k in X1 if k.startswith("cal_")])
    X2 = reversal.indicators(px, ohlcv["SOXL"]).add_prefix("soxl_")
    X3 = cycle.indicators(px, cycle.load_fred_extra(cache / "fred_monthly"))
    X3 = X3[[k for k in X3 if k not in X1]].add_prefix("cyc_")
    X = pd.concat([X1, X2, X3], axis=1).reindex(c.index)
    X["vxn_minus_vix"] = X["vxn"] - X["vix"]
    rsi = X["sox_rsi_14"]
    keep = zone.notna() & rsi.notna()
    return X[keep], zone[keep], rsi[keep]


def auc_table(X: pd.DataFrame, y: pd.Series, min_n: int = 200) -> pd.DataFrame:
    rows = []
    for c in X:
        row = dict(indicator=c)
        for name, (a, b) in {"all": (ERA_A[0], ERA_B[1]), "2010-17": ERA_A, "2018-": ERA_B}.items():
            d = pd.concat([X[c], y], axis=1, keys=["x", "y"]).loc[a:b].dropna()
            ok = len(d) >= min_n and d.y.nunique() == 2 and d.x.nunique() > 1
            row[f"auc_{name}"] = roc_auc_score(d.y, d.x) if ok else np.nan
        rows.append(row)
    T = pd.DataFrame(rows)
    # 向き（大きいほど底圏か、小さいほど底圏か）は全期間で決め、両期間で同じ向きに効いているかを見る
    sign = np.sign(T["auc_all"] - 0.5)
    T["direction"] = np.where(sign > 0, "高いほど底圏", "低いほど底圏")
    for k in ("all", "2010-17", "2018-"):
        T[f"strength_{k}"] = (T[f"auc_{k}"] - 0.5) * sign * 2  # 0 = 無関係、1 = 完全に分離
    T["stable"] = (T["strength_2010-17"] > 0.1) & (T["strength_2018-"] > 0.1)
    T["min_strength"] = T[["strength_2010-17", "strength_2018-"]].min(axis=1)
    return T.sort_values("min_strength", ascending=False)


def addon_rules(X: pd.DataFrame, y: pd.Series, rsi: pd.Series, cands: list[str], base_thr: float = 40) -> pd.DataFrame:
    """RSI<40 に「指標 X が 2010-17 の RSI<40 日の上位/下位 50% 側」を足したときの成績。閾値は 2010-17 のみで決める。"""
    base = rsi < base_thr
    A = X.loc[ERA_A[0]:ERA_A[1]]
    rows = []
    for c in cands:
        d = pd.concat([X[c], y], axis=1, keys=["x", "y"])[base].dropna()
        da = d.loc[ERA_A[0]:ERA_A[1]]
        if len(da) < 50 or da.y.nunique() < 2:
            continue
        hi_better = roc_auc_score(da.y, da.x) >= 0.5
        thr = A.loc[base.loc[ERA_A[0]:ERA_A[1]], c].median()
        sig = (X[c] >= thr) if hi_better else (X[c] <= thr)
        for era, (a, b) in {"2010-17": ERA_A, "2018-": ERA_B}.items():
            m_base = base.loc[a:b]
            m = m_base & sig.loc[a:b].fillna(False)
            yy = y.loc[a:b]
            rows.append(dict(indicator=c, rule=f"{'≥' if hi_better else '≤'} {thr:.3g}", era=era,
                             base_precision=yy[m_base].mean(), base_months=yy[m_base].index.to_period("M").nunique(),
                             precision=yy[m].mean(), months=yy[m].index.to_period("M").nunique(),
                             zone_days_kept=(yy[m] == 1).sum() / max((yy[m_base] == 1).sum(), 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("output/bottom_screen"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    px, ohlcv = load(a.cache)
    X, y, rsi = build(px, ohlcv, a.cache)
    T_all = auc_table(X, y)
    sub = rsi < 40
    T_sub = auc_table(X[sub], y[sub], min_n=80)
    top = T_sub[T_sub.stable].indicator.head(12).tolist()
    R = addon_rules(X, y, rsi, top)
    T_all.to_csv(a.out / "auc_all_days.csv", index=False)
    T_sub.to_csv(a.out / "auc_within_rsi40.csv", index=False)
    R.to_csv(a.out / "addon_rules.csv", index=False)
    cols = ["indicator", "direction", "auc_all", "strength_2010-17", "strength_2018-", "stable"]
    L = ["# 底圏とそれ以外を分ける指標（2010 年〜、実 SOXL）\n",
         f"- 底圏 = SOXL が前後 3 か月の最安値 +20% 以内（全日の {y.mean():.0%}）。RSI(14)<40 の日のうち底圏は {y[sub].mean():.0%}",
         "- strength = |AUC−0.5|×2（0 = 無関係、1 = 完全に分離）。stable = 2010-17 と 2018- の両方で同じ向きに 0.1 超\n",
         f"## 全日（指標数 {len(T_all)}、stable {int(T_all.stable.sum())}）上位 25\n", md_table(T_all.head(25)[cols]), "",
         f"## RSI<40 の日の中で底圏をさらに分ける指標（stable {int(T_sub.stable.sum())}）上位 25\n",
         md_table(T_sub.head(25)[cols]), "",
         "## RSI<40 に上乗せした場合（閾値は 2010-17 の RSI<40 日の中央値、2018- は検証）\n", md_table(R)]
    (a.out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
