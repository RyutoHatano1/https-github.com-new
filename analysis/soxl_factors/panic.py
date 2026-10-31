"""パニック局面（VIX 高 + ^SOX 大幅下落）で買った後に「反発した局面」と「さらに崩れた局面」を分ける指標の探索。

実行: cd analysis && python -m soxl_factors.panic --out output/panic
事例数が少ない（十数局面）ため、ここで見つかる差は仮説生成用。多数の指標から選ぶと偶然きれいに分かれるものが出る点に注意。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import cycle
from .data import load
from .report import md_table
from .scale_in import build_prices

# 指標 -> 説明（局面初日に観測可能な値）
CANDIDATES = {
    "vix": "VIX",
    "vix_term": "VIX/VIX3M（2006年〜）",
    "sox_dd_252": "SOX 1年高値からの下落率",
    "sox_dd_ath": "SOX 過去最高値からの下落率",
    "days_since_high": "1年高値からの経過営業日",
    "sox_ret_21": "SOX 直近1か月リターン（下落の速さ）",
    "spy_dd_252": "S&P500 1年高値からの下落率",
    "sox_minus_spy_dd": "SOX下落率 − S&P500下落率（半導体固有の下げ）",
    "breadth_sma200": "主要7社のうち200日線上の割合",
    "baa10y": "社債スプレッド Baa−10年債",
    "baa10y_chg126": "社債スプレッドの半年変化",
    "nfci": "金融環境指数（高い=逼迫）",
    "dff_chg252": "政策金利の1年変化（マイナス=利下げ済み）",
    "dgs2_chg126": "2年金利の半年変化",
    "curve_10y3m": "長短金利差 10年−3か月",
    "curve_min_504": "過去2年の長短金利差の最小値（逆転の深さ）",
    "sahm": "サーム・ルール（失業率の急上昇）",
    "unrate_vs_12m_min": "失業率 − 過去12か月最低",
    "cfnai_ma3": "シカゴ連銀全米活動指数 3か月平均",
    "icsa_yoy": "新規失業保険申請 前年比",
    "ip_semi_yoy": "半導体生産 前年比",
    "orders_yoy": "電子機器新規受注 前年比",
    "kr_exports_yoy": "韓国輸出 前年比",
    "copper_gold_chg126": "銅/金 半年変化",
}


def episodes(P: pd.DataFrame, vix: pd.Series, vix_min: float = 30, dd_max: float = -0.30, gap: int = 21) -> pd.DataFrame:
    cond = ((vix >= vix_min) & (P.dd <= dd_max)).fillna(False).to_numpy()
    pos = np.where(cond)[0]
    groups: list[list[int]] = []
    for p in pos:
        if not groups or p - groups[-1][-1] > gap:
            groups.append([p])
        else:
            groups[-1].append(p)
    rows = []
    s, x1 = P.soxl, P.onex
    for g in groups:
        i0 = g[0]
        if i0 + 1 + 252 >= len(P):
            continue
        b = i0 + 1  # 翌日の引けで買う
        rows.append(dict(start=P.index[i0], end=P.index[g[-1]], panic_days=len(g),
                         soxl_1y=s.iloc[b + 252] / s.iloc[b] - 1, sox_1y=x1.iloc[b + 252] / x1.iloc[b] - 1,
                         soxl_6m=s.iloc[b + 126] / s.iloc[b] - 1,
                         soxl_maxdd_1y=s.iloc[b:b + 253].min() / s.iloc[b] - 1, i0=i0))
    E = pd.DataFrame(rows)
    E["success"] = E.soxl_1y > 0
    return E


def panel(px: pd.DataFrame, P: pd.DataFrame, fred: dict) -> pd.DataFrame:
    ind = cycle.indicators(px, fred).reindex(P.index)
    sox, spy = px["SOX"].reindex(P.index), px["SPY"].reindex(P.index)
    hi = sox.rolling(252, min_periods=1).max()
    # 1 年高値をつけた日からの経過営業日
    since = pd.Series(np.nan, index=P.index)
    last = 0
    for i, (v, h) in enumerate(zip(sox.to_numpy(), hi.to_numpy())):
        if v >= h:
            last = i
        since.iloc[i] = i - last
    spy_dd = spy / spy.rolling(252, min_periods=1).max() - 1
    t10y3m = px["FRED_T10Y3M"].reindex(P.index) if "FRED_T10Y3M" in px else pd.Series(np.nan, index=P.index)
    X = pd.DataFrame({
        "vix": px["VIX"].reindex(P.index), "vix_term": (px["VIX"] / px["VIX3M"]).reindex(P.index),
        "sox_dd_252": P.dd, "sox_dd_ath": ind.dd_ath, "days_since_high": since,
        "sox_ret_21": np.log(sox / sox.shift(21)), "spy_dd_252": spy_dd, "sox_minus_spy_dd": P.dd - spy_dd,
        "curve_min_504": t10y3m.rolling(504, min_periods=250).min(),
    })
    for c in CANDIDATES:
        if c not in X and c in ind:
            X[c] = ind[c]
    return X[list(CANDIDATES)]


def separation(E: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in X:
        v = X[c].iloc[E.i0.to_numpy()].to_numpy()
        ok = ~np.isnan(v)
        s, f = v[ok & E.success.to_numpy()], v[ok & ~E.success.to_numpy()]
        if len(s) < 2 or len(f) < 2:
            rows.append(dict(indicator=c, desc=CANDIDATES[c], n_success=len(s), n_fail=len(f)))
            continue
        u = stats.mannwhitneyu(s, f, alternative="two-sided")
        auc = u.statistic / (len(s) * len(f))  # 成功局面の値が失敗局面より大きい確率
        rows.append(dict(indicator=c, desc=CANDIDATES[c], n_success=len(s), n_fail=len(f),
                         median_success=np.median(s), median_fail=np.median(f), auc_success_higher=auc,
                         separation=abs(auc - 0.5) * 2, p_mw=u.pvalue))
    T = pd.DataFrame(rows).sort_values("separation", ascending=False)
    # 候補全体で多重比較補正（参考）
    ok = T.p_mw.notna()
    T.loc[ok, "q_bh"] = stats.false_discovery_control(T.loc[ok, "p_mw"].to_numpy())
    return T


def run(px, out: Path, cache: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    P = build_prices(px)
    fred = cycle.load_fred_extra(cache / "fred_monthly")
    X = panel(px, P, fred)
    E = episodes(P, px["VIX"].reindex(P.index))
    vals = X.iloc[E.i0.to_numpy()].reset_index(drop=True)
    table = pd.concat([E.drop(columns="i0"), vals], axis=1)
    table.to_csv(out / "panic_episodes.csv", index=False)
    sep = separation(E, X)
    sep.to_csv(out / "separation.csv", index=False)
    res = dict(E=E, table=table, sep=sep)
    L = ["# パニック局面で「反発」と「さらに崩落」を分ける指標\n",
         "- パニック局面: VIX≥30 かつ ^SOX が 1 年高値から −30% 以下。21 営業日以上空いたら別局面。初日の翌日に SOXL を買う",
         "- 成功 = 1 年後の SOXL がプラス。2010-03 以前の SOXL は合成\n",
         "## 局面一覧\n",
         md_table(table[["start", "end", "panic_days", "soxl_6m", "soxl_1y", "sox_1y", "soxl_maxdd_1y", "success"]]), "",
         "## 成功/失敗の分離度（AUC: 成功局面の値の方が大きい確率。0.5=無関係、0 か 1 に近いほど分かれる）\n",
         md_table(sep), "", "## 局面ごとの指標値\n", md_table(table.drop(columns=["end"]), floatfmt=".2f")]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/panic"))
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    a = ap.parse_args()
    px, _ = load(a.cache)
    run(px, a.out, a.cache)
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
