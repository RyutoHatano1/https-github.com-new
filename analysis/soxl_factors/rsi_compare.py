"""RSI の種類（対象・計算法・期間）の比較と、「RSI<40 + 1 日急落 + 失業率」ルールの検証。

実行: cd analysis && python -m soxl_factors.rsi_compare --cache data_cache_latest
- 底圏 = SOXL（2010-03 以前は合成）が前後 3 か月の最安値 +20% 以内。
- RSI の比較は、各 RSI の閾値を「点灯日の割合が ^SOX RSI14(Wilder)<40 と同じ」になるよう 2010-17 で合わせて公平に比べる。
- 不確実性は月単位のクラスター・ブートストラップ（同じ月の日々は独立でないため）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import cycle
from .data import load
from .report import md_table
from .scale_in import build_prices

ERA_A, ERA_B, ERA_OLD = ("2010-03-11", "2017-12-31"), ("2018-01-01", "2100-01-01"), ("1996-01-01", "2010-03-10")


def rsi(s: pd.Series, n: int, method: str) -> pd.Series:
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    if method == "wilder":
        a, b = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    elif method == "sma":
        a, b = up.rolling(n).mean(), dn.rolling(n).mean()
    else:  # ema
        a, b = up.ewm(span=n, adjust=False, min_periods=n).mean(), dn.ewm(span=n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + a / b)


def frame(px: pd.DataFrame, cache: Path) -> pd.DataFrame:
    P = build_prices(px)
    c = P["soxl"]
    sox = px["SOX"].reindex(c.index)
    lo = c.rolling(127, center=True).min()
    fred = cycle.load_fred_extra(cache / "fred_monthly")
    D = pd.DataFrame({
        "soxl": c, "sox": sox, "sox_1d": sox.pct_change(), "vix": px["VIX"].reindex(c.index),
        "zone": (c <= lo * 1.2).astype(float).where(lo.notna()),
        "swing_low": (c == lo).astype(float).where(lo.notna()),
        "unrate_rise": cycle._on_calendar(fred["UNRATE"] - fred["UNRATE"].rolling(12).min(), cycle.FRED_EXTRA["UNRATE"], c.index),
    })
    for h, nm in ((21, "r1m"), (63, "r3m"), (252, "r1y")):
        D[nm] = c.shift(-h) / c - 1
    D["dd3m"] = c[::-1].rolling(63, min_periods=1).min()[::-1].shift(-1) / c - 1
    return D.loc[c.index[252]:]


def variants(D: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for src in ("sox", "soxl"):
        for m in ("wilder", "sma", "ema"):
            for n in (7, 9, 14, 21):
                out[f"{src.upper()} RSI{n} ({ {'wilder': 'ワイルダー', 'sma': '単純平均', 'ema': '指数平均'}[m]})"] = rsi(D[src], n, m)
    return out


def stats(D: pd.DataFrame, m: pd.Series) -> dict:
    g = D[m.fillna(False)]
    return dict(日数=len(g), 月数=g.index.to_period("M").nunique(), 底圏=g.zone.mean(),
                最安値の日を捉えた数=int(g.swing_low.sum()), 一か月中央値=g.r1m.median(), 一年中央値=g.r1y.median(),
                一年プラス=(g.r1y.dropna() > 0).mean(), 三か月内最大下落中央値=g.dd3m.median())


def cluster_boot(D: pd.DataFrame, m: pd.Series, col: str, fn, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """月をまとめて復元抽出した 90% 区間。"""
    g = D[m.fillna(False)].dropna(subset=[col])
    if g.empty:
        return np.nan, np.nan
    months = g.index.to_period("M")
    groups = [x[col].to_numpy() for _, x in g.groupby(months)]
    rng = np.random.default_rng(seed)
    vals = [fn(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])) for _ in range(n_boot)]
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def compare_rsi(D: pd.DataFrame) -> pd.DataFrame:
    V = variants(D)
    ref = V["SOX RSI14 (ワイルダー)"]
    A = slice(*ERA_A)
    rate = (ref.loc[A] < 40).mean()
    rows = []
    for name, r in V.items():
        thr = r.loc[A].quantile(rate)  # 2010-17 で点灯頻度を揃える
        row = dict(RSI=name, 揃えた閾値=thr)
        for era, sl in (("2010-17", ERA_A), ("2018-", ERA_B), ("1996-2009", ERA_OLD)):
            d = D.loc[sl[0]:sl[1]]
            rr = r.loc[sl[0]:sl[1]]
            m = rr < thr
            row[f"底圏_{era}"] = d.zone[m].mean()
            ok = d.zone.notna() & rr.notna()
            row[f"AUC_{era}"] = roc_auc_score(d.zone[ok], -rr[ok])
        d = D.loc[ERA_A[0]:ERA_B[1]]
        row["最安値の日を捉えた数_2010-"] = int(d.swing_low[(r.loc[d.index] < thr)].sum())
        row["閾値40での点灯日数_2010-"] = int((r.loc[d.index] < 40).sum())
        rows.append(row)
    T = pd.DataFrame(rows)
    T["平均AUC_2010-"] = T[["AUC_2010-17", "AUC_2018-"]].mean(axis=1)
    return T.sort_values("平均AUC_2010-", ascending=False)


def combo_rules(D: pd.DataFrame) -> pd.DataFrame:
    r = rsi(D.sox, 14, "wilder")
    base = r < 40
    rules = {
        "RSI<40": base,
        "RSI<40 + SOX 1日 -3%以下": base & (D.sox_1d <= -0.03),
        "RSI<40 + SOX 1日 -5%以下": base & (D.sox_1d <= -0.05),
        "RSI<40 + SOX 1日 -7%以下": base & (D.sox_1d <= -0.07),
        "RSI<40 + SOX 1日 -5%以下 + 失業率上昇<0.4": base & (D.sox_1d <= -0.05) & (D.unrate_rise < 0.4),
        "RSI<40 + 失業率上昇<0.4": base & (D.unrate_rise < 0.4),
        "参考: 全日": r.notna(),
    }
    rows = []
    for era, sl in (("2010-", (ERA_A[0], ERA_B[1])), ("2010-17", ERA_A), ("2018-", ERA_B), ("1996-2009(合成)", ERA_OLD)):
        d = D.loc[sl[0]:sl[1]]
        for name, m in rules.items():
            mm = m.loc[d.index]
            row = dict(期間=era, ルール=name, **stats(d, mm))
            if era == "2010-":
                row["底圏_90%区間"] = "{:.0%}〜{:.0%}".format(*cluster_boot(d, mm, "zone", np.mean))
                row["一年プラス_90%区間"] = "{:.0%}〜{:.0%}".format(*cluster_boot(d, mm, "r1y", lambda x: (x > 0).mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("output/rsi_compare"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    px, _ = load(a.cache)
    D = frame(px, a.cache)
    T = compare_rsi(D)
    C = combo_rules(D)
    T.to_csv(a.out / "rsi_variants.csv", index=False)
    C.to_csv(a.out / "combo_rules.csv", index=False)
    L = ["# RSI の種類の比較と組み合わせルールの検証\n",
         "- 底圏 = SOXL（2010-03 以前は合成）が前後 3 か月の最安値 +20% 以内",
         "- RSI の閾値は 2010-17 で点灯頻度を ^SOX RSI14(ワイルダー)<40 と揃えた。AUC は閾値によらない判別力（0.5 = 無関係）\n",
         "## RSI の種類\n", md_table(T), "", "## 組み合わせルール\n", md_table(C)]
    (a.out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
