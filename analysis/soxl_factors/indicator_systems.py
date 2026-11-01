"""スーパートレンド・MACD・ストキャスティクス・移動平均（SMA/EMA）と、現行の RSI40 買い / RSI75 売りの比較。

実行: cd analysis && python -m soxl_factors.indicator_systems --cache data_cache_latest
- 判断は引け、売買は翌営業日の引け（これまでの検証と同じ）。ノーポジ中は短期金利。手数料は含めない。
- 税引後は日本の 20.315% を売却益ごとに概算（損失繰越なし。売買回数が多いほど不利に出る）。
- 指標の定義は一般的な既定値（スーパートレンド ATR10×3、MACD 12/26/9、ストキャスティクス 14/3）。
- MACD の 0 ライン（MACD>0）は「EMA12 > EMA26」と同じ条件。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest
from .data import load
from .features import _rsi
from .report import md_table
from .scale_in import build_prices

TAX = 0.20315
ERAS = {"1996-2009(合成)": ("1996-01-01", "2010-03-10"), "2010-17": ("2010-03-11", "2017-12-31"),
        "2018-": ("2018-01-01", "2100-01-01"), "2010-通し": ("2010-03-11", "2100-01-01")}


# ---------------------------------------------------------------- 指標
def atr(h, l, c, n):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def supertrend(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 10, mult: float = 3.0) -> pd.Series:
    """上昇トレンドなら True。最終バンドは前日の終値との位置関係で更新する一般的な定義。"""
    a = atr(h, l, c, n)
    hl2 = (h + l) / 2
    bu, bl = (hl2 + mult * a).to_numpy(), (hl2 - mult * a).to_numpy()
    cv = c.to_numpy()
    fu, fl = bu.copy(), bl.copy()
    up = np.zeros(len(c), bool)
    for i in range(1, len(c)):
        if np.isnan(bu[i]) or np.isnan(fu[i - 1]):
            up[i] = cv[i] >= (hl2.iloc[i] if not np.isnan(hl2.iloc[i]) else cv[i])
            continue
        fu[i] = bu[i] if (bu[i] < fu[i - 1] or cv[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (bl[i] > fl[i - 1] or cv[i - 1] < fl[i - 1]) else fl[i - 1]
        if up[i - 1]:
            up[i] = cv[i] >= fl[i]
        else:
            up[i] = cv[i] > fu[i]
    return pd.Series(up, index=c.index).where(a.notna())


def macd(c: pd.Series, fast=12, slow=26, sig=9) -> tuple[pd.Series, pd.Series]:
    m = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    return m, m.ewm(span=sig, adjust=False).mean()


def stoch(h, l, c, n=14, d=3) -> tuple[pd.Series, pd.Series]:
    k = (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min()) * 100
    return k, k.rolling(d).mean()


# ---------------------------------------------------------------- 売買の状態機械
def positions(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """引け時点の「持ちたい状態」。ノーポジで entry なら 1、保有中に exit なら 0。"""
    en, ex = entry.fillna(False).to_numpy(bool), exit_.fillna(False).to_numpy(bool)
    s = np.zeros(len(en))
    hold = False
    for i in range(len(en)):
        if hold and ex[i]:
            hold = False
        elif not hold and en[i]:
            hold = True
        s[i] = 1.0 if hold else 0.0
    return pd.Series(s, index=entry.index)


def run(state: pd.Series, price: pd.Series, rf: pd.Series) -> dict:
    """state_t（引けで決定）→ 翌日引けで約定 → その翌日のリターンから効く。"""
    r = price.pct_change().fillna(0.0)
    w = state.shift(2).fillna(0.0)
    ret = w * r + (1 - w) * rf
    # 取引ごとの損益（約定は決定の翌日の引け）
    exe = state.shift(1).fillna(0.0)
    d = exe.diff().fillna(exe)
    buys, sells = price.index[d > 0], price.index[d < 0]
    trades = []
    for b in buys:
        s = sells[sells > b]
        trades.append((price[b], price[s[0]] if len(s) else np.nan))
    return dict(ret=ret, w=w, trades=trades)


def after_tax(trades, last_px: float) -> float:
    net = 1.0
    for bp, sp in trades:
        if np.isnan(sp):
            g = last_px / bp
            net *= g - (TAX * (g - 1) if g > 1 else 0)  # 最後に売ったとして課税
        else:
            g = sp / bp
            net *= g - (TAX * (g - 1) if g > 1 else 0)
    return net


def systems(D: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """名前 -> (entry, exit)。"""
    S: dict[str, tuple[pd.Series, pd.Series]] = {}
    always = pd.Series(True, index=D.index)
    never = pd.Series(False, index=D.index)
    S["買い持ち"] = (always, never)
    rsi = D.rsi
    S["RSI40買い→RSI75売り（現行）"] = (rsi < 40, rsi >= 75)
    for (n, m) in ((10, 3.0), (10, 2.0), (14, 3.0), (20, 4.0)):
        st = D[f"st_sox_{n}_{m}"]
        S[f"スーパートレンド SOX ({n},{m:g})"] = (st == 1, st == 0)
    st = D["st_soxl_10_3.0"]
    S["スーパートレンド SOXL (10,3)"] = (st == 1, st == 0)
    S["MACD シグナル線クロス (SOX)"] = (D.macd > D.macd_sig, D.macd < D.macd_sig)
    S["MACD 0ライン (SOX)"] = (D.macd > 0, D.macd < 0)
    k, dd = D.stoch_k, D.stoch_d
    S["ストキャス 20以下で買い→80以上で売り"] = ((k < 20) & (k > dd), (k > 80) & (k < dd))
    # 移動平均（SMA = 単純平均、EMA = 指数平均）
    sox = D.sox
    for n in (50, 100, 200):
        S[f"SOX > SMA{n} で保有"] = (sox > D[f"sma{n}"], sox < D[f"sma{n}"])
        S[f"SOX > EMA{n} で保有"] = (sox > D[f"ema{n}"], sox < D[f"ema{n}"])
    S["ゴールデンクロス SMA50 > SMA200"] = (D.sma50 > D.sma200, D.sma50 < D.sma200)
    S["EMA20 > EMA50"] = (D.ema20 > D.ema50, D.ema20 < D.ema50)
    # 組み合わせ: 現行ルールに「下落トレンド入りで売る（損切り）」を足す
    S["RSI40買い→RSI75売り or SMA200割れ"] = (rsi < 40, (rsi >= 75) | (sox < D.sma200))
    st10 = D["st_sox_10_3.0"]
    S["RSI40買い→RSI75売り or スーパートレンド下転換"] = (rsi < 40, (rsi >= 75) | (st10 == 0))
    S["RSI40買い→RSI75売り or MACDデッドクロス"] = (rsi < 40, (rsi >= 75) | (D.macd < D.macd_sig))
    # 組み合わせ: RSI40 で買うのは上昇トレンド中だけ
    S["RSI40買い（スーパートレンド上昇中のみ）→RSI75売り"] = (rsi.lt(40) & (st10 == 1), rsi >= 75)
    return S


def frame(px: pd.DataFrame, ohlcv: dict) -> pd.DataFrame:
    P = build_prices(px)
    sox = ohlcv["SOX"].reindex(P.index)
    D = pd.DataFrame({"soxl": P.soxl, "rf": P.rf})
    D["rsi"] = _rsi(px["SOX"].dropna(), 14).reindex(P.index)
    for (n, m) in ((10, 3.0), (10, 2.0), (14, 3.0), (20, 4.0)):
        D[f"st_sox_{n}_{m}"] = supertrend(sox.High, sox.Low, sox.Close, n, m).astype(float)
    sl = ohlcv["SOXL"].reindex(P.index)
    # SOXL の OHLC は上場後のみ。上場前は ^SOX の判定で代用
    st_l = supertrend(sl.High, sl.Low, sl.Close, 10, 3.0).astype(float)
    D["st_soxl_10_3.0"] = st_l.where(st_l.notna(), D["st_sox_10_3.0"])
    D["sox"] = sox.Close
    for n in (20, 50, 100, 200):
        D[f"sma{n}"] = sox.Close.rolling(n).mean()
        D[f"ema{n}"] = sox.Close.ewm(span=n, adjust=False, min_periods=n).mean()
    D["macd"], D["macd_sig"] = macd(sox.Close)
    D["stoch_k"], D["stoch_d"] = stoch(sox.High, sox.Low, sox.Close)
    return D.loc["1996":]


def evaluate(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, yrs = [], []
    for name, (en, ex) in systems(D).items():
        for era, (a, b) in ERAS.items():
            d = D.loc[a:b]
            st = positions(en.loc[d.index], ex.loc[d.index])
            res = run(st, d.soxl, d.rf)
            m = backtest.metrics(res["ret"], d.rf)
            rows.append(dict(期間=era, 方法=name, 最終倍率=float((1 + res["ret"]).cumprod().iloc[-1]),
                             税引後倍率=after_tax(res["trades"], float(d.soxl.iloc[-1])),
                             年率=m["CAGR"], sharpe=m["sharpe"], 最大下落=m["maxDD"], 投資期間=float(res["w"].mean()),
                             年間売買回数=len(res["trades"]) / m["years"]))
        row = dict(方法=name)
        for y in range(2010, 2025):
            d = D.loc[(f"{y}-01-01" if y > 2010 else "2010-03-11"):]
            res = run(positions(en.loc[d.index], ex.loc[d.index]), d.soxl, d.rf)
            row[str(y)] = after_tax(res["trades"], float(d.soxl.iloc[-1]))
        yrs.append(row)
    return pd.DataFrame(rows), pd.DataFrame(yrs).set_index("方法")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("output/indicator_systems"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    px, ohlcv = load(a.cache)
    D = frame(px, ohlcv)
    R, Y = evaluate(D)
    R.to_csv(a.out / "summary.csv", index=False)
    Y.to_csv(a.out / "after_tax_by_start_year.csv")
    base = "RSI40買い→RSI75売り（現行）"
    beat = (Y.gt(Y.loc[base], axis=1)).sum(axis=1).rename("現行に税引後で勝った開始年の数（15年中）")
    L = ["# スーパートレンド・MACD・ストキャスティクス・移動平均と現行ルールの比較\n",
         "- 判断は引け、売買は翌営業日の引け。待機中は短期金利。税引後は 20.315% を売却益ごとに概算\n"]
    for era in ERAS:
        L += [f"## {era}\n", md_table(R[R.期間 == era].drop(columns="期間")), ""]
    L += ["## 開始年をずらした場合（税引後、終点は直近）\n", md_table(beat.reset_index()), "",
          md_table(Y.T.reset_index().rename(columns={"index": "開始年"}))]
    (a.out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
