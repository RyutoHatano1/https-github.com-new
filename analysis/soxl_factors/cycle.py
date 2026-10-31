"""中期（数か月〜1年）の底・天井の探索。テクニカル・半導体/景気サイクル・金利/信用・センチメント。

実行: cd analysis && python -m soxl_factors.cycle --out output/cycle
- サイクルの谷/山は ^SOX のジグザグ（既定 25% 反転）で定義し、谷から +10% 以内を底圏、山から −10% 以内を天井圏とする。
- 月次統計は公表ラグ分ずらしてから日次に展開（最新ヴィンテージを使うため改定の影響は残る）。
- パーセンタイルは各日までのデータのみで計算（拡張窓）。
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from . import backtest  # noqa: E402
from .data import FRED_URL, load, synthetic_leveraged  # noqa: E402
from .report import md_table  # noqa: E402

# FRED 系列 -> 観測日（月初/日付）から利用可能になるまでの暦日数（保守的に設定）
FRED_EXTRA = {
    "IPG3344S": 50,          # 鉱工業生産: 半導体・電子部品（翌月中旬公表）
    "A34SNO": 65,            # 製造業新規受注: コンピュータ・電子製品
    "CFNAI": 60,             # シカゴ連銀全米活動指数
    "UNRATE": 40,            # 失業率（翌月第 1 金曜）
    "SAHMREALTIME": 40,      # サーム・ルール（リアルタイム）
    "UMCSENT": 35,           # ミシガン消費者信頼感（確報）
    "XTEXVA01KRM667S": 60,   # 韓国 輸出額（OECD 経由）
    "BAA10Y": 1, "DFF": 1, "DGS2": 1,
}

HORIZONS = (63, 126, 252)

# 指標 -> 底側（"low": 小さいほど底らしい / "high": 大きいほど底らしい）
BOTTOM_SIDE = {
    "dd_ath": "low", "dd_252": "low", "dist_sma200": "low", "dist_sma1260": "low",
    "ret_126": "low", "ret_252": "low", "sox_vs_spy_252": "low", "breadth_sma200": "low",
    "rv63": "high", "vix": "high", "vix_max63": "high",
    "ip_semi_yoy": "low", "ip_semi_accel": "high", "orders_yoy": "low", "orders_accel": "high",
    "kr_exports_yoy": "low", "kr_exports_accel": "high", "cfnai_ma3": "low",
    "unrate_vs_12m_min": "high", "sahm": "high", "umcsent": "low", "umcsent_chg12m": "low",
    "curve_10y3m": "high", "curve_chg126": "high", "dgs2_chg126": "low", "dff_chg252": "low",
    "baa10y": "high", "baa10y_chg126": "high", "nfci": "high", "icsa_yoy": "high",
    "copper_gold_chg126": "low", "dxy_chg126": "high",
}

GROUP = {
    "price": ["dd_ath", "dd_252", "dist_sma200", "dist_sma1260", "ret_126", "ret_252", "sox_vs_spy_252", "breadth_sma200"],
    "vol_sentiment": ["rv63", "vix", "vix_max63", "umcsent", "umcsent_chg12m"],
    "semi_cycle": ["ip_semi_yoy", "ip_semi_accel", "orders_yoy", "orders_accel", "kr_exports_yoy", "kr_exports_accel"],
    "macro": ["cfnai_ma3", "unrate_vs_12m_min", "sahm", "icsa_yoy"],
    "rates_credit": ["curve_10y3m", "curve_chg126", "dgs2_chg126", "dff_chg252", "baa10y", "baa10y_chg126", "nfci"],
    "cross_asset": ["copper_gold_chg126", "dxy_chg126"],
}


def load_fred_extra(cache: Path, refresh: bool = False) -> dict[str, pd.Series]:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for sid in FRED_EXTRA:
        f = cache / f"{sid}.csv"
        if refresh or not f.exists():
            f.write_text(requests.get(FRED_URL.format(sid=sid), timeout=60).text)
        df = pd.read_csv(f)
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0])
        out[sid] = s.dropna()
    return out


def _on_calendar(s: pd.Series, lag_days: int, cal: pd.DatetimeIndex) -> pd.Series:
    """観測日 + lag_days 以降に利用可能として営業日カレンダーへ前方補完。"""
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s.reindex(s.index.union(cal)).ffill().reindex(cal)


def _yoy(s: pd.Series) -> pd.Series:
    """月次系列の前年比（対数）。"""
    return np.log(s / s.shift(12))


def indicators(px: pd.DataFrame, fred: dict[str, pd.Series]) -> pd.DataFrame:
    sox = px["SOX"].dropna()
    cal = sox.index
    d = {}
    d["dd_ath"] = sox / sox.cummax() - 1
    d["dd_252"] = sox / sox.rolling(252).max() - 1
    d["dist_sma200"] = np.log(sox / sox.rolling(200).mean())
    d["dist_sma1260"] = np.log(sox / sox.rolling(1260).mean())
    d["ret_126"], d["ret_252"] = np.log(sox / sox.shift(126)), np.log(sox / sox.shift(252))
    d["sox_vs_spy_252"] = d["ret_252"] - np.log(px["SPY"] / px["SPY"].shift(252)).reindex(cal)
    cons = [c for c in ("NVDA", "TSM", "ASML", "AVGO", "AMD", "MU", "AMAT") if c in px]
    cp = px[cons].reindex(cal)
    sma = cp.rolling(200).mean()
    above = (cp > sma).astype(float).where(sma.notna())
    d["breadth_sma200"] = above.mean(axis=1).where(above.notna().sum(axis=1) >= 3)
    d["rv63"] = np.log(sox).diff().rolling(63).std() * np.sqrt(252)
    vix = px["VIX"].reindex(cal)
    d["vix"], d["vix_max63"] = vix, vix.rolling(63).max()

    m = {sid: s for sid, s in fred.items()}
    lag = FRED_EXTRA
    for sid, nm in (("IPG3344S", "ip_semi"), ("A34SNO", "orders"), ("XTEXVA01KRM667S", "kr_exports")):
        if sid in m:
            y = _yoy(m[sid])
            d[f"{nm}_yoy"] = _on_calendar(y, lag[sid], cal)
            d[f"{nm}_accel"] = _on_calendar(y - y.shift(6), lag[sid], cal)  # 前年比の 6 か月変化（底打ち/ピークアウト）
    if "CFNAI" in m:
        d["cfnai_ma3"] = _on_calendar(m["CFNAI"].rolling(3).mean(), lag["CFNAI"], cal)
    if "UNRATE" in m:
        u = m["UNRATE"]
        d["unrate_vs_12m_min"] = _on_calendar(u - u.rolling(12).min(), lag["UNRATE"], cal)
    if "SAHMREALTIME" in m:
        d["sahm"] = _on_calendar(m["SAHMREALTIME"], lag["SAHMREALTIME"], cal)
    if "UMCSENT" in m:
        d["umcsent"] = _on_calendar(m["UMCSENT"], lag["UMCSENT"], cal)
        d["umcsent_chg12m"] = _on_calendar(m["UMCSENT"].diff(12), lag["UMCSENT"], cal)

    def daily(sid):
        return _on_calendar(m[sid], lag[sid], cal) if sid in m else None

    t10y3m = px.get("FRED_T10Y3M")
    if t10y3m is not None:
        t10y3m = t10y3m.reindex(cal)
        d["curve_10y3m"], d["curve_chg126"] = t10y3m, t10y3m.diff(126)
    if (s := daily("DGS2")) is not None:
        d["dgs2_chg126"] = s.diff(126)
    if (s := daily("DFF")) is not None:
        d["dff_chg252"] = s.diff(252)
    if (s := daily("BAA10Y")) is not None:
        d["baa10y"], d["baa10y_chg126"] = s, s.diff(126)
    if "FRED_NFCI" in px:
        d["nfci"] = px["FRED_NFCI"].reindex(cal)
    if "FRED_ICSA" in px:
        d["icsa_yoy"] = np.log(px["FRED_ICSA"] / px["FRED_ICSA"].shift(252)).reindex(cal)
    cg = np.log(px["COPPER"] / px["GOLD"]).reindex(cal)
    d["copper_gold_chg126"] = cg.diff(126)
    d["dxy_chg126"] = np.log(px["DXY"] / px["DXY"].shift(126)).reindex(cal)
    return pd.DataFrame(d).replace([np.inf, -np.inf], np.nan)


def oriented_percentile(ind: pd.DataFrame, min_periods: int = 756) -> pd.DataFrame:
    """拡張窓パーセンタイル（その日までのデータのみ）を「1 に近いほど底らしい」向きに揃える。"""
    out = {}
    for c in ind:
        p = ind[c].expanding(min_periods=min_periods).rank(pct=True)
        out[c] = p if BOTTOM_SIDE[c] == "high" else 1 - p
    return pd.DataFrame(out)


def zigzag(c: pd.Series, thr: float = 0.25) -> pd.DataFrame:
    """thr 以上の反転で確定する山/谷。最後の未確定の極値は含めない。"""
    v, idx = c.to_numpy(), c.index
    pivots, trend = [], 0
    hi_i = lo_i = 0
    for i in range(1, len(v)):
        if v[i] > v[hi_i]:
            hi_i = i
        if v[i] < v[lo_i]:
            lo_i = i
        if trend >= 0 and v[i] <= v[hi_i] * (1 - thr):
            if trend == 1 or not pivots:
                pivots.append((idx[hi_i], v[hi_i], "peak"))
            trend, lo_i = -1, i
        elif trend <= 0 and v[i] >= v[lo_i] * (1 + thr):
            if trend == -1 or not pivots:
                pivots.append((idx[lo_i], v[lo_i], "trough"))
            trend, hi_i = 1, i
    return pd.DataFrame(pivots, columns=["date", "price", "kind"])


def zones(c: pd.Series, piv: pd.DataFrame, band: float = 0.10) -> pd.DataFrame:
    """各谷（山）の前後の区間（隣の山（谷）まで）で、谷から +band 以内（山から −band 以内）を底圏（天井圏）とする。"""
    bottom = pd.Series(0.0, index=c.index)
    top = pd.Series(0.0, index=c.index)
    dates = list(piv.date)
    for k, row in piv.iterrows():
        lo = dates[k - 1] if k > 0 else c.index[0]
        hi = dates[k + 1] if k + 1 < len(dates) else c.index[-1]
        seg = c[(c.index > lo) & (c.index < hi)]
        if row.kind == "trough":
            bottom[seg.index[seg <= row.price * (1 + band)]] = 1.0
        else:
            top[seg.index[seg >= row.price * (1 - band)]] = 1.0
    # 最後のピボット以降は未確定なので評価から除外
    last = dates[-1] if dates else c.index[0]
    bottom[c.index > last] = np.nan
    top[c.index > last] = np.nan
    return pd.DataFrame({"bottom_zone": bottom, "top_zone": top})


def ic_table(P: pd.DataFrame, fwd: pd.DataFrame, split) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        y = fwd[f"fwd_{h}"]
        for c in P:
            d = pd.concat([P[c].rename("x"), y.rename("y")], axis=1).dropna()
            dd, dh = d[d.index < split], d[d.index >= split]
            if len(dd) < 500:
                continue
            xr, yr = (dd.x.rank() - dd.x.rank().mean()) / dd.x.rank().std(), (dd.y.rank() - dd.y.rank().mean()) / dd.y.rank().std()
            res = sm.OLS(yr.to_numpy(), sm.add_constant(xr.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": int(1.5 * h)})
            rows.append(dict(indicator=c, h=h, ic_disc=dd.x.corr(dd.y, method="spearman"), t_hac=res.tvalues[1],
                             p_hac=res.pvalues[1], ic_hold=dh.x.corr(dh.y, method="spearman") if len(dh) > 100 else np.nan,
                             n_disc_years=len(dd) / 252, n_hold_years=len(dh) / 252))
    t = pd.DataFrame(rows)
    t["q_bh"] = multipletests(t.p_hac, method="fdr_bh")[1]
    # 向きを揃えたパーセンタイルなので、底指標として正しければ IC > 0
    t["robust"] = (t.q_bh < 0.05) & (t.ic_disc > 0) & (t.ic_hold > 0.5 * t.ic_disc)
    return t.sort_values(["h", "p_hac"])


def zone_skill(P: pd.DataFrame, Z: pd.DataFrame, split, q: float = 0.15) -> pd.DataFrame:
    """パーセンタイル >= 1-q を底シグナル、<= q を天井シグナルとして、底圏/天井圏の的中率・捕捉率。"""
    rows = []
    for c in P:
        row = dict(indicator=c)
        for side, sig, zone in (("bottom", P[c] >= 1 - q, "bottom_zone"), ("top", P[c] <= q, "top_zone")):
            for per, m in (("disc", P.index < split), ("hold", P.index >= split)):
                d = pd.concat([sig.where(P[c].notna()).rename("s"), Z[zone].rename("z")], axis=1)[m].dropna()
                d = d.astype(float)
                prec = d.z[d.s == 1].mean()
                row[f"{side}_prec_{per}"] = prec
                row[f"{side}_lift_{per}"] = prec / d.z.mean() if d.z.mean() > 0 else np.nan
                row[f"{side}_recall_{per}"] = d.s[d.z == 1].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_snapshot(P: pd.DataFrame, piv: pd.DataFrame) -> pd.DataFrame:
    """各山/谷の日の向き揃えパーセンタイル（1 = 底らしさ最大）。"""
    rows = []
    for _, r in piv.iterrows():
        if r.date in P.index:
            rows.append(dict(date=r.date, kind=r.kind, **P.loc[r.date].round(2).to_dict()))
    return pd.DataFrame(rows)


def drawdown_entries(px: pd.DataFrame, start, thresholds=(-0.2, -0.3, -0.4, -0.5), vix_min=None) -> pd.DataFrame:
    """SOX が 52 週高値から thr 以上下落した最初の日（エピソード化: 高値更新までは 1 回のみ）に買った場合の将来リターン。

    SOXL は上場前を合成 3 倍で補った系列。各エピソードを列挙して最悪値も示す。
    """
    sox = px["SOX"].dropna()
    lev = synthetic_leveraged(px["SOX"], px["IRX"]).reindex(sox.index)
    actual = px["SOXL"].reindex(sox.index)
    first = actual.first_valid_index()
    soxl = pd.concat([lev[lev.index < first] * actual[first] / lev[first], actual[actual.index >= first]])
    dd = sox / sox.rolling(252).max() - 1
    vix = px["VIX"].reindex(sox.index)
    rows = []
    for thr in thresholds:
        armed = True
        for i, t in enumerate(sox.index):
            if t < start:
                continue
            if dd.iloc[i] > -0.05:
                armed = True  # 高値圏に戻ったら次のエピソードを許可
            if armed and dd.iloc[i] <= thr and (vix_min is None or vix.iloc[i] >= vix_min):
                armed = False
                r = dict(threshold=thr, date=t, sox_dd=dd.iloc[i], vix=vix.iloc[i])
                for h in (63, 126, 252):
                    j = i + 1 + h
                    if j < len(sox):
                        # 翌日の引けで買い、h 日後の引けで評価
                        r[f"sox_{h}"] = sox.iloc[j] / sox.iloc[i + 1] - 1
                        r[f"soxl_{h}"] = soxl.iloc[j] / soxl.iloc[i + 1] - 1
                # 買ってから 252 日以内の SOXL の最大下落
                w = soxl.iloc[i + 1:i + 253]
                r["soxl_max_loss_252"] = w.min() / w.iloc[0] - 1 if len(w) else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def run(px, out: Path, cache: Path, holdout_frac: float = 0.3, zz: float = 0.25, cost_bps: float = 5.0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fred = load_fred_extra(cache / "fred_monthly")
    ind = indicators(px, fred)
    P = oriented_percentile(ind)
    sox = px["SOX"].dropna()
    start = P.dropna(how="all").index[0]
    P, sox = P.loc[start:], sox.loc[start:]
    split = sox.index[int(len(sox) * (1 - holdout_frac))]

    fwd = pd.DataFrame({f"fwd_{h}": np.log(sox.shift(-h) / sox) for h in HORIZONS})
    piv = zigzag(px["SOX"].dropna(), zz)
    piv = piv[piv.date >= start].reset_index(drop=True)
    Z = zones(sox, piv)

    ic = ic_table(P, fwd, split)
    ic.to_csv(out / "ic.csv", index=False)
    skill = zone_skill(P, Z, split)
    skill.to_csv(out / "zone_skill.csv", index=False)
    snap = pivot_snapshot(P, piv)
    snap.to_csv(out / "pivot_snapshot.csv", index=False)

    # 複合スコア: 探索期間で (a) fwd_126 の IC が FDR<10% で正、かつ (b) 底圏の lift>1.5 の指標を選び、
    # 向き揃えパーセンタイルを単純平均（重みの推定はしない）
    ic126 = ic[ic.h == 126].set_index("indicator")
    sel = [c for c in P if c in ic126.index and ic126.loc[c, "q_bh"] < 0.10 and ic126.loc[c, "ic_disc"] > 0
           and skill.set_index("indicator").loc[c, "bottom_lift_disc"] > 1.5]
    comp = P[sel].mean(axis=1, skipna=True).where(P[sel].notna().sum(axis=1) >= max(2, len(sel) // 2)) if sel else None

    res = dict(start=start, end=sox.index[-1], split=split, piv=piv, ic=ic, skill=skill, snap=snap, sel=sel, zz=zz,
               base_bottom=float(Z.bottom_zone.mean()), base_top=float(Z.top_zone.mean()))

    if comp is not None:
        comp = comp.rename("composite")
        cic = []
        for h in HORIZONS:
            d = pd.concat([comp, fwd[f"fwd_{h}"]], axis=1).dropna()
            for per, m in (("disc", d.index < split), ("hold", d.index >= split)):
                dd = d[m]
                cic.append(dict(h=h, period=per, ic=dd.iloc[:, 0].corr(dd.iloc[:, 1], method="spearman")))
        res["comp_ic"] = pd.DataFrame(cic)
        bins = pd.cut(comp, [0, 0.3, 0.5, 0.7, 1.0], labels=["<0.3", "0.3-0.5", "0.5-0.7", ">0.7"])
        qt = []
        for per, m in (("disc", sox.index < split), ("hold", sox.index >= split)):
            d = pd.concat([bins.rename("bin"), fwd["fwd_126"], Z.bottom_zone, Z.top_zone], axis=1)[m].dropna(subset=["bin", "fwd_126"])
            for b, g in d.groupby("bin", observed=True):
                qt.append(dict(period=per, composite=str(b), n_days=len(g), mean_fwd126_sox=g.fwd_126.mean(),
                               hit=(g.fwd_126 > 0).mean(), p_bottom_zone=g.bottom_zone.mean(), p_top_zone=g.top_zone.mean()))
        res["comp_bins"] = pd.DataFrame(qt)
        comp_skill = zone_skill(comp.to_frame(), Z, split)
        res["comp_skill"] = comp_skill

        # 中期戦略（月次リバランス、翌日約定）: SOXL（上場前は合成）を composite に応じて保有
        rf_annual = px["IRX"].reindex(sox.index)
        lev = synthetic_leveraged(px["SOX"], px["IRX"]).reindex(sox.index)
        actual = px["SOXL"].reindex(sox.index)
        first = actual.first_valid_index()
        price = pd.concat([lev[lev.index < first] * actual[first] / lev[first], actual[actual.index >= first]])
        ret = price.pct_change()
        rf = (rf_annual.ffill() / 100 / 252).fillna(0.0)
        rules = {
            "buy_hold": pd.Series(1.0, index=sox.index),
            "comp>0.5": (comp > 0.5).astype(float),
            "comp>0.4": (comp > 0.4).astype(float),
            "scaled(comp)": comp.clip(0, 1),
            "scaled(comp)*voltarget60": comp.clip(0, 1) * (0.6 / (3 * ind["rv63"].reindex(sox.index))).clip(0, 1),
            "voltarget60": (0.6 / (3 * ind["rv63"].reindex(sox.index))).clip(0, 1),
        }
        rows, runs = [], {}
        for n, w in rules.items():
            bt = backtest.run(w.reindex(sox.index).fillna(0.0), ret, rf, freq="M", exec_lag=1, cost_bps=cost_bps)
            runs[n] = bt
            for per, m in (("disc", bt.index < split), ("hold", bt.index >= split)):
                mm = m & (bt.index >= comp.first_valid_index())
                rows.append(dict(rule=n, period=per, **backtest.metrics(bt.ret[mm], rf, bt.w, bt.turnover)))
        res["strat"] = pd.DataFrame(rows)
        hm = sox.index >= split
        res["boot"] = pd.DataFrame([dict(rule=n, **backtest.block_bootstrap_sharpe(runs[n].ret[hm], rf, runs["buy_hold"].ret[hm], block=63))
                                    for n in rules if n != "buy_hold"])

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        axes[0].semilogy(sox.index, sox, lw=0.8, color="k")
        for _, r in piv.iterrows():
            axes[0].plot(r.date, r.price, "v" if r.kind == "peak" else "^", color="r" if r.kind == "peak" else "g")
        axes[0].fill_between(sox.index, sox.min(), sox.max(), where=Z.bottom_zone == 1, color="g", alpha=0.15, step="mid")
        axes[0].fill_between(sox.index, sox.min(), sox.max(), where=Z.top_zone == 1, color="r", alpha=0.12, step="mid")
        axes[0].set_ylabel("^SOX (log)")
        axes[1].plot(comp.index, comp, lw=0.8)
        axes[1].axhline(0.5, color="k", lw=0.5)
        for ax in axes:
            ax.axvline(split, color="k", ls="--", lw=0.8)
        axes[1].set_ylabel("bottom score")
        fig.tight_layout()
        fig.savefig(out / "fig_cycle.png", dpi=130)
        plt.close(fig)

    res["dd_entries"] = drawdown_entries(px, start)
    res["dd_entries"].to_csv(out / "drawdown_entries.csv", index=False)
    lev_all = synthetic_leveraged(px["SOX"], px["IRX"]).reindex(sox.index)
    base = {}
    for h in (63, 126, 252):
        base[f"sox_{h}"] = (sox.shift(-h) / sox - 1).mean()
        base[f"soxl_{h}"] = (lev_all.shift(-h) / lev_all - 1).mean()
    res["dd_base"] = base
    write_report(res, out)
    return res


def write_report(res: dict, out: Path):
    L = [f"# 中期の底・天井（^SOX サイクル {int(res['zz'] * 100)}% ジグザグ）\n",
         f"- 期間 {res['start']:%Y-%m-%d}〜{res['end']:%Y-%m-%d}、探索/保留の境界 **{res['split']:%Y-%m-%d}**",
         f"- 底圏の日の割合 {res['base_bottom']:.1%}、天井圏 {res['base_top']:.1%}（谷 +10% 以内 / 山 −10% 以内）",
         "- 指標は「1 に近いほど底らしい」向きの拡張窓パーセンタイル（天井は 0 に近いほど）\n",
         "## サイクルの山と谷\n", md_table(res["piv"], floatfmt=".1f"), ""]
    ic = res["ic"]
    L += ["## 将来リターンとの順位相関（向き揃え済み: 正 = 底指標として機能）\n",
          f"検定数 {len(ic)}、FDR<5% {int((ic.q_bh < 0.05).sum())}、robust {int(ic.robust.sum())}\n"]
    for h in HORIZONS:
        L += [f"### {h} 営業日先\n", md_table(ic[ic.h == h].head(12)[["indicator", "ic_disc", "t_hac", "q_bh", "ic_hold", "robust"]]), ""]
    sk = res["skill"].sort_values("bottom_lift_disc", ascending=False)
    L += ["## 底圏/天井圏の的中率（上位 15% パーセンタイルで点灯）\n",
          "prec = 点灯日のうち実際に底圏（天井圏）だった割合、lift = prec / 基準率、recall = 底圏の日のうち点灯していた割合\n",
          md_table(sk[["indicator", "bottom_prec_disc", "bottom_lift_disc", "bottom_recall_disc", "bottom_prec_hold",
                       "bottom_lift_hold", "bottom_recall_hold"]]), ""]
    sk = res["skill"].sort_values("top_lift_disc", ascending=False)
    L += [md_table(sk[["indicator", "top_prec_disc", "top_lift_disc", "top_recall_disc", "top_prec_hold", "top_lift_hold",
                       "top_recall_hold"]]), ""]
    L += ["## 各山/谷での指標（向き揃えパーセンタイル: 谷で 1 に近く、山で 0 に近いほど良い指標）\n",
          md_table(res["snap"], floatfmt=".2f"), ""]
    de = res["dd_entries"]
    if len(de):
        L += ["## 52 週高値からの下落で買った場合（エピソード単位、翌日引けで買い）\n",
              "全日平均（基準）: " + ", ".join(f"{k} {v:+.1%}" for k, v in res["dd_base"].items()) + "\n"]
        agg = de.groupby("threshold").agg(n=("date", "size"), sox_126_med=("sox_126", "median"), soxl_126_med=("soxl_126", "median"),
                                          soxl_126_min=("soxl_126", "min"), soxl_252_med=("soxl_252", "median"),
                                          soxl_252_win=("soxl_252", lambda s: (s > 0).mean()), soxl_252_min=("soxl_252", "min"),
                                          worst_drawdown=("soxl_max_loss_252", "min")).reset_index()
        L += [md_table(agg), "", md_table(de, floatfmt=".2f"), ""]
    if res["sel"]:
        L += [f"## 複合ボトムスコア（探索期間で選定: {', '.join(res['sel'])}）\n", md_table(res["comp_ic"]), "",
              md_table(res["comp_bins"]), "", md_table(res["comp_skill"]), "",
              "### 中期戦略（月次リバランス、翌日約定、SOXL・上場前は合成）\n", md_table(res["strat"]), "",
              "保留期間の Sharpe 90% 区間と対買い持ち差（ブロック 63 日）\n", md_table(res["boot"]), "",
              "![cycle](fig_cycle.png)"]
    else:
        L.append("探索期間の基準を満たす指標がなく、複合スコアは作成していない。")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/cycle"))
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--zigzag", type=float, default=0.25)
    a = ap.parse_args()
    px, _ = load(a.cache)
    run(px, a.out, a.cache, zz=a.zigzag)
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
