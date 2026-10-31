"""底・天井と関わる指標の探索（逆張り: 連続下落、RSI、BB 等）。

実行: cd analysis && python -m soxl_factors.reversal --out output/reversal
約定は「シグナル日 t の引けで判定 → t+1 の寄付きで売買」を既定とする（引けで判定してから注文できる現実的な前提）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from . import backtest, features  # noqa: E402
from .data import load  # noqa: E402
from .evaluation import split_date  # noqa: E402
from .report import md_table  # noqa: E402

HORIZONS = (1, 3, 5, 10, 21)

# 指標名 -> 底側（"low": 値が小さいほど底らしい / "high": 大きいほど底らしい）
BOTTOM_SIDE = {
    "ret1": "low", "ret2": "low", "ret5": "low", "ret10": "low",
    "rsi2": "low", "rsi3": "low", "rsi14": "low", "bbz20": "low",
    "dist_sma10": "low", "dist_sma20": "low", "dist_sma50": "low", "drawdown_252": "low",
    "ibs": "low", "breadth_sma50": "low",
    "vix_vs_sma20": "high", "vix_term": "high", "logvol_z63": "high",
}


def _streak(cond: pd.Series) -> pd.Series:
    """cond が連続して True の日数（False の日は 0）。"""
    c = cond.fillna(False).astype(bool)
    return c.groupby((~c).cumsum()).cumsum().astype(float)


def indicators(px: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """bars（対象の OHLCV）の引け時点で観測可能な逆張り系指標。"""
    c, h, lo = bars["Close"], bars["High"], bars["Low"]
    lc = np.log(c)
    d = {f"ret{n}": lc.diff(n) for n in (1, 2, 5, 10)}
    d.update({f"rsi{n}": features._rsi(c, n) for n in (2, 3, 14)})
    d["bbz20"] = features._z(c, 20)
    for n in (10, 20, 50):
        d[f"dist_sma{n}"] = np.log(c / c.rolling(n).mean())
    d["drawdown_252"] = c / c.rolling(252).max() - 1
    rng = (h - lo).where(h > lo)
    d["ibs"] = (c - lo) / rng  # 0 = 安値引け, 1 = 高値引け
    d["down_streak"] = _streak(c.diff() < 0)
    d["up_streak"] = _streak(c.diff() > 0)
    if "VIX" in px:
        vix = px["VIX"].reindex(c.index)
        d["vix_vs_sma20"] = vix / vix.rolling(20).mean() - 1
        if "VIX3M" in px:
            d["vix_term"] = vix / px["VIX3M"].reindex(c.index)
    cons = [k for k in features.CONSTITUENTS if k in px]
    if cons:
        cp = px[cons].reindex(c.index)
        sma = cp.rolling(50).mean()
        above = (cp > sma).astype(float).where(sma.notna())
        d["breadth_sma50"] = above.mean(axis=1).where(above.notna().sum(axis=1) >= 3)
    if "Volume" in bars and bars["Volume"].gt(0).mean() > 0.9:
        d["logvol_z63"] = features._z(np.log(bars["Volume"].replace(0, np.nan)), 63)
    return pd.DataFrame(d).replace([np.inf, -np.inf], np.nan)


def forward_returns(bars: pd.DataFrame, entry: str = "next_open") -> pd.DataFrame:
    """t の引けで判定して entry で買い、t+h の引けで評価した対数リターンと、21 日の MFE/MAE。"""
    o, h, lo, c = bars["Open"], bars["High"], bars["Low"], bars["Close"]
    if entry == "next_open":
        e = o.shift(-1)
    elif entry == "close":
        e = c
    else:
        raise ValueError(entry)
    out = {f"fwd_{k}": np.log(c.shift(-k) / e) for k in HORIZONS}
    # 21 日間の最大含み益・最大含み損（エントリー価格基準）
    out["mfe_21"] = h[::-1].rolling(21).max()[::-1].shift(-1) / e - 1
    out["mae_21"] = lo[::-1].rolling(21).min()[::-1].shift(-1) / e - 1
    return pd.DataFrame(out)


def swing_labels(c: pd.Series, k: int = 10) -> pd.DataFrame:
    """前後 k 日で最安値なら底、最高値なら天井（ラベル作成にのみ未来を使う）。"""
    w = 2 * k + 1
    lo = c.rolling(w, center=True).min()
    hi = c.rolling(w, center=True).max()
    return pd.DataFrame({"bottom": (c == lo).astype(float).where(lo.notna()),
                         "top": (c == hi).astype(float).where(hi.notna())})


def build_signals(ind: pd.DataFrame, split: pd.Timestamp) -> tuple[dict, dict]:
    """（底シグナル, 天井シグナル）。連続日数は固定閾値、連続値は探索期間の分位点で閾値化。"""
    bottom, top = {}, {}
    for n in (2, 3, 4, 5):
        bottom[f"down_streak>={n}"] = ind["down_streak"] >= n
    for n in (3, 4, 5, 6):
        top[f"up_streak>={n}"] = ind["up_streak"] >= n
    bottom["rsi2<10"], bottom["rsi2<5"], bottom["rsi14<30"] = ind["rsi2"] < 10, ind["rsi2"] < 5, ind["rsi14"] < 30
    top["rsi2>90"], top["rsi2>95"], top["rsi14>70"] = ind["rsi2"] > 90, ind["rsi2"] > 95, ind["rsi14"] > 70
    disc = ind[ind.index < split]
    for col, side in BOTTOM_SIDE.items():
        if col not in ind:
            continue
        for q in (0.05, 0.10):
            lo_thr, hi_thr = disc[col].quantile(q), disc[col].quantile(1 - q)
            lo_sig, hi_sig = ind[col] <= lo_thr, ind[col] >= hi_thr
            b, t = (lo_sig, hi_sig) if side == "low" else (hi_sig, lo_sig)
            tag = f"{int(q * 100)}%"
            bottom[f"{col} bottom{tag}"] = b.where(ind[col].notna())
            top[f"{col} top{tag}"] = t.where(ind[col].notna())
    return bottom, top


def event_study(signals: dict, side: str, fwd: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    """シグナル日の将来リターンを全日平均と比較。探索期間で HAC 検定、保留期間で再現性。"""
    rows = []
    for name, sig in signals.items():
        s = sig.astype(float)
        starts = (s == 1) & (s.shift(1) != 1)
        for h in HORIZONS:
            y = fwd[f"fwd_{h}"]
            d = pd.concat([s.rename("s"), y.rename("y"), fwd["mfe_21"], fwd["mae_21"]], axis=1).dropna(subset=["s", "y"])
            row = dict(side=side, signal=name, h=h)
            for per, m in (("disc", d.index < split), ("hold", d.index >= split)):
                dd = d[m]
                on = dd[dd.s == 1]
                row[f"n_{per}"] = len(on)
                row[f"episodes_{per}"] = int(starts.reindex(dd.index).fillna(False).sum())
                row[f"mean_{per}"] = on.y.mean()
                row[f"excess_{per}"] = on.y.mean() - dd.y.mean()
                row[f"hit_{per}"] = (on.y > 0).mean()
                row[f"base_hit_{per}"] = (dd.y > 0).mean()
                if h == 21:
                    row[f"mfe21_{per}"], row[f"mae21_{per}"] = on.mfe_21.mean(), on.mae_21.mean()
                if per == "disc" and len(on) >= 10 and dd.s.nunique() > 1:
                    res = sm.OLS(dd.y, sm.add_constant(dd.s)).fit(cov_type="HAC", cov_kwds={"maxlags": max(5, int(1.5 * h))})
                    row["t_hac"], row["p_hac"] = res.tvalues["s"], res.pvalues["s"]
            rows.append(row)
    return pd.DataFrame(rows)


def finalize_events(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ok = ev["p_hac"].notna()
    ev.loc[ok, "q_bh"] = multipletests(ev.loc[ok, "p_hac"], method="fdr_bh")[1]
    want = np.where(ev.side == "bottom", 1, -1)  # 底シグナルは正の超過、天井シグナルは負の超過が「当たり」
    ev["right_direction_disc"] = np.sign(ev.excess_disc) == want
    ev["robust"] = ((ev.q_bh < 0.05) & ev.right_direction_disc & (np.sign(ev.excess_hold) == want)
                    & (ev.excess_hold.abs() >= 0.5 * ev.excess_disc.abs()) & (ev.n_hold >= 10))
    return ev


def swing_auc(ind: pd.DataFrame, labels: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    rows = []
    cols = list(BOTTOM_SIDE) + ["down_streak", "up_streak"]
    for col in cols:
        if col not in ind:
            continue
        side = BOTTOM_SIDE.get(col, "high" if col == "down_streak" else "low")
        bscore = -ind[col] if side == "low" else ind[col]
        row = dict(indicator=col)
        for lab, score in (("bottom", bscore), ("top", -bscore)):
            for per, m in (("disc", ind.index < split), ("hold", ind.index >= split)):
                d = pd.concat([score.rename("x"), labels[lab].rename("y")], axis=1)[m].dropna()
                row[f"auc_{lab}_{per}"] = roc_auc_score(d.y, d.x) if d.y.nunique() == 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def overnight_split(signals: dict, side: str, bars: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    """シグナル翌日のリターンを 夜間（引け→翌寄付き）と 日中以降（翌寄付き→5 日後引け）に分解した超過リターン。"""
    o, c = bars["Open"], bars["Close"]
    parts = {"overnight": np.log(o.shift(-1) / c), "open_to_5d": np.log(c.shift(-5) / o.shift(-1)),
             "close_to_5d": np.log(c.shift(-5) / c)}
    rows = []
    for name, sig in signals.items():
        row = dict(side=side, signal=name)
        for per, m in (("disc", bars.index < split), ("hold", bars.index >= split)):
            on = (sig == 1) & m
            for k, y in parts.items():
                row[f"{k}_{per}"] = y[on].mean() - y[m].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def bottom_precision(signals: dict, c: pd.Series, split: pd.Timestamp, k: int = 10, tol: int = 2) -> pd.DataFrame:
    """シグナルが出た日が「前後 k 日の安値から ±tol 日以内」だった割合（=的中率）と基準率に対する倍率。"""
    lab = swing_labels(c, k)["bottom"]
    near = lab.rolling(2 * tol + 1, center=True, min_periods=1).max()
    rows = []
    for name, sig in signals.items():
        row = dict(signal=name)
        for per, m in (("disc", c.index < split), ("hold", c.index >= split)):
            d = pd.concat([sig.astype(float).rename("s"), near.rename("y")], axis=1)[m].dropna()
            p = d.y[d.s == 1].mean()
            row[f"precision_{per}"], row[f"base_{per}"], row[f"lift_{per}"] = p, d.y.mean(), p / d.y.mean()
            # 実際の底のうち、±tol 日以内にシグナルが出ていた割合（=捕捉率）
            sw = d.index[(lab.reindex(d.index) == 1).to_numpy()]
            s_near = d.s.rolling(2 * tol + 1, center=True, min_periods=1).max()
            row[f"recall_{per}"] = s_near.reindex(sw).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def simulate(bars: pd.DataFrame, entry: pd.Series, exit_sig: pd.Series, max_hold: int,
             rf: pd.Series, cost_bps: float) -> tuple[pd.Series, pd.DataFrame]:
    """t 引けで判定 → t+1 寄付きで売買。ポジション日は実リターン、ノーポジ日は短期金利。"""
    o, c = bars["Open"].to_numpy(), bars["Close"].to_numpy()
    ent = entry.reindex(bars.index).fillna(False).to_numpy(bool)
    ex = exit_sig.reindex(bars.index).fillna(False).to_numpy(bool)
    rfv = rf.reindex(bars.index).fillna(0.0).to_numpy()
    cost = cost_bps / 1e4
    n = len(c)
    r = np.zeros(n)
    holding, go_in, go_out, days, e_px, e_i = False, False, False, 0, 0.0, 0
    trades = []
    for i in range(1, n):
        if holding and go_out:
            r[i] = o[i] / c[i - 1] - 1 - cost
            trades.append((bars.index[e_i], bars.index[i], o[i] / e_px - 1 - 2 * cost, days))
            holding = go_out = False
        elif holding:
            r[i] = c[i] / c[i - 1] - 1
        elif go_in:
            r[i] = c[i] / o[i] - 1 - cost
            holding, go_in, days, e_px, e_i = True, False, 0, o[i], i
        else:
            r[i] = rfv[i]
        if holding:
            days += 1
            go_out = ex[i] or days >= max_hold
        else:
            go_in = ent[i]
    tr = pd.DataFrame(trades, columns=["entry", "exit", "ret", "days"])
    return pd.Series(r, index=bars.index), tr


def strategy_grid(bars: pd.DataFrame, ind: pd.DataFrame, sox: pd.Series, rf: pd.Series,
                  split: pd.Timestamp, cost_bps: float) -> tuple[pd.DataFrame, dict]:
    c = bars["Close"]
    disc = ind.index < split
    entries = {
        "rsi2<10": ind.rsi2 < 10, "rsi2<5": ind.rsi2 < 5,
        "down>=2": ind.down_streak >= 2, "down>=3": ind.down_streak >= 3, "down>=4": ind.down_streak >= 4,
        "bbz<-2": ind.bbz20 < -2, "ibs<0.15": ind.ibs < 0.15,
        "ret5<=q10": ind.ret5 <= ind.ret5[disc].quantile(0.10),
    }
    sma200 = sox.rolling(200).mean().reindex(c.index)
    filters = {"none": pd.Series(True, index=c.index), "SOX>SMA200": (sox.reindex(c.index) > sma200)}
    exits = {
        "rsi2>70|10d": (ind.rsi2 > 70, 10), "close>sma5|10d": (c > c.rolling(5).mean(), 10),
        "time5d": (pd.Series(False, index=c.index), 5), "time10d": (pd.Series(False, index=c.index), 10),
    }
    rows, runs = [], {}
    bh = c.pct_change()
    runs["buy_hold"] = bh
    for per, m in (("disc", c.index < split), ("hold", c.index >= split)):
        rows.append(dict(rule="buy_hold", entry="-", filter="-", exit="-", period=per,
                         **backtest.metrics(bh[m], rf), exposure=1.0, trades_per_year=0, win_rate=np.nan, avg_trade=np.nan))
    for en, es in entries.items():
        for fn, fs in filters.items():
            for xn, (xs, mh) in exits.items():
                name = f"{en} & {fn} -> {xn}"
                r, tr = simulate(bars, es & fs, xs, mh, rf, cost_bps)
                runs[name] = r
                for per, m in (("disc", c.index < split), ("hold", c.index >= split)):
                    t = tr[(tr.entry >= c.index[m][0]) & (tr.entry <= c.index[m][-1])]
                    yrs = m.sum() / 252
                    inpos = pd.Series(False, index=c.index)
                    for e, x in zip(t.entry, t.exit):
                        inpos[(c.index >= e) & (c.index < x)] = True
                    rows.append(dict(rule=name, entry=en, filter=fn, exit=xn, period=per, **backtest.metrics(r[m], rf),
                                     exposure=inpos[m].mean(), trades_per_year=len(t) / yrs,
                                     win_rate=(t.ret > 0).mean() if len(t) else np.nan,
                                     avg_trade=t.ret.mean() if len(t) else np.nan))
    tab = pd.DataFrame(rows)
    d = tab[(tab.period == "disc") & (tab.rule != "buy_hold")]
    best = d.sort_values("sharpe", ascending=False).head(5).rule.tolist()
    tab["top5_on_disc"] = tab.rule.isin(best)
    return tab, runs


def event_path(c: pd.Series, sig: pd.Series, pre: int = 5, post: int = 21) -> pd.Series:
    """シグナル開始日を 0 とした平均累積対数リターン（-pre〜+post）。"""
    lc = np.log(c)
    starts = np.where((sig == 1) & (sig.shift(1) != 1))[0]
    paths = [lc.iloc[i - pre:i + post + 1].to_numpy() - lc.iloc[i] for i in starts if i - pre >= 0 and i + post < len(lc)]
    return pd.Series(np.nanmean(paths, axis=0), index=range(-pre, post + 1)) if paths else pd.Series(dtype=float)


def run(px, ohlcv, out: Path, target: str = "SOXL", holdout_frac: float = 0.3, cost_bps: float = 5.0,
        strategies: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bars = ohlcv[target].dropna(subset=["Open", "Close"])
    ind = indicators(px, bars)
    start = ind["drawdown_252"].first_valid_index()
    bars, ind = bars.loc[start:], ind.loc[start:]
    split = split_date(bars.index, holdout_frac)
    fwd = forward_returns(bars)
    bottom, top = build_signals(ind, split)
    ev = finalize_events(pd.concat([event_study(bottom, "bottom", fwd, split), event_study(top, "top", fwd, split)]))
    ev.to_csv(out / "events.csv", index=False)
    auc = swing_auc(ind, swing_labels(bars["Close"], 10), split)
    auc.to_csv(out / "swing_auc.csv", index=False)

    # 底シグナルをトレンド（SOX と 200 日線）で条件付け
    sox = px["SOX"]
    up = (sox > sox.rolling(200).mean()).reindex(bars.index)
    cond_rows = []
    for name in ("down_streak>=2", "down_streak>=3", "rsi2<10", "rsi2<5", "ibs bottom10%", "bbz20 bottom5%"):
        if name not in bottom:
            continue
        for trend, tm in (("SOX>SMA200", up == True), ("SOX<SMA200", up == False)):  # noqa: E712
            for per, pm in (("disc", bars.index < split), ("hold", bars.index >= split)):
                m = tm & pm
                on = fwd.loc[m & (bottom[name] == 1), "fwd_5"].dropna()
                base = fwd.loc[m, "fwd_5"].dropna()
                cond_rows.append(dict(signal=name, trend=trend, period=per, n=len(on), mean_fwd5=on.mean(),
                                      excess_fwd5=on.mean() - base.mean(), hit=(on > 0).mean()))
    cond = pd.DataFrame(cond_rows)

    res = dict(target=target, start=bars.index[0], end=bars.index[-1], split=split, events=ev, auc=auc, cond=cond,
               overnight=pd.concat([overnight_split(bottom, "bottom", bars, split), overnight_split(top, "top", bars, split)]),
               precision=bottom_precision(bottom, bars["Close"], split),
               base_mfe=dict(mean_disc=fwd.loc[fwd.index < split, "fwd_21"].mean(), mean_hold=fwd.loc[fwd.index >= split, "fwd_21"].mean(),
                             mfe_disc=fwd.loc[fwd.index < split, "mfe_21"].mean(), mfe_hold=fwd.loc[fwd.index >= split, "mfe_21"].mean(),
                             mae_disc=fwd.loc[fwd.index < split, "mae_21"].mean(), mae_hold=fwd.loc[fwd.index >= split, "mae_21"].mean()),
               base_rate_bottom=float(swing_labels(bars["Close"], 10)["bottom"].mean()))

    # 図: イベント後の平均経路
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, sig in (("down_streak>=3", bottom["down_streak>=3"]), ("rsi2<10", bottom["rsi2<10"]),
                      ("rsi2>90", top["rsi2>90"]), ("up_streak>=5", top["up_streak>=5"])):
        p = event_path(bars["Close"], sig.astype(float))
        ax.plot(p.index, p.values, label=name)
    ax.plot(range(-5, 22), np.log(bars["Close"]).diff().mean() * np.arange(-5, 22), "k--", lw=0.8, label="average day")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("days from signal start (close)")
    ax.set_ylabel(f"mean cum. log return ({target})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_event_paths.png", dpi=130)
    plt.close(fig)

    if strategies:
        rf = (px["IRX"].ffill() / 100 / 252).reindex(bars.index).fillna(0.0)
        tab, runs = strategy_grid(bars, ind, sox, rf, split, cost_bps)
        tab.to_csv(out / "strategies.csv", index=False)
        res["strategies"] = tab
        hold = tab[(tab.period == "hold") & (tab.rule != "buy_hold")]
        bh_sh = tab[(tab.period == "hold") & (tab.rule == "buy_hold")].sharpe.iloc[0]
        res["frac_beat_bh_hold"] = float((hold.sharpe > bh_sh).mean())
        res["n_configs"] = int(hold.shape[0])
        hm = bars.index >= split
        boots = []
        for n in tab[tab.top5_on_disc].rule.unique():
            boots.append(dict(rule=n, **backtest.block_bootstrap_sharpe(runs[n][hm], rf, runs["buy_hold"][hm])))
        res["boot"] = pd.DataFrame(boots)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(hold.sharpe, bins=25, alpha=0.8)
        ax.axvline(bh_sh, color="r", label=f"buy & hold {bh_sh:.2f}")
        ax.set_xlabel("holdout Sharpe of each mean-reversion config")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_strategy_sharpe_hist.png", dpi=130)
        plt.close(fig)
    write_report(res, out, cost_bps)
    return res


def write_report(res: dict, out: Path, cost_bps: float):
    ev = res["events"]
    L = [f"# 底・天井シグナル解析（{res['target']}）\n",
         f"- 期間 {res['start']:%Y-%m-%d}〜{res['end']:%Y-%m-%d}、探索/保留の境界 **{res['split']:%Y-%m-%d}**",
         "- 約定: シグナル日の引けで判定 → 翌営業日の寄付きで買い。リターンは寄付きから h 日後の引けまでの対数リターン",
         "- 閾値の分位点は探索期間のみから計算。`excess` = シグナル日の平均 − 同期間の全日平均\n",
         f"検定数 {ev.p_hac.notna().sum()}、FDR<5% {int((ev.q_bh < 0.05).sum())}、robust {int(ev.robust.sum())}\n"]
    cols = ["signal", "h", "n_disc", "episodes_disc", "excess_disc", "hit_disc", "base_hit_disc", "t_hac", "q_bh",
            "n_hold", "excess_hold", "hit_hold", "base_hit_hold", "robust"]
    for side in ("bottom", "top"):
        L.append(f"## {'底' if side == 'bottom' else '天井'}シグナル（5 日先、p 値順 上位 15）\n")
        sub = ev[(ev.side == side) & (ev.h == 5)].sort_values("p_hac").head(15)[cols]
        L.append(md_table(sub, floatfmt=".3f"))
        L.append("")
    L.append("## robust と判定されたもの（全ホライズン）\n")
    rb = ev[ev.robust][cols]
    L.append(md_table(rb, floatfmt=".3f") if len(rb) else "（なし）")
    L.append("")
    L.append("## 21 日の最大含み益/含み損（主要シグナル）\n")
    k = ev[(ev.h == 21) & ev.signal.isin(["down_streak>=2", "down_streak>=3", "down_streak>=4", "rsi2<10", "rsi2<5",
                                          "rsi14<30", "rsi2>90", "rsi14>70", "up_streak>=5"])]
    L.append(md_table(k[["side", "signal", "n_disc", "mean_disc", "mfe21_disc", "mae21_disc", "n_hold", "mean_hold",
                         "mfe21_hold", "mae21_hold"]]))
    b = res["base_mfe"]
    L.append(f"\n全日平均（基準）: 探索 平均 {b['mean_disc']:.3f} / MFE {b['mfe_disc']:.3f} / MAE {b['mae_disc']:.3f}、"
             f"保留 平均 {b['mean_hold']:.3f} / MFE {b['mfe_hold']:.3f} / MAE {b['mae_hold']:.3f}")
    L.append("\n![paths](fig_event_paths.png)\n")
    L.append("## 夜間ギャップ分解（超過リターン: 引け→翌寄付き / 翌寄付き→5 日後 / 引け→5 日後）\n")
    on = res["overnight"]
    keys = ["down_streak>=2", "down_streak>=3", "down_streak>=4", "rsi2<10", "rsi2<5", "ret5 bottom5%", "ibs bottom10%",
            "rsi2>90", "rsi14>70", "up_streak>=4", "bbz20 top5%"]
    L.append(md_table(on[on.signal.isin(keys)], floatfmt=".4f"))
    L.append("\n## 底の的中率（シグナル日が前後 10 日安値の ±2 日以内だった割合）と捕捉率\n")
    pr = res["precision"].sort_values("lift_disc", ascending=False).head(15)
    L.append(md_table(pr))
    L.append(f"## 実際のスイング底/天井（前後 10 日の極値）の判別力 AUC（0.5 = 無情報、底の基準率 {res['base_rate_bottom']:.3f}）\n")
    L.append(md_table(res["auc"].sort_values("auc_bottom_disc", ascending=False)))
    L.append("\n## トレンド別の底シグナル（5 日先）\n")
    L.append(md_table(res["cond"]))
    if "strategies" in res:
        t = res["strategies"]
        L.append(f"\n## 逆張り売買ルール（{res['n_configs']} 通り、片道 {cost_bps}bp）\n")
        L.append(f"保留期間で買い持ちの Sharpe を上回った設定の割合: **{res['frac_beat_bh_hold']:.0%}**\n")
        show = t[t.top5_on_disc | (t.rule == "buy_hold")][["rule", "period", "CAGR", "vol", "sharpe", "maxDD",
                                                            "exposure", "trades_per_year", "win_rate", "avg_trade"]]
        L.append("探索期間 Sharpe 上位 5 設定と買い持ち\n")
        L.append(md_table(show))
        L.append("\n保留期間の Sharpe 90% 区間と対買い持ち差（ブロック・ブートストラップ）\n")
        L.append(md_table(res["boot"]))
        L.append("\n![hist](fig_strategy_sharpe_hist.png)\n全設定は `strategies.csv`。")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/reversal"))
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--target", default="SOXL", choices=["SOXL", "SOX"])
    ap.add_argument("--cost-bps", type=float, default=5.0)
    a = ap.parse_args()
    px, ohlcv = load(a.cache)
    run(px, ohlcv, a.out / a.target, target=a.target, cost_bps=a.cost_bps, strategies=a.target == "SOXL")
    print(f"レポート: {a.out / a.target / 'report.md'}")


if __name__ == "__main__":
    main()
