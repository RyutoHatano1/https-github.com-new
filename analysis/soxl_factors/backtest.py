"""低頻度ルールのバックテスト。パラメータは探索期間で選び、保留期間の成績を別に報告する。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    """各期間（W: 週, M: 月, D: 毎日）の最終営業日に True。"""
    if freq == "D":
        return pd.Series(True, index=idx)
    key = idx.to_period(freq)
    s = pd.Series(np.arange(len(idx)), index=idx)
    return s.groupby(key).transform("max") == s


def run(weight: pd.Series, ret: pd.Series, rf: pd.Series, freq: str = "W",
        exec_lag: int = 1, cost_bps: float = 5.0) -> pd.DataFrame:
    """weight_t は t の引けで決まる目標比率。リバランス日だけ更新し、exec_lag 日遅れで約定。

    exec_lag=0: t の引けで約定（MOC 前提、やや楽観）。exec_lag=1: 翌日引けで約定（保守的）。
    """
    w = weight.where(rebalance_mask(weight.index, freq)).ffill().fillna(0.0).clip(0, 1)
    held = w.shift(1 + exec_lag).fillna(0.0)  # held_t が r_t（t-1→t）に掛かる
    turnover = held.diff().abs().fillna(held.abs())
    r = held * ret.fillna(0.0) + (1 - held) * rf.fillna(0.0) - turnover * cost_bps / 1e4
    return pd.DataFrame({"ret": r, "w": held, "turnover": turnover})


def metrics(r: pd.Series, rf: pd.Series, w: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    ex = r - rf.reindex(r.index).fillna(0.0)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    dd = eq / eq.cummax() - 1
    down = ex[ex < 0].std() * np.sqrt(252)
    out = dict(
        CAGR=eq.iloc[-1] ** (1 / yrs) - 1,
        vol=r.std() * np.sqrt(252),
        sharpe=ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
        sortino=ex.mean() * 252 / down if down > 0 else np.nan,
        maxDD=dd.min(),
        calmar=(eq.iloc[-1] ** (1 / yrs) - 1) / abs(dd.min()) if dd.min() < 0 else np.nan,
        years=yrs,
    )
    if w is not None:
        out["exposure"] = float(w.reindex(r.index).mean())
    if turnover is not None:
        out["trades_per_year"] = float((turnover.reindex(r.index) > 1e-9).sum() / yrs)
    return out


def block_bootstrap_sharpe(r: pd.Series, rf: pd.Series, bench: pd.Series | None = None,
                           block: int = 21, n_boot: int = 2000, seed: int = 0) -> dict:
    """定常ブロック・ブートストラップで Sharpe（と対ベンチ差）の 90% 区間。"""
    ex = (r - rf.reindex(r.index).fillna(0.0)).dropna()
    exb = (bench - rf.reindex(bench.index).fillna(0.0)).reindex(ex.index).to_numpy() if bench is not None else None
    a = ex.to_numpy()
    n = len(a)
    rng = np.random.default_rng(seed)
    sh, diff = [], []
    for _ in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = 0
        while i < n:
            s = rng.integers(n)
            L = min(rng.geometric(1 / block), n - i)
            idx[i:i + L] = (s + np.arange(L)) % n
            i += L
        x = a[idx]
        s1 = x.mean() / x.std() * np.sqrt(252)
        sh.append(s1)
        if exb is not None:
            xb = exb[idx]
            diff.append(s1 - xb.mean() / xb.std() * np.sqrt(252))
    out = dict(sharpe_lo=np.percentile(sh, 5), sharpe_hi=np.percentile(sh, 95))
    if diff:
        out.update(diff_lo=np.percentile(diff, 5), diff_hi=np.percentile(diff, 95),
                   p_diff_le0=float(np.mean(np.array(diff) <= 0)))
    return out


def candidate_rules(X: pd.DataFrame, sox: pd.Series, split: pd.Timestamp) -> dict[str, tuple[str, pd.Series]]:
    """（ファミリー名, 目標比率）の辞書。閾値の分位点は探索期間のみから計算する。"""
    rules: dict[str, tuple[str, pd.Series]] = {"buy_hold": ("buy_hold", pd.Series(1.0, index=X.index))}
    rv = X["sox_rv21"]
    rv_disc = rv[rv.index < split]
    for n in (50, 100, 150, 200):
        sma = sox.rolling(n).mean()
        rules[f"trend_sma{n}"] = ("trend", (sox > sma).astype(float).where(sma.notna(), 0.0))
    for q in (0.5, 0.7, 0.8, 0.9):
        thr = rv_disc.quantile(q)
        rules[f"volfilter_q{int(q * 100)}"] = ("volfilter", (rv < thr).astype(float).where(rv.notna(), 0.0))
    for n in (100, 200):
        sma = sox.rolling(n).mean()
        for q in (0.7, 0.9):
            thr = rv_disc.quantile(q)
            rules[f"trend{n}_vol{int(q * 100)}"] = ("trend_vol", ((sox > sma) & (rv < thr)).astype(float).where(sma.notna(), 0.0))
    for tv in (0.4, 0.6, 0.8):
        # SOXL のボラ ≈ 3 × SOX のボラ
        rules[f"voltarget_{int(tv * 100)}"] = ("voltarget", (tv / (3 * rv)).clip(0, 1).fillna(0.0))
    return rules


def evaluate_rules(rules: dict[str, tuple[str, pd.Series]], ret: pd.Series, rf: pd.Series, split: pd.Timestamp,
                   freq: str, exec_lag: int, cost_bps: float) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, runs = [], {}
    for name, (fam, w) in rules.items():
        w = w.reindex(ret.index)
        bt = run(w, ret, rf, freq, exec_lag, cost_bps)
        runs[name] = bt
        # モデル系ルールは OOS 予測が存在する日以降だけを評価する
        valid = bt.index >= (w.first_valid_index() or bt.index[0])
        for period, mask in (("disc", valid & (bt.index < split)), ("hold", valid & (bt.index >= split))):
            m = metrics(bt["ret"][mask], rf, bt["w"], bt["turnover"])
            if m:
                rows.append(dict(rule=name, family=fam, period=period, **m))
    tab = pd.DataFrame(rows)
    # 各ファミリーで探索期間 Sharpe 最良を選ぶ（保留期間の成績は選択に使わない）
    disc = tab[tab.period == "disc"]
    chosen = disc.loc[disc.groupby("family")["sharpe"].idxmax(), "rule"].tolist()
    tab["chosen_on_disc"] = tab["rule"].isin(chosen)
    return tab, runs
