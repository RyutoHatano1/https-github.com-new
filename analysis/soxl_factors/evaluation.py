"""単変量の予測力評価、同時点の要因分解、レバレッジ減価、季節性、レジーム分析。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def split_date(index: pd.DatetimeIndex, holdout_frac: float) -> pd.Timestamp:
    return index[int(len(index) * (1 - holdout_frac))]


def _hac_rank_t(x: pd.Series, y: pd.Series, lags: int) -> tuple[float, float]:
    """順位化した x, y の回帰係数（=Spearman 相関）の Newey-West t 値と p 値。

    重複する多日リターンは自己相関を持つため、通常の t 値は過大になる。
    """
    xr = stats.zscore(x.rank().to_numpy())
    yr = stats.zscore(y.rank().to_numpy())
    res = sm.OLS(yr, sm.add_constant(xr)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[1]), float(res.pvalues[1])


def ic_table(X: pd.DataFrame, Y: pd.DataFrame, split: pd.Timestamp, min_obs: int = 250) -> pd.DataFrame:
    """全特徴量 × 全ターゲットの Spearman IC。探索期間で検定し、保留期間で再現性を確認する。"""
    rows = []
    for tgt in Y:
        h = int(tgt.rsplit("_", 1)[1])
        lags = max(5, int(1.5 * h))
        for feat in X:
            d = pd.concat([X[feat], Y[tgt]], axis=1, keys=["x", "y"]).dropna()
            disc, hold = d[d.index < split], d[d.index >= split]
            if len(disc) < min_obs or d["x"].nunique() < 2 or disc["x"].nunique() < 2:
                continue
            ic_d = disc["x"].corr(disc["y"], method="spearman")
            t, p = _hac_rank_t(disc["x"], disc["y"], lags)
            ic_h = hold["x"].corr(hold["y"], method="spearman") if len(hold) >= 60 and hold["x"].nunique() > 1 else np.nan
            yearly = disc.groupby(disc.index.year).apply(
                lambda g: g["x"].corr(g["y"], method="spearman") if len(g) > 50 and g["x"].nunique() > 1 else np.nan
            ).dropna()
            consist = float((np.sign(yearly) == np.sign(ic_d)).mean()) if len(yearly) else np.nan
            rows.append(dict(target=tgt, feature=feat, n_disc=len(disc), n_hold=len(hold),
                             ic_disc=ic_d, t_hac=t, p_hac=p, ic_hold=ic_h,
                             yearly_sign_consistency=consist, n_years=len(yearly)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh"] = multipletests(out["p_hac"], method="fdr_bh")[1]
    out["same_sign_hold"] = np.sign(out["ic_disc"]) == np.sign(out["ic_hold"])
    out["robust"] = (out["q_bh"] < 0.05) & out["same_sign_hold"] & (out["ic_hold"].abs() >= 0.5 * out["ic_disc"].abs())
    return out.sort_values(["target", "p_hac"]).reset_index(drop=True)


def quantile_table(x: pd.Series, y: pd.Series, q: int = 5) -> pd.DataFrame:
    d = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    d["bin"] = pd.qcut(d["x"].rank(method="first"), q, labels=[f"Q{i + 1}" for i in range(q)])
    g = d.groupby("bin", observed=True)["y"]
    return pd.DataFrame({"mean": g.mean(), "median": g.median(), "hit_rate": g.apply(lambda s: (s > 0).mean()),
                         "n": g.size(), "x_lo": d.groupby("bin", observed=True)["x"].min(),
                         "x_hi": d.groupby("bin", observed=True)["x"].max()})


CONTEMP_FACTORS = {
    "QQQ": ("ret", "QQQ"), "dTNX": ("diff", "TNX"), "dlogVIX": ("ret", "VIX"),
    "DXY": ("ret", "DXY"), "USDJPY": ("ret", "USDJPY"), "HYG": ("ret", "HYG"),
    "COPPER": ("ret", "COPPER"), "OIL": ("ret", "OIL"), "BTC": ("ret", "BTC"),
    "TWII": ("ret", "TWII"),
}


def contemporaneous_factors(px: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, (kind, col) in CONTEMP_FACTORS.items():
        if col in px:
            out[name] = np.log(px[col]).diff() if kind == "ret" else px[col].diff()
    return pd.DataFrame(out)


def contemporaneous_decomposition(px: pd.DataFrame, target: str = "SOX") -> dict:
    """同日の値動きの説明（取引シグナルではない）。標準化回帰 + 単独 R² + drop-one ΔR²。"""
    y = np.log(px[target]).diff().rename("y")
    F = contemporaneous_factors(px)
    # BTC は 2014 年以降のみ。全期間版と BTC 抜き版を分ける
    results = {}
    for label, cols in (("all", list(F.columns)), ("ex_btc", [c for c in F if c != "BTC"])):
        dd = pd.concat([y, F[cols]], axis=1).dropna()
        Z = (dd - dd.mean()) / dd.std()
        full = sm.OLS(Z["y"], sm.add_constant(Z[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows = []
        for c in cols:
            uni = sm.OLS(Z["y"], sm.add_constant(Z[[c]])).fit()
            others = [o for o in cols if o != c]
            drop = sm.OLS(Z["y"], sm.add_constant(Z[others])).fit() if others else None
            rows.append(dict(factor=c, std_beta=full.params[c], t_hac=full.tvalues[c],
                             r2_alone=uni.rsquared, delta_r2_drop=full.rsquared - (drop.rsquared if drop else 0)))
        results[label] = dict(table=pd.DataFrame(rows).sort_values("delta_r2_drop", ascending=False),
                              r2=full.rsquared, n=len(dd), start=dd.index[0], end=dd.index[-1])
    return results


def rolling_betas(px: pd.DataFrame, target: str = "SOX", window: int = 252,
                  cols=("QQQ", "dTNX", "DXY", "dlogVIX")) -> pd.DataFrame:
    y = np.log(px[target]).diff()
    F = contemporaneous_factors(px)
    F = F[[c for c in cols if c in F]]
    d = pd.concat([y.rename("y"), F], axis=1).dropna()
    out = {}
    for c in F:
        # 単回帰ベータ（多重共線性で解釈が崩れないよう単独で）
        out[c] = d["y"].rolling(window).cov(d[c]) / d[c].rolling(window).var()
    return pd.DataFrame(out)


def leverage_drag(px: pd.DataFrame, window: int = 63, L: float = 3.0) -> dict:
    """非重複窓で 実SOXLの対数リターン − L×SOXの対数リターン を SOX 日次分散の和に回帰。

    連続時間近似では差 ≈ −(L²−L)/2 × Σr² − コスト。L=3 なら傾き ≈ −3。
    """
    d = pd.concat([np.log(px["SOXL"]).diff(), np.log(px["SOX"]).diff()], axis=1, keys=["soxl", "sox"]).dropna()
    blk = np.arange(len(d)) // window
    g = d.groupby(blk)
    agg = pd.DataFrame({"gap": g["soxl"].sum() - L * g["sox"].sum(), "var": g["sox"].apply(lambda s: (s ** 2).sum()),
                        "n": g.size(), "end": g.apply(lambda x: x.index[-1])})
    agg = agg[agg["n"] == window]
    res = sm.OLS(agg["gap"], sm.add_constant(agg["var"])).fit(cov_type="HC1")
    beta = sm.OLS(d["soxl"], sm.add_constant(d["sox"])).fit()
    return dict(slope=res.params["var"], slope_se=res.bse["var"], intercept_ann=res.params["const"] * 252 / window,
                theory_slope=-(L ** 2 - L) / 2, n_windows=len(agg), r2=res.rsquared, blocks=agg,
                daily_beta=beta.params["sox"], daily_r2=beta.rsquared)


def seasonality(y1: pd.Series) -> dict[str, pd.DataFrame]:
    """翌日リターンの曜日・月・月替わり・SQ週別の平均と Kruskal-Wallis 検定。"""
    y1 = y1.dropna()
    idx = y1.index
    # 曜日は「シグナル日 t」ではなく「リターンが実現する日 t+1」でまとめる方が直感的だが、
    # 取引判断は t 時点で行うため t の曜日で集計する
    keys = {
        "weekday(t)": pd.Series(idx.day_name(), index=idx),
        "month(t)": pd.Series(idx.month, index=idx),
    }
    out = {}
    for name, k in keys.items():
        g = y1.groupby(k)
        tab = pd.DataFrame({"mean_bp": g.mean() * 1e4, "hit_rate": g.apply(lambda s: (s > 0).mean()), "n": g.size()})
        p = stats.kruskal(*[s.to_numpy() for _, s in g]).pvalue
        out[name] = (tab, p)
    return out


def regime_table(price: pd.Series, X: pd.DataFrame, Y: pd.DataFrame, split: pd.Timestamp,
                 tgt: str = "fwd_ret_21") -> pd.DataFrame:
    """レジーム別の将来リターン/ボラ。閾値は探索期間のみで決める。"""
    disc = X.index < split
    regimes = {}
    if "sox_above_sma200" in X:
        regimes["SOX vs SMA200"] = X["sox_above_sma200"].map({1.0: "above", 0.0: "below"})
    if "vix" in X:
        regimes["VIX"] = pd.cut(X["vix"], [0, 15, 20, 30, np.inf], labels=["<15", "15-20", "20-30", ">30"])
    if "sox_rv21" in X:
        qs = X.loc[disc, "sox_rv21"].quantile([1 / 3, 2 / 3]).to_numpy()
        regimes["SOX RV21 tercile"] = pd.cut(X["sox_rv21"], [-np.inf, qs[0], qs[1], np.inf], labels=["low", "mid", "high"])
    if "sox_drawdown_252" in X:
        regimes["SOX drawdown"] = pd.cut(X["sox_drawdown_252"], [-1, -0.4, -0.2, -0.1, 0.0001],
                                         labels=["<-40%", "-40~-20%", "-20~-10%", ">-10%"])
    rows = []
    for rname, lab in regimes.items():
        for period, mask in (("disc", X.index < split), ("hold", X.index >= split)):
            d = pd.concat([lab.rename("lab"), Y[tgt], Y["fwd_vol_21"]], axis=1)[mask].dropna()
            for lv, g in d.groupby("lab", observed=True):
                rows.append(dict(regime=rname, level=str(lv), period=period, n=len(g),
                                 mean_fwd=g[tgt].mean(), median_fwd=g[tgt].median(),
                                 hit_rate=(g[tgt] > 0).mean(), mean_fwd_vol=g["fwd_vol_21"].mean()))
    return pd.DataFrame(rows)
