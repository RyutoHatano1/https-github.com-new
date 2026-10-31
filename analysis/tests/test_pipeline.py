"""合成データでの検証。結果の代用ではなく、パイプラインの正しさ（先読みなし・既知の信号を回収・
偽の信号を拒否・レバレッジ減価の理論値）を確かめるためのもの。

実行: cd analysis && python -m pytest -q tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soxl_factors import backtest, evaluation, features
from soxl_factors.data import YAHOO_TICKERS, synthetic_leveraged
from soxl_factors.pipeline import run_pipeline

PLANT_K = 0.004  # DXY の 21 日リターンの符号が翌日の SOX リターンに与える影響（日次）


def make_market(n: int = 3800, seed: int = 1) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    r = {name: rng.normal(0.0002, 0.012, n) for name in YAHOO_TICKERS}
    r["QQQ"] = rng.normal(0.0004, 0.013, n)
    px = pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in r.items()}, index=idx)
    for k in ("VIX", "VIX9D", "VIX3M", "VXN"):
        px[k] = 20 * np.exp(0.3 * np.sin(np.arange(n) / 50) + rng.normal(0, 0.05, n))
    px["TNX"] = 3 + np.cumsum(rng.normal(0, 0.05, n))
    px["IRX"] = 1.0
    # 仕込み: SOX_t のリターン = 1.3*QQQ_t - k*sign(DXY の 21 日リターン_{t-1}) + ノイズ
    dxy21 = np.log(px["DXY"] / px["DXY"].shift(21)).shift(1).fillna(0).to_numpy()
    rs = 1.3 * r["QQQ"] - PLANT_K * np.sign(dxy21) + rng.normal(0, 0.012, n)
    px["SOX"] = 300 * np.exp(np.cumsum(rs))
    soxl = synthetic_leveraged(px["SOX"], px["IRX"])
    px["SOXL"] = soxl.where(idx >= idx[300])  # 上場前の期間を作る
    hl = np.abs(rng.normal(0, 0.01, n)) + 0.002
    ohlcv = {
        "SOX": pd.DataFrame({"Open": px["SOX"], "High": px["SOX"] * (1 + hl), "Low": px["SOX"] * (1 - hl),
                             "Close": px["SOX"], "Volume": 0.0}, index=idx),
        "SOXL": pd.DataFrame({"Open": px["SOXL"], "High": px["SOXL"], "Low": px["SOXL"], "Close": px["SOXL"],
                              "Volume": rng.lognormal(15, 0.3, n)}, index=idx).where(px["SOXL"].notna(), axis=0),
    }
    return px, ohlcv


@pytest.fixture(scope="module")
def market():
    return make_market()


def test_no_lookahead(market):
    """日付 D で打ち切ったデータで作った特徴量が、全データで作った D の特徴量と一致する。"""
    px, ohlcv = market
    X_full, _ = features.build_features(px, ohlcv)
    for D in px.index[[700, 1500, 2600]]:
        X_cut, _ = features.build_features(px.loc[:D], {k: v.loc[:D] for k, v in ohlcv.items()})
        # 月替わりフラグは取引所カレンダーから事前に分かる情報なので除外
        cols = [c for c in X_cut if c != "cal_turn_of_month"]
        a, b = X_full.loc[D, cols].astype(float), X_cut.loc[D, cols].astype(float)
        bad = [c for c in cols if not (np.isclose(a[c], b[c], equal_nan=True))]
        assert not bad, f"先読みの疑い: {bad}"


def test_targets_are_future(market):
    px, _ = market
    Y = features.build_targets(px["SOX"])
    lp = np.log(px["SOX"])
    t = px.index[1000]
    assert np.isclose(Y.loc[t, "fwd_ret_21"], lp.iloc[1021] - lp.iloc[1000])
    rv = lp.diff().iloc[1001:1022].std() * np.sqrt(252)
    assert np.isclose(Y.loc[t, "fwd_vol_21"], rv)


def test_ic_recovers_planted_and_controls_fdr(market):
    px, ohlcv = market
    X, _ = features.build_features(px, ohlcv)
    Y = features.build_targets(px["SOX"])[["fwd_ret_1", "fwd_ret_5"]]
    idx = X.index[300:-30]
    X, Y = X.loc[idx], Y.loc[idx]
    ic = evaluation.ic_table(X, Y, evaluation.split_date(idx, 0.3))
    hit = ic[(ic.feature == "dxy_ret21") & (ic.target == "fwd_ret_1")].iloc[0]
    assert hit.ic_disc < 0 and hit.robust
    # 仕込んでいない純ノイズ系列（OIL, GOLD, BTC 等）由来は robust にならないはず
    noise = ic[ic.feature.str.startswith(("oil_", "gold_", "btc_", "n225_", "ks11_"))]
    assert noise.robust.mean() <= 0.05


def test_leverage_drag_matches_theory(market):
    px, _ = market
    ld = evaluation.leverage_drag(px)
    assert abs(ld["slope"] - ld["theory_slope"]) < 0.6
    assert abs(ld["daily_beta"] - 3) < 0.02


def test_backtest_timing():
    """exec_lag=1 なら t のシグナルは r_{t+2} から効く（t+1 の引けで約定）。"""
    idx = pd.bdate_range("2020-01-01", periods=10)
    w = pd.Series(0.0, index=idx)
    w.iloc[3] = 1.0
    ret = pd.Series(0.01, index=idx)
    bt = backtest.run(w, ret, pd.Series(0.0, index=idx), freq="D", exec_lag=1, cost_bps=0)
    assert bt["w"].iloc[5] == 1.0 and bt["w"].iloc[4] == 0.0 and bt["w"].iloc[6] == 0.0


def test_pipeline_end_to_end(market, tmp_path):
    px, ohlcv = market
    res = run_pipeline(px, ohlcv, tmp_path, extended=True, n_perm=2, quick=True)
    assert (tmp_path / "report.md").exists()
    wf = res["wf"].set_index(["target", "model"])
    # 仕込んだ信号は fx グループに属するので、重要度で上位に来るはず
    assert res["importance"]["ridge:fwd_ret_21"].idxmax() == "fx"
    assert wf.loc[("fwd_ret_21", "ridge"), "mean_fold_ic"] > 0.05
