"""逆張りモジュールの機械的な正しさ（連続日数、スイング判定、約定タイミング）。"""
import numpy as np
import pandas as pd

from soxl_factors import reversal


def test_streak_counts():
    s = pd.Series([True, True, False, True, True, True, False])
    assert reversal._streak(s).tolist() == [1, 2, 0, 1, 2, 3, 0]


def test_swing_labels_marks_local_extrema():
    c = pd.Series([5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3], dtype=float)
    lab = reversal.swing_labels(c, k=2)
    assert lab["bottom"].iloc[4] == 1 and lab["top"].iloc[10] == 1
    assert lab["bottom"].iloc[[3, 5]].sum() == 0


def test_simulate_enters_next_open_and_exits_next_open():
    idx = pd.bdate_range("2024-01-01", periods=8)
    bars = pd.DataFrame({"Open": [10, 10, 11, 12, 13, 14, 15, 16.0], "Close": [10, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5]},
                        index=idx)
    entry = pd.Series(False, index=idx)
    entry.iloc[1] = True  # 2 日目の引けでシグナル → 3 日目寄付きで買い
    r, tr = reversal.simulate(bars, entry, pd.Series(False, index=idx), max_hold=2, rf=pd.Series(0.0, index=idx), cost_bps=0)
    assert np.isclose(r.iloc[2], 11.5 / 11 - 1)          # 寄付き→引け
    assert np.isclose(r.iloc[3], 12.5 / 11.5 - 1)        # 保有中は引け→引け
    assert np.isclose(r.iloc[4], 13 / 12.5 - 1)          # 2 日保有後、翌寄付きで手仕舞い
    assert r.iloc[5] == 0 and len(tr) == 1 and np.isclose(tr.ret.iloc[0], 13 / 11 - 1)


def test_zigzag_and_zones():
    from soxl_factors import cycle

    c = pd.Series([100, 120, 140, 100, 90, 95, 130, 150, 110.0], index=pd.bdate_range("2020-01-01", periods=9))
    piv = cycle.zigzag(c, 0.25)
    # 系列の最初の点は、そこから 25% 上昇した時点で谷として確定する
    assert list(piv.kind) == ["trough", "peak", "trough", "peak"]
    assert piv.price.tolist() == [100, 140, 90, 150]
    z = cycle.zones(c, piv, band=0.10)
    assert z.bottom_zone.iloc[4] == 1 and z.bottom_zone.iloc[5] == 1 and z.bottom_zone.iloc[3] == 0
    assert np.isnan(z.bottom_zone.iloc[8])  # 最後のピボット以降は未確定


def test_monthly_release_lag():
    from soxl_factors import cycle

    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    cal = pd.bdate_range("2024-01-01", "2024-04-30")
    d = cycle._on_calendar(s, 50, cal)
    assert np.isnan(d.loc["2024-02-19"]) and d.loc["2024-02-20"] == 1.0 and d.loc["2024-03-22"] == 2.0
