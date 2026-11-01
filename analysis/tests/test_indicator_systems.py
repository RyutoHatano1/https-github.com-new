"""スーパートレンド・状態機械・約定タイミングの確認。"""
import numpy as np
import pandas as pd

from soxl_factors import indicator_systems as ins


def bars(close):
    c = pd.Series(close, index=pd.bdate_range("2020-01-01", periods=len(close)), dtype=float)
    return c * 1.01, c * 0.99, c


def test_supertrend_follows_trend():
    up = np.linspace(100, 200, 60)
    down = np.linspace(200, 120, 40)
    h, l, c = bars(np.r_[up, down])
    st = ins.supertrend(h, l, c, 10, 3.0)
    assert st.iloc[55] == 1          # 上昇の終盤は上昇トレンド
    assert st.iloc[-1] == 0          # 大きく下げた後は下落トレンド
    flips = (st.diff().abs() == 1).sum()
    assert flips <= 2                # 一方向の値動きで何度も反転しない


def test_positions_and_execution_lag():
    idx = pd.bdate_range("2020-01-01", periods=8)
    entry = pd.Series([0, 1, 0, 0, 0, 0, 0, 0], index=idx).astype(bool)
    exit_ = pd.Series([0, 0, 0, 1, 0, 0, 0, 0], index=idx).astype(bool)
    st = ins.positions(entry, exit_)
    assert st.tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    price = pd.Series([10, 10, 11, 12, 13, 14, 15, 16.0], index=idx)
    res = ins.run(st, price, pd.Series(0.0, index=idx))
    # 1 日目引けで判断 → 2 日目引けで買い → 3 日目のリターンから。3 日目引けで判断 → 4 日目引けで売り
    assert res["w"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert res["trades"] == [(11.0, 13.0)]
