"""特徴量（ファクター）とターゲットの生成。

規約: 日付 t の特徴量は「t の米国引け時点までに観測可能な情報」のみで作る。
ターゲットは t の引けから t+h の引けまでの対数リターン等（未来）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONSTITUENTS = ("NVDA", "TSM", "ASML", "AVGO", "AMD", "MU", "AMAT")
HORIZONS = (1, 5, 21, 63)


def _lr(s: pd.Series, n: int) -> pd.Series:
    return np.log(s / s.shift(n))


def _rv(s: pd.Series, n: int) -> pd.Series:
    return np.log(s).diff().rolling(n).std() * np.sqrt(252)


def _z(s: pd.Series, n: int) -> pd.Series:
    m, sd = s.rolling(n).mean(), s.rolling(n).std()
    return (s - m) / sd


def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def _col(px: pd.DataFrame, name: str) -> pd.Series:
    return px[name] if name in px else pd.Series(np.nan, index=px.index)


def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    ym = idx.to_period("M")
    pos = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).rank(method="first")
    size = pd.Series(1, index=idx).groupby(ym).transform("size")
    df["cal_turn_of_month"] = ((pos <= 3) | (pos == size)).astype(float)
    # 第3金曜（月次オプション満期）の週
    first = idx.to_period("M").to_timestamp()
    third_fri = first + pd.to_timedelta((4 - first.weekday) % 7 + 14, unit="D")
    delta = (third_fri - idx).days
    df["cal_opex_week"] = ((delta >= 0) & (delta <= 4)).astype(float)
    df["cal_month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["cal_month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    for i, d in enumerate(("mon", "tue", "wed", "thu")):
        df[f"cal_dow_{d}"] = (idx.weekday == i).astype(float)
    return df


def build_features(px: pd.DataFrame, ohlcv: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    sox = px["SOX"]
    f: dict[str, pd.Series] = {}
    groups: dict[str, list[str]] = {}

    def add(group: str, name: str, s: pd.Series):
        f[name] = s
        groups.setdefault(group, []).append(name)

    # --- モメンタム（原指数 ^SOX 基準: レバレッジの経路依存を排除し長い履歴を使う）
    for n in (1, 5, 21, 63, 126, 252):
        add("momentum", f"sox_ret_{n}", _lr(sox, n))
    add("momentum", "sox_mom_12_1", _lr(sox.shift(21), 231))

    # --- 平均回帰 / 過熱
    add("mean_reversion", "sox_rsi_2", _rsi(sox, 2))
    add("mean_reversion", "sox_rsi_14", _rsi(sox, 14))
    for n in (20, 50, 200):
        add("mean_reversion", f"sox_dist_sma{n}", np.log(sox / sox.rolling(n).mean()))
    add("mean_reversion", "sox_bb_z20", _z(sox, 20))
    add("mean_reversion", "sox_drawdown_252", sox / sox.rolling(252).max() - 1)

    # --- トレンド
    sma50, sma200 = sox.rolling(50).mean(), sox.rolling(200).mean()
    add("trend", "sox_above_sma200", (sox > sma200).astype(float).where(sma200.notna()))
    add("trend", "sox_golden_cross", (sma50 > sma200).astype(float).where(sma200.notna()))
    add("trend", "sox_sma200_slope21", np.log(sma200 / sma200.shift(21)))

    # --- 実現ボラティリティ
    rv5, rv21, rv63 = _rv(sox, 5), _rv(sox, 21), _rv(sox, 63)
    add("realized_vol", "sox_rv5", rv5)
    add("realized_vol", "sox_rv21", rv21)
    add("realized_vol", "sox_rv63", rv63)
    add("realized_vol", "sox_rv_ratio_5_63", rv5 / rv63)
    add("realized_vol", "sox_rv21_chg21", np.log(rv21 / rv21.shift(21)))
    r = np.log(sox).diff()
    add("realized_vol", "sox_downvol21", r.clip(upper=0).rolling(21).std() * np.sqrt(252))
    add("realized_vol", "sox_skew63", r.rolling(63).skew())
    add("realized_vol", "sox_kurt63", r.rolling(63).kurt())
    if "SOX" in ohlcv and {"High", "Low"} <= set(ohlcv["SOX"].columns):
        hl = np.log(ohlcv["SOX"]["High"] / ohlcv["SOX"]["Low"]).replace([np.inf, -np.inf], np.nan)
        hl = hl.where(hl > 0)  # 高値=安値の欠損日を除外
        add("realized_vol", "sox_parkinson21",
            np.sqrt((hl ** 2).rolling(21, min_periods=15).mean() / (4 * np.log(2)) * 252))

    # --- インプライド・ボラティリティ
    vix, vxn = _col(px, "VIX"), _col(px, "VXN")
    add("implied_vol", "vix", vix)
    add("implied_vol", "vix_z252", _z(vix, 252))
    add("implied_vol", "vix_chg5", _lr(vix, 5))
    add("implied_vol", "vxn", vxn)
    add("implied_vol", "vix_term_9d_30d", _col(px, "VIX9D") / vix)
    add("implied_vol", "vix_term_30d_3m", vix / _col(px, "VIX3M"))
    add("implied_vol", "vrp_qqq", vxn / 100 - _rv(_col(px, "QQQ"), 21))

    # --- 出来高（SOXL 上場後のみ）
    if "SOXL" in ohlcv and "Volume" in ohlcv["SOXL"]:
        v = np.log(ohlcv["SOXL"]["Volume"].replace(0, np.nan))
        add("volume", "soxl_logvol_z63", _z(v, 63))
        dv = ohlcv["SOXL"]["Volume"] * ohlcv["SOXL"]["Close"]
        add("volume", "soxl_dollarvol_21_252", np.log(dv.rolling(21).mean() / dv.rolling(252).mean()))

    # --- 相対強度・ブレッドス
    qqq, spy = _col(px, "QQQ"), _col(px, "SPY")
    for n in (21, 63, 126):
        add("relative_strength", f"sox_vs_qqq_{n}", _lr(sox, n) - _lr(qqq, n))
    add("relative_strength", "sox_vs_spy_63", _lr(sox, 63) - _lr(spy, 63))
    add("relative_strength", "sox_vs_iwm_63", _lr(sox, 63) - _lr(_col(px, "IWM"), 63))
    cons = [c for c in CONSTITUENTS if c in px]
    if cons:
        cp = px[cons]
        above = (cp > cp.rolling(50).mean()).astype(float).where(cp.rolling(50).mean().notna())
        add("breadth", "cons_frac_above_sma50", above.mean(axis=1, skipna=True).where(above.notna().sum(axis=1) >= 3))
        r21 = np.log(cp / cp.shift(21))
        add("breadth", "cons_dispersion21", r21.std(axis=1).where(r21.notna().sum(axis=1) >= 3))
        if "NVDA" in px:
            add("breadth", "nvda_vs_sox_21", _lr(px["NVDA"], 21) - _lr(sox, 21))

    # --- ベータ・相関レジーム
    rq = np.log(qqq).diff()
    add("beta_corr", "sox_beta_qqq_63", r.rolling(63).cov(rq) / rq.rolling(63).var())
    add("beta_corr", "sox_corr_qqq_63", r.rolling(63).corr(rq))
    tnx = _col(px, "TNX")
    add("beta_corr", "sox_corr_dtnx_63", r.rolling(63).corr(tnx.diff()))

    # --- 金利
    irx = _col(px, "IRX")
    add("rates", "tnx", tnx)
    add("rates", "tnx_chg5", tnx.diff(5))
    add("rates", "tnx_chg21", tnx.diff(21))
    add("rates", "curve_10y_3m", tnx - irx)
    add("rates", "curve_10y_3m_chg21", (tnx - irx).diff(21))
    add("rates", "tlt_ret21", _lr(_col(px, "TLT"), 21))
    for sid, nm in (("DFII10", "real10y"), ("T10YIE", "breakeven10y")):
        c = f"FRED_{sid}"
        if c in px:
            add("rates", f"{nm}_chg21", px[c].diff(21))

    # --- クレジット・金融環境
    add("credit", "hyg_vs_ief_21", _lr(_col(px, "HYG"), 21) - _lr(_col(px, "IEF"), 21))
    add("credit", "lqd_vs_ief_21", _lr(_col(px, "LQD"), 21) - _lr(_col(px, "IEF"), 21))
    if "FRED_BAMLH0A0HYM2" in px:
        oas = px["FRED_BAMLH0A0HYM2"]
        add("credit", "hy_oas", oas)
        add("credit", "hy_oas_chg21", oas.diff(21))
        add("credit", "hy_oas_z252", _z(oas, 252))
    if "FRED_NFCI" in px:
        add("credit", "nfci", px["FRED_NFCI"])
    if "FRED_ICSA" in px:
        add("macro", "icsa_chg21", np.log(px["FRED_ICSA"] / px["FRED_ICSA"].shift(21)))

    # --- 為替
    for nm in ("DXY", "USDJPY", "USDKRW", "USDTWD"):
        add("fx", f"{nm.lower()}_ret21", _lr(_col(px, nm), 21))
    add("fx", "dxy_ret5", _lr(_col(px, "DXY"), 5))

    # --- コモディティ・暗号資産
    cg = np.log(_col(px, "COPPER") / _col(px, "GOLD"))
    add("commodity", "copper_gold_chg21", cg.diff(21))
    add("commodity", "copper_gold_chg63", cg.diff(63))
    add("commodity", "oil_ret21", _lr(_col(px, "OIL"), 21))
    add("commodity", "gold_ret21", _lr(_col(px, "GOLD"), 21))
    add("crypto", "btc_ret5", _lr(_col(px, "BTC"), 5))
    add("crypto", "btc_ret21", _lr(_col(px, "BTC"), 21))

    # --- アジア（同日の米国引けより前に確定）
    for nm in ("TWII", "KS11", "N225"):
        add("asia", f"{nm.lower()}_ret1", _lr(_col(px, nm), 1))
        add("asia", f"{nm.lower()}_ret5", _lr(_col(px, nm), 5))

    # --- カレンダー
    cal = _calendar(px.index)
    for c in cal:
        add("calendar", c, cal[c])

    X = pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)
    # 全欠損の列（取得失敗した系列由来）を除去
    keep = [c for c in X if X[c].notna().any()]
    groups = {g: [c for c in cs if c in keep] for g, cs in groups.items()}
    groups = {g: cs for g, cs in groups.items() if cs}
    return X[keep], groups


def build_targets(price: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    lp = np.log(price)
    y = {f"fwd_ret_{h}": lp.shift(-h) - lp for h in horizons}
    r = lp.diff()
    # t+1..t+21 の実現ボラ（未来）
    y["fwd_vol_21"] = r.rolling(21).std().shift(-21) * np.sqrt(252)
    return pd.DataFrame(y)
