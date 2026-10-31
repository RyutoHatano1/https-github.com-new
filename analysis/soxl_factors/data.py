"""データ取得: Yahoo Finance (yfinance) と FRED。ローカルCSVキャッシュ対応。

全系列は米国営業日 (^SOX / SOXL の取引日) に揃える。
FRED 系列は公表ラグを考慮して営業日単位で後ろにずらす（先読みバイアス対策）。
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 名前 -> Yahoo ティッカー
YAHOO_TICKERS: dict[str, str] = {
    # 対象と原指数
    "SOXL": "SOXL", "SOX": "^SOX", "SMH": "SMH",
    # 広い株式市場
    "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM",
    # 主要構成銘柄
    "NVDA": "NVDA", "TSM": "TSM", "ASML": "ASML", "AVGO": "AVGO",
    "AMD": "AMD", "MU": "MU", "AMAT": "AMAT",
    # ボラティリティ
    "VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M", "VXN": "^VXN",
    # 金利 (^TNX 等は利回り×10ではなく % 表示)
    "TNX": "^TNX", "FVX": "^FVX", "IRX": "^IRX", "TYX": "^TYX",
    # 債券・クレジット
    "TLT": "TLT", "IEF": "IEF", "HYG": "HYG", "LQD": "LQD",
    # 為替
    "DXY": "DX-Y.NYB", "USDJPY": "JPY=X", "USDKRW": "KRW=X", "USDTWD": "TWD=X",
    # コモディティ・暗号資産
    "COPPER": "HG=F", "GOLD": "GC=F", "OIL": "CL=F", "BTC": "BTC-USD",
    # アジア（米国の同日引けより前に引ける）
    "TWII": "^TWII", "KS11": "^KS11", "N225": "^N225",
}

# OHLCV が必要な系列
OHLCV_NAMES = ("SOXL", "SOX")

# FRED 系列 -> 公表ラグ（営業日）。日次系列は翌営業日公表、週次は概ね5営業日後。
FRED_SERIES: dict[str, int] = {
    "BAMLH0A0HYM2": 1,  # ICE BofA US High Yield OAS
    "T10Y2Y": 1,
    "T10Y3M": 1,
    "DFII10": 1,        # 10年実質金利 (TIPS)
    "T10YIE": 1,        # 10年ブレークイーブン
    "NFCI": 5,          # シカゴ連銀 金融環境指数（週次）
    "ICSA": 5,          # 新規失業保険申請件数（週次）
}

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _retry(fn, tries: int = 4):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # ネットワーク一時エラーのみを想定
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def download_yahoo(start: str, end: str | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """終値 (配当・分割調整済) のワイド表と、OHLCV 対象の個別表を返す。"""
    import yfinance as yf

    tickers = list(YAHOO_TICKERS.values())
    raw = _retry(lambda: yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        progress=False, group_by="column", threads=True,
    ))
    if raw is None or raw.empty:
        raise RuntimeError("yfinance から空のデータが返りました（ネットワーク遮断の可能性）")

    inv = {v: k for k, v in YAHOO_TICKERS.items()}
    close = raw["Close"].rename(columns=inv)
    missing = [k for k in YAHOO_TICKERS if k not in close.columns or close[k].dropna().empty]
    if missing:
        print(f"[warn] 取得できなかった系列: {missing}")

    ohlcv = {}
    for name in OHLCV_NAMES:
        t = YAHOO_TICKERS[name]
        df = pd.DataFrame({f: raw[f][t] for f in ("Open", "High", "Low", "Close", "Volume")})
        ohlcv[name] = df.dropna(subset=["Close"])
    return close, ohlcv


def download_fred(start: str) -> pd.DataFrame:
    frames = {}
    import requests

    for sid in FRED_SERIES:
        try:
            txt = _retry(lambda: requests.get(FRED_URL.format(sid=sid), timeout=30).text)
            df = pd.read_csv(io.StringIO(txt))
            df.columns = ["date", sid]
            df["date"] = pd.to_datetime(df["date"])
            s = pd.to_numeric(df.set_index("date")[sid], errors="coerce")
            frames[sid] = s[s.index >= start]
        except Exception as e:
            print(f"[warn] FRED {sid} 取得失敗: {e}")
    return pd.DataFrame(frames)


def align(close: pd.DataFrame, ohlcv: dict[str, pd.DataFrame], fred: pd.DataFrame | None,
          max_ffill: int = 5) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """米国営業日 (^SOX の取引日) に揃える。休場差は最大 max_ffill 日だけ前方補完。"""
    cal = ohlcv["SOX"].index
    if "SOXL" in ohlcv:
        cal = cal.union(ohlcv["SOXL"].index)
    cal = cal.sort_values()

    # 週末に取引のある系列 (BTC 等) は「その営業日の終値時点までに観測された最新値」を使う
    px = close.sort_index().reindex(close.index.union(cal)).ffill(limit=max_ffill).reindex(cal)

    if fred is not None and not fred.empty:
        f = fred.sort_index().reindex(fred.index.union(cal)).ffill(limit=10).reindex(cal)
        for sid, lag in FRED_SERIES.items():
            if sid in f:
                px[f"FRED_{sid}"] = f[sid].shift(lag)
    ohlcv = {k: v.reindex(cal) for k, v in ohlcv.items()}
    return px, ohlcv


def load(cache_dir: Path, start: str = "1995-01-01", end: str | None = None,
         refresh: bool = False, use_fred: bool = True):
    """キャッシュがあれば読み、無ければダウンロードして保存する。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f_close, f_fred = cache_dir / "close.csv", cache_dir / "fred.csv"
    f_ohlcv = {n: cache_dir / f"ohlcv_{n}.csv" for n in OHLCV_NAMES}

    if refresh or not f_close.exists():
        close, ohlcv = download_yahoo(start, end)
        close.to_csv(f_close)
        for n, df in ohlcv.items():
            df.to_csv(f_ohlcv[n])
        if use_fred:
            fred = download_fred(start)
            fred.to_csv(f_fred)

    close = pd.read_csv(f_close, index_col=0, parse_dates=True)
    ohlcv = {n: pd.read_csv(p, index_col=0, parse_dates=True) for n, p in f_ohlcv.items() if p.exists()}
    fred = pd.read_csv(f_fred, index_col=0, parse_dates=True) if (use_fred and f_fred.exists()) else None
    return align(close, ohlcv, fred)


def synthetic_leveraged(sox_close: pd.Series, rf_annual: pd.Series, leverage: float = 3.0,
                        expense: float = 0.0075, spread: float = 0.005) -> pd.Series:
    """原指数から日次リバランス L 倍 ETF を合成する（SOXL 上場前の期間の延長用）。

    日次リターン = L*r - (経費率 + (L-1)*(rf+spread))/252。
    ^SOX は価格指数で配当を含まないため、実際の SOXL よりやや不利に出る。
    """
    r = sox_close.pct_change()
    rf = (rf_annual.ffill() / 100.0).reindex(r.index).fillna(0.0)
    cost = (expense + (leverage - 1.0) * (rf + spread)) / 252.0
    lev = (leverage * r - cost).clip(lower=-0.999)
    out = (1.0 + lev.fillna(0.0)).cumprod()
    return out.where(sox_close.notna())


def build_target_price(px: pd.DataFrame, extended: bool) -> pd.Series:
    """解析対象の価格系列。extended=True なら上場前を合成で補い、実 SOXL に接続する。"""
    actual = px["SOXL"].dropna()
    if not extended:
        return actual
    syn = synthetic_leveraged(px["SOX"], px.get("IRX", pd.Series(0.0, index=px.index)))
    first = actual.index[0]
    scale = actual.iloc[0] / syn.loc[first]
    joined = pd.concat([syn.loc[:first].iloc[:-1] * scale, actual])
    return joined.dropna()


def tracking_report(px: pd.DataFrame) -> dict:
    """合成系列と実 SOXL の重複期間での一致度。"""
    syn = synthetic_leveraged(px["SOX"], px.get("IRX", pd.Series(0.0, index=px.index)))
    both = pd.concat([px["SOXL"].pct_change(), syn.pct_change()], axis=1, keys=["actual", "syn"]).dropna()
    diff = np.log1p(both["actual"]) - np.log1p(both["syn"])
    return {
        "n_days": int(len(both)),
        "daily_corr": float(both.corr().iloc[0, 1]),
        "ann_tracking_diff_log": float(diff.mean() * 252),
        "ann_tracking_error": float(diff.std() * np.sqrt(252)),
    }
