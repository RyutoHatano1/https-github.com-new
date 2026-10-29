"""実行例: cd analysis && python -m soxl_factors --out output"""
import argparse
from pathlib import Path

from .data import load
from .pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="SOXL factor analysis")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--end", default=None, help="この日付の前日までを取得（当日の未確定足を除く）")
    ap.add_argument("--refresh", action="store_true", help="キャッシュを無視して再取得")
    ap.add_argument("--no-fred", action="store_true")
    ap.add_argument("--extended", action="store_true", help="SOXL 上場前を ^SOX から合成した 3 倍系列で延長")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--exec-lag", type=int, default=1, help="0=シグナル当日引け約定, 1=翌日引け約定")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--freq", default="W", choices=["D", "W", "M"])
    a = ap.parse_args()

    px, ohlcv = load(a.cache, a.start, end=a.end, refresh=a.refresh, use_fred=not a.no_fred)
    out = a.out / ("extended" if a.extended else "actual")
    run_pipeline(px, ohlcv, out, extended=a.extended, holdout_frac=a.holdout_frac,
                 exec_lag=a.exec_lag, cost_bps=a.cost_bps, freq=a.freq)
    print(f"レポート: {out / 'report.md'}")


if __name__ == "__main__":
    main()
