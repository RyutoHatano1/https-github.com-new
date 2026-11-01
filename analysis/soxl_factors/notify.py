"""SOXL の RSI 売買シグナルを LINE（Messaging API のプッシュ）で通知する。

ルール: 半導体指数 ^SOX の RSI(14)（ワイルダー）が 40 未満で買い、75 以上で売る（2010 年以降の検証に基づく）。
毎回、過去データ（2010-03 以降の実 SOXL）から勝率・期待値（平均）を計算し直して添える。

実行: cd analysis && python -m soxl_factors.notify [--dry-run]
環境変数: LINE_CHANNEL_ACCESS_TOKEN, LINE_USER_ID（無ければ送信せず標準出力に表示）
          NOTIFY_DAILY=true/false（売買が必要な日以外も日報を送るか。既定 false＝買い・売りの日と異常時だけ）
終了コード: 0 = 正常（または休場日でスキップ）、1 = データ異常・送信失敗（GitHub Actions を失敗にして気づけるように）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .exit_rules import rsi_exit, simulate
from .features import _rsi

BUY, SELL = 40.0, 75.0
START = "2010-03-11"  # SOXL 上場日
TICKERS = {"SOX": "^SOX", "SOXL": "SOXL", "NVDA": "NVDA", "VXN": "^VXN", "VIX": "^VIX", "IRX": "^IRX"}
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
FRED_UNRATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
MAX_TEXT = 4800  # LINE のテキストは 1 通 5000 文字まで。余裕を持って分割


# ---------------------------------------------------------------- データ取得
def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def fetch_market(start: str = "2009-01-01") -> pd.DataFrame:
    import yfinance as yf

    raw = _retry(lambda: yf.download(list(TICKERS.values()), start=start, auto_adjust=True, progress=False,
                                     group_by="column", threads=False))
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance から空のデータ")
    inv = {v: k for k, v in TICKERS.items()}
    return raw["Close"].rename(columns=inv)


def fetch_unrate() -> pd.Series:
    txt = _retry(lambda: requests.get(FRED_UNRATE, timeout=30).text)
    df = pd.read_csv(io.StringIO(txt))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def unrate_rise_on(cal: pd.DatetimeIndex, u: pd.Series, lag_days: int = 40) -> pd.Series:
    """失業率 − 過去 12 か月最低。公表ラグ（約 40 日）分ずらして、その日に分かっていた値にする。"""
    rise = u - u.rolling(12).min()
    rise.index = rise.index + pd.Timedelta(days=lag_days)
    return rise.reindex(rise.index.union(cal)).ffill().reindex(cal)


# ---------------------------------------------------------------- 異常検知
def check_data(px: pd.DataFrame, now_utc: datetime) -> tuple[list[str], list[str], bool]:
    """(致命的な異常, 警告, 休場等でスキップすべきか)"""
    crit, warn = [], []
    core = px[["SOX", "SOXL"]].dropna()
    if len(core) < 300:
        return ["SOX/SOXL のデータが不足"], warn, False
    last = core.index[-1]
    today_et = pd.Timestamp(now_utc).tz_convert("America/New_York").tz_localize(None).normalize()
    lag_bd = len(pd.bdate_range(last + pd.Timedelta(days=1), today_et))
    skip = False
    if lag_bd >= 3:
        crit.append(f"最新データが古い（最終 {last:%Y-%m-%d}、{lag_bd} 営業日遅れ）")
    elif lag_bd >= 1:
        skip = True  # 祝日休場、またはデータ未更新。重複通知を避けて送らない
        warn.append(f"本日分のデータなし（最終 {last:%Y-%m-%d}）。休場日か更新遅れ")
    if px.loc[last, ["SOX", "SOXL"]].isna().any():
        crit.append("最新日の SOX/SOXL が欠損")
    r = core.pct_change().iloc[-1]
    if abs(r["SOX"]) > 0.25 or abs(r["SOXL"]) > 0.70:
        crit.append(f"前日比が異常に大きい（SOX {r['SOX']:+.1%}、SOXL {r['SOXL']:+.1%}）。分割・データ誤りの可能性")
    # SOXL は SOX の約 3 倍で動く。直近 5 日でズレが大きければデータ不整合を疑う
    rr = core.pct_change().iloc[-5:]
    dev = (rr["SOXL"] - 3 * rr["SOX"]).abs().max()
    if dev > 0.08:
        warn.append(f"SOXL と 3×SOX の日次リターンのズレが大きい（最大 {dev:.1%}）")
    for k in ("NVDA", "VXN", "VIX"):
        if k not in px or pd.isna(px[k].reindex([last]).iloc[0]):
            warn.append(f"{k} の最新値が取得できず")
    return crit, warn, skip


# ---------------------------------------------------------------- 集計
def _fwd(c: pd.Series) -> pd.DataFrame:
    out = {lab: c.shift(-h) / c - 1 for lab, h in (("1か月", 21), ("3か月", 63), ("6か月", 126), ("1年", 252))}
    lo = c.rolling(127, center=True).min()
    out["底圏"] = (c <= lo * 1.2).astype(float).where(lo.notna())
    return pd.DataFrame(out)


def cond_stats(mask: pd.Series, F: pd.DataFrame, horizons=("1か月", "3か月", "1年")) -> dict:
    m = mask.reindex(F.index).fillna(False).astype(bool)
    g = F[m]
    out = {"日数": int(m.sum()), "月数": int(g.index.to_period("M").nunique())}
    for h in horizons:
        x = g[h].dropna()
        out[h] = (float((x > 0).mean()), float(x.mean()), float(x.median()), len(x)) if len(x) else None
    z = g["底圏"].dropna()
    out["底圏"] = float(z.mean()) if len(z) else None
    return out


def compute(px: pd.DataFrame, unrate: pd.Series | None) -> dict:
    px = px.copy()
    sox = px["SOX"].dropna()
    rsi_all = _rsi(sox, 14)
    c = px["SOXL"].dropna()
    c = c.loc[START:]
    idx = c.index
    rsi = rsi_all.reindex(idx)
    F = _fwd(c)
    sox_c = sox.reindex(idx)
    sox_1d = sox_c.pct_change()
    nvda_rel = (np.log(px["NVDA"] / px["NVDA"].shift(21)) - np.log(px["SOX"] / px["SOX"].shift(21))).reindex(idx)
    vxn = px["VXN"].reindex(idx) if "VXN" in px else pd.Series(np.nan, index=idx)
    urise = unrate_rise_on(idx, unrate) if unrate is not None else pd.Series(np.nan, index=idx)

    # ルール（RSI<40 で買い、RSI>=75 で売る）の取引履歴と現在の状態
    rf = (px["IRX"].ffill() / 100 / 252).reindex(idx).fillna(0.0) if "IRX" in px else pd.Series(0.0, index=idx)
    _, _, trades = simulate(c, sox_c, rsi, rsi < BUY, rf, rsi_exit(SELL))
    closed = [t for t in trades if t["exits"] and t.get("exit_i") is not None]
    rets = [t["exits"][-1][1] / t["entry_px"] - 1 for t in closed]
    days = [t["exits"][-1][0] - t["entry_i"] for t in closed]
    holding = bool(trades) and trades[-1].get("exit_i") is None
    last = idx[-1]
    r_now, r_prev = float(rsi.iloc[-1]), float(rsi.iloc[-2])
    below = rsi < BUY
    streak = int(below.astype(int)[::-1].cumprod().sum()) if r_now < BUY else 0

    # 条件 -> (マスク, 今日の値)。信頼度は過去の検証（analysis/results）に基づく注記
    conds = {
        "失業率の上昇が0.4pt未満（景気後退でない）": (urise < 0.4, f"{urise.iloc[-1]:+.1f}pt" if pd.notna(urise.iloc[-1]) else "不明"),
        "NVDAが半導体指数より強い（21日）": (nvda_rel >= 0, f"{nvda_rel.iloc[-1]:+.1%}" if pd.notna(nvda_rel.iloc[-1]) else "不明"),
        "VXN（ナスダック恐怖指数）25以上": (vxn >= 25, f"{vxn.iloc[-1]:.1f}" if pd.notna(vxn.iloc[-1]) else "不明"),
        "半導体指数が1日で-5%以上下落": (sox_1d <= -0.05, f"{sox_1d.iloc[-1]:+.1%}"),
    }
    cond_today = {k: (bool(m.iloc[-1]) if pd.notna(m.iloc[-1]) else None, v) for k, (m, v) in conds.items()}
    buy_stats = {"RSI<40の全日": cond_stats(below, F)}
    for k, (m, _) in conds.items():
        buy_stats[f"RSI<40 + {k}"] = cond_stats(below & m.fillna(False), F)
    combo = below.copy()
    for k, (m, _) in conds.items():
        if cond_today[k][0]:
            combo &= m.fillna(False)
    buy_stats["今日と同じ条件がすべてそろった日"] = cond_stats(combo, F)
    sell_stats = {"RSI≥75の全日（売らずに持っていた場合）": cond_stats(rsi >= SELL, F, ("1か月", "3か月", "6か月"))}

    info = dict(
        date=last, rsi=r_now, rsi_prev=r_prev, soxl=float(c.iloc[-1]), soxl_1d=float(c.pct_change().iloc[-1]),
        sox_1d=float(sox_1d.iloc[-1]), streak=streak, holding=holding,
        entry=(idx[trades[-1]["entry_i"]], float(trades[-1]["entry_px"])) if holding else None,
        buy_signal=(not holding and r_now < BUY), sell_signal=(holding and r_now >= SELL),
        cross_below=(r_now < BUY <= r_prev), cross_above=(r_prev < SELL <= r_now),
        cond_today=cond_today, buy_stats=buy_stats, sell_stats=sell_stats,
        trades=dict(n=len(rets), win=float(np.mean([x > 0 for x in rets])) if rets else None,
                    mean=float(np.mean(rets)) if rets else None, median=float(np.median(rets)) if rets else None,
                    days=float(np.median(days)) if days else None),
    )
    return info


# ---------------------------------------------------------------- メッセージ
# 各条件の信頼度（1996-2009 / 2010-17 / 2018- での検証結果の要約）
RELIABILITY = {
    "失業率の上昇が0.4pt未満（景気後退でない）": "安全装置。古い時期でも有効",
    "NVDAが半導体指数より強い（21日）": "2010年以降の2期間で安定（仮説）",
    "VXN（ナスダック恐怖指数）25以上": "時期で効果が逆転。参考程度",
    "半導体指数が1日で-5%以上下落": "2000年代は逆効果。参考程度",
}
SMALL_MONTHS = 12


def _pct(x, sign=True):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.0%}" if sign else f"{x:.0%}")


def _stat_line(s: dict, h: str) -> str:
    v = s.get(h)
    if not v:
        return f"  {h}: データなし"
    win, mean, med, n = v
    return f"  {h}後: 勝率{win:.0%} 期待値{mean:+.0%}（中央値{med:+.0%}）"


def build_message(info: dict, warnings: list[str], daily: bool) -> str | None:
    d = info["date"]
    # 日報なしの設定では、ルール上の売買が必要な日（買い・売り）だけ送る
    event = info["buy_signal"] or info["sell_signal"]
    if not event and not daily:
        return None
    L = []
    if info["buy_signal"]:
        L.append(f"🟢 買いシグナル（RSI {BUY:.0f}未満）")
    elif info["sell_signal"]:
        L.append(f"🔴 売りシグナル（RSI {SELL:.0f}以上）")
    elif info["cross_below"]:
        L.append(f"🟡 RSIが{BUY:.0f}を下回りました（ルール上は保有中のため新規の買いなし）")
    elif info["cross_above"]:
        L.append(f"🟡 RSIが{SELL:.0f}を上回りました（ルール上はノーポジション）")
    else:
        L.append("📊 SOXL 日報")
    L.append(f"{d:%Y/%m/%d} 米国引け")
    L.append(f"SOX RSI14: {info['rsi']:.1f}（前日 {info['rsi_prev']:.1f}）")
    L.append(f"SOXL: {info['soxl']:.2f}（前日比 {info['soxl_1d']:+.1%}）／半導体指数 {info['sox_1d']:+.1%}")
    if info["streak"]:
        L.append(f"RSI40未満 {info['streak']}日目")
    if info["holding"]:
        ed, ep = info["entry"]
        L.append(f"ルール上: 保有中（{ed:%Y/%m/%d} {ep:.2f}で買い、{info['soxl'] / ep - 1:+.1%}）")
    else:
        L.append("ルール上: ノーポジション")
    if info["buy_signal"]:
        L.append("→ ルール: 翌営業日の引けで買い")
    if info["sell_signal"]:
        L.append("→ ルール: 翌営業日の引けで売り")
    if not event:
        L.append(f"買い基準まで {info['rsi'] - BUY:+.1f}／売り基準まで {SELL - info['rsi']:.1f}")

    if info["buy_signal"] or info["cross_below"] or (daily and info["rsi"] < BUY):
        L.append("")
        L.append("【過去にRSI<40の日に買った場合（2010〜）】")
        s = info["buy_stats"]["RSI<40の全日"]
        for h in ("1か月", "3か月", "1年"):
            L.append(_stat_line(s, h))
        L.append(f"  （3か月以内にさらに下げるのが普通。期待値は平均で、大反発に引っ張られやすい）")
        L.append(f"  底圏（前後3か月の最安値+20%以内）だった割合 {_pct(s['底圏'], False)}（{s['月数']}か月分）")
        L.append("【今日の追加条件】")
        for k, (ok, val) in info["cond_today"].items():
            mark = "✓" if ok else ("✗" if ok is False else "?")
            st = info["buy_stats"][f"RSI<40 + {k}"]
            y = st.get("1年")
            tail = (f"1年勝率{y[0]:.0%}・平均{y[1]:+.0%}/中央値{y[2]:+.0%}・底圏{_pct(st['底圏'], False)}（{st['月数']}か月）"
                    if y else "データなし")
            L.append(f" {mark} {k} [{val}] → {tail}　※{RELIABILITY.get(k, '')}")
        cs = info["buy_stats"]["今日と同じ条件がすべてそろった日"]
        y = cs.get("1年")
        if y and cs["月数"] >= SMALL_MONTHS:
            L.append(f" ◎ 全部そろった日: 1年勝率{y[0]:.0%}・平均{y[1]:+.0%}/中央値{y[2]:+.0%}・底圏{_pct(cs['底圏'], False)}（{cs['月数']}か月）")
        elif y:
            L.append(f" ◎ 全部そろった日: {cs['月数']}か月分しかなく参考程度（1年勝率{y[0]:.0%}・中央値{y[2]:+.0%}）")
        else:
            L.append(" ◎ 全部そろった日: 事例なし")
    if info["sell_signal"] or info["cross_above"]:
        L.append("")
        L.append("【RSI≥75の日以降も持ち続けた場合（2010〜）】")
        s = info["sell_stats"]["RSI≥75の全日（売らずに持っていた場合）"]
        for h in ("1か月", "3か月", "6か月"):
            L.append(_stat_line(s, h))
    t = info["trades"]
    if t["n"]:
        L.append("")
        L.append(f"【RSI40買い→RSI75売りの過去{t['n']}回】勝率{t['win']:.0%} 1回平均{t['mean']:+.0%}"
                 f"（中央値{t['median']:+.0%}）保有中央値{t['days']:.0f}日")
    if warnings:
        L.append("")
        L += [f"⚠ {w}" for w in warnings]
    L.append("※過去データの集計で、将来の結果を保証しません。")
    return "\n".join(L)


def send_line(text: str, token: str, to: str) -> None:
    chunks = [text[i:i + MAX_TEXT] for i in range(0, len(text), MAX_TEXT)][:5]  # 1 回のプッシュは最大 5 メッセージ
    body = {"to": to, "messages": [{"type": "text", "text": ch} for ch in chunks]}
    resp = requests.post(LINE_PUSH_URL, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                         data=json.dumps(body), timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"LINE 送信失敗: HTTP {resp.status_code} {resp.text[:200]}")


def _summary(text: str):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("```\n" + text + "\n```\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="LINE に送らず表示のみ")
    ap.add_argument("--test", action="store_true", help="合図が無くても現在の状況をテスト送信する（設定確認用）")
    a = ap.parse_args(argv)
    token, to = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"), os.environ.get("LINE_USER_ID")
    daily = os.environ.get("NOTIFY_DAILY", "false").lower() == "true"
    dry = a.dry_run or not (token and to)

    def deliver(text: str):
        print(text)
        _summary(text)
        if not dry:
            send_line(text, token, to)

    try:
        px = fetch_market()
    except Exception as e:
        try:
            deliver(f"⚠ SOXL通知: 価格データを取得できませんでした（{e}）")
        except Exception as e2:
            print(f"::error::{e2}")
        return 1
    warnings = []
    try:
        unrate = fetch_unrate()
    except Exception as e:
        unrate = None
        warnings.append(f"失業率（FRED）を取得できず: {e}")
    crit, warn, skip = check_data(px, datetime.now(timezone.utc))
    warnings += warn
    for w in warnings:
        print(f"::warning::{w}")
    if crit:
        deliver("⚠ SOXL通知: データ異常のため本日の判定を止めました\n" + "\n".join(f"・{c}" for c in crit))
        return 1
    if skip:
        print("本日分のデータが無いため通知をスキップ")
        return 0
    info = compute(px, unrate)
    if not (0 <= info["rsi"] <= 100):
        deliver(f"⚠ SOXL通知: RSI の計算値が異常（{info['rsi']}）")
        return 1
    text = build_message(info, warnings, daily or a.test)
    if a.test and text:
        text = "🧪 テスト送信（LINE の設定確認用）\n" + text
    if text:
        try:
            deliver(text)
        except Exception as e:
            print(f"::error::{e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
