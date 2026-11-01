"""LINE 通知の判定・異常検知・送信形式のテスト（ネットワーク不要）。"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from soxl_factors import notify


def market(n=3000, seed=3, end="2026-10-30"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    r = rng.normal(0.0006, 0.018, n)
    sox = 1000 * np.exp(np.cumsum(r))
    soxl = 10 * np.exp(np.cumsum(3 * r - 0.0001))
    nvda = 50 * np.exp(np.cumsum(r + rng.normal(0, 0.01, n)))
    return pd.DataFrame({"SOX": sox, "SOXL": soxl, "NVDA": nvda, "VXN": 25 + rng.normal(0, 3, n),
                         "VIX": 20 + rng.normal(0, 3, n), "IRX": 2.0}, index=idx)


def unrate():
    return pd.Series(4.0, index=pd.date_range("2005-01-01", "2026-09-01", freq="MS"))


def after_close(day):
    return datetime(day.year, day.month, day.day, 22, 17, tzinfo=timezone.utc)


def test_clean_data_passes():
    px = market()
    crit, warn, skip = notify.check_data(px, after_close(px.index[-1]))
    assert not crit and not skip


def test_holiday_skips_and_stale_is_critical():
    px = market()
    last = px.index[-1]
    _, _, skip = notify.check_data(px, after_close(last + pd.offsets.BDay(1)))
    assert skip
    crit, _, _ = notify.check_data(px, after_close(last + pd.offsets.BDay(4)))
    assert crit and "古い" in crit[0]


def test_abnormal_jump_is_critical():
    px = market()
    px.iloc[-1, px.columns.get_loc("SOXL")] *= 3  # 分割の取り違えのような跳ね
    crit, _, _ = notify.check_data(px, after_close(px.index[-1]))
    assert any("前日比" in c for c in crit)


def test_signals_and_message():
    px = market()
    info = notify.compute(px, unrate())
    assert 0 <= info["rsi"] <= 100
    # 買い/売りシグナルは保有状態と RSI に整合する
    assert info["buy_signal"] == (not info["holding"] and info["rsi"] < notify.BUY)
    assert info["sell_signal"] == (info["holding"] and info["rsi"] >= notify.SELL)
    msg = notify.build_message(info, ["テスト警告"], daily=True)
    assert msg and "SOX RSI14" in msg and "⚠ テスト警告" in msg and len(msg) < 5000
    # 日報なし設定では、売買が必要な日以外（RSI の通過だけの日を含む）は送らない
    quiet = dict(info, buy_signal=False, sell_signal=False, cross_below=True, cross_above=False)
    assert notify.build_message(quiet, [], daily=False) is None
    buy = dict(info, buy_signal=True, sell_signal=False, holding=False, entry=None)
    assert notify.build_message(buy, [], daily=False).startswith("🟢")


def test_forced_buy_message_has_stats():
    px = market()
    # 最後の 6 日を急落させて RSI を 40 未満にする
    px.iloc[-6:, px.columns.get_loc("SOX")] *= np.linspace(0.97, 0.80, 6)
    px.iloc[-6:, px.columns.get_loc("SOXL")] *= np.linspace(0.91, 0.50, 6)
    info = notify.compute(px, unrate())
    assert info["rsi"] < notify.BUY
    # 保有中で RSI が前日から 40 未満ならイベントではないが、日報には過去の勝率を載せる
    msg = notify.build_message(info, [], daily=True)
    assert "【過去にRSI<40の日に買った場合" in msg and "勝率" in msg


def test_send_line_payload(monkeypatch):
    sent = {}

    class Resp:
        status_code = 200
        text = "{}"

    def fake_post(url, headers, data, timeout):
        sent.update(url=url, headers=headers, data=data)
        return Resp()

    monkeypatch.setattr(notify.requests, "post", fake_post)
    notify.send_line("あ" * 6000, "TOKEN", "U123")
    import json
    body = json.loads(sent["data"])
    assert sent["url"] == notify.LINE_PUSH_URL and sent["headers"]["Authorization"] == "Bearer TOKEN"
    assert body["to"] == "U123" and len(body["messages"]) == 2 and all(m["type"] == "text" for m in body["messages"])


def test_send_line_failure_raises(monkeypatch):
    class Resp:
        status_code = 401
        text = "invalid token"

    monkeypatch.setattr(notify.requests, "post", lambda *a, **k: Resp())
    with pytest.raises(RuntimeError):
        notify.send_line("x", "T", "U")


def test_test_flag_forces_message(monkeypatch, capsys):
    px = market()
    monkeypatch.setattr(notify, "fetch_market", lambda: px)
    monkeypatch.setattr(notify, "fetch_unrate", unrate)
    monkeypatch.setattr(notify, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: after_close(px.index[-1]))}))
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
    assert notify.main(["--test"]) == 0
    assert "🧪 テスト送信" in capsys.readouterr().out
