"""下落局面での分割買い（価格分割・時間分割）の検証。

実行: cd analysis && python -m soxl_factors.scale_in --out output/scale_in
- きっかけは ^SOX の 52 週高値からの下落率。高値の -5% 以内に戻ったら全て売って現金に戻す（次の下落に備える）。
- 判断は引け、約定は翌営業日の引け。現金には短期金利（13 週 T-bill）が付く。
- SOXL の上場前（2010-03 以前）は ^SOX から合成した 3 倍系列。1 倍は ^SOX から年 0.35% の経費を差し引いた系列。
- プランはすべて事前に固定し、成績を見て選び直していない。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import backtest  # noqa: E402
from .data import load, synthetic_leveraged  # noqa: E402
from .report import md_table  # noqa: E402

RECOVER = -0.05  # 52 週高値の -5% 以内に戻ったら手仕舞い・次の局面を待つ


@dataclass(frozen=True)
class Plan:
    name: str
    asset: str                      # "soxl" or "onex"
    levels: tuple[float, ...] = ()  # 価格分割: 下落率がこの水準に達するたびに 1/len を買う
    trigger: float = -0.20          # 時間分割の開始水準
    n_slices: int = 0               # 時間分割の回数（0 なら価格分割）
    interval: int = 21              # 時間分割の間隔（営業日）


PLANS = [
    Plan("一括(SOXL, -30%で全額)", "soxl", levels=(-0.30,)),
    Plan("価格分割(SOXL, -20/-30/-40/-50%で1/4ずつ)", "soxl", levels=(-0.20, -0.30, -0.40, -0.50)),
    Plan("時間分割(SOXL, -20%から毎月1/4×4回)", "soxl", trigger=-0.20, n_slices=4),
    Plan("価格分割(1倍, -20/-30/-40/-50%で1/4ずつ)", "onex", levels=(-0.20, -0.30, -0.40, -0.50)),
    Plan("時間分割(1倍, -20%から毎月1/4×4回)", "onex", trigger=-0.20, n_slices=4),
]


def build_prices(px: pd.DataFrame) -> pd.DataFrame:
    sox = px["SOX"].dropna()
    lev = synthetic_leveraged(px["SOX"], px["IRX"]).reindex(sox.index)
    actual = px["SOXL"].reindex(sox.index)
    first = actual.first_valid_index()
    soxl = pd.concat([lev[lev.index < first] * actual[first] / lev[first], actual[actual.index >= first]])
    onex = (1 + sox.pct_change().fillna(0.0) - 0.0035 / 252).cumprod()
    rf = (px["IRX"].reindex(sox.index).ffill() / 100 / 252).fillna(0.0)
    dd = sox / sox.rolling(252, min_periods=1).max() - 1
    return pd.DataFrame({"sox": sox, "soxl": soxl, "onex": onex, "rf": rf, "dd": dd})


def simulate(P: pd.DataFrame, plan: Plan, start) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1 単位の資金で開始。日次の資産額・保有比率と、局面（エピソード）ごとの記録を返す。"""
    P = P.loc[start:]
    price, rf, dd = P[plan.asset].to_numpy(), P.rf.to_numpy(), P.dd.to_numpy()
    idx = P.index
    n = len(P)
    cash, units = 1.0, 0.0
    value = np.zeros(n)
    expo = np.zeros(n)
    armed, active = dd[0] > RECOVER, False
    filled: set[float] = set()
    budget = 0.0
    pending_buy, pending_sell = 0.0, False
    schedule: list[int] = []
    episodes, ep = [], None
    for i in range(n):
        if i > 0:
            cash *= 1 + rf[i]
        # 前日の引けで決めた注文を本日の引けで約定
        if pending_sell and units > 0:
            cash += units * price[i]
            units = 0.0
            ep.update(end=idx[i], end_value=cash)
            episodes.append(ep)
            ep, active = None, False
        pending_sell = False
        if pending_buy > 0:
            amt = min(pending_buy, cash)
            units += amt / price[i]
            cash -= amt
            ep["deployed"] += amt / ep["start_value"]
        pending_buy = 0.0
        v = cash + units * price[i]
        value[i], expo[i] = v, units * price[i] / v
        if ep is not None:
            ep["worst"] = min(ep["worst"], v / ep["start_value"] - 1)

        # 本日の引けで判断
        if dd[i] > RECOVER:
            if active and units > 0:
                pending_sell = True
            elif active:  # 何も買えないまま回復（時間分割の途中など）
                ep.update(end=idx[i], end_value=v)
                episodes.append(ep)
                ep, active = None, False
            armed, filled, schedule = True, set(), []
            continue
        if not armed and not active:
            continue
        if plan.n_slices:
            if not active and dd[i] <= plan.trigger:
                active, armed, budget = True, False, v
                schedule = [i + k * plan.interval for k in range(plan.n_slices)]
                ep = dict(start=idx[i], start_dd=dd[i], start_value=v, deployed=0.0, worst=0.0, min_dd=dd[i])
            if active and i in schedule:
                pending_buy = budget / plan.n_slices
        else:
            hit = [lv for lv in plan.levels if dd[i] <= lv and lv not in filled]
            if hit:
                if not active:
                    active, armed, budget = True, False, v
                    ep = dict(start=idx[i], start_dd=dd[i], start_value=v, deployed=0.0, worst=0.0, min_dd=dd[i])
                filled.update(hit)
                pending_buy = budget * len(hit) / len(plan.levels)
        if ep is not None:
            ep["min_dd"] = min(ep["min_dd"], dd[i])
    if ep is not None:  # 現在進行中の局面
        ep.update(end=pd.NaT, end_value=value[-1], open=True)
        episodes.append(ep)
    daily = pd.DataFrame({"value": value, "exposure": expo}, index=idx)
    E = pd.DataFrame(episodes)
    if len(E):
        E["return"] = E.end_value / E.start_value - 1
        E["days"] = [(e - s).days if pd.notna(e) else np.nan for s, e in zip(E.start, E.end)]
        E["open"] = E["open"].fillna(False).astype(bool) if "open" in E else False
    return daily, E


def lump_vs_slices(P: pd.DataFrame, start, horizons=(252, 504), n_slices: int = 4, interval: int = 21) -> pd.DataFrame:
    """毎月 1 日目を開始日として「一括」と「毎月 1/4 × 4 回」を比較（未投入分は短期金利）。h 日後の資金倍率。"""
    P = P.loc[start:]
    rf_idx = (1 + P.rf).cumprod()
    mon = P.index.to_period("M")
    starts = P.index[np.r_[True, mon[1:] != mon[:-1]]][1:]  # 各月の最初の営業日（初月は途中開始のため除外）
    rows = []
    for t0 in starts:
        i0 = P.index.get_loc(t0)
        for h in horizons:
            iend = i0 + 1 + h
            if iend >= len(P):
                continue
            row = dict(start=t0, h=h, dd=P.dd.iloc[i0])
            for asset in ("soxl", "onex"):
                px_ = P[asset].to_numpy()
                row[f"lump_{asset}"] = px_[iend] / px_[i0 + 1] - 1
                val = 0.0
                for k in range(n_slices):
                    ib = i0 + 1 + k * interval
                    # 投入までは現金で金利を受け取り、投入後は資産で運用
                    cash_growth = rf_idx.iloc[ib] / rf_idx.iloc[i0 + 1]
                    val += cash_growth * px_[iend] / px_[ib] / n_slices
                row[f"slices_{asset}"] = val - 1
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame, rf: pd.Series, periods: dict) -> list[dict]:
    r = daily.value.pct_change().fillna(0.0)
    out = []
    for per, (a, b) in periods.items():
        m = (r.index >= a) & (r.index <= b)
        out.append(dict(period=per, **backtest.metrics(r[m], rf), exposure=daily.exposure[m].mean()))
    return out


def run(px, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    P = build_prices(px)
    start = P.index[252]  # 52 週高値が定義できてから
    first_actual = px["SOXL"].first_valid_index()
    periods = {"全期間": (start, P.index[-1]), "合成期(〜2010-03)": (start, first_actual),
               "実SOXL期(2010-03〜)": (first_actual, P.index[-1])}
    rows, eps, curves = [], {}, {}
    for name, col in (("買い持ち(SOXL)", "soxl"), ("買い持ち(1倍)", "onex")):
        v = P.loc[start:, col] / P.loc[start, col]
        d = pd.DataFrame({"value": v, "exposure": 1.0})
        curves[name] = v
        rows += [dict(plan=name, **s) for s in summarize(d, P.rf, periods)]
    for plan in PLANS:
        daily, E = simulate(P, plan, start)
        curves[plan.name] = daily.value
        eps[plan.name] = E
        rows += [dict(plan=plan.name, **s) for s in summarize(daily, P.rf, periods)]
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "summary.csv", index=False)

    ep_rows = []
    for name, E in eps.items():
        if not len(E):
            continue
        closed = E[~E["open"]]
        ep_rows.append(dict(plan=name, n=len(closed), win_rate=(closed["return"] > 0).mean(),
                            median=closed["return"].median(), mean=closed["return"].mean(), worst_episode=closed["return"].min(),
                            worst_interim=E.worst.min(), median_days=closed.days.median()))
        E.to_csv(out / f"episodes_{name.split('(')[0]}_{'soxl' if 'SOXL' in name else 'onex'}.csv", index=False)
    ep_sum = pd.DataFrame(ep_rows)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, v in curves.items():
        ax.semilogy(v.index, v, lw=1, label=name)
    ax.axvline(first_actual, color="k", ls="--", lw=0.7)
    ax.set_ylabel("value of 1 unit (log)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_scale_in.png", dpi=130)
    plt.close(fig)

    lv = lump_vs_slices(P, start)
    lv.to_csv(out / "lump_vs_slices.csv", index=False)
    lrows = []
    for cond, m in (("全ての月", lv.dd <= 1), ("SOX が高値から-20%以上下落中に開始", lv.dd <= -0.20)):
        for h in (252, 504):
            d = lv[m & (lv.h == h)]
            for k in ("lump_soxl", "slices_soxl", "lump_onex", "slices_onex"):
                lrows.append(dict(start_condition=cond, years=h // 252, method=k, n_starts=len(d), median=d[k].median(),
                                  p10=d[k].quantile(0.10), worst=d[k].min(), loss_prob=(d[k] < 0).mean(),
                                  big_loss_prob=(d[k] < -0.5).mean()))
            for a in ("soxl", "onex"):
                lrows.append(dict(start_condition=cond, years=h // 252, method=f"分割が一括に勝った割合({a})",
                                  n_starts=len(d), median=(d[f"slices_{a}"] > d[f"lump_{a}"]).mean()))
    lsum = pd.DataFrame(lrows)
    res = dict(tab=tab, ep_sum=ep_sum, lsum=lsum, eps=eps, start=start, end=P.index[-1], first_actual=first_actual)
    write_report(res, out)
    return res


def write_report(res: dict, out: Path):
    L = ["# 分割買いの検証\n",
         f"- 期間 {res['start']:%Y-%m-%d}〜{res['end']:%Y-%m-%d}（{res['first_actual']:%Y-%m} 以前の SOXL は合成）",
         "- ^SOX の 52 週高値からの下落率で買い、高値の -5% 以内に戻ったら翌日に全て売る。待機中の現金は短期金利で運用",
         "- 判断は引け、約定は翌営業日の引け。売買コストは含めていない（回数が少ないため影響は小さい）\n",
         "## 全体成績\n", md_table(res["tab"][["plan", "period", "CAGR", "vol", "sharpe", "maxDD", "exposure"]]), "",
         "## 局面ごとの成績（買い始め → 回復して売るまで、資金全体に対するリターン）\n", md_table(res["ep_sum"]), ""]
    L += ["## 一括 vs 時間分割（毎月 1/4 × 4 回）: 任意の月に始めた場合の 1・2 年後\n",
          "lump = 一括、slices = 分割。median = 中央値、p10 = 悪い方から 10% の値、loss_prob = 損失確率、big_loss_prob = 半分以上失う確率\n",
          md_table(res["lsum"]), ""]
    for name, E in res["eps"].items():
        if len(E):
            L += [f"### {name}\n", md_table(E[["start", "start_dd", "min_dd", "deployed", "end", "days", "return", "worst", "open"]]), ""]
    L.append("![curves](fig_scale_in.png)")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/scale_in"))
    ap.add_argument("--cache", type=Path, default=Path("data_cache"))
    a = ap.parse_args()
    px, _ = load(a.cache)
    run(px, a.out)
    print(f"レポート: {a.out / 'report.md'}")


if __name__ == "__main__":
    main()
