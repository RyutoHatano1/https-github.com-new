"""解析全体の実行とレポート生成。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, evaluation, features, models, report
from .data import build_target_price, tracking_report


def run_pipeline(px: pd.DataFrame, ohlcv: dict[str, pd.DataFrame], out: Path, extended: bool = False,
                 holdout_frac: float = 0.3, exec_lag: int = 1, cost_bps: float = 5.0, freq: str = "W",
                 n_perm: int = 5, quick: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    price = build_target_price(px, extended)
    X, groups = features.build_features(px, ohlcv)
    Y = features.build_targets(price)

    # 解析窓: 対象価格があり、252 日の特徴量ウォームアップが済んだ日
    start = max(price.index[0], X["sox_ret_252"].first_valid_index())
    idx = X.index[(X.index >= start) & X.index.isin(price.index)]
    X, Y = X.loc[idx], Y.reindex(idx)
    split = evaluation.split_date(idx, holdout_frac)
    rf = (px["IRX"].ffill() / 100 / 252).reindex(idx).fillna(0.0) if "IRX" in px else pd.Series(0.0, index=idx)

    res: dict = dict(sample_start=idx[0], sample_end=idx[-1], split=split, n_days=len(idx),
                     n_features=X.shape[1], extended=extended)

    # 1. 単変量 IC（探索期間で検定 → 保留期間で再現性）
    ic = evaluation.ic_table(X, Y, split)
    ic.to_csv(out / "ic_table.csv", index=False)
    report.plot_ic_heatmap(ic, out / "fig_ic_heatmap.png")
    res["ic"] = ic

    # 上位ファクターの五分位表（fwd_ret_21）
    q_tabs = {}
    top21 = ic[(ic.target == "fwd_ret_21")].sort_values("p_hac").head(6)["feature"]
    for f in top21:
        q_tabs[f] = {p: evaluation.quantile_table(X.loc[m, f], Y.loc[m, "fwd_ret_21"])
                     for p, m in (("disc", idx < split), ("hold", idx >= split))}
    res["quantiles"] = q_tabs

    # 2. 同時点の要因分解（取引不可・説明用）
    res["contemp"] = evaluation.contemporaneous_decomposition(px.loc[px.index >= idx[0]])
    rb = evaluation.rolling_betas(px.loc[px.index >= idx[0]])
    report.plot_rolling_betas(rb, out / "fig_rolling_betas.png")

    # 3. レバレッジ減価（実 SOXL の期間）
    if px["SOXL"].notna().sum() > 300:
        ld = evaluation.leverage_drag(px)
        report.plot_leverage_drag(ld, out / "fig_leverage_drag.png")
        res["drag"] = ld
        res["tracking"] = tracking_report(px)

    # 4. 季節性・5. レジーム
    res["season"] = evaluation.seasonality(Y["fwd_ret_1"])
    res["regimes"] = evaluation.regime_table(price, X, Y, split)
    res["regimes"].to_csv(out / "regimes.csv", index=False)

    # 6. 多変量 walk-forward
    wf_rows, imps, preds = [], {}, {}
    tasks = [("fwd_ret_21", 21), ("fwd_vol_21", 21)] + ([] if quick else [("fwd_ret_5", 5)])
    for tgt, h in tasks:
        for mname in models.MODELS:
            r = models.walk_forward(X, Y[tgt], h, mname, groups, n_perm=n_perm)
            preds[(tgt, mname)] = r["pred"]
            imps[f"{mname}:{tgt}"] = r["importance"]
            wf_rows.append(dict(target=tgt, model=mname, oos_start=r["pred"].index.min(), n_oos=len(r["pred"]),
                                n_features=len(r["features"]), mean_fold_ic=r["mean_fold_ic"], pooled_ic=r["oos_ic"], oos_r2_vs_zero=r["oos_r2_zero"],
                                oos_r2_vs_train_mean=r["oos_r2_mean"], hit_rate=r["hit_rate"],
                                frac_folds_ic_pos=r["frac_folds_ic_pos"]))
    res["wf"] = pd.DataFrame(wf_rows)
    res["wf_dropped_features"] = sorted(set(X.columns) - set(r["features"]))
    imp = pd.DataFrame(imps)
    res["importance"] = imp
    imp.to_csv(out / "group_importance.csv")
    report.plot_importance(imp, out / "fig_group_importance.png")

    # 7. バックテスト（低頻度ルール）
    ret = price.pct_change().reindex(idx)
    rules = backtest.candidate_rules(X, px["SOX"].reindex(idx), split)
    for mname in models.MODELS:
        p = preds[("fwd_ret_21", mname)].reindex(idx)
        rules[f"model_{mname}_21d"] = ("model", (p > 0).astype(float).where(p.notna()))
    tab, runs = backtest.evaluate_rules(rules, ret, rf, split, freq, exec_lag, cost_bps)
    tab.to_csv(out / "backtest.csv", index=False)
    chosen = sorted(tab.loc[tab.chosen_on_disc, "rule"].unique())
    boots = []
    hold = idx >= split
    bh = runs["buy_hold"]["ret"][hold]
    for n in chosen:
        b = backtest.block_bootstrap_sharpe(runs[n]["ret"][hold], rf, None if n == "buy_hold" else bh)
        boots.append(dict(rule=n, **b))
    res["backtest"], res["boot"] = tab, pd.DataFrame(boots)
    res["n_rule_trials"] = len(rules)
    report.plot_equity(runs, chosen, split, out / "fig_equity.png")

    write_report(res, out, exec_lag, cost_bps, freq)
    return res


def _pct(v):
    return f"{v * 100:.1f}%"


def write_report(res: dict, out: Path, exec_lag: int, cost_bps: float, freq: str):
    L = []
    a = L.append
    a("# SOXL ファクター解析レポート\n")
    a(f"- 解析期間: {res['sample_start']:%Y-%m-%d} 〜 {res['sample_end']:%Y-%m-%d}（{res['n_days']} 営業日）"
      f"{'・上場前は ^SOX から合成した 3 倍系列' if res['extended'] else '・実 SOXL のみ'}")
    a(f"- 探索期間 / 保留期間の境界: **{res['split']:%Y-%m-%d}**（保留期間は検定・パラメータ選択に一切使っていない）")
    a(f"- 特徴量数: {res['n_features']}、バックテスト: リバランス={freq}、約定遅延={exec_lag}日、片道コスト={cost_bps}bp\n")

    ic = res["ic"]
    a("## 1. 単変量の予測力（Spearman IC）\n")
    a("探索期間で Newey-West t 値 → Benjamini-Hochberg FDR 補正。`robust` = FDR<5% かつ保留期間で同符号・"
      "IC が探索期間の半分以上残ったもの。\n")
    a(f"検定数: {len(ic)}、FDR<5%: {(ic.q_bh < 0.05).sum()}、robust: {ic.robust.sum()}\n")
    cols = ["feature", "ic_disc", "t_hac", "q_bh", "ic_hold", "yearly_sign_consistency", "robust"]
    for tgt in ic.target.unique():
        sub = ic[ic.target == tgt].head(12)[cols]
        a(f"### {tgt}（p 値順 上位 12）\n")
        a(md_or_empty(sub))
    a("![IC](fig_ic_heatmap.png)\n")

    a("### 上位ファクターの五分位（fwd_ret_21、対数リターン）\n")
    for f, d in res["quantiles"].items():
        a(f"**{f}**\n")
        tab = pd.concat({k: v[["mean", "hit_rate", "n"]] for k, v in d.items()}, axis=1)
        tab.columns = [f"{p}_{c}" for p, c in tab.columns]
        a(report.md_table(tab, index=True))
        a("")

    a("## 2. 同日の値動きの要因分解（^SOX 日次、説明であって予測ではない）\n")
    for k in ("ex_btc", "all"):
        c = res["contemp"][k]
        a(f"**{k}**: R²={c['r2']:.3f}, n={c['n']}（{c['start']:%Y-%m-%d}〜{c['end']:%Y-%m-%d}）\n")
        a(report.md_table(c["table"]))
        a("")
    a("![betas](fig_rolling_betas.png)\n")

    if "drag" in res:
        d, t = res["drag"], res["tracking"]
        a("## 3. レバレッジ減価（ボラティリティ・ドラッグ）\n")
        a(f"- 日次: SOXL = {d['daily_beta']:.3f} × SOX（R²={d['daily_r2']:.4f}）")
        a(f"- 63 日窓の [log SOXL − 3·log SOX] を窓内 Σr² に回帰: 傾き {d['slope']:.2f} ± {d['slope_se']:.2f}"
          f"（理論値 {d['theory_slope']:.1f}）、切片（年率コスト）{_pct(d['intercept_ann'])}、窓数 {d['n_windows']}")
        a(f"- 合成 3 倍系列との一致: 日次相関 {t['daily_corr']:.4f}、年率トラッキング差 {_pct(t['ann_tracking_diff_log'])}、"
          f"TE {_pct(t['ann_tracking_error'])}\n")
        a("![drag](fig_leverage_drag.png)\n")

    a("## 4. 季節性（翌日リターン）\n")
    for k, (tab, p) in res["season"].items():
        a(f"**{k}**（Kruskal-Wallis p={p:.3f}）\n")
        a(report.md_table(tab, index=True, floatfmt=".2f"))
        a("")

    a("## 5. レジーム別の将来 21 日（閾値は探索期間で決定）\n")
    a(report.md_table(res["regimes"]))
    a("")

    a("## 6. 多変量 walk-forward（拡張窓・h 日パージ・252 日ごと再学習）\n")
    a(report.md_table(res["wf"]))
    a("")
    if res["wf_dropped_features"]:
        a(f"欠損 >10% のためモデルから除外: {', '.join(res['wf_dropped_features'])}\n")
    a("グループ置換重要度（OOS IC の低下量、大きいほど重要）\n")
    a(report.md_table(res["importance"].round(4), index=True, floatfmt=".4f"))
    a("\n![imp](fig_group_importance.png)\n")

    a("## 7. 低頻度ルールのバックテスト\n")
    a(f"試行したルール数: {res['n_rule_trials']}（多重試行による Sharpe の上振れに注意）。"
      "各ファミリーで探索期間 Sharpe 最良を選び、保留期間の成績を見る。\n")
    bt = res["backtest"]
    show = bt[bt.chosen_on_disc][["rule", "family", "period", "CAGR", "vol", "sharpe", "maxDD", "calmar",
                                  "exposure", "trades_per_year"]]
    a(report.md_table(show))
    a("\n保留期間の Sharpe 90% 区間（定常ブロック・ブートストラップ、ブロック 21 日）\n")
    a(report.md_table(res["boot"]))
    a("\n![eq](fig_equity.png)\n")
    a("全ルールの成績は `backtest.csv`。\n")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")

    summary = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in res.items()
               if isinstance(v, (int, float, str, bool, pd.Timestamp))}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def md_or_empty(df: pd.DataFrame) -> str:
    return report.md_table(df) + "\n" if len(df) else "（該当なし）\n"
