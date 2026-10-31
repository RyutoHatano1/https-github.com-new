# SOXL ファクター解析

SOXL（半導体指数 ^SOX の日次 3 倍ブル ETF）の値動きに、どのファクターが関わるかを調べるパイプライン。
低頻度（週次〜月次）の売買を前提にしている。

## 実行

```bash
cd analysis
pip install -r requirements.txt
python -m soxl_factors --out output                # 実 SOXL の期間（2010〜）
python -m soxl_factors --out output --extended     # 上場前を ^SOX から合成した 3 倍系列で延長（ドットコム崩壊・2008 年を含む）
python -m pytest -q tests                          # 合成データで先読み・既知信号の回収を検証
python -m soxl_factors.reversal --target SOXL      # 底・天井シグナル（連続下落・RSI・IBS 等）の検証。--target SOX で 1995 年〜
python -m soxl_factors.cycle                       # 中期の底・天井（景気・半導体サイクル・金利/信用・センチメント指標）
python -m soxl_factors.scale_in                    # 下落時の分割買い（価格分割・時間分割、SOXL と 1 倍の比較）
python -m soxl_factors.panic                       # パニック局面（VIX≥30 python -m soxl_factors.scale_in                    # 下落時の分割買い（価格分割・時間分割、SOXL と 1 倍の比較） SOX −30%）で反発した局面と崩れた局面の違い
python -m soxl_factors.scatter [--extended]          # 恐怖指数・失業率・RSI・移動平均と SOXL の日次散布図（--extended で 1996 年〜合成 SOXL）
python -m soxl_factors.vxn_rsi                     # 2010 年〜: VXN × RSI と前後 3 か月の底圏（scatter --extended の後に実行）
python -m soxl_factors.bottom_screen               # 2010 年〜: 底圏とそれ以外を分ける指標の総当たりと RSI<40 への上乗せ
python -m soxl_factors.buy_patterns [--reset 40] [--unrate-max 0.4] [--extended]  # RSI<40（+失業率）をきっかけにした購入パターンの平均購入単価比較
python -m soxl_factors.rsi_compare                 # RSI の種類（対象・計算法・期間）の比較と組み合わせルールの検証
python -m soxl_factors.exit_rules                  # 売り方の比較（RSI 水準・保有期間・トレーリングストップ・移動平均割れ・利益確定）
python -m soxl_factors.rsi_swing_vs_hold           # RSI40 で買い RSI75 で売る（複利） vs 持ち続け
python -m soxl_factors.exit_compare                # 売り方 RSI75 / +50% / 早い方 の比較（複利・税引後）
```

出力（`output/actual/` または `output/extended/`）: `report.md`, `ic_table.csv`, `backtest.csv`,
`group_importance.csv`, `regimes.csv`, 図 (`fig_*.png`)。
データは Yahoo Finance（yfinance）と FRED から取得し、`data_cache/` に CSV でキャッシュする（再取得は `--refresh`）。

## 何を計算するか

| 区分 | 内容 |
|---|---|
| ファクター（約 85） | モメンタム（1〜252 日, 12-1）、平均回帰（RSI, SMA 乖離, BB, 52 週高値からの下落率）、トレンド（SMA200, ゴールデンクロス）、実現ボラ（5/21/63 日, Parkinson, 下方ボラ, 歪度・尖度）、インプライド・ボラ（VIX, VXN, 期間構造, VRP）、出来高、相対強度（vs QQQ/SPY/IWM）、ブレッドス（主要 7 銘柄の SMA50 上回り比率・分散）、ベータ/相関レジーム、金利（10 年, 10y-3m, 実質金利, BEI）、クレジット（HYG/IEF, HY OAS, NFCI）、為替（DXY, 円, ウォン, 台湾ドル）、コモディティ（銅/金, 原油）、BTC、アジア指数（台湾・韓国・日本）、失業保険申請、カレンダー（曜日, 月, 月替わり, SQ 週） |
| ターゲット | 将来 1/5/21/63 日の対数リターン、将来 21 日の実現ボラ |
| 単変量 | Spearman IC、Newey-West t 値（重複リターン対策）、BH-FDR、年別の符号一貫性、保留期間での再現 |
| 同時点 | ^SOX 日次の要因分解（標準化ベータ、単独 R²、drop-one ΔR²）、252 日ローリング・ベータ |
| 構造 | レバレッジ減価（実測の傾きと理論値 −3 の比較）、合成 3 倍系列と実 SOXL の一致度 |
| 多変量 | Ridge（時系列内側検証で正則化）と勾配ブースティング、拡張窓 walk-forward、h 日パージ、グループ置換重要度 |
| 売買ルール | トレンド・ボラフィルター・ボラターゲット・モデル予測。週次リバランス、翌日約定、コスト込み。探索期間で選択し保留期間で評価、ブロック・ブートストラップの Sharpe 区間 |

## 先読み・過剰適合への対策

- 日付 t の特徴量は t の米国引けまでに観測可能な情報のみ。FRED は公表ラグ分ずらす。テスト `test_no_lookahead` が、途中で打ち切ったデータで作った特徴量と全データ版の一致を確認する。
- 系列の後ろ 30%（既定）を保留期間とし、検定・閾値・パラメータ選択には使わない。
- 試行したルール数をレポートに明記する。

## 限界

- 価格・マクロデータのみを使う。半導体の受注・在庫サイクル、決算日、FOMC 日程などのイベントは含まない。
- ^SOX は価格指数なので、合成 3 倍系列には配当が入らない。
- 過去の関係が将来も続く保証はない。これは研究用の解析で、投資助言ではない。
