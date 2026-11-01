# SOXL RSI の LINE 通知の設定

毎営業日、米国の引け後（日本時間 7:17 頃）に GitHub Actions が次の処理を行います。

1. 半導体指数 ^SOX の RSI(14)（ワイルダー方式）を計算する。
2. ルール（RSI<40 で買い、RSI≥75 で売る）の判定をする。
3. 過去の勝率と期待値を添えて LINE に送る。

LINE Notify は 2025 年 3 月 31 日に終了したため、LINE 公式アカウントの **Messaging API**（プッシュメッセージ）を使います。
無料のコミュニケーションプランでは、月 200 通まで送れます（売買の日だけなら年に数十通程度）。

## 1. LINE 側（一度だけ）

LINE の画面構成は変わることがあるので、見つからない場合は公式の案内を確認してください。

1. LINE 公式アカウントを作る（LINE Official Account Manager、無料プラン）。
2. 公式アカウントの設定で **Messaging API を有効にする**。プロバイダーは新規作成で構いません。
3. LINE Developers コンソールで、そのチャネルの「Messaging API 設定」を開き、**チャネルアクセストークン（長期）** を発行する。
4. 同じチャネルの「チャネル基本設定」で、**あなたのユーザー ID**（`U` で始まる文字列）を控える。
5. 自分の LINE で、その公式アカウントを**友だち追加**する。QR コードは「Messaging API 設定」にあります。
6. （おすすめ）LINE Official Account Manager の「応答設定」で、あいさつメッセージと応答メッセージをオフにする。オンのままだと、こちらから送ったときに定型文が返ってくることがあります。

## 2. GitHub 側

1. リポジトリの **Settings → Secrets and variables → Actions** を開き、次の 2 つを登録する。
   - `LINE_CHANNEL_ACCESS_TOKEN`: 手順 3 のトークン
   - `LINE_USER_ID`: 手順 4 のユーザー ID
2. 既定では、**ルール上の買い（🟢）・売り（🔴）が出た日と、異常の警告だけ**を送ります。毎日の日報も欲しい場合は、同じ画面の **Variables** に `NOTIFY_DAILY=true` を入れてください。
3. ワークフローは **既定ブランチ（master）にマージされてから**動きます。これは GitHub の定時実行（schedule）の仕様です。
4. 動作確認は Actions タブ →「SOXL RSI LINE通知」→ **Run workflow** で行います。
   - `test_send` にチェックを入れると、売買の合図が無い日でも現在の状況を **LINE にテスト送信**します。設定が正しいかの確認に使ってください。
   - `dry_run` にチェックを入れると、送らずに結果だけ表示します。

## 異常時の動き

- **送信を止めて警告し、ワークフローを失敗にするケース:** データが 3 営業日以上古い、欠損、前日比の異常（分割の取り違えなど）、RSI の計算値の異常、LINE 送信の失敗。このときは LINE に「⚠」の警告を送ります。
- **送らずに正常終了するケース:** 休場日や、データがまだ更新されていない日。
- **警告を添えて通常どおり送るケース:** 失業率・NVDA・VXN が取れない、SOXL と 3×SOX のズレが大きい。

## 手元での確認

```bash
cd analysis
pip install -r requirements.txt
python -m soxl_factors.notify --dry-run   # LINE に送らず本文を表示
python -m pytest -q tests/test_notify.py
```
