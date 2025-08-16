# jpgovsummary

政府の会議資料を要約するツール

## 概要

jpgovsummaryは、政府の会議資料（gov.jp）を自動的に要約し、関連資料を抽出するツールです。HTMLやPDFの会議資料を解析し、以下の情報を抽出します：

- 会議の名称と回数
- 会議の要約
- 関連資料の一覧

## 機能

- HTML/PDFの会議資料の読み込み
- メインコンテンツの抽出
- 会議の要約生成
- 関連資料の列挙と分類

## インストール

```bash
pip install jpgovsummary
```

## 使用方法

```bash
jpgovsummary <会議のURL>
```

### オプション

- `--model`: 使用するOpenAIモデルを指定

## 開発環境のセットアップ

1. リポジトリのクローン
```bash
git clone https://github.com/yourusername/jpgovsummary.git
cd jpgovsummary
```

2. 依存関係のインストール
```bash
poetry install
```

3. 開発用コマンドの実行
```bash
poetry run jpgovsummary
```

## ライセンス

MIT License
