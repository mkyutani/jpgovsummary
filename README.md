# jpgovsummary

政府の会議資料を要約するツール

## 概要

jpgovsummaryは、政府の会議資料（gov.jp）を自動的に要約し、関連資料を抽出するツールです。HTMLやPDFの会議資料を解析し、AI技術を活用して以下の情報を抽出・生成します：

- 会議の名称と回数
- 会議の要約
- 関連資料の一覧と内容要約
- 統合された最終要約

## 機能

- **多様な文書形式のサポート**: HTML/PDFの会議資料の読み込み
- **メインコンテンツの自動抽出**: ノイズを除去した純粋な会議内容の抽出
- **AI要約生成**: OpenAI APIを使用した高品質な要約生成
- **関連資料の自動発見**: 会議ページから関連PDFや文書の自動検出
- **文書タイプ別処理**: PowerPoint、従来の会議資料、参加者情報など文書タイプに応じた最適化された要約
- **統合要約機能**: 複数の関連資料を統合した包括的な要約
- **バッチ処理**: 人的介入なしでの自動処理モード
- **Bluesky連携**: 生成された要約のSNS投稿機能（オプション）

## 技術スタック

- **Python 3.12+**
- **LangChain**: AI エージェント フレームワーク
- **LangGraph**: ワークフロー管理
- **OpenAI API**: テキスト生成・要約
- **Requests**: HTTPリクエスト
- **BeautifulSoup4**: HTML解析
- **MarkItDown**: 文書変換
- **PyPDF2**: PDF処理

## インストール

### PyPIからのインストール（予定）
```bash
pip install jpgovsummary
```

### 開発版のインストール
```bash
git clone https://github.com/mkyutani/jpgovsummary.git
cd jpgovsummary
poetry install
```

## 環境設定

### 必須の環境変数

```bash
# OpenAI API キー（必須）
export OPENAI_API_KEY="your-openai-api-key"
```

### オプションの環境変数

```bash
# Bluesky設定（SNS投稿機能を使用する場合のみ）
export SSKY_USER="your-handle.bsky.social:your-app-password"
```

`.env`ファイルでの設定も可能です。

## 使用方法

### 基本的な使用法

```bash
# 政府会議ページのURL指定
jpgovsummary https://www.kantei.go.jp/jp/singi/example/

# ローカルPDFファイルの処理
jpgovsummary /path/to/document.pdf
```

### オプション

```bash
jpgovsummary <URL_or_FILE_PATH> [options]
```

- `--model MODEL`: 使用するOpenAIモデルを指定（デフォルト: gpt-4o-mini）
- `--batch`: バッチモード（人的介入なし）で実行
- `--skip-bluesky-posting`: Bluesky投稿をスキップ
- `--overview-only`: 概要のみを生成（関連文書処理なし）

### 使用例

```bash
# バッチモードで実行（自動処理）
jpgovsummary https://www.kantei.go.jp/jp/singi/example/ --batch

# より高性能なモデルを使用
jpgovsummary document.pdf --model gpt-4o

# 概要のみを素早く生成
jpgovsummary https://www.kantei.go.jp/jp/singi/example/ --overview-only
```

### Devcontainer環境での使用

Devcontainer外からバッチモードで実行するための便利なスクリプトを提供しています:

```bash
# スクリプトの使用
./scripts/batch-summary.sh <URL_or_FILE_PATH>

# 例
./scripts/batch-summary.sh https://www.kantei.go.jp/jp/singi/example/
./scripts/batch-summary.sh /path/to/document.pdf
```

このスクリプトは `--batch` フラグ付きで実行されるため、人的介入なしで自動的に処理され、Blueskyにも自動投稿されます。

## 出力例

ツールの最終出力は非常にシンプルです：

```
第5回デジタル社会推進会議では、デジタル田園都市国家構想の推進について議論された。主な議題は地方のDX推進、デジタル人材の育成、オンライン行政サービスの拡充であった。特に、中小企業のデジタル化支援と高齢者向けデジタルサービスの普及が重点課題として挙げられた。
https://www.kantei.go.jp/jp/singi/it2/dgov/dai5/
```

**出力形式**:
- 1行目: 生成された要約文
- 2行目: 処理対象のURL

**処理中のログ**（標準エラー出力）では以下のような情報が表示されます：
- メインコンテンツ抽出の進捗
- 関連資料の発見と選択
- 各文書の要約生成
- 統合要約の作成
- 人間レビューのインタラクション

## アーキテクチャ

jpgovsummaryは以下のコンポーネントで構成されています：

1. **メインコンテンツ抽出器**: HTML/PDFからメインコンテンツを抽出
2. **概要生成器**: 基本的な会議概要を生成
3. **資料列挙器**: 関連資料を自動発見
4. **資料選択器**: 重要度に基づいて処理対象を選択
5. **文書要約器**: 個別文書の詳細要約
6. **要約統合器**: 複数要約の統合
7. **要約最終化器**: 最終レビューと調整
8. **Bluesky投稿器**: SNS投稿機能

## 開発

### 開発環境のセットアップ

1. リポジトリのクローン
```bash
git clone https://github.com/mkyutani/jpgovsummary.git
cd jpgovsummary
```

2. 依存関係のインストール
```bash
poetry install
```

3. 開発用実行
```bash
poetry run jpgovsummary
```

### コード品質

プロジェクトでは以下のツールを使用してコード品質を維持しています：

- **Ruff**: リンティングとフォーマッティング

```bash
# リンティング実行
poetry run ruff check

# フォーマッティング実行
poetry run ruff format
```

## 貢献

プルリクエストや Issues は歓迎します。大きな変更を行う前に、Issue で議論することをお勧めします。

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 作者

- Miki Yutani (mkyutani@gmail.com)

## リポジトリ

- GitHub: https://github.com/mkyutani/jpgovsummary
- Keywords: government, japan, summary
