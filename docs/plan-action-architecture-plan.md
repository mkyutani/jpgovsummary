# jpgovsummary アーキテクチャ変更プラン: StateベースからPlan-Action/ReActへの移行

## エグゼクティブサマリー

現在のStateベース（TypedDict + 23フィールド）のLangGraphワークフローを、**Plan-Action/ReActパターン**に移行し、コンテキスト消費の大きい操作を**サブエージェント**に分離します。

### 主要目標
1. **コンテキスト管理の最適化**: Stateの肥大化を防ぎ、必要な情報のみを保持
2. **サブエージェント化**: PDF要約などの重い処理を独立したコンテキストで実行
3. **LangChain最新機能の活用**: 0.2.74で利用可能な機能を最大限活用

### 期待される効果
- **トークン使用量**: 35-50%削減（1文書あたり6K-8Kトークン削減）
- **処理速度**: 並列サブエージェント実行により向上
- **保守性**: モジュール化による明確な責任分離

---

## 現状分析サマリー

### 現在のアーキテクチャの課題

**1. State肥大化の問題**
- 23個のフィールドを持つ巨大なTypedDict
- `messages` リストが無制限に成長（7エージェント × 3メッセージ = 21+メッセージ）
- `target_report_summaries` が文書数に応じて線形増加
- すべてのエージェントが全Stateにアクセス可能（不要な結合）

**2. コンテキスト消費の分析**

| 操作 | トークン/実行 | 優先度 | 現在の問題 |
|------|--------------|--------|-----------|
| PowerPoint分析 | 2000-3000 | 最高 | 複数LLM呼び出し、20ページバッチ処理 |
| Document Type検出 | 1000-1500 | 高 | 全文書で実行、7カテゴリ分類 |
| Overview生成 | 1500-2500 | 中 | 169行の巨大プロンプト |
| HTML抽出 | 1000-2000 | 中 | リトライ時に2倍のコスト |

**合計**: 1文書あたり11K-17Kトークン（複数文書で累積）

**3. 現在のLangChain使用状況**
- バージョン: langgraph 0.2.74（Plan-Action/ReActに必要な機能は利用可能）
- 使用中: StateGraph, add_messages, conditional_edges, MemorySaver
- 未使用: @tool decorator, bind_tools(), sub-graphs, parallel execution

---

## 提案アーキテクチャ

### アプローチ1: Plan-Actionパターン（推奨）

#### コンセプト
**明示的なPlanningフェーズ**を導入し、実行計画を作成してからアクションを実行。Stateは最小限の制御情報のみを保持。

#### アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────┐
│                     MAIN WORKFLOW GRAPH                      │
│                                                              │
│  START → [input_analyzer] → [action_planner] → [executor]  │
│                                ↓                    ↓        │
│                          [plan_state]        [action_state]  │
│                                                      ↓        │
│                                              [sub-agent]     │
│                                                   ↓          │
│                                              [integrator]    │
│                                                   ↓          │
│                                              [finalizer]     │
└─────────────────────────────────────────────────────────────┘

SUB-AGENTS (独立コンテキスト):
┌──────────────────────┐  ┌──────────────────────┐
│ DocumentTypeDetector │  │ PowerPointSummarizer │
│  - 入力: PDF pages   │  │  - 入力: PDF pages   │
│  - 出力: type        │  │  - 出力: summary     │
└──────────────────────┘  └──────────────────────┘
┌──────────────────────┐  ┌──────────────────────┐
│   WordSummarizer     │  │   HTMLProcessor      │
│  - 入力: PDF pages   │  │  - 入力: HTML bytes  │
│  - 出力: summary     │  │  - 出力: markdown    │
└──────────────────────┘  └──────────────────────┘
```

#### 新しいState設計（最小化）

```python
class PlanState(TypedDict):
    """Planningフェーズの状態（軽量）"""
    input_url: str
    input_type: Literal["html_meeting", "pdf_file"]
    overview: str  # 生成された概要
    discovered_documents: list[str]  # URL リスト
    action_plan: ActionPlan  # 実行計画（構造化）

class ActionPlan(BaseModel):
    """実行計画の構造"""
    steps: list[ActionStep]
    reasoning: str

class ActionStep(BaseModel):
    """個別のアクションステップ"""
    action_type: Literal["summarize_pdf", "extract_html", "integrate"]
    target: str  # URL or identifier
    params: dict[str, Any]
    priority: int

class ExecutionState(TypedDict):
    """実行フェーズの状態"""
    plan: ActionPlan
    completed_actions: list[CompletedAction]
    current_step_index: int
    results: dict[str, Any]  # アクション結果の格納

class CompletedAction(BaseModel):
    """完了したアクションの記録"""
    step: ActionStep
    result: Any
    tokens_used: int
    timestamp: str
```

#### ワークフローフロー

**Phase 1: Planning（計画生成）**
```
1. input_analyzer
   - 入力タイプ判定（HTML meeting or PDF file）
   - 軽量分析のみ実行

2. action_planner
   - HTML meeting の場合:
     a. HTMLProcessor sub-agentを呼び出し（markdown変換）
     b. 関連文書を発見
     c. 優先度付け
     d. ActionPlan生成（どの文書を要約するか）

   - PDF file の場合:
     a. 単一文書要約のActionPlan生成
```

**Phase 2: Execution（実行）**
```
3. executor
   - ActionPlanの各ステップを実行
   - Sub-agentを並列/順次呼び出し
   - 結果をExecutionStateに蓄積（軽量）

   ステップ例:
   - DocumentTypeDetector sub-agent → 文書タイプ判定
   - PowerPointSummarizer sub-agent → PPT要約（重い処理を隔離）
   - WordSummarizer sub-agent → Word文書要約
```

**Phase 3: Integration（統合）**
```
4. integrator
   - 全サブエージェント結果を統合
   - 最終サマリー生成

5. finalizer
   - 文字数制限チェック
   - ユーザーレビュー（batch modeでスキップ）
   - Bluesky投稿（オプション）
```

#### サブエージェントの実装

**1. PowerPointSummarizer Sub-Agent（最優先）**

```python
class PowerPointSummarizerGraph:
    """PowerPoint要約専用のサブグラフ"""

    def __init__(self):
        self.graph = StateGraph(PowerPointState)
        self.graph.add_node("extract_title", self._extract_title)
        self.graph.add_node("score_slides", self._score_slides)
        self.graph.add_node("select_content", self._select_content)
        self.graph.add_node("generate_summary", self._generate_summary)
        # 独立したコンテキストで実行

    class PowerPointState(TypedDict):
        pdf_pages: list[str]  # 入力のみ
        title: str
        scored_slides: list[ScoredSlide]
        selected_content: str
        summary: str  # 出力のみ
```

**メリット**:
- メインワークフローから2000-3000トークン削減
- 並列実行可能（複数PPT文書を同時処理）
- エラー隔離（PPT処理失敗してもメインフローに影響しない）

**2. DocumentTypeDetector Sub-Agent**

```python
class DocumentTypeDetectorGraph:
    """文書タイプ検出専用サブグラフ"""

    class DetectorState(TypedDict):
        pdf_pages: list[str]  # 最初の10ページのみ
        document_type: str
        confidence_scores: dict[str, float]
        detection_detail: str
```

**メリット**:
- 1000-1500トークン削減
- 検出ロジックの独立性向上
- 再利用性（他のワークフローでも使用可能）

**3. HTMLProcessor Sub-Agent**

```python
class HTMLProcessorGraph:
    """HTML処理専用サブグラフ"""

    class HTMLState(TypedDict):
        html_bytes: bytes
        url: str
        markdown: str
        main_content: str  # ヘッダー/フッター除去済み
```

#### LangChain機能の活用

**利用可能な機能（0.2.74）**:

1. **Sub-graph composition**
```python
# サブグラフの定義
ppt_summarizer = PowerPointSummarizerGraph().graph.compile()

# メイングラフに統合
main_graph.add_node(
    "summarize_powerpoint",
    lambda state: ppt_summarizer.invoke({"pdf_pages": state["pdf_pages"]})
)
```

2. **並列実行**
```python
# 複数文書を並列処理
async def process_documents_parallel(documents: list[str]):
    tasks = [
        ppt_summarizer.ainvoke({"pdf_pages": load_pdf(url)})
        for url in documents
    ]
    results = await asyncio.gather(*tasks)
    return results
```

3. **Tool formalization**
```python
from langchain_core.tools import tool

@tool
def load_pdf_document(url: str) -> list[str]:
    """Load PDF and extract pages as text."""
    return load_pdf_as_text(url)

@tool
def detect_document_type(pages: list[str]) -> str:
    """Detect document type from PDF pages."""
    # Sub-agentを呼び出し
    return detector_graph.invoke({"pdf_pages": pages})
```

---

### アプローチ2: ReActパターン（代替案）

#### コンセプト
エージェントが**Reasoning → Action → Observation**ループを実行。各ステップで次のアクションを動的に決定。

#### アーキテクチャ図

```
┌────────────────────────────────────────┐
│         REACT LOOP WORKFLOW            │
│                                        │
│  START → [reasoner] ⟲                 │
│              ↓                         │
│         [thought]                      │
│              ↓                         │
│         [action_selector]              │
│              ↓                         │
│         [tool_executor]                │
│              ↓                         │
│         [observation]                  │
│              ↓                         │
│         [should_continue?]             │
│          ↙         ↘                   │
│    YES (loop)    NO (終了)            │
└────────────────────────────────────────┘
```

#### State設計

```python
class ReActState(TypedDict):
    """ReAct パターンの状態"""
    input: str  # 元のURL/ファイルパス
    thoughts: list[Thought]  # 思考の履歴
    actions: list[Action]  # 実行したアクション
    observations: list[Observation]  # 観察結果
    final_answer: str | None

class Thought(BaseModel):
    reasoning: str
    next_action: str

class Action(BaseModel):
    tool_name: str
    params: dict[str, Any]

class Observation(BaseModel):
    result: Any
    reflection: str
```

#### メリット
- 動的な意思決定（計画の柔軟な変更）
- エラー対応の自動化（失敗時の再計画）
- LLMの推論能力を最大活用

#### デメリット
- トークン消費が多い（各ループでLLM呼び出し）
- 実行時間が長くなる可能性
- デバッグが困難（非決定的な動作）

---

## 推奨実装プラン: **Plan-Action パターン**

### 理由

| 観点 | Plan-Action | ReAct |
|------|-------------|-------|
| トークン効率 | ⭐⭐⭐⭐⭐ 事前計画で無駄削減 | ⭐⭐⭐ ループで消費増加 |
| 実行速度 | ⭐⭐⭐⭐⭐ 並列実行可能 | ⭐⭐ 順次実行が基本 |
| 予測可能性 | ⭐⭐⭐⭐⭐ 明確な計画 | ⭐⭐ 非決定的 |
| エラー処理 | ⭐⭐⭐⭐ サブエージェント隔離 | ⭐⭐⭐⭐ 自動リトライ |
| 保守性 | ⭐⭐⭐⭐⭐ モジュール化 | ⭐⭐⭐ 複雑なループ |

**結論**: jpgovsummaryのユースケース（バッチ処理、予測可能な文書要約）には**Plan-Actionが最適**

---

## 段階的移行ロードマップ

### Phase 1: 基礎準備（1-2週間）

**目標**: 既存コードを壊さずに新機能を追加

**実装内容**:
1. **Tool formalization**
   - `@tool` デコレータで既存関数をラップ
   - `load_pdf_as_text`, `load_html_as_markdown` など

2. **Sub-agent prototyping**
   - PowerPointSummarizer サブグラフを作成（最優先）
   - 独立したファイルに実装: `src/jpgovsummary/subagents/powerpoint_summarizer.py`

3. **State schema設計**
   - `PlanState`, `ExecutionState` の TypedDict 定義
   - 既存 `State` との共存を許可

**Critical Files**:
- 新規作成: `src/jpgovsummary/subagents/powerpoint_summarizer.py`
- 新規作成: `src/jpgovsummary/state_v2.py`（新State定義）
- 修正: `src/jpgovsummary/tools/pdf_loader.py`（@tool追加）

### Phase 2: サブエージェント実装（2-3週間）

**目標**: コンテキスト消費の大きい操作をサブエージェント化

**実装順序**（優先度順）:

1. **PowerPointSummarizer** (最優先)
   - 現在の `document_summarizer.py` からPPT処理を抽出
   - 独立したStateGraphとして実装
   - 並列実行のテスト

2. **DocumentTypeDetector**
   - 文書タイプ検出を独立化
   - 全文書タイプで再利用

3. **WordSummarizer**
   - Word文書処理を独立化

4. **HTMLProcessor**
   - HTML処理とmain content抽出を統合

**Critical Files**:
- 新規: `src/jpgovsummary/subagents/document_type_detector.py`
- 新規: `src/jpgovsummary/subagents/word_summarizer.py`
- 新規: `src/jpgovsummary/subagents/html_processor.py`
- 修正: `src/jpgovsummary/agents/document_summarizer.py`（既存コードの移動）

### Phase 3: Plan-Action統合（2-3週間）

**目標**: メインワークフローをPlan-Actionパターンに移行

**実装内容**:

1. **Planner agent作成**
   - `action_planner.py` を新規作成
   - 入力分析→実行計画生成ロジック

2. **Executor agent作成**
   - `action_executor.py` を新規作成
   - ActionPlanに基づいてサブエージェント呼び出し

3. **メインワークフロー再構築**
   - `jpgovwatcher.py` を `jpgovwatcher_v2.py` にコピー
   - 新しいグラフ構造を実装
   - 既存版との共存（フィーチャーフラグで切り替え）

**Critical Files**:
- 新規: `src/jpgovsummary/agents/action_planner.py`
- 新規: `src/jpgovsummary/agents/action_executor.py`
- 新規: `src/jpgovsummary/jpgovwatcher_v2.py`
- 修正: `src/jpgovsummary/cli.py`（v1/v2切り替えフラグ追加）

### Phase 4: 並列化とパフォーマンス最適化（1-2週間）

**目標**: 複数文書の並列処理実装

**実装内容**:
1. Async sub-agent呼び出し
2. 並列文書処理のテスト
3. Token使用量のモニタリング

### Phase 5: 既存版の削除とクリーンアップ（1週間）

**目標**: 旧アーキテクチャの削除、v2をデフォルトに

**実装内容**:
1. 旧 `jpgovwatcher.py` の削除
2. `jpgovwatcher_v2.py` → `jpgovwatcher.py` へリネーム
3. 不要な State フィールドの削除
4. ドキュメント更新

---

## 技術的詳細

### Sub-graph実装パターン

```python
# src/jpgovsummary/subagents/powerpoint_summarizer.py

from langgraph.graph import StateGraph, END
from typing import TypedDict

class PowerPointState(TypedDict):
    """PowerPoint要約サブエージェント専用の状態"""
    pdf_pages: list[str]  # 入力
    title: str
    scored_slides: list[dict]
    summary: str  # 出力

class PowerPointSummarizer:
    def __init__(self, model):
        self.model = model
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(PowerPointState)

        # ノード追加
        graph.add_node("extract_title", self._extract_title)
        graph.add_node("score_slides", self._score_slides)
        graph.add_node("generate_summary", self._generate_summary)

        # エッジ追加
        graph.add_edge("extract_title", "score_slides")
        graph.add_edge("score_slides", "generate_summary")
        graph.add_edge("generate_summary", END)

        graph.set_entry_point("extract_title")
        return graph

    def _extract_title(self, state: PowerPointState) -> PowerPointState:
        # 既存の extract_powerpoint_title() ロジックを移植
        llm = self.model.llm()
        # ... (実装)
        return {"title": extracted_title}

    def _score_slides(self, state: PowerPointState) -> PowerPointState:
        # 既存の extract_titles_and_score() ロジックを移植
        # ... (実装)
        return {"scored_slides": scored}

    def _generate_summary(self, state: PowerPointState) -> PowerPointState:
        # 既存の powerpoint_based_summarize() ロジックを移植
        # ... (実装)
        return {"summary": final_summary}

    def invoke(self, input_data: dict) -> dict:
        """サブグラフを実行"""
        compiled = self.graph.compile()
        result = compiled.invoke(input_data)
        return result
```

### メインワークフローでの呼び出し

```python
# src/jpgovsummary/agents/action_executor.py

from jpgovsummary.subagents.powerpoint_summarizer import PowerPointSummarizer

def execute_summarize_action(action: ActionStep, state: ExecutionState) -> dict:
    """ActionStepを実行してサブエージェントを呼び出す"""

    if action.action_type == "summarize_pdf":
        # 1. 文書タイプ検出
        doc_type = detect_document_type(action.target)

        # 2. タイプに応じたサブエージェント選択
        if doc_type == "PowerPoint":
            summarizer = PowerPointSummarizer(model=Model())
            pdf_pages = load_pdf_as_text(action.target)
            result = summarizer.invoke({"pdf_pages": pdf_pages})
            return {
                "summary": result["summary"],
                "source": action.target
            }
        elif doc_type == "Word":
            # Word用サブエージェント
            pass

    return {}
```

### 並列実行パターン

```python
import asyncio

async def execute_plan_parallel(plan: ActionPlan) -> list[dict]:
    """ActionPlanの並列可能なステップを同時実行"""

    # 優先度でグループ化
    priority_groups = {}
    for step in plan.steps:
        priority_groups.setdefault(step.priority, []).append(step)

    all_results = []

    # 優先度順に実行（同じ優先度は並列）
    for priority in sorted(priority_groups.keys()):
        steps = priority_groups[priority]

        # 並列実行
        tasks = [execute_summarize_action_async(step) for step in steps]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)

    return all_results
```

---

## 検証計画

### 成功基準

1. **トークン削減**: 35-50%削減を達成
2. **処理速度**: 複数文書処理時に30-40%高速化
3. **後方互換性**: 既存の入出力フォーマット維持
4. **テストカバレッジ**: 新コードで80%以上

### テスト戦略

**Phase 1テスト**:
- サブエージェント単体テスト
- PowerPoint文書10件で要約品質評価

**Phase 2テスト**:
- エンドツーエンドテスト（HTML meeting page）
- トークン使用量の計測

**Phase 3テスト**:
- 並列実行のストレステスト
- エラーケースのテスト

**テスト対象URL**:
- 内閣官房会議ページ（複数PDF）
- 単一PDF文書
- PowerPoint資料のみのページ

---

## リスクと緩和策

| リスク | 影響 | 確率 | 緩和策 |
|--------|------|------|--------|
| LangGraph API変更 | 高 | 低 | バージョン固定、段階的移行 |
| トークン削減未達 | 中 | 中 | Phase 1でプロトタイプ検証 |
| 並列実行の複雑化 | 中 | 中 | 順次実行版も維持 |
| 既存機能の破壊 | 高 | 低 | v1/v2共存期間を設ける |
| パフォーマンス劣化 | 中 | 低 | ベンチマーク比較 |

---

## Critical Files Summary

### 新規作成ファイル（Phase 1-3）

```
src/jpgovsummary/
├── state_v2.py                          # 新State定義
├── jpgovwatcher_v2.py                   # 新メインワークフロー
├── subagents/
│   ├── __init__.py
│   ├── powerpoint_summarizer.py         # PPT要約サブエージェント
│   ├── document_type_detector.py        # 文書タイプ検出
│   ├── word_summarizer.py               # Word要約
│   └── html_processor.py                # HTML処理
└── agents/
    ├── action_planner.py                # 計画生成エージェント
    └── action_executor.py               # 実行エージェント
```

### 修正対象ファイル

```
src/jpgovsummary/
├── cli.py                               # v1/v2切り替えフラグ
├── tools/
│   ├── pdf_loader.py                    # @tool追加
│   └── html_loader.py                   # @tool追加
└── agents/
    └── document_summarizer.py           # ロジック移植後削減
```

---

## 確定した実装方針

✅ **アーキテクチャパターン**: Plan-Action（事前計画→実行の2フェーズ）
✅ **サブエージェント優先順位**: PowerPointSummarizer → DocumentTypeDetector → WordSummarizer
✅ **移行戦略**: v1/v2共存方式（jpgovwatcher_v2.py作成、CLIフラグで切り替え）

---

## 実装詳細スケジュール

### Phase 1: 基礎準備（1-2週間）- 既存コードに影響なし

**Week 1-2の成果物**:
1. `src/jpgovsummary/state_v2.py` - 新State定義
2. `src/jpgovsummary/subagents/powerpoint_summarizer.py` - PPTサブエージェント
3. Tool formalization（`@tool`デコレータ追加）

**開始条件**: なし（すぐ開始可能）
**完了条件**: PowerPointSummarizerが単体で動作、トークン削減を実測

---

### Phase 2: サブエージェント拡張（2-3週間）

**Week 3-5の成果物**:
1. `src/jpgovsummary/subagents/document_type_detector.py`
2. `src/jpgovsummary/subagents/word_summarizer.py`
3. `src/jpgovsummary/subagents/html_processor.py`

**開始条件**: Phase 1完了
**完了条件**: 全サブエージェントが単体テスト通過

---

### Phase 3: Plan-Action統合（2-3週間）

**Week 6-8の成果物**:
1. `src/jpgovsummary/agents/action_planner.py`
2. `src/jpgovsummary/agents/action_executor.py`
3. `src/jpgovsummary/jpgovwatcher_v2.py`
4. `src/jpgovsummary/cli.py` - `--use-v2` フラグ追加

**開始条件**: Phase 2完了
**完了条件**: エンドツーエンドテスト通過、v1と同等の出力品質

---

### Phase 4: 並列化最適化（1-2週間）

**Week 9-10の成果物**:
1. Async sub-agent実装
2. 並列文書処理のパフォーマンステスト

**開始条件**: Phase 3完了
**完了条件**: 複数文書処理で30-40%高速化を達成

---

### Phase 5: クリーンアップ（1週間）

**Week 11の成果物**:
1. 旧コード削除（v1版）
2. `jpgovwatcher_v2.py` → `jpgovwatcher.py` にリネーム
3. CLAUDE.md更新

**開始条件**: Phase 4完了 + 2週間の本番運用テスト
**完了条件**: v2がデフォルト、ドキュメント更新完了

---

## 検証とテスト計画

### 各Phaseの検証方法

**Phase 1検証**:
- PowerPoint文書10件でトークン削減を実測（目標: 2000-3000トークン削減）
- 要約品質の目視確認（既存版と比較）

**Phase 2検証**:
- 各サブエージェントの単体テスト
- 文書タイプ別のテストケース作成

**Phase 3検証**:
```bash
# v1版（既存）
poetry run jpgovsummary https://www.kantei.go.jp/jp/singi/example/ > v1_output.txt

# v2版（新規）
poetry run jpgovsummary https://www.kantei.go.jp/jp/singi/example/ --use-v2 > v2_output.txt

# 出力比較（文字数、要約品質、実行時間）
diff v1_output.txt v2_output.txt
```

**Phase 4検証**:
- 複数文書（5-10個のPDF）を含む会議ページでベンチマーク
- トークン使用量のログ出力と比較

**Phase 5検証**:
- 本番環境で2週間運用テスト
- エラーレート、実行時間、要約品質の統計分析

### テストケースURL例

1. **複数PDF会議ページ**: 内閣官房の定期会議ページ（5-10個のPDF）
2. **PowerPoint heavy**: プレゼン資料が多い会議
3. **Word heavy**: Word文書のみの会議
4. **単一PDFファイル**: 直接PDFファイルパス指定
5. **会議議事録検出**: 議事録が存在するケース

---

## リスク管理

### 高リスク項目と緩和策

| リスク | 緩和策 | 担当Phase |
|--------|--------|----------|
| サブエージェントのトークン削減未達 | Phase 1で早期プロトタイプ検証、実測で判断 | Phase 1 |
| v2の要約品質劣化 | 各PhaseでA/Bテスト、品質基準を設定 | Phase 3 |
| 並列実行の複雑化 | 順次実行版も維持、フラグで切り替え可能に | Phase 4 |
| LangGraph APIの互換性問題 | バージョン固定（0.2.74）、アップグレード時は別PR | 全Phase |

---

## 成功基準（KPI）

### 必達目標

1. **トークン削減**: 35%以上削減（Phase 3終了時点）
2. **要約品質**: 既存版と同等以上（人間評価 or LLM-as-judge）
3. **実行時間**: 単一文書で既存版と同等、複数文書で30%以上高速化（Phase 4）
4. **後方互換性**: 出力フォーマット（2行形式）維持
5. **テストカバレッジ**: 新コードで80%以上

### ストレッチ目標

1. **トークン削減**: 50%削減達成
2. **並列実行**: 10文書を5文書分の時間で処理
3. **エラー率**: 既存版の50%以下

---

## Critical Files（実装対象ファイル一覧）

### 新規作成（Phase 1-3）

```
src/jpgovsummary/
├── state_v2.py                          # 新State定義（PlanState, ExecutionState）
├── jpgovwatcher_v2.py                   # 新メインワークフロー
├── subagents/
│   ├── __init__.py
│   ├── powerpoint_summarizer.py         # Phase 1
│   ├── document_type_detector.py        # Phase 2
│   ├── word_summarizer.py               # Phase 2
│   └── html_processor.py                # Phase 2
└── agents/
    ├── action_planner.py                # Phase 3
    └── action_executor.py               # Phase 3
```

### 修正対象

```
src/jpgovsummary/
├── cli.py                               # --use-v2 フラグ追加（Phase 3）
├── tools/
│   ├── pdf_loader.py                    # @tool追加（Phase 1）
│   └── html_loader.py                   # @tool追加（Phase 1）
└── agents/
    └── document_summarizer.py           # ロジック移植後はコメントアウト（Phase 5で削除）
```

### 削除対象（Phase 5）

```
src/jpgovsummary/
├── jpgovwatcher.py                      # v2にリネーム後削除
├── state.py                             # state_v2.pyに統合後削除
└── agents/
    ├── main_content_extractor.py        # html_processor sub-agentに統合
    └── document_summarizer.py           # 各type-specific sub-agentに分割
```

---

## 次のアクション

このプランが承認されたら、以下の順で実装を開始します：

1. **ブランチ作成**: `feature/plan-action-architecture`
2. **Phase 1 Week 1開始**:
   - `state_v2.py` 作成
   - `powerpoint_summarizer.py` プロトタイプ実装
   - 単体テストとトークン削減の実測
3. **定期レポート**: 各Phase終了時に進捗と実測結果を報告

準備完了次第、実装を開始できます。
