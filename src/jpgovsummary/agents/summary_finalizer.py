import sys
import re
from typing import Dict, Any, NamedTuple
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

from .. import Model, State, logger


class QualityEvaluation(NamedTuple):
    """品質評価結果"""
    technical_detail: int   # 技術詳細保持度 (1-5)
    practical_value: int    # 実務価値維持度 (1-5)
    concreteness: int       # 具体性レベル (1-5)
    reader_utility: int     # 読者有用性 (1-5)
    overall_score: int      # 総合評価 (1-5)
    needs_improvement: bool # 改善要否
    improved_summary: str   # 改善された要約（改善要の場合）


def summary_finalizer(state: State) -> State:
    """
    Summary finalizer agent for final summary quality assurance and character limit validation.
    Provides bidirectional Q&A functionality for iterative improvement and automatic shortening.
    """
    logger.info("🟢 最終調整を行います")

    llm = Model().llm()
    
    # Get current data
    final_summary = state.get("final_summary", "")
    overview = state.get("overview", "")
    url = state.get("url", "")
    target_report_summaries = state.get("target_report_summaries", [])
    overview_only = state.get("overview_only", False)
    batch = state.get("batch", False)
    messages = state.get("messages", [])
    
    # 会議ページかどうかを判定：初期値で設定されたフラグを使用（summary_integratorと同じロジック）
    is_meeting_page = state.get("is_meeting_page", False)  # デフォルトは個別文書として扱う
    
    # Determine what to review based on mode
    # overview_onlyまたは議事録検出時はoverviewを使用
    use_overview_mode = (
        overview_only or 
        state.get("meeting_minutes_detected", False)
    )
    
    if use_overview_mode:
        current_summary = overview
    else:
        current_summary = final_summary
    
    # Initialize review session if not exists
    if "review_session" not in state:
        state["review_session"] = {
            "original_summary": current_summary,
            "improvements": []
        }
    
    review_session = state["review_session"]
    
    while True:
        try:
            _display_current_summary(current_summary, url=url)

            # Check character limit before approval
            total_chars = len(current_summary) + len(url) + 1
            if total_chars > 300:
                # Simple character limit logic: 500+ total chars -> 500 chars, otherwise -> 300 chars
                target_total_chars = 500 if total_chars >= 500 else 300
                
                logger.warning(f"⚠️ 要約が{total_chars}文字で長すぎるため{target_total_chars}文字以内に再生成します")
                shortened_summary = _generate_shortened_summary(
                    llm, current_summary, overview, target_report_summaries, url, is_meeting_page, target_total_chars
                )
                
                # Update the summary
                current_summary = shortened_summary
                if use_overview_mode:
                    state["overview"] = current_summary
                else:
                    state["final_summary"] = current_summary
                    final_summary = current_summary
                
                review_session["improvements"].append({
                    "request": f"Auto-shorten from {total_chars} to fit {target_total_chars} char limit",
                    "result": shortened_summary
                })
                continue

            if batch:
                logger.info("バッチモードのため人間レビューをスキップします")
                state["review_approved"] = True
                break

            user_input = _enhanced_input("OK または ^D で承認、改善要求の入力、または Enter でエディター起動します\nYou>")

            # Check if user wants to approve
            if _is_positive_response(user_input):
                # Approve and finish
                state["review_approved"] = True
                break
            elif user_input.strip():
                # Process 1-line improvement request directly
                new_summary = _generate_improved_summary(llm, current_summary, user_input, overview, target_report_summaries, url, is_meeting_page)
                if new_summary and new_summary != current_summary:
                    current_summary = new_summary
                    if use_overview_mode:
                        state["overview"] = current_summary
                    else:
                        state["final_summary"] = current_summary
                        final_summary = current_summary
                    
                    review_session["improvements"].append({
                        "request": user_input,
                        "result": new_summary
                    })
                else:
                    logger.error("❌ 改善要求を処理できませんでした")
            else:
                # Empty input - launch fullscreen editor with current summary pre-filled
                editor_content = f"""# Summary (edit directly if needed)
{current_summary}

# Improvement instructions (optional)


# How to use:
# - Edit the summary above directly, OR
# - Write improvement instructions below, OR
# - Both approaches work!
# 
# Note for improvement instructions:
# - Use ## or lower for section headings (# is system reserved)
# - Example: ## Content to add, ### Detail items, etc.
# - Structured instructions enable more accurate improvements
# 
# Save with Ctrl+S when done, or Ctrl+Q to cancel.
"""

                # Calculate cursor position to place it at the start of improvement instructions section
                lines_before_improvement = editor_content.split('\n')
                improvement_line_index = -1
                for i, line in enumerate(lines_before_improvement):
                    if line.strip() == '# Improvement instructions (optional)':
                        improvement_line_index = i + 1  # +1 to place cursor right after the header
                        break
                
                cursor_position = 0
                if improvement_line_index > 0:
                    cursor_position = len('\n'.join(lines_before_improvement[:improvement_line_index])) + 1

                result = _fullscreen_editor(initial_content=editor_content, cursor_position=cursor_position)

                if result and result.strip():
                    new_summary = _process_editor_result(llm, result, current_summary, overview, target_report_summaries, url, is_meeting_page)
                    if new_summary:
                        current_summary = new_summary
                        if use_overview_mode:
                            state["overview"] = current_summary
                        else:
                            state["final_summary"] = current_summary
                            final_summary = current_summary
                        
                        review_session["improvements"].append({
                            "request": "Editor input",
                            "result": new_summary
                        })
                    else:
                        logger.error("❌ エディター入力を処理できませんでした")
                else:
                    logger.info("変更はありませんでした")
                
        except KeyboardInterrupt:
            logger.info("キーボード中断により現在の要約を使用")
            state["review_approved"] = False
            break
        except EOFError:
            logger.info("EOF検出により現在の要約を使用")
            state["review_approved"] = False
            break
    
    # Update review session
    state["review_session"] = review_session

    # Display final confirmed summary
    logger.info("✅ レビュー完了！")
    _display_current_summary(current_summary, url=url)

    # Update messages with final reviewed summary
    message = AIMessage(content=f"{current_summary}\n{url}")
    system_message = HumanMessage(content="要約の品質を確認し、必要に応じて改善してください。")

    # Add review metadata to state
    state["review_completed"] = True
    state["final_review_summary"] = current_summary

    return {**state, "messages": [system_message, message]}


def _generate_improved_summary(llm, current_summary: str, improvement_request: str, 
                             overview: str, summaries: list, url: str, is_meeting_page: bool) -> str:
    """Generate an improved summary based on human feedback"""
    
    source_context = ""
    if summaries:
        source_context = "\n\n".join([
            f"【{s.name}】\n{s.content}" for s in summaries if s.content
        ])
    
    # Calculate max characters based on URL length
    url_length = len(url)
    max_chars = max(50, 300 - url_length - 1)
    
    # Handle improvement request
    # 会議 or 文書に応じて表現を変更
    subject_type = "会議" if is_meeting_page else "文書"
    subject_expression = "「会議名」では〜が議論された" if is_meeting_page else "「文書名」によれば〜"
    
    prompt = PromptTemplate(
        input_variables=["current_summary", "improvement_request", "overview", "source_context", "max_chars", "subject_type", "subject_expression"],
        template="""現在の{subject_type}要約に対して改善要求がありました。要求に従って{subject_type}要約を改善してください。

# 改善要求
{{improvement_request}}

# 現在の{subject_type}要約
{{current_summary}}

# {subject_type}概要情報
{{overview}}

# {subject_type}で扱われた内容
{{source_context}}

# 改善要件
- 改善要求に具体的に対応する
- {{max_chars}}文字以下で作成する
- 実際に書かれている内容のみを使用する
- 推測や創作は行わない
- 重要な情報を漏らさない
- 読みやすく論理的な構成にする
- {subject_type}名を適切に含める
- 「{subject_expression}」の形式で表現する（会議名の前に「会議では」は付けない）
- 文書の場合、「では」の重複を避ける：文書名に既に「では」が含まれている場合は追加しない
- 文書名の前に番号（文書1、文書2など）は付けない
- {subject_type}名、タイトルは必ず「」（鍵括弧）で囲む
- より適切な日本語の文章に推敲する
- **以下の情報は要約に含めない：**
  - 開会・閉会・進行に関する情報（「開会した」「閉会した」「進行した」等）
  - 開催日時・時間に関する情報（「○月○日」「午前」「午後」「○時」等）
  - 開催場所・会場に関する情報（「○○省」「○○ビル」「オンライン」「ハイブリッド」等）
  - 会議の形式・構成に関する情報（「書面開催」「対面開催」「Web会議」等）
  - {subject_type}の出席者・参加者情報
  - 会議の場合、どんな資料が配布されたかの情報
""".format(subject_type=subject_type, subject_expression=subject_expression)
    )
    
    try:
        response = llm.invoke(prompt.format(
            current_summary=current_summary,
            improvement_request=improvement_request,
            overview=overview,
            source_context=source_context,
            max_chars=max_chars,
            subject_type=subject_type,
            subject_expression=subject_expression
        ))
        improved_summary = response.content.strip()
        
        return improved_summary
    except Exception as e:
        logger.error(f"❌ 要約改善中にエラーが発生: {str(e)}")
        return current_summary

def _generate_shortened_summary(llm, current_summary: str, overview: str, summaries: list, url: str, is_meeting_page: bool, target_total_chars: int) -> str:
    """3段階の要約短縮処理：1.短縮 → 2.品質確認 → 3.品質改善"""
    
    source_context = ""
    if summaries:
        source_context = "\n\n".join([
            f"【{s.name}】\n{s.content}" for s in summaries if s.content
        ])
    
    # Calculate max characters based on URL length
    url_length = len(url)
    max_chars = max(50, target_total_chars - url_length - 1)
    
    # 会議 or 文書に応じて表現を変更
    subject_type = "会議" if is_meeting_page else "文書"
    subject_expression = "「会議名」では〜が議論された" if is_meeting_page else "「文書名」によれば〜"
    
    prompt = PromptTemplate(
        input_variables=["current_summary", "overview", "source_context", "max_chars", "subject_type", "subject_expression"],
        template="""承認された{subject_type}要約を{max_chars}文字以下に短縮し、品質確認・改善を行ってください。

# 承認された{subject_type}要約
{current_summary}

# {subject_type}概要情報
{overview}

# {subject_type}で扱われた内容
{source_context}

# 処理手順
## 手順1: 短縮
- {max_chars}文字以下で作成する（厳守）
- 承認された{subject_type}要約の主要な内容と意図を保持する
- 最も重要な情報を優先的に含める

## 手順2: 品質確認
以下の観点で短縮結果を評価してください：
- **技術詳細の保持**: 具体的な技術名称・手法・数値が適切に保持されているか
- **実務価値の維持**: 読者が実務で活用できる具体的情報が含まれているか
- **具体性の確保**: 単独で読んで理解できる具体的な内容になっているか
- **有用性の確保**: 政策・技術的価値が伝わる内容になっているか

## 手順3: 品質改善（必要に応じて）
手順2で問題が発見された場合、以下の点を改善してください：
- 重要な技術詳細の復活・追加
- 実務価値を高める具体的情報の強化
- 理解しやすさの向上
- 政策・技術的価値の明確化
- **重要**: 改善後も{max_chars}文字以下を厳守してください

# 表現・構成要件
- 実際に書かれている内容のみを使用する（推測や創作は行わない）
- 読みやすく論理的な構成にする
- {subject_type}名を適切に含める
- 「{subject_expression}」の形式で表現する（会議名の前に「会議では」は付けない）
- 文書の場合、「では」の重複を避ける：文書名に既に「では」が含まれている場合は追加しない
- 文書名の前に番号（文書1、文書2など）は付けない
- 人間の改善意図を可能な限り反映する
- より適切な日本語の文章に推敲する

# 除外すべき情報
- 開会・閉会・進行に関する情報（「開会した」「閉会した」「進行した」等）
- 開催日時・時間に関する情報（「○月○日」「午前」「午後」「○時」等）
- 開催場所・会場に関する情報（「○○省」「○○ビル」「オンライン」「ハイブリッド」等）
- 会議の形式・構成に関する情報（「書面開催」「対面開催」「Web会議」等）
- {subject_type}の出席者・参加者情報
- 会議の場合、どんな資料が配布されたかの情報

# 出力
最終的に短縮・品質確認・改善を完了した要約のみを出力してください（処理手順や説明は不要）。
        """)
    
    try:
        response = llm.invoke(prompt.format(
            current_summary=current_summary,
            overview=overview,
            source_context=source_context,
            max_chars=max_chars,
            subject_type=subject_type,
            subject_expression=subject_expression
        ))
        
        result_summary = response.content.strip()
        return result_summary
        
    except Exception as e:
        logger.error(f"❌ 要約短縮中にエラーが発生: {str(e)}")
        return current_summary


def _is_positive_response(user_input: str) -> bool:
    """肯定的な応答かどうかを判定"""
    positive_keywords = [
        # English
        "ok", "okay", "gj", "good", "great", "nice", "perfect", "yes", "yep", "yeah", "fine", "excellent", "awesome", "cool", "okay", "go",
        # Japanese
        "いいね", "良い", "よい", "承認", "はい", "オーケー", "グッド", "ナイス", "完璧", "最高", "素晴らしい", "いい", "よし",
        # Emoji/symbols
        "👍", "✅", "🆗", "👌", "💯", "🎉", "😊", "😍", "🥰",
        # Variations
        "おk", "おｋ", "ｏｋ", "ＯＫ", "オーキー", "だいじょうぶ", "大丈夫", "問題ない", "もんだいない"
    ]
    
    # Check exact matches (case insensitive)
    normalized_input = user_input.lower().strip()
    return normalized_input in positive_keywords


def _process_editor_result(llm, editor_result: str, current_summary: str, overview: str, summaries: list, url: str, is_meeting_page: bool) -> str:
    """エディタ結果を処理して新しいサマリーを生成"""
    
    lines = editor_result.strip().split('\n')
    
    # Find the sections
    current_section = []
    improvement_section = []
    
    in_current = False
    in_improvement = False
    
    for line in lines:
        if line.strip().startswith('# Summary'):
            in_current = True
            in_improvement = False
            continue
        elif line.strip().startswith('# Improvement instructions'):
            in_current = False
            in_improvement = True
            continue
        elif line.strip().startswith('# How to use'):
            in_current = False
            in_improvement = False
            continue
        
        if in_current:
            current_section.append(line)
        elif in_improvement:
            improvement_section.append(line)
    
    # Extract edited summary and improvement requests
    edited_summary = '\n'.join(current_section).strip()
    improvement_request = '\n'.join(improvement_section).strip()
    
    # Check if user modified the summary directly and/or provided improvement instructions
    has_direct_edit = edited_summary and edited_summary != current_summary
    has_improvement_request = improvement_request
    
    if has_direct_edit and has_improvement_request:
        logger.info(f"{improvement_request.replace('\n', ' ')}")
        updated_summary = _generate_improved_summary(llm, edited_summary, improvement_request, overview, summaries, url, is_meeting_page)
    elif has_direct_edit:
        updated_summary = edited_summary
    elif has_improvement_request:
        # Only improvement request
        logger.info(f"{improvement_request.replace('\n', ' ')}")
        updated_summary = _generate_improved_summary(llm, current_summary, improvement_request, overview, summaries, url, is_meeting_page)
    else:
        # No changes made
        logger.info("変更は検出されませんでした")
        updated_summary = current_summary

    return updated_summary.strip().replace('\n', '')

def _display_current_summary(final_summary: str, url: str) -> None:
    """現在のサマリーを表示する"""
    summary_message = f"{final_summary}\n{url}"
    logger.info(f"📄 {final_summary}")
    logger.info(f"🔗 {url}")

def _fullscreen_editor(initial_content: str = "", cursor_position: int = None) -> str:
    """Full-screen editor using prompt_toolkit"""
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.layout import Layout
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.formatted_text import HTML
        import os
        
        # Create buffer for text input with proper initialization
        from prompt_toolkit.document import Document
        buffer = Buffer(multiline=True)
        
        # Set initial content explicitly
        if initial_content:
            buffer.text = initial_content
            # Set cursor position - use provided position or default to end
            if cursor_position is not None and 0 <= cursor_position <= len(initial_content):
                buffer.cursor_position = cursor_position
            else:
                buffer.cursor_position = len(initial_content)
        
        # Create key bindings
        kb = KeyBindings()
        
        @kb.add('c-s')  # Ctrl+S to save and exit
        def _(event):
            event.app.exit(result=buffer.text)
        
        @kb.add('c-q')  # Ctrl+Q to quit without saving
        def _(event):
            event.app.exit(result=initial_content)
        
        @kb.add('c-x', 'c-c')  # Ctrl+X Ctrl+C to save and exit
        def _(event):
            event.app.exit(result=buffer.text)
        
        @kb.add('c-c')  # Ctrl+C to raise KeyboardInterrupt
        def _(event):
            raise KeyboardInterrupt()
        
        # Help overlay state
        help_visible = [False]  # Use list to make it mutable in nested function
        
        @kb.add('c-g')  # Ctrl+G for help
        def _(event):
            # Toggle help overlay
            help_visible[0] = not help_visible[0]
            event.app.invalidate()  # Refresh display
        
        # Create dynamic status line
        def get_status_text():
            line_count = buffer.document.line_count
            cursor_line = buffer.document.cursor_position_row + 1
            cursor_col = buffer.document.cursor_position_col + 1
            char_count = len(buffer.text)
            return f'行 {cursor_line}/{line_count}  列 {cursor_col}  文字数 {char_count}'
        
        # Help content function
        def get_help_content():
            if help_visible[0]:
                return HTML('''<style bg="ansiyellow" fg="ansiblack">
=== 全画面エディタ ヘルプ ===

キーバインド:
^S (Ctrl+S)     : 保存して終了
^Q (Ctrl+Q)     : キャンセル（保存しない）
^G (Ctrl+G)     : このヘルプを表示/非表示
^C (Ctrl+C)     : 強制終了

編集機能:
- 通常の文字入力、削除、改行が可能
- 日本語入力対応
- 上下左右矢印キーでカーソル移動
- Home/End キーで行の始端/終端へ移動

このエディタで長文の改善要求や
フィードバックを快適に入力できます。

もう一度 ^G を押すとヘルプを閉じます
=== ヘルプ終了 ===
</style>''')
            else:
                return HTML('')
        
        # Create layout with nano-style interface
        main_content = [
            # Header with title
            Window(
                content=FormattedTextControl(
                    HTML(f'<style bg="ansiblue" fg="ansiwhite"><b> Fullscreen Editor </b></style>')
                ),
                height=1,
                dont_extend_height=True,
            ),
            # Main editing area - takes remaining space
            Window(
                content=BufferControl(buffer=buffer),
                wrap_lines=True,
            ),
        ]
        
        # Add help overlay if visible
        help_window = Window(
            content=FormattedTextControl(get_help_content),
            height=lambda: 18 if help_visible[0] else 0,
            dont_extend_height=True,
        )
        main_content.append(help_window)
        
        # Add status and help bar
        main_content.extend([
            # Status line
            Window(
                content=FormattedTextControl(
                    lambda: HTML(f'<style bg="ansigray" fg="ansiwhite"> {get_status_text()} </style>')
                ),
                height=1,
                dont_extend_height=True,
            ),
            # Bottom help bar (nano-style)
            Window(
                content=FormattedTextControl(
                    HTML('<style bg="ansiwhite" fg="ansiblack">'
                         ' ^S 保存終了   ^Q キャンセル   ^G ヘルプ   ^C 中断 '
                         '</style>')
                ),
                height=1,
                dont_extend_height=True,
            ),
        ])
        
        root_container = HSplit(main_content)
        
        layout = Layout(root_container)
        
        # Create application
        app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=False  # WSL2環境での安定性のため無効化
        )

        # Force refresh after initialization
        def refresh_on_start():
            app.invalidate()

        # Run the application
        result = app.run()
        return result.strip() if result else initial_content
        
    except Exception as e:
        logger.error(f"❌ フルスクリーンエディターエラー: {type(e).__name__}: {str(e)}")
        return initial_content

def _enhanced_input(prompt_text: str) -> str:
    """Enhanced input with prompt_toolkit support for Japanese input"""

    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.history import InMemoryHistory
        
        # Create history for this session
        history = InMemoryHistory()
        result = prompt(f"{prompt_text} ", history=history)
        
        return result.strip()
        
    except (EOFError, KeyboardInterrupt):
        # Re-raise these as they should be handled by the main loop
        raise
