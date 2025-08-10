import sys
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

from .. import Model, State, logger


def human_reviewer(state: State) -> State:
    """
    Human-in-the-loop reviewer agent for final summary quality assurance.
    Provides bidirectional Q&A functionality for iterative improvement.
    """
    logger.info("human_reviewer")

    llm = Model().llm()
    
    # Get current data
    final_summary = state.get("final_summary", "")
    overview = state.get("overview", "")
    url = state.get("url", "")
    target_report_summaries = state.get("target_report_summaries", [])
    overview_only = state.get("overview_only", False)
    
    # Determine what to review based on mode
    if overview_only:
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
                # Generate shortened version
                print(f"📏 Summary is {total_chars} chars (exceeds 300 limit).")
                print("✨ Generating shortened version...")
                shortened_summary = _generate_shortened_summary(
                    llm, current_summary, overview, target_report_summaries, url
                )
                
                # Update the summary
                current_summary = shortened_summary
                if overview_only:
                    state["overview"] = current_summary
                else:
                    state["final_summary"] = current_summary
                    final_summary = current_summary
                
                review_session["improvements"].append({
                    "request": f"Auto-shorten from {total_chars} to fit 300 char limit",
                    "result": shortened_summary
                })
                continue

            print("💬 OK to approve, improvement request, or Enter for editor")
            user_input = _enhanced_input("You>")

            # Check if user wants to approve
            if _is_positive_response(user_input):
                # Approve and finish
                state["review_approved"] = True
                break
            elif user_input.strip():
                # Process 1-line improvement request directly
                print(f"🔄")
                new_summary = _generate_improved_summary(llm, current_summary, user_input, overview, target_report_summaries, url)
                if new_summary and new_summary != current_summary:
                    current_summary = new_summary
                    if overview_only:
                        state["overview"] = current_summary
                    else:
                        state["final_summary"] = current_summary
                        final_summary = current_summary
                    
                    review_session["improvements"].append({
                        "request": user_input,
                        "result": new_summary
                    })
                else:
                    print("❌ Could not process improvement request.")
            else:
                # Empty input - launch fullscreen editor with current summary pre-filled
                editor_content = f"""# Summary (edit directly if needed)
{current_summary}

# Improvement instructions (optional)


# How to use:
# - Edit the summary above directly, OR
# - Write improvement instructions below, OR
# - Both approaches work!
# Save with Ctrl+S when done.
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

                result = _enhanced_input("サマリー編集・改善要求", fullscreen=True, initial_content=editor_content, cursor_position=cursor_position)

                if result and result.strip():
                    new_summary = _process_editor_result(llm, result, current_summary, overview, target_report_summaries, url)
                    if new_summary:
                        current_summary = new_summary
                        if overview_only:
                            state["overview"] = current_summary
                        else:
                            state["final_summary"] = current_summary
                            final_summary = current_summary
                        
                        review_session["improvements"].append({
                            "request": "Editor input",
                            "result": new_summary
                        })
                    else:
                        print("❌ Could not process editor input.")
                else:
                    print("📝 No changes made.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Review interrupted. Using current summary.")
            state["review_approved"] = False
            break
        except EOFError:
            print("\n\n⚠️  Input ended. Using current summary.")
            state["review_approved"] = False
            break
    
    # Update review session
    state["review_session"] = review_session

    # Display final confirmed summary
    print("\n✅ Review completed!")
    _display_current_summary(current_summary, url=url)

    # Update messages with final reviewed summary
    message = HumanMessage(content=f"{current_summary}\n{url}")

    # Add review metadata to state
    state["review_completed"] = True
    state["final_review_summary"] = current_summary

    return {**state, "messages": [message]}



def _generate_improved_summary(llm, current_summary: str, improvement_request: str, 
                             overview: str, summaries: list, url: str) -> str:
    """Generate an improved summary based on human feedback"""
    
    source_context = ""
    if summaries:
        source_context = "\n\n".join([
            f"【{s.name}】\n{s.content}" for s in summaries if s.content
        ])
    
    # Calculate max characters based on URL length
    url_length = len(url)
    max_chars = max(50, 300 - url_length - 1)
    
    # Check if the improvement request contains new material to be summarized
    if _contains_substantial_material(improvement_request):
        # Handle new material summarization
        prompt = PromptTemplate(
            input_variables=["improvement_request", "overview", "source_context", "max_chars"],
            template="""
            人間から新しい資料内容が提供されました。この内容を要約してください。

            **提供された新しい資料:**
            {improvement_request}

            **参考情報（概要）:**
            {overview}

            **既存の元資料の要約:**
            {source_context}

            **要約要件:**
            - 提供された新しい資料の内容を中心に要約する
            - {max_chars}文字以下で作成する
            - 実際に書かれている内容のみを使用する
            - 推測や創作は行わない
            - 重要な情報を漏らさない
            - 読みやすく論理的な構成にする
            - 会議名や資料名を適切に含める
            - 新しい資料の構造や要点を適切に反映する
            """
        )
        
        try:
            response = llm.invoke(prompt.format(
                improvement_request=improvement_request,
                overview=overview,
                source_context=source_context,
                max_chars=max_chars
            ))
            improved_summary = response.content.strip()
            return improved_summary
        except Exception as e:
            logger.error(f"Error in new material summarization: {str(e)}")
            return current_summary
    
    else:
        # Handle regular improvement request
        prompt = PromptTemplate(
            input_variables=["current_summary", "improvement_request", "overview", "source_context", "max_chars"],
            template="""
            現在の要約に対して改善要求がありました。要求に従って要約を改善してください。

            **改善要求:**
            {improvement_request}

            **現在の要約:**
            {current_summary}

            **概要情報:**
            {overview}

            **元資料の要約:**
            {source_context}

            **改善要件:**
            - 改善要求に具体的に対応する
            - {max_chars}文字以下で作成する
            - 実際に書かれている内容のみを使用する
            - 推測や創作は行わない
            - 重要な情報を漏らさない
            - 読みやすく論理的な構成にする
            - 会議名や資料名を適切に含める
            """
        )
    
    try:
        response = llm.invoke(prompt.format(
            current_summary=current_summary,
            improvement_request=improvement_request,
            overview=overview,
            source_context=source_context,
            max_chars=max_chars
        ))
        improved_summary = response.content.strip()
        
        return improved_summary
    except Exception as e:
        logger.error(f"Error in summary improvement: {str(e)}")
        return current_summary


def _generate_shortened_summary(llm, current_summary: str, overview: str, summaries: list, url: str) -> str:
    """Generate a shortened version of the summary to fit 300 character limit"""
    
    source_context = ""
    if summaries:
        source_context = "\n\n".join([
            f"【{s.name}】\n{s.content}" for s in summaries if s.content
        ])
    
    # Calculate max characters based on URL length
    url_length = len(url)
    max_chars = max(50, 300 - url_length - 1)
    
    prompt = PromptTemplate(
        input_variables=["current_summary", "overview", "source_context", "max_chars"],
        template="""
        承認された要約が文字数制限を超えているため、短縮版を作成してください。
        人間が承認した内容の意図と重要な情報を保持しながら、文字数制限内に収めてください。

        **承認された要約:**
        {current_summary}

        **概要情報:**
        {overview}

        **元資料の要約:**
        {source_context}

        **短縮要件:**
        - {max_chars}文字以下で作成する（厳守）
        - 承認された要約の主要な内容と意図を保持する
        - 最も重要な情報を優先的に含める
        - 実際に書かれている内容のみを使用する
        - 推測や創作は行わない
        - 読みやすく論理的な構成にする
        - 会議名や資料名を適切に含める
        - 人間の改善意図を可能な限り反映する
        """
    )
    
    try:
        response = llm.invoke(prompt.format(
            current_summary=current_summary,
            overview=overview,
            source_context=source_context,
            max_chars=max_chars
        ))
        shortened_summary = response.content.strip()
        
        return shortened_summary
    except Exception as e:
        logger.error(f"Error in summary shortening: {str(e)}")
        return current_summary

def _is_positive_response(user_input: str) -> bool:
    """肯定的な応答かどうかを判定"""
    positive_keywords = [
        # English
        "ok", "okay", "gj", "good", "great", "nice", "perfect", "yes", "yep", "yeah", "fine", "excellent", "awesome", "cool",
        # Japanese
        "いいね", "良い", "よい", "承認", "はい", "オーケー", "グッド", "ナイス", "完璧", "最高", "素晴らしい", "いい", "よし",
        # Emoji/symbols
        "👍", "✅", "🆗", "👌", "💯", "🎉", "😊", "😍", "🥰",
        # Variations
        "おk", "ｏｋ", "ＯＫ", "オーキー", "だいじょうぶ", "大丈夫", "問題ない", "もんだいない"
    ]
    
    # Check exact matches (case insensitive)
    normalized_input = user_input.lower().strip()
    return normalized_input in positive_keywords


def _process_editor_result(llm, editor_result: str, current_summary: str, overview: str, summaries: list, url: str) -> str:
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
    
    # Check if user modified the summary directly
    if edited_summary and edited_summary != current_summary:
        print(f"Direct edit detected: using edited summary")
        return edited_summary
    elif improvement_request:
        print(f"🔄 {improvement_request}")
        return _generate_improved_summary(llm, current_summary, improvement_request, overview, summaries, url)
    else:
        # No changes made
        print("No changes detected")
        return current_summary

def _contains_substantial_material(user_input: str) -> bool:
    """Check if user input contains substantial new material content"""
    
    # Check for indicators of new material content
    material_indicators = [
        "以下の内容を要約",
        "以下の資料を要約",
        "以下のテキストを要約",
        "新しい資料",
        "現在の要約は破棄",
        "要約を破棄",
        "###",  # Markdown headers
        "##",   # Markdown headers
        "①", "②", "③", "④", "⑤",  # Numbered lists
        "１章", "２章", "３章",  # Chapter indicators
        "〇", "○",  # Bullet points
        "・",   # Bullet points
    ]
    
    # Check for substantial length (likely contains material)
    if len(user_input) > 500:
        return True
    
    # Check for material indicators
    for indicator in material_indicators:
        if indicator in user_input:
            return True
    
    # Check for structured content patterns
    lines = user_input.split('\n')
    structured_lines = 0
    for line in lines:
        line = line.strip()
        if line and (line.startswith('・') or line.startswith('○') or line.startswith('〇') or 
                    line.startswith('①') or line.startswith('②') or line.startswith('③') or
                    line.startswith('④') or line.startswith('⑤') or line.startswith('#')):
            structured_lines += 1
    
    # If many lines are structured content, likely new material
    if structured_lines >= 5:
        return True
    
    return False

def _display_current_summary(final_summary: str, url: str) -> None:
    """現在のサマリーを表示する"""
    summary_chars = len(final_summary)
    url_chars = len(url)
    # 要約 + 改行1文字 + URL = 合計文字数
    total_chars = summary_chars + url_chars + 1
    
    print(f"📄 Current Summary (summary: {summary_chars}, URL: {url_chars}, total: {total_chars} chars):")
    print("-" * 50)
    print(final_summary)
    print("-" * 50)
    print(f"🔗 URL: {url}")

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
        
    except ImportError as e:
        print(f"⚠️ No prompt_toolkit: {e}")
        return initial_content
    except Exception as e:
        print(f"⚠️ Exception occurred: {type(e).__name__}")
        print(f"   Detail: {str(e)}")
        return initial_content

def _enhanced_input(prompt_text: str, fullscreen: bool = False, initial_content: str = "", cursor_position: int = None) -> str:
    """Enhanced input with prompt_toolkit support for Japanese input"""
    
    # Full-screen editor mode
    if fullscreen:
        return _fullscreen_editor(initial_content, cursor_position)
    
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.history import InMemoryHistory
        
        # Create history for this session
        history = InMemoryHistory()
        result = prompt(f"{prompt_text} ", history=history, default=initial_content)
        
        return result.strip()
        
    except ImportError:
        # prompt_toolkitが利用できない場合のフォールバック
        print("⚠️  prompt_toolkit not available. Using standard input.")
        return _safe_input_fallback(prompt_text, initial_content)
    except (EOFError, KeyboardInterrupt):
        # Re-raise these as they should be handled by the main loop
        raise

def _safe_input_fallback(prompt: str, fallback_content: str = "") -> str:
    """Fallback input function when prompt_toolkit is not available"""
    try:
        return input(prompt).strip()
    except UnicodeDecodeError as e:
        print(f"❌ Character encoding error occurred: {e}")
        print("💡 Input contains unsupported characters.")
        return fallback_content
    except (EOFError, KeyboardInterrupt):
        # Re-raise these as they should be handled by the main loop
        raise