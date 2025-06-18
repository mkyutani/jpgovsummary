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
    
    # Initialize review session if not exists
    if "review_session" not in state:
        state["review_session"] = {
            "iteration": 0,
            "qa_history": [],
            "original_summary": final_summary,
            "improvements": []
        }
    
    review_session = state["review_session"]
    
    # Display current summary for human review
    print("\n" + "="*80)
    print("🔍 HUMAN REVIEW SESSION - FINAL SUMMARY QUALITY CHECK")
    print("="*80)
    _display_current_summary(final_summary, url=url)
    
    if review_session["iteration"] > 0:
        print(f"\n📊 Review Iteration: {review_session['iteration']}")
        print(f"💬 Previous Q&A exchanges: {len(review_session['qa_history'])}")
    
    # Interactive review loop
    print("\n💡 Tips: 改行のみ（空入力）で全画面エディタが起動します")
    
    while True:
        try:
            user_input = _enhanced_input("\nYou>")
            
            # Check for empty input (just Enter) - launch fullscreen editor
            if not user_input.strip():
                print("\n📝 全画面エディタを起動します。詳細な改善要求やフィードバックを入力してください。")
                user_input = _enhanced_input("詳細な改善要求", fullscreen=True)
                if not user_input.strip():
                    print("❌ Please provide some input.")
                    continue
            
            # Check for help request
            if user_input.lower() in ["help", "h", "ヘルプ"]:
                _display_help()
                continue
            
            # Use LLM to classify the user's intent
            action_type = _classify_user_intent(llm, user_input)
            
            if action_type == "approve":
                # Check character limit before approval
                total_chars = len(final_summary) + len(url) + 1
                if total_chars <= 300:
                    # Approve and finish
                    print("\n✅ Summary approved! Finishing review session.")
                    state["review_approved"] = True
                    break
                else:
                    # Generate shortened version
                    print(f"\n📏 Summary is {total_chars} chars (exceeds 300 limit).")
                    print("✨ Generating shortened version...")
                    shortened_summary = _generate_shortened_summary(
                        llm, final_summary, overview, target_report_summaries, url
                    )
                    
                    # Update the summary
                    final_summary = shortened_summary
                    state["final_summary"] = final_summary
                    review_session["improvements"].append({
                        "request": f"Auto-shorten from {total_chars} to fit 300 char limit",
                        "result": shortened_summary
                    })
                    
                    _display_current_summary(final_summary, url)
                    print("✅ Shortened version generated!")
                    print("\n💬 Please review the shortened version. You can approve, improve further, or provide feedback.")
                
            elif action_type == "question":
                # Extract question or ask for clarification
                question = _extract_question_from_input(llm, user_input)
                if question:
                    ai_response = _ask_ai_question(llm, question, final_summary, overview, target_report_summaries)
                    print(f"\nAI> {ai_response}")
                    
                    # Record Q&A
                    review_session["qa_history"].append({
                        "type": "human_question",
                        "content": question,
                        "ai_response": ai_response
                    })
                else:
                    print("❓ Could not extract a clear question.")
                
            elif action_type == "improve":
                # Extract improvement request
                improvement_request = _extract_improvement_request(llm, user_input)
                if improvement_request:
                    print("✨ Generating improved summary...")
                    improved_summary = _generate_improved_summary(
                        llm, final_summary, improvement_request, overview, target_report_summaries, url
                    )
                    
                    # Apply the improvement immediately
                    final_summary = improved_summary
                    state["final_summary"] = final_summary
                    review_session["improvements"].append({
                        "request": improvement_request,
                        "result": improved_summary
                    })
                    
                    _display_current_summary(final_summary, url)
                    print("✅ Improvement applied!")
                else:
                    print("📝 Could not extract a clear improvement request.")
                
            elif action_type == "source":
                # Review source materials
                _display_source_materials(overview, target_report_summaries)
                

            elif action_type == "cancel":
                # Cancel review
                print("\n❌ Review cancelled. Using current summary.")
                state["review_approved"] = False
                break
                
            else:
                print("❌ Could not understand your request.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Review interrupted. Using current summary.")
            state["review_approved"] = False
            break
        except EOFError:
            print("\n\n⚠️  Input ended. Using current summary.")
            state["review_approved"] = False
            break
    
    # Update review session
    review_session["iteration"] += 1
    state["review_session"] = review_session
    
    # Display final confirmed summary
    print("\n✅ Review completed!")
    _display_current_summary(final_summary, url=url)
    
    # Update messages with final reviewed summary
    message = HumanMessage(content=f"{final_summary}\n{url}")
    
    # Add review metadata to state
    state["review_completed"] = True
    state["final_review_summary"] = final_summary
    
    return {**state, "messages": [message]}


def _ask_ai_question(llm, question: str, summary: str, overview: str, summaries: list) -> str:
    """Ask AI a specific question about the summary"""
    
    source_context = ""
    if summaries:
        source_context = "\n\n".join([
            f"【{s.name}】\n{s.content}" for s in summaries if s.content
        ])
    
    prompt = PromptTemplate(
        input_variables=["question", "summary", "overview", "source_context"],
        template="""
        人間から以下の質問を受けました。要約の内容と元資料を参照して、正確で詳細な回答をしてください。

        **質問:** {question}

        **現在の要約:**
        {summary}

        **概要情報:**
        {overview}

        **元資料の要約:**
        {source_context}

        **回答要件:**
        - 質問に対して具体的で正確な回答をする
        - 根拠となる資料や情報を明示する
        - 不明な点があれば素直に「不明」と答える
        - 推測や創作は行わない
        - 必要に応じて追加の質問を提案する
        """
    )
    
    try:
        response = llm.invoke(prompt.format(
            question=question,
            summary=summary,
            overview=overview,
            source_context=source_context
        ))
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error in AI question answering: {str(e)}")
        return f"申し訳ありませんが、質問への回答中にエラーが発生しました: {str(e)}"


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





def _classify_user_intent(llm, user_input: str) -> str:
    """Classify user's natural language input into action categories"""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
        ユーザーの自然言語入力を慎重に分析して、最も適切なアクション分類を決定してください。

        **入力:** {user_input}

        **分析手順:**
        1. まず入力の主要な意図を特定する
        2. 各カテゴリの定義と照らし合わせる
        3. 最も適合度の高いカテゴリを選択する

        **分類カテゴリ（優先度順）:**
        
        **approve** - 要約を承認・完了する
        キーワード: 「承認」「OK」「良い」「完了」「終了」「はい」「いいね」「大丈夫」「問題ない」「採用」
        判定基準: 現在の要約に満足し、作業を完了したい意図が明確
        
        **question** - AIに質問する
        キーワード: 「質問」「なぜ」「どうして」「教えて」「？」「根拠」「理由」「詳しく」
        判定基準: 疑問符があるか、説明や詳細を求める意図が明確
        
        **source** - 元資料を確認する
        キーワード: 「資料」「ソース」「元」「確認」「見たい」「原文」「出典」
        判定基準: 元となる資料や文書を見たい意図が明確
        
        **cancel** - レビューをキャンセルする
        キーワード: 「キャンセル」「中止」「やめる」「終わり」「停止」「中断」
        判定基準: 作業を中止したい意図が明確
        
                 **improve** - 改善を要求する（デフォルト）
         キーワード: 「改善」「修正」「変更」「直して」「もっと」「作り直し」「全面的に」「追加」「削除」
         アドバイス: 「〇〇は××です」「実際には」「正確には」「補足すると」「ちなみに」
         判定基準: 上記以外のすべて、変更・改善を求める意図、または情報提供・アドバイス

                 **判定ルール:**
         - 複数の意図が混在する場合は、最も強い意図を選ぶ
         - 曖昧な場合や判断に迷う場合は "improve" を選ぶ
         - 単語だけでなく、文脈や全体的な意図を重視する
         - ユーザーが何かを変えたい・良くしたいと思っている場合は "improve"
         - ユーザーからの情報提供やアドバイス、事実の訂正も "improve" として扱う
         - 「〇〇は××です」のような情報提供は要約改善のための材料として "improve"

        **出力:** 5つのカテゴリ（approve, question, improve, source, cancel）のうち1つのみを英語小文字で返す
        """
    )
    
    try:
        response = llm.invoke(prompt.format(user_input=user_input))
        action_type = response.content.strip().lower()
        
        # Validate the response
        valid_actions = ["approve", "question", "improve", "source", "cancel"]
        if action_type in valid_actions:
            return action_type
        else:
            return "improve"  # Default to improve for unclear inputs
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return "unknown"  # Return unknown on error


def _extract_question_from_input(llm, user_input: str) -> str:
    """Extract a clear question from user's natural language input"""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
        ユーザーの入力から質問を抽出してください。

        **入力:** {user_input}

        **抽出要件:**
        - 明確な質問文として整理する
        - 「？」で終わる疑問文にする
        - 要約に関する質問として適切に整形する
        - 質問が不明確な場合は空文字列を返す

        **例:**
        - 入力: "この部分について詳しく教えて" → 出力: "この部分について詳しく教えてください？"
        - 入力: "なぜこの結論になったの" → 出力: "なぜこの結論になったのですか？"
        - 入力: "根拠は" → 出力: "この要約の根拠は何ですか？"

        **出力:** 質問文のみ（説明不要）
        """
    )
    
    try:
        response = llm.invoke(prompt.format(user_input=user_input))
        question = response.content.strip()
        return question if question and question != "空文字列" else ""
    except Exception as e:
        logger.error(f"Error in question extraction: {str(e)}")
        return ""


def _extract_improvement_request(llm, user_input: str) -> str:
    """Extract improvement request from user's natural language input"""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
        ユーザーの入力から改善要求を抽出してください。

        **入力:** {user_input}

        **抽出要件:**
        - 具体的な改善指示として整理する
        - 何をどのように改善したいかを明確にする
        - 要約の改善に関する指示として適切に整形する
        - 改善要求が不明確な場合は空文字列を返す

        **例:**
        - 入力: "もっと簡潔にして" → 出力: "要約をより簡潔で読みやすくしてください"
        - 入力: "数字を追加して" → 出力: "具体的な数字やデータを追加してください"
        - 入力: "結論を明確に" → 出力: "結論部分をより明確に表現してください"

        **出力:** 改善要求文のみ（説明不要）
        """
    )
    
    try:
        response = llm.invoke(prompt.format(user_input=user_input))
        improvement = response.content.strip()
        return improvement if improvement and improvement != "空文字列" else ""
    except Exception as e:
        logger.error(f"Error in improvement extraction: {str(e)}")
        return ""





def _display_current_summary(final_summary: str, url: str) -> None:
    """現在のサマリーを表示する"""
    summary_chars = len(final_summary)
    url_chars = len(url)
    # 要約 + 改行1文字 + URL = 合計文字数
    total_chars = summary_chars + url_chars + 1
    
    print(f"\n📄 Current Summary (summary: {summary_chars}, URL: {url_chars}, total: {total_chars} chars):")
    print("-" * 50)
    print(final_summary)
    print("-" * 50)
    print(f"🔗 URL: {url}")






def _fullscreen_editor(prompt_text: str, default: str = "") -> str:
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
        
        # Create buffer for text input
        from prompt_toolkit.document import Document
        buffer = Buffer(multiline=True, document=Document(default))
        
        # Create key bindings
        kb = KeyBindings()
        
        @kb.add('c-s')  # Ctrl+S to save and exit
        def _(event):
            event.app.exit(result=buffer.text)
        
        @kb.add('c-q')  # Ctrl+Q to quit without saving
        def _(event):
            event.app.exit(result=default)
        
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
                    HTML(f'<style bg="ansiblue" fg="ansiwhite"><b> 全画面エディタ - {prompt_text} </b></style>')
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
        
        # Run the application
        result = app.run()
        return result.strip() if result else default
        
    except ImportError as e:
        print(f"⚠️  prompt_toolkitが利用できません: {e}")
        print("📝 標準入力を使用します。")
        return _safe_input_fallback(prompt_text, default)
    except Exception as e:
        print(f"⚠️  全画面エディタでエラーが発生しました: {e}")
        print(f"   エラーの種類: {type(e).__name__}")
        print(f"   詳細: {str(e)}")
        print("📝 標準入力を使用します。")
        return _safe_input_fallback(prompt_text, default)


def _is_wsl_environment() -> bool:
    """Check if running in WSL environment"""
    import os
    try:
        # Check multiple WSL indicators
        return (
            'microsoft' in os.uname().release.lower() or
            'wsl' in os.environ.get('WSL_DISTRO_NAME', '').lower() or
            os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
        )
    except:
        return False


def _enhanced_input(prompt_text: str, fullscreen: bool = False, default: str = "") -> str:
    """Enhanced input with prompt_toolkit support for Japanese input"""
    
    # Full-screen editor mode
    if fullscreen:
        return _fullscreen_editor(prompt_text, default)
    
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.history import InMemoryHistory
        
        # Create history for this session
        history = InMemoryHistory()
        result = prompt(f"{prompt_text} ", history=history, default=default)
        
        # If empty input, launch fullscreen editor
        if not result.strip():
            editor_result = _fullscreen_editor(prompt_text.rstrip(': '), default)
            if editor_result is not None:
                return editor_result
            else:
                print("キャンセルされました。")
                return ""
        
        return result.strip()
        
    except ImportError:
        # prompt_toolkitが利用できない場合のフォールバック
        print("⚠️  prompt_toolkitが利用できません。標準入力を使用します。")
        return _safe_input_fallback(prompt_text, default)
    except (EOFError, KeyboardInterrupt):
        # Re-raise these as they should be handled by the main loop
        raise


def _safe_input_fallback(prompt: str, default: str = "") -> str:
    """Fallback input function when prompt_toolkit is not available"""
    try:
        return input(prompt).strip()
    except UnicodeDecodeError as e:
        print(f"❌ 文字エンコーディングエラーが発生しました: {e}")
        print("💡 入力に使用できない文字が含まれています。")
        return default
    except (EOFError, KeyboardInterrupt):
        # Re-raise these as they should be handled by the main loop
        raise


def _safe_input(prompt: str, default: str = "") -> str:
    """Wrapper function for backward compatibility"""
    return _enhanced_input(prompt, fullscreen=False, default=default)


def _classify_confirmation_with_feedback(llm, user_input: str) -> dict:
    """Classify user response to accept/reject with potential additional feedback
    
    Returns:
        dict: {
            "action": "accept" | "reject" | "improve",
            "feedback": str | None
        }
    """
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
        ユーザーの入力を分析して、以下のいずれかに分類してください：

        1. "accept" - 承認・受け入れ（yes, ok, いいね、承認、など）
        2. "reject" - 拒否・却下（no, だめ、却下、やり直し、など）  
        3. "improve" - 追加の改善提案が含まれている

        ユーザー入力: "{user_input}"

        **分類ルール:**
        - 明確に受け入れる意思が示されている場合は "accept"
        - 明確に拒否する意思が示されている場合は "reject"
        - 具体的な改善点や変更要求が含まれている場合は "improve"
        - 曖昧な場合は "reject" を選ぶ

        **出力形式:**
        action: [accept/reject/improve]
        feedback: [改善提案がある場合のみ、その内容を抽出]

        例：
        - "yes" → action: accept, feedback: 
        - "もう少し詳しく" → action: improve, feedback: もう少し詳しく説明してほしい
        - "数字を具体的にして、あと結論を最初に" → action: improve, feedback: 数字を具体的にして、結論を最初に持ってきてほしい
        """
    )
    
    try:
        response = llm.invoke(prompt.format(user_input=user_input))
        content = response.content.strip()
        
        action = "reject"  # default
        feedback = None
        
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("action:"):
                action_part = line.split(":", 1)[1].strip()
                if action_part in ["accept", "reject", "improve"]:
                    action = action_part
            elif line.startswith("feedback:"):
                feedback_part = line.split(":", 1)[1].strip()
                if feedback_part:
                    feedback = feedback_part
        
        return {"action": action, "feedback": feedback}
        
    except Exception as e:
        logger.error(f"Error in confirmation classification: {str(e)}")
        return {"action": "reject", "feedback": None}


def _classify_confirmation(llm, user_input: str) -> bool:
    """Classify user's natural language input as acceptance or rejection"""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
        ユーザーの自然言語入力が「承認・受け入れ」か「拒否・却下」かを判定してください。

        **入力:** {user_input}

        **判定基準:**
        
        **承認・受け入れの例:**
        - 「はい」「いいえ」「OK」「良い」「承認」「受け入れます」
        - 「それで良い」「問題ない」「大丈夫」「採用」
        - 「yes」「accept」「approve」「good」「fine」
        - 「そうしてください」「お願いします」「進めて」
        
        **拒否・却下の例:**
        - 「いいえ」「だめ」「NG」「拒否」「却下」「やめて」
        - 「良くない」「問題がある」「不適切」「違う」
        - 「no」「reject」「deny」「bad」「wrong」
        - 「やり直し」「別の方法で」「変更して」

        **出力要件:**
        - 承認の場合: "true"
        - 拒否の場合: "false"
        - 判断に迷う場合は文脈から最も適切な方を選ぶ
        - 不明確な場合は "false" を返す（安全側に倒す）

        **出力:** "true" または "false" のみ（説明不要）
        """
    )
    
    try:
        response = llm.invoke(prompt.format(user_input=user_input))
        result = response.content.strip().lower()
        
        # Validate and convert to boolean
        if result == "true":
            return True
        elif result == "false":
            return False
        else:
            # If LLM returns unexpected format, default to False (safe side)
            logger.warning(f"Unexpected confirmation classification result: {result}")
            return False
    except Exception as e:
        logger.error(f"Error in confirmation classification: {str(e)}")
        return False


def _display_help() -> None:
    """Display help information for the human reviewer"""
    print("\n" + "="*60)
    print("📖 HUMAN REVIEWER HELP")
    print("="*60)
    print("\n🎯 利用可能なアクション:")
    print("   • 質問する: 要約について詳しく聞く")
    print("   • 改善要求: 具体的な改善点を指摘")
    print("   • ソース確認: 元資料を確認")
    print("   • 承認: 要約を承認して完了")
    print("   • キャンセル: レビューを中止")
    print("   • 全画面エディタ: 空入力（Enterのみ）で起動")
    
    print("\n📝 入力方法:")
    print("   • 通常入力: そのまま入力してEnter")
    print("   • 全画面エディタ: 空入力（Enterのみ）で自動起動")
    print("   • 日本語入力: WSL環境でも快適に入力可能")
    print("   • 履歴呼び出し: 上下矢印キーで過去の入力を呼び出し")
    print("   • 全画面終了: Ctrl+S (保存), Ctrl+Q (キャンセル)")
    
    print("\n💡 入力例:")
    print("   • 「この部分をもっと詳しく説明して」")
    print("   • 「数字を具体的にして、結論を最初に」")
    print("   • 「全体的に作り直して時系列で整理」")
    print("   • 「OK」「承認」「いいね」(承認)")
    print("   • 「source」「ソース」(元資料確認)")
    
    print("\n⌨️  ショートカット:")
    print("   • Ctrl+C: レビュー中断")
    print("   • Enter: 空入力で全画面エディタ起動")
    print("   • Ctrl+S: 全画面エディタで保存して終了")
    print("   • Ctrl+Q: 全画面エディタでキャンセル")
    print("="*60)


def _display_source_materials(overview: str, summaries: list) -> None:
    """Display source materials for human review"""
    
    print("\n" + "="*60)
    print("📚 SOURCE MATERIALS REVIEW")
    print("="*60)
    
    if overview:
        print(f"\n📋 Overview:\n{'-'*30}\n{overview}\n")
    
    if summaries:
        print(f"📄 Document Summaries ({len(summaries)} documents):")
        for i, summary in enumerate(summaries, 1):
            print(f"\n{i}. 【{summary.name}】")
            print(f"   URL: {summary.url}")
            print(f"   Content: {summary.content[:200]}{'...' if len(summary.content) > 200 else ''}")
    else:
        print("\n❌ No document summaries available.")
    
    _enhanced_input("\n📖 Press Enter to continue...", default="") 