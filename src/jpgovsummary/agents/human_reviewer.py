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
    print(f"\n📄 Current Summary ({len(final_summary)} characters):")
    print("-" * 50)
    print(final_summary)
    print("-" * 50)
    print(f"🔗 URL: {url}")
    
    if review_session["iteration"] > 0:
        print(f"\n📊 Review Iteration: {review_session['iteration']}")
        print(f"💬 Previous Q&A exchanges: {len(review_session['qa_history'])}")
    
    # Interactive review loop
    while True:
        try:
            user_input = _safe_input("\nYou> ")
            
            if not user_input:
                print("❌ Please provide some input.")
                continue
            
            # Use LLM to classify the user's intent
            action_type = _classify_user_intent(llm, user_input)
            
            if action_type == "approve":
                # Approve and finish
                print("\n✅ Summary approved! Finishing review session.")
                state["review_approved"] = True
                break
                
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
                    improved_summary = _generate_improved_summary(
                        llm, final_summary, improvement_request, overview, target_report_summaries, url
                    )
                    
                    print(f"\nAI> Improved Summary:\n{'-'*50}\n{improved_summary}\n{'-'*50}")
                    
                    # Ask for confirmation in natural language
                    confirmation_input = _safe_input("\nYou> ", default="")
                    
                    if confirmation_input:
                        result = _classify_confirmation_with_feedback(llm, confirmation_input)
                        
                        if result["action"] == "accept":
                            final_summary = improved_summary
                            state["final_summary"] = final_summary
                            review_session["improvements"].append({
                                "request": improvement_request,
                                "result": improved_summary
                            })
                            print("✅ Improvement applied!")
                        elif result["action"] == "improve" and result["feedback"]:
                            # Apply additional improvement
                            further_improved_summary = _generate_improved_summary(
                                llm, improved_summary, result["feedback"], overview, target_report_summaries, url
                            )
                            final_summary = further_improved_summary
                            state["final_summary"] = final_summary
                            review_session["improvements"].append({
                                "request": f"{improvement_request} + {result['feedback']}",
                                "result": further_improved_summary
                            })
                            print(f"\nAI> Further Improved Summary:\n{'-'*50}\n{further_improved_summary}\n{'-'*50}")
                            print("✅ Additional improvement applied!")
                        else:
                            print("❌ Improvement rejected, keeping current summary.")
                    else:
                        print("❌ No response provided, keeping current summary.")
                else:
                    print("📝 Could not extract a clear improvement request.")
                
            elif action_type == "source":
                # Review source materials
                _display_source_materials(overview, target_report_summaries)
                
            elif action_type == "regenerate":
                # Extract feedback for regeneration
                feedback = _extract_regeneration_feedback(llm, user_input)
                if feedback:
                    regenerated_summary = _regenerate_summary_with_feedback(
                        llm, final_summary, feedback, overview, target_report_summaries, url
                    )
                    
                    print(f"\nAI> Regenerated Summary:\n{'-'*50}\n{regenerated_summary}\n{'-'*50}")
                    
                    # Ask for confirmation in natural language
                    confirmation_input = _safe_input("\nYou> ", default="")
                    
                    if confirmation_input:
                        result = _classify_confirmation_with_feedback(llm, confirmation_input)
                        
                        if result["action"] == "accept":
                            final_summary = regenerated_summary
                            state["final_summary"] = final_summary
                            review_session["improvements"].append({
                                "request": f"Regeneration: {feedback}",
                                "result": regenerated_summary
                            })
                            print("✅ Regenerated summary applied!")
                        elif result["action"] == "improve" and result["feedback"]:
                            # Apply additional improvement to regenerated summary
                            further_improved_summary = _generate_improved_summary(
                                llm, regenerated_summary, result["feedback"], overview, target_report_summaries, url
                            )
                            final_summary = further_improved_summary
                            state["final_summary"] = final_summary
                            review_session["improvements"].append({
                                "request": f"Regeneration: {feedback} + {result['feedback']}",
                                "result": further_improved_summary
                            })
                            print(f"\nAI> Further Improved Summary:\n{'-'*50}\n{further_improved_summary}\n{'-'*50}")
                            print("✅ Additional improvement applied!")
                        else:
                            print("❌ Regeneration rejected, keeping current summary.")
                    else:
                        print("❌ No response provided, keeping current summary.")
                else:
                    print("📝 Could not extract clear feedback.")
                
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
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error in summary improvement: {str(e)}")
        return current_summary


def _regenerate_summary_with_feedback(llm, current_summary: str, feedback: str,
                                    overview: str, summaries: list, url: str) -> str:
    """Completely regenerate summary with comprehensive feedback"""
    
    source_context = ""
    if summaries:
        source_context = "\n\n".join([
            f"【{s.name}】\n{s.content}" for s in summaries if s.content
        ])
    
    # Calculate max characters based on URL length
    url_length = len(url)
    max_chars = max(50, 300 - url_length - 1)
    
    prompt = PromptTemplate(
        input_variables=["feedback", "current_summary", "overview", "source_context", "max_chars"],
        template="""
        包括的なフィードバックに基づいて要約を完全に再生成してください。

        **フィードバック:**
        {feedback}

        **参考（現在の要約）:**
        {current_summary}

        **概要情報:**
        {overview}

        **元資料の要約:**
        {source_context}

        **再生成要件:**
        - フィードバックを全面的に反映する
        - {max_chars}文字以下で作成する
        - 実際に書かれている内容のみを使用する
        - 推測や創作は行わない
        - より良い構成と表現を心がける
        - 重要な情報を漏らさない
        - 会議名や資料名を適切に含める
        - 読み手にとって価値のある要約にする
        """
    )
    
    try:
        response = llm.invoke(prompt.format(
            feedback=feedback,
            current_summary=current_summary,
            overview=overview,
            source_context=source_context,
            max_chars=max_chars
        ))
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error in summary regeneration: {str(e)}")
        return current_summary


def _classify_user_intent(llm, user_input: str) -> str:
    """Classify user's natural language input into action categories"""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
        ユーザーの自然言語入力を以下のアクション分類のいずれかに分類してください。

        **入力:** {user_input}

        **分類カテゴリ:**
        - approve: 要約を承認・完了する（例：「承認」「OK」「良い」「完了」「終了」）
        - question: AIに質問する（例：「質問」「なぜ」「どうして」「教えて」「？」）
        - improve: 改善を要求する（例：「改善」「修正」「変更」「直して」「もっと」）
        - source: 元資料を確認する（例：「資料」「ソース」「元」「確認」「見たい」）
        - regenerate: 完全に再生成する（例：「再生成」「作り直し」「最初から」「全面的に」）
        - cancel: レビューをキャンセルする（例：「キャンセル」「中止」「やめる」「終わり」）

        **出力要件:**
        - 上記6つのカテゴリのうち1つだけを返す
        - 英語の小文字で返す（approve, question, improve, source, regenerate, cancel）
        - 判断に迷う場合は最も近いカテゴリを選ぶ
        - 分類できない場合は "unknown" を返す
        """
    )
    
    try:
        response = llm.invoke(prompt.format(user_input=user_input))
        action_type = response.content.strip().lower()
        
        # Validate the response
        valid_actions = ["approve", "question", "improve", "source", "regenerate", "cancel"]
        if action_type in valid_actions:
            return action_type
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return "unknown"


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


def _extract_regeneration_feedback(llm, user_input: str) -> str:
    """Extract comprehensive feedback for summary regeneration"""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
        ユーザーの入力から要約の再生成に必要な包括的なフィードバックを抽出してください。

        **入力:** {user_input}

        **抽出要件:**
        - 要約の再生成に必要な詳細な指示として整理する
        - 構成、内容、スタイル等の改善点を明確にする
        - 完全な作り直しに必要な情報を含める
        - フィードバックが不明確な場合は空文字列を返す

        **例:**
        - 入力: "全体的に作り直して、もっと詳しく" → 出力: "要約全体を作り直し、より詳細で具体的な内容にしてください"
        - 入力: "構成を変えて時系列で" → 出力: "要約の構成を時系列順に変更して再生成してください"

        **出力:** フィードバック文のみ（説明不要）
        """
    )
    
    try:
        response = llm.invoke(prompt.format(user_input=user_input))
        feedback = response.content.strip()
        return feedback if feedback and feedback != "空文字列" else ""
    except Exception as e:
        logger.error(f"Error in feedback extraction: {str(e)}")
        return ""


def _safe_input(prompt: str, default: str = "") -> str:
    """Safely get user input with Unicode error handling"""
    try:
        return input(prompt).strip()
    except UnicodeDecodeError as e:
        print(f"❌ 文字エンコーディングエラーが発生しました: {e}")
        print("💡 入力に使用できない文字が含まれています。")
        return default
    except (EOFError, KeyboardInterrupt):
        # Re-raise these as they should be handled by the main loop
        raise


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
    
    _safe_input("\n📖 Press Enter to continue...", default="") 