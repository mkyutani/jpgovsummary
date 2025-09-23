from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
import re

from .. import Model, State, logger


def extract_context_from_messages(messages: list) -> dict:
    """メッセージ履歴から文脈情報を抽出"""
    context = {
        "meeting_info": {},
        "document_contexts": [],
        "processing_notes": []
    }
    
    for message in messages:
        if isinstance(message, AIMessage) and "##" in message.content:
            # 要約系エージェントのメッセージを解析
            content = message.content
            
            # 会議概要の情報を抽出
            if "会議概要生成結果" in content:
                meeting_match = re.search(r"\*\*会議名\*\*:\s*([^\n]+)", content)
                if meeting_match:
                    context["meeting_info"]["name"] = meeting_match.group(1).strip()
                
                minutes_match = re.search(r"\*\*議事録検出\*\*:\s*([^\n]+)", content)
                if minutes_match:
                    context["meeting_info"]["has_minutes"] = minutes_match.group(1).strip() == "有"
            
            # 個別文書要約の情報を抽出
            elif "個別文書要約結果" in content:
                doc_context = {}
                
                name_match = re.search(r"\*\*文書名\*\*:\s*([^\n]+)", content)
                if name_match:
                    doc_context["name"] = name_match.group(1).strip()
                
                type_match = re.search(r"\*\*文書タイプ\*\*:\s*([^\n]+)", content)
                if type_match:
                    doc_context["type"] = type_match.group(1).strip()
                
                reason_match = re.search(r"\*\*選択理由\*\*:\s*([^\n]+)", content)
                if reason_match:
                    doc_context["selection_reason"] = reason_match.group(1).strip()
                
                if doc_context:
                    context["document_contexts"].append(doc_context)
        
        elif isinstance(message, HumanMessage) and "最高スコア" in message.content:
            # 選択系エージェントの結果を記録
            context["processing_notes"].append(message.content.strip())
    
    return context


def _format_context_info(context: dict) -> str:
    """文脈情報をプロンプト用にフォーマット"""
    info_parts = []
    
    # 会議情報
    if context["meeting_info"]:
        meeting_info = context["meeting_info"]
        if "name" in meeting_info:
            info_parts.append(f"会議名: {meeting_info['name']}")
        if "has_minutes" in meeting_info:
            info_parts.append(f"議事録の有無: {'有' if meeting_info['has_minutes'] else '無'}")
    
    # 文書情報
    if context["document_contexts"]:
        info_parts.append("選択された文書:")
        for i, doc in enumerate(context["document_contexts"], 1):
            doc_info = f"  {i}. {doc.get('name', '不明')}"
            if "type" in doc:
                doc_info += f" ({doc['type']})"
            if "selection_reason" in doc:
                doc_info += f" - 選択理由: {doc['selection_reason']}"
            info_parts.append(doc_info)
    
    # 処理ノート
    if context["processing_notes"]:
        info_parts.append("処理履歴:")
        for note in context["processing_notes"]:
            info_parts.append(f"  - {note}")
    
    return "\n".join(info_parts) if info_parts else "文脈情報なし"


def summary_integrator(state: State) -> State:
    """複数の資料の要約を統合し、最終的な要約を生成するエージェント"""
    logger.info("● 各資料の要約を統合します")

    llm = Model().llm()

    # 必要なデータを取得
    target_report_summaries = state.get("target_report_summaries", [])
    overview = state.get("overview", "")
    url = state.get("url", "")
    messages = state.get("messages", [])
    
    # メッセージ履歴から文脈情報を抽出
    context = extract_context_from_messages(messages)
    
    # 会議ページかどうかを判定：初期値で設定されたフラグを使用
    is_meeting_page = state.get("is_meeting_page", False)  # デフォルトは個別文書として扱う

    # URLの長さに基づいて動的に文字数制限を計算
    url_length = len(url)
    max_chars = max(50, 500 - url_length - 1)  # 最低50文字は確保

    if not target_report_summaries:
        final_summary = overview if overview else "文書の要約がないため要約を統合できませんでした。"
        message = HumanMessage(content=f"{final_summary}\n{url}")

        logger.info("資料の要約がないため要約を統合できませんでした。")

        return {**state, "messages": [message], "final_summary": final_summary}

    # 各資料の要約を1つのテキストに結合
    summaries_text = "\n\n".join(
        [
            f"【{summary.name}】\n{summary.content}"
            for summary in target_report_summaries
            if summary.content
        ]
    )

    # 実質的な内容があるかをチェック
    valid_summaries = [
        summary for summary in target_report_summaries
        if summary.content.strip() and 
           not summary.content.strip().endswith("について：") and
           len(summary.content.strip()) > 1
    ]

    if not valid_summaries:
        final_summary = overview if overview else ""
        if not final_summary:
            final_summary = ""
        message = HumanMessage(content=f"{final_summary}\n{url}")

        logger.warning("⚠️ 有効な要約がないため要約を統合できませんでした。")

        return {**state, "messages": [message], "final_summary": final_summary}

    try:
        # Step 1: 内容をまとめる（会議 or 文書に応じて表現を変更）
        subject_type = "会議" if is_meeting_page else "文書"
        subject_expression = "「会議名」では〜が議論された" if is_meeting_page else "「文書名」では〜と記載されている"
        
        combined_summary_prompt = PromptTemplate(
            input_variables=["summaries", "max_chars", "subject_type", "subject_expression"],
            template="""
以下の{subject_type}で扱われた複数の内容をまとめて、{{max_chars}}文字以下の簡潔な{subject_type}要約を作成してください。

**重要な制約：**
- 実際に書かれている内容のみを使用してください
- 推測や補完、創作は一切行わないでください
- 「について：」の後に実質的な内容がない場合は空文字列を返してください
- 意味のある内容、検討事項、結論、データがない場合は要約を作成しないでください

**統合方針：**
- 重要な情報を漏らさないようにしながら、重複を避け、論理的な流れを保ってください
- {subject_type}名は「について：」の前の部分から取得してください
- {subject_type}で扱われた複数の内容を適切にまとめてください
- 「{subject_expression}」の形式で表現してください（会議名の前に「会議では」は付けない）
- 文書名の前に番号（文書1、文書2など）は付けないでください
- 文書の場合、「では」の重複を避けてください：文書名に既に「では」が含まれている場合は追加しない
- {subject_type}名、タイトルは必ず「」（鍵括弧）で囲んでください

# {subject_type}で扱われた内容:
{{summaries}}

# 出力要件
- {{max_chars}}文字以下の{subject_type}要約文
- 箇条書きではなく、文章形式で
- 実際に書かれている内容のみを含める
- {subject_type}名を含める
- 専門用語は適切に使用
- 内容の重複を避ける
- {subject_type}が主語となる表現を使用
- 実質的内容がない場合は空文字列を返す
- より適切な日本語の文章に推敲する
""".format(subject_type=subject_type, subject_expression=subject_expression),
        )

        # 会議で扱われた内容を統合
        combined_result = llm.invoke(
            combined_summary_prompt.format(summaries=summaries_text, max_chars=max_chars)
        )
        combined_summary = combined_result.content.strip()

        # 統合結果が空または無意味な場合のチェック
        if not combined_summary or len(combined_summary) < 1:
            final_summary = overview if overview else ""
            if not final_summary:
                final_summary = ""
            message = HumanMessage(content=f"{final_summary}\n{url}")

            logger.warning("⚠️ 統合要約が短すぎるかありません")

            return {**state, "messages": [message], "final_summary": final_summary}

        # Step 2: 統合した要約とoverviewを合わせて最終要約を作成
        final_summary_prompt = PromptTemplate(
            input_variables=["combined_summary", "overview", "max_chars", "context_info", "subject_type", "subject_expression"],
            template="""
以下の{subject_type}情報をもとに、{{max_chars}}文字以下で最終的な{subject_type}要約を作成してください。

**重要な制約：**
- 実際に書かれている内容のみを使用してください
- 推測や補完、創作は一切行わないでください
- {subject_type}の目的や結論を創作しないでください
- overviewとcombined_summaryの両方に実質的内容がない場合は空文字列を返してください

**統合方針：**
- overviewに{subject_type}名が含まれている場合は、必ず要約文中に残してください
- overviewが提供されていない場合は、{subject_type}情報から{subject_type}名を抽出して使用してください
- 「第1回○○{subject_type}」などの正式名称や回数情報を省略しないでください
- 重要な情報を漏らさず、重複を避け、論理的な流れを保ってください
- 文脈情報を考慮して、より一貫性のある要約を作成してください
- 「{subject_expression}」の形式で表現してください（会議名の前に「会議では」は付けない）
- 文書名の前に番号（文書1、文書2など）は付けないでください
- 文書の場合、「では」の重複を避けてください：文書名に既に「では」が含まれている場合は追加しない
- {subject_type}名、タイトルは必ず「」（鍵括弧）で囲んでください

**文脈情報：**
{{context_info}}

# {subject_type}概要
{{overview}}

# {subject_type}で扱われた内容
{{combined_summary}}

# 出力要件
- {{max_chars}}文字以下の{subject_type}要約文
- 箇条書きではなく、文章形式でまとめる
- {subject_type}名（回数含む）が含まれていること
- 専門用語は適切に使用する
- 内容の重複を避ける
- {subject_type}が主語となる表現を使用
- 実質的内容がない場合は空文字列を返す
- より適切な日本語の文章に推敲する
            """.format(subject_type=subject_type, subject_expression=subject_expression),
        )

        # 文脈情報をフォーマット
        context_info = _format_context_info(context)
        
        # 最終要約を生成
        final_result = llm.invoke(
            final_summary_prompt.format(
                combined_summary=combined_summary,
                overview=overview,
                max_chars=max_chars,
                context_info=context_info,
                subject_type=subject_type,
                subject_expression=subject_expression
            )
        )
        final_summary = final_result.content.strip()

        # Step 3: "作成した要約\nURL"の形式でmessagesに格納
        summary_message = f"{final_summary}\n{url}"

        message = HumanMessage(content=summary_message)
        system_message = HumanMessage(content="複数の要約を統合して、最終的な要約を作成してください。")

        logger.info(summary_message.replace('\n', '\\n'))
        logger.info(f"✅ 要約を統合しました({len(summary_message)}文字)")
    
        return {**state, "messages": [system_message, message], "final_summary": final_summary}

    except Exception as e:
        # エラー時はoverviewをそのまま使用
        final_summary = overview if overview else "要約の統合中にエラーが発生しました。"
        message = HumanMessage(content=f"{final_summary}\n{url}")
        system_message = HumanMessage(content="複数の要約を統合して、最終的な要約を作成してください。")

        logger.error(f"❌ 要約統合中にエラーが発生: {str(e)}")

        return {**state, "messages": [system_message, message], "final_summary": final_summary}
