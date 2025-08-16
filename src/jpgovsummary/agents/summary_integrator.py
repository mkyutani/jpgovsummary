from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from .. import Model, State, logger


def summary_integrator(state: State) -> State:
    """複数の資料の要約を統合し、最終的な要約を生成するエージェント"""
    logger.info("summary_integrator")

    llm = Model().llm()

    # 必要なデータを取得
    target_report_summaries = state.get("target_report_summaries", [])
    overview = state.get("overview", "")
    url = state.get("url", "")

    # URLの長さに基づいて動的に文字数制限を計算
    url_length = len(url)
    max_chars = max(50, 500 - url_length - 1)  # 最低50文字は確保

    if not target_report_summaries:
        logger.warning("No report summaries found")
        final_summary = overview if overview else "要約を生成できませんでした。"
        message = HumanMessage(content=f"{final_summary}\n{url}")
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
        logger.warning("No valid summaries with substantial content found")
        final_summary = overview if overview else ""
        if not final_summary:
            final_summary = ""
        message = HumanMessage(content=f"{final_summary}\n{url}")
        return {**state, "messages": [message], "final_summary": final_summary}

    try:
        # Step 1: 資料の要約をまとめる
        combined_summary_prompt = PromptTemplate(
            input_variables=["summaries", "max_chars"],
            template="""
            以下の複数の資料の要約をまとめて、{max_chars}文字以下の簡潔な要約を作成してください。

            **重要な制約：**
            - 実際に書かれている内容のみを使用してください
            - 推測や補完、創作は一切行わないでください
            - 「について：」の後に実質的な内容がない場合は空文字列を返してください
            - 意味のある議論内容、検討事項、結論、データがない場合は要約を作成しないでください

            **統合方針：**
            - 重要な情報を漏らさないようにしながら、重複を避け、論理的な流れを保ってください
            - 会議名や資料名は「について：」の前の部分から取得してください
            - 複数の資料がある場合は、適切にまとめてください

            # 資料の要約:
            {summaries}

            # 出力要件
            - {max_chars}文字以下の要約文
            - 箇条書きではなく、文章形式で
            - 実際に書かれている内容のみを含める
            - 会議名や資料名を含める
            - 専門用語は適切に使用
            - 内容の重複を避ける
            - 実質的内容がない場合は空文字列を返す
            """,
        )

        # 資料の要約を統合
        combined_result = llm.invoke(
            combined_summary_prompt.format(summaries=summaries_text, max_chars=max_chars)
        )
        combined_summary = combined_result.content.strip()

        # 統合結果が空または無意味な場合のチェック
        if not combined_summary or len(combined_summary) < 1:
            logger.warning("Combined summary is empty or too short")
            final_summary = overview if overview else ""
            if not final_summary:
                final_summary = ""
            message = HumanMessage(content=f"{final_summary}\n{url}")
            return {**state, "messages": [message], "final_summary": final_summary}

        # Step 2: 統合した要約とoverviewを合わせて最終要約を作成
        final_summary_prompt = PromptTemplate(
            input_variables=["combined_summary", "overview", "max_chars"],
            template="""
            以下の要約内容をもとに、{max_chars}文字以下で全体の要約を作成してください。

            **重要な制約：**
            - 実際に書かれている内容のみを使用してください
            - 推測や補完、創作は一切行わないでください
            - 会議の目的や結論を創作しないでください
            - overviewとcombined_summaryの両方に実質的内容がない場合は空文字列を返してください

            **統合方針：**
            - overviewに会議名や資料名が含まれている場合は、必ず要約文中に残してください
            - overviewが提供されていない場合は、統合要約から会議名や資料名を抽出して使用してください
            - 「第1回○○会議」などの正式名称や回数情報を省略しないでください
            - 重要な情報を漏らさず、重複を避け、論理的な流れを保ってください

            # overview
            {overview}

            # 関連資料の要約
            {combined_summary}

            # 出力要件
            - {max_chars}文字以下の要約文
            - 箇条書きではなく、文章形式でまとめる
            - 会議名や資料名（回数含む）が含まれていること
            - 専門用語は適切に使用する
            - 内容の重複を避ける
            - 実質的内容がない場合は空文字列を返す
            """,
        )

        # 最終要約を生成
        final_result = llm.invoke(
            final_summary_prompt.format(
                combined_summary=combined_summary, overview=overview, max_chars=max_chars
            )
        )
        final_summary = final_result.content.strip()

        # Step 3: "作成した要約\nURL"の形式でmessagesに格納
        message = HumanMessage(content=f"{final_summary}\n{url}")

        return {**state, "messages": [message], "final_summary": final_summary}

    except Exception as e:
        logger.error(f"Error in summary integration: {str(e)}")
        # エラー時はoverviewをそのまま使用
        final_summary = overview if overview else "要約の統合中にエラーが発生しました。"
        message = HumanMessage(content=f"{final_summary}\n{url}")

        return {**state, "messages": [message], "final_summary": final_summary}
