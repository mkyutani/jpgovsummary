from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from .. import Config, Model, State, logger

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
    max_chars = max(50, 300 - url_length - 1)  # 最低50文字は確保
    
    if not target_report_summaries:
        logger.warning("No report summaries found")
        final_summary = overview if overview else "要約を生成できませんでした。"
        message = HumanMessage(content=f"{final_summary}\n{url}")
        return {
            **state,
            "messages": [message],
            "final_summary": final_summary
        }
    
    # 各資料の要約を1つのテキストに結合（summaryが辞書として扱われるように修正）
    summaries_text = "\n\n".join([
        f"【{summary.get('name', '')}】\n{summary.get('content', '')}" 
        for summary in target_report_summaries 
        if summary.get('content', '')
    ])
    
    try:
        # Step 1: 資料の要約をまとめる
        combined_summary_prompt = PromptTemplate(
            input_variables=["summaries", "max_chars"],
            template="""
            以下の複数の資料の要約をまとめて、{max_chars}文字以下の簡潔な要約を作成してください。
            重要な情報を漏らさないようにしながら、重複を避け、論理的な流れを保ってください。

            # 資料の要約:
            {summaries}
            
            # 出力形式
            - {max_chars}文字以下の要約文
            - 箇条書きではなく、文章形式で
            - 専門用語は適切に使用
            - 内容の重複を避ける
            """
        )
        
        # 資料の要約を統合
        combined_result = llm.invoke(combined_summary_prompt.format(
            summaries=summaries_text,
            max_chars=max_chars
        ))
        combined_summary = combined_result.content.strip()
        
        # Step 2: 統合した要約とoverviewを合わせて最終要約を作成
        final_summary_prompt = PromptTemplate(
            input_variables=["combined_summary", "overview", "max_chars"],
            template="""
            以下の要約内容をもとに、{max_chars}文字以下で全体の要約を作成してください。

            【重要な指示】
            - overviewに含まれている会議名や資料名（例：「第1回○○会議」など）は、省略せず必ず要約文中に残してください。
            - 特に会議の回数や正式名称が失われないようにしてください。
            - 重要な情報を漏らさず、重複を避け、論理的な流れを保ってください。

            # overview
            {overview}

            # 関連資料の要約
            {combined_summary}

            # 出力形式
            - {max_chars}文字以下の要約文
            - 箇条書きではなく、文章形式でまとめる
            - 会議名や資料名（回数含む）が必ず含まれていること
            - 専門用語は適切に使用する
            - 内容の重複を避ける
            - 会議の目的や結論を明確にする
            """
        )
        
        # 最終要約を生成
        final_result = llm.invoke(final_summary_prompt.format(
            combined_summary=combined_summary,
            overview=overview,
            max_chars=max_chars
        ))
        final_summary = final_result.content.strip()
        
        # Step 3: "作成した要約\nURL"の形式でmessagesに格納
        message = HumanMessage(content=f"{final_summary}\n{url}")
        
        return {
            **state,
            "messages": [message],
            "final_summary": final_summary
        }
        
    except Exception as e:
        logger.error(f"Error in summary integration: {str(e)}")
        # エラー時はoverviewをそのまま使用
        final_summary = overview if overview else "要約の統合中にエラーが発生しました。"
        message = HumanMessage(content=f"{final_summary}\n{url}")
        
        return {
            **state,
            "messages": [message],
            "final_summary": final_summary
        } 