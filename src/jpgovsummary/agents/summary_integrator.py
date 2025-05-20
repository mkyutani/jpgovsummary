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
            input_variables=["summaries"],
            template="""
            以下の複数の資料の要約をまとめて、200文字程度の簡潔な要約を作成してください。
            重要な情報を漏らさないようにしながら、重複を避け、論理的な流れを保ってください。

            # 資料の要約:
            {summaries}
            
            # 出力形式
            - 200文字程度の要約文
            - 箇条書きではなく、文章形式で
            - 専門用語は適切に使用
            - 内容の重複を避ける
            """
        )
        
        # 資料の要約を統合
        combined_result = llm.invoke(combined_summary_prompt.format(
            summaries=summaries_text
        ))
        combined_summary = combined_result.content.strip()
        
        # Step 2: 統合した要約とoverviewを合わせて最終要約を作成
        final_summary_prompt = PromptTemplate(
            input_variables=["combined_summary", "overview"],
            template="""
            以下の会議の概要と関連資料の要約をまとめて、200文字程度の最終的な要約を作成してください。
            重要な情報を漏らさないようにしながら、重複を避け、論理的な流れを保ってください。

            # 会議の概要:
            {overview}
            
            # 関連資料の要約:
            {combined_summary}
            
            # 出力形式
            - 200文字程度の要約文
            - 箇条書きではなく、文章形式で
            - 専門用語は適切に使用
            - 内容の重複を避ける
            - 会議の目的や結論を明確に
            """
        )
        
        # 最終要約を生成
        final_result = llm.invoke(final_summary_prompt.format(
            combined_summary=combined_summary,
            overview=overview
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