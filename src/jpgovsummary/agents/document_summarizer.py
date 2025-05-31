from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from .. import Model, State, Summary, TargetReportList, logger
from ..tools import load_pdf_as_text


def document_summarizer(state: State) -> State:
    """PDF文書を要約するエージェント"""

    logger.info("document_summarizer")

    llm = Model().llm()
    parser = JsonOutputParser(pydantic_object=Summary)

    # 現在のインデックスを取得
    current_index = state.get("target_report_index", 0)
    target_reports = state.get("target_reports", TargetReportList(reports=[]))

    # 初期値を設定
    summary_obj = None
    message = None
    target_report_index = current_index + 1

    try:
        if current_index >= len(target_reports):
            logger.info("All documents have been summarized")
            return state

        # 現在の文書のURLを取得
        current_report = target_reports[current_index]
        url = current_report.url
        name = current_report.name

        logger.info(f"Processing: {name} {url}")

        # PDFを読み込んでテキストを抽出
        texts = load_pdf_as_text(url)
        if not texts:
            logger.warning(f"Failed to load PDF: {url}")
            summary_obj = Summary(url=url, name=name, content="")
            message = HumanMessage(
                content=f"文書: {name}\nURL: {url}\n\n要約: (PDFを読み込めませんでした)"
            )
        else:
            # テキストを直接ドキュメントに変換
            docs = [Document(page_content=t) for t in texts]

            # まず要約を生成し、その後JSONに変換する2段階のプロセス
            # ステップ1: テキストの要約
            map_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                以下の文章を要約してください。
                重要な情報を漏らさないように、かつ簡潔にまとめてください。
                文章中に資料名、文書名、タイトルなどが含まれている場合は、それも抽出してください。

                文章：
                {text}
                """,
            )

            combine_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                以下の要約を1つの文章にまとめてください。
                重複を避け、論理的な流れを保ちながら、重要な情報をすべて含めてください。

                **必須の出力形式：**
                「[資料名]」について：[要約内容]

                **手順：**
                1. まず要約全体から、この文書の正式な資料名・文書名・タイトルを特定してください
                2. 資料名を「」で囲み、その後に「について：」を付けてください
                3. その後に要約内容を続けてください

                **例：**
                - 「デジタル庁個人情報保護ガイドライン」について：このガイドラインでは個人情報の適切な取り扱いについて...
                - 「第3回検討会資料1-2」について：本資料では新しい認証システムの検討状況について...
                - 「令和6年度ICT基盤整備計画」について：本計画では次年度のシステム整備について...

                **注意：**
                - 必ず「について：」を含む形式で出力してください
                - 資料名が不明確な場合は「資料」として出力してください
                - 出力は必ず上記の形式で始めてください

                要約：
                {text}
                """,
            )

            # 要約を生成
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=True,
            )

            summary_result = chain.invoke(docs)
            summary_content = summary_result["output_text"]

            # 既に資料名が含まれているかチェックし、含まれていない場合のみ付加
            if not ("について：" in summary_content or "について:" in summary_content):
                summary_content_with_name = f"「{name}」について：{summary_content}"
            else:
                summary_content_with_name = summary_content

            # ステップ2: 要約をJSONに変換するプロンプト
            json_prompt = PromptTemplate(
                input_variables=["content", "url", "name"],
                template="""
                以下の要約内容をJSON形式に変換してください。

                文書名: {name}
                URL: {url}
                要約: {content}

                JSON形式の出力形式：
                {format_instructions}
                """,
            )

            # JSON形式に変換
            json_chain = json_prompt | llm
            json_result = json_chain.invoke(
                {
                    "content": summary_content_with_name,
                    "url": url,
                    "name": name,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            # JSONのパースを行う
            # エラーは外側のtry-exceptでキャッチされる
            summary_obj = parser.parse(json_result.content)

            # メッセージも作成
            message = HumanMessage(
                content=f"文書: {name}\nURL: {url}\n\n要約:\n{json_result.content}"
            )

    except Exception as e:
        logger.error(f"Error occurred while summarizing document: {str(e)}")
        if current_index < len(target_reports):
            current_report = target_reports[current_index]
            summary_obj = Summary(url=current_report.url, name=current_report.name, content="")
        else:
            summary_obj = Summary(url="", name="", content="")

        message = HumanMessage(
            content=f"文書: {summary_obj.name}\nURL: {summary_obj.url}\n\n要約: (エラーのため要約できませんでした)"
        )

    # 既存のsummariesを取得し、新しい要約を追加
    current_summaries = state.get("target_report_summaries", [])
    new_summaries = current_summaries + ([summary_obj] if summary_obj else [])

    # 新しい状態を返す
    return {
        **state,
        "messages": [message] if message else [],
        "target_report_summaries": new_summaries,
        "target_report_index": target_report_index,
    }
