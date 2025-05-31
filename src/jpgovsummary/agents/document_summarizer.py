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
                以下の文章を分析し、段階的に要約を作成してください。

                ## ステップ1: 文書種類の判定
                まず、この文章がどのような種類の文書かを判定してください：
                
                **判定基準：**
                - 「表紙・タイトルページ」: タイトル、組織名、日付のみで実質的な内容が少ない
                - 「目次・概要」: 章立てや概要のみで詳細な説明がない  
                - 「本文・詳細資料」: 具体的な内容、説明、データ、議論等が含まれている
                
                **判定結果**: [表紙・タイトルページ/目次・概要/本文・詳細資料]
                **判定理由**: [具体的な根拠を記述]

                ## ステップ2: 要約方針の決定
                ステップ1の判定結果に基づいて要約方針を決定してください：
                
                - 「表紙・タイトルページ」→ タイトル、組織名、基本情報のみを簡潔に記述
                - 「目次・概要」→ 構成や概要の要点を整理
                - 「本文・詳細資料」→ 重要な内容を論理的に要約
                
                **採用する方針**: [選択した方針を記述]

                ## ステップ3: 要約の作成
                ステップ2で決定した方針に従って要約を作成してください。
                
                **重要な制約：**
                - 文章に実際に書かれている内容のみを使用してください
                - 推測や補完、創作は一切行わないでください
                - 表紙・タイトルページの場合は詳細な説明を創作しないでください
                
                文章：
                {text}
                
                ## 出力形式
                **文書種類**: [判定結果]
                **要約**: [作成した要約]
                """,
            )

            combine_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                以下の要約を1つの文章にまとめてください。
                各ページの文書種類判定結果を考慮し、重複を避け、論理的な流れを保ちながら、重要な情報をすべて含めてください。

                **統合の方針：**
                - 表紙・タイトルページからは基本情報（資料名、組織名等）を抽出
                - 目次・概要からは全体構成を把握
                - 本文・詳細資料からは具体的な内容を要約
                - 各ページの判定結果に基づいて適切な重み付けを行う

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
                - 推測や創作は行わず、実際に書かれている内容のみを統合してください

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

            # 表紙・タイトルページの場合、overviewが空ならタイトル情報で置き換える
            if current_index == 0:  # 最初の文書のみチェック
                current_overview = state.get("overview", "")
                if not current_overview or current_overview.strip() == "":
                    # 「～について：」の資料名部分を抽出
                    import re
                    match = re.search(r'「([^」]+)」について[：:]', summary_content)
                    if match:
                        title = match.group(1).strip()
                        if len(title) > 3:
                            state["overview"] = f"「{title}」"
                            logger.info(f"Updated overview with title: 「{title}」")

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
