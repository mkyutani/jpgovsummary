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
                - **抽出されたテキストが不十分または意味のある内容がない場合は、要約を空文字列で返してください**
                - **OCRエラーや文字化けなど、判読不能なテキストしかない場合は空文字列を返してください**
                
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

                **事前チェック（重要）：**
                まず、入力された要約を分析してください：
                - すべてのページの要約が空文字列または「要約:」だけの場合は空文字列を返す
                - 箇条書き記号（⚫、●、•、-等）のみで構成されている場合は空文字列を返す
                - OCRエラーや文字化けしたタイトル（例：「pan L租t'aLon」）のみの場合は空文字列を返す
                - 意味のあるテキスト内容が一切含まれていない場合は空文字列を返す

                **特別処理（表紙情報がある場合）：**
                表紙・タイトルページから適切なタイトル情報が抽出できるが、本文・詳細資料からの要約が空または無意味な場合：
                - 「[適切なタイトル]について：」の形式で出力する
                - タイトル後のコロンの後は何も追加せず、そのまま終了する
                - 文書の説明や構成の説明は一切追加しない

                **統合処理（実質的内容がある場合）：**
                実質的な議論内容、検討事項、結論、データなどが含まれている場合のみ：
                - 表紙・タイトルページからは基本情報（資料名、組織名等）を抽出
                - 目次・概要からは全体構成を把握
                - 本文・詳細資料からは具体的な内容を要約
                - 各ページの判定結果に基づいて適切な重み付けを行う

                **出力形式：**
                1. 完全に無効な場合：空文字列
                2. 表紙情報のみの場合：「[タイトル]について：」
                3. 実質的内容がある場合：「[タイトル]について：[要約内容]」

                **手順：**
                1. まず事前チェックを実行し、完全に無効かを判定
                2. 完全に無効な場合は空文字列を返す
                3. 表紙情報から適切なタイトルを特定
                4. 本文・詳細資料に実質的な内容があるかを確認
                5. 実質的な内容がない場合は「[タイトル]について：」で終了
                6. 実質的な内容がある場合は「[タイトル]について：[内容]」を作成

                **例：**
                - 実質的内容がある場合：「デジタル庁個人情報保護ガイドライン」について：このガイドラインでは個人情報の適切な取り扱いについて...
                - 表紙情報のみの場合：「第3回検討会資料1-2」について：
                - 完全に無効な場合：（空文字列）

                **注意：**
                - 推測や創作は一切行わず、実際に書かれている内容のみを使用する
                - 表紙情報のみの場合は文書の説明や構成の説明は追加しない
                - タイトル後のコロンの後に無意味な説明を追加しない

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
                verbose=False,
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
