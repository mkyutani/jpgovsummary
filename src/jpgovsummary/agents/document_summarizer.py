from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from .. import Model, State, Summary, TargetReportList, logger
from ..tools import load_pdf_as_text


def extract_document_structure(texts: list[str], llm) -> tuple[str, str]:
    """p10までから文書構成・要点を抽出
    
    Returns:
        tuple[str, str]: (抽出内容, 該当ページ情報)
    """
    
    # p10までのテキストをマージ（最大10ページ）
    pages_to_analyze = min(10, len(texts))
    merged_pages = []
    
    for i in range(pages_to_analyze):
        merged_pages.append(f"--- ページ {i+1} ---\n{texts[i]}")
    
    merged_text = "\n\n".join(merged_pages)
    
    structure_prompt = PromptTemplate(
        input_variables=["text", "pages_count"],
        template="""
        以下はPDF文書のp1-p{pages_count}の内容です。
        文書の構成や要点を示す部分を特定し、該当箇所があれば抽出してください。

        **抽出対象：**
        - 目次、Contents、構成（複数ページにわたる場合も含む）
        - 目次（続き）、Contents (continued)
        - 本日の論点、論点、検討事項
        - 議題、アジェンダ、Agenda
        - 概要、要点、ポイント
        - 今回の内容、本日の内容
        - その他、文書の全体構成や主要論点を示す項目リスト

        **抽出ルール：**
        - 該当箇所の内容をそのまま抽出（見出し含む）
        - **複数ページにわたる目次は統合して抽出**
        - 目次の一部分のみの場合も抽出対象とする
        - 複数箇所ある場合はすべて抽出
        - 要約や解釈は行わない
        - 該当箇所がない場合は「なし」と回答

        **特別注意：**
        - 目次が複数ページに分かれている場合、すべてのページの目次部分を統合して抽出
        - 「目次（続き）」「Contents (continued)」なども目次として認識
        - ページ番号や章番号の連続性から目次の継続を判断
        - 各ページの目次部分を見つけた場合は、ページ境界を超えて統合

        **出力形式：**
        該当箇所がある場合：
        PAGES: [該当ページ番号をカンマ区切りで列挙（例：1,2,3）]
        CONTENT: [抽出内容（複数ページの場合は統合）]
        
        該当箇所がない場合：
        PAGES: なし
        CONTENT: なし

        文書内容：
        {text}
        """,
    )
    
    chain = structure_prompt | llm
    result = chain.invoke({"text": merged_text, "pages_count": pages_to_analyze})
    
    # 結果をパース
    result_text = result.content.strip()
    
    # PAGES: とCONTENT: を抽出
    pages_info = "不明"
    content = "なし"
    
    lines = result_text.split('\n')
    for line in lines:
        if line.startswith('PAGES:'):
            pages_info = line.replace('PAGES:', '').strip()
        elif line.startswith('CONTENT:'):
            content = line.replace('CONTENT:', '').strip()
            # 残りの行も content に含める
            idx = lines.index(line)
            if idx + 1 < len(lines):
                remaining_lines = lines[idx + 1:]
                content += '\n' + '\n'.join(remaining_lines)
            break
    
    return content, pages_info


def generate_structure_based_summary(structure_content: str, document_name: str, llm) -> str:
    """抽出した構成・要点から全体要約を生成"""
    
    summary_prompt = PromptTemplate(
        input_variables=["structure", "doc_name"],
        template="""
        以下は文書「{doc_name}」から抽出した構成・要点です。
        この情報を基に文書全体の要約を作成してください。

        **要約方針：**
        - 抽出された構成・要点から文書の全体像を把握
        - 主要なテーマや論点を特定
        - 文書の目的や内容の概要を簡潔にまとめる
        - 実際に記載されている内容のみを使用

        **制約：**
        - 推測や補完は行わない
        - 構成・要点から読み取れる範囲での要約に留める
        - 簡潔で分かりやすい文章にする

        抽出された構成・要点：
        {structure}

        **出力形式：**
        「{doc_name}」について：[要約内容]
        """,
    )
    
    chain = summary_prompt | llm
    result = chain.invoke({"structure": structure_content, "doc_name": document_name})
    return result.content.strip()


def traditional_summarize(texts: list[str], llm) -> str:
    """従来の全文要約処理"""
    
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
    return summary_result["output_text"]


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
            
            # 要約内容をログに出力
            logger.info(f"Summary: {summary_obj.content.replace('\n', '\\n')}")
            
            message = HumanMessage(
                content=f"文書: {name}\nURL: {url}\n\n要約: (PDFを読み込めませんでした)"
            )
        else:
            # 文書構成・要点の抽出を試行
            if len(texts) > 10:
                logger.info(f"Extracting document structure from first 10 pages (total pages: {len(texts)})")
            else:
                logger.info(f"Extracting document structure")
            structure_content, pages_info = extract_document_structure(texts, llm)
            
            if structure_content != "なし" and structure_content.strip():
                # 構成・要点が見つかった場合
                logger.info(f"Structure found on pages: {pages_info}")
                summary_content = generate_structure_based_summary(
                    structure_content, name, llm
                )
            else:
                # 構成・要点が見つからない場合は全ページ探索
                logger.info(f"No structure found. Full-page analysis for: {name} (total pages: {len(texts)})")
                summary_content = traditional_summarize(texts, llm)

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
            parsed_dict = parser.parse(json_result.content)
            # 辞書をSummaryオブジェクトに変換
            summary_obj = Summary(**parsed_dict)

            # 要約内容をログに出力
            logger.info(f"Summary: {summary_obj.content.replace('\n', '\\n')}")

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
