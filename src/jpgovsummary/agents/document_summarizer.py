from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from .. import Model, State, Summary, TargetReportList, logger
from ..tools import load_pdf_as_text


def extract_document_structure(texts: list[str]) -> tuple[str, str]:
    """p10までから文書構成・要点を抽出
    
    Returns:
        tuple[str, str]: (抽出内容, 該当ページ情報)
    """
    llm = Model().llm()
    
    # p10までのテキストをマージ（最大10ページ）
    pages_to_analyze = min(10, len(texts))
    merged_pages = []
    
    for i in range(pages_to_analyze):
        merged_pages.append(f"--- ページ {i+1} ---\n{texts[i]}")
    
    merged_text = "\n\n".join(merged_pages)
    
    structure_prompt = PromptTemplate(
        input_variables=["text", "pages_count"],
        template="""以下はPDF文書のp1-p{pages_count}の内容です。
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
{text}""",
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


def generate_structure_based_summary(structure_content: str, document_name: str) -> str:
    """抽出した構成・要点から全体要約を生成"""
    llm = Model().llm()
    
    summary_prompt = PromptTemplate(
        input_variables=["structure", "doc_name"],
        template="""以下は文書「{doc_name}」から抽出した構成・要点です。
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
        「{doc_name}」について：[要約内容]""",
    )
    
    chain = summary_prompt | llm
    result = chain.invoke({"structure": structure_content, "doc_name": document_name})
    return result.content.strip()


def detect_document_type(texts: list[str]) -> tuple[str, dict]:
    """文書タイプを判定する
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        tuple[str, dict]: (判定結果, 詳細情報)
            判定結果: "word" | "powerpoint" | "other"
            詳細情報: {"scores": {...}, "reasoning": {...}, "conclusion": str}
    """
    llm = Model().llm()
    # 最初の数ページを分析用に取得（最大5ページ）
    pages_to_analyze = min(5, len(texts))
    sample_texts = texts[:pages_to_analyze]
    
    # ページ数に応じてサンプルテキストを準備
    if pages_to_analyze == 1:
        merged_text = f"ページ1:\n{sample_texts[0]}"
    else:
        merged_text = "\n\n".join([f"ページ{i+1}:\n{text}" for i, text in enumerate(sample_texts)])
    
    # 文書判定プロンプト
    detection_prompt = PromptTemplate(
        input_variables=["text", "total_pages"],
        template="""### 目的
        以下のPDFから抽出されたテキストを分析し、元の文書がWord文書かPowerPoint資料のどちらで作成されたかを判定してください。
        判定は文書の構造パターン、テキストの配置方法、情報の組織化方式に基づいて行ってください。

        ### 分析対象
        総ページ数: {total_pages}ページ
        分析対象: 最初の{pages_count}ページ

        PDFテキスト:
        {text}

        ### 分析の観点
        以下の文書特徴を重点的に分析してください：

        **1. 文書構造の分析**
        - テキストの流れ：連続した文章か、断片的な情報か
        - ページ構成：統一された書式か、ページごとに独立したレイアウトか
        - 情報の階層：章節構造（1.1、1.2など）か、タイトル+内容の繰り返しか

        **2. テキスト配置パターン**
        - 文章の長さ：長い段落が主体か、短いフレーズが主体か
        - 箇条書きの頻度：箇条書き（・、●、-、数字）がどの程度使用されているか
        - タイトルの扱い：各ページに明確なタイトルがあるか

        **3. 情報提示方式**
        - 説明の詳しさ：詳細な説明文か、要点を絞った簡潔な表現か
        - 図表の扱い：文章に埋め込まれているか、独立して配置されているか
        - 全体の目的：詳細な報告・説明か、プレゼンテーション・概要説明か

        ### 判定カテゴリー
        1. Word文書 - 連続したテキスト、段落構造、報告書形式
        2. PowerPoint資料 - スライド形式、箇条書き、タイトル+内容構造

        ### Word文書の特徴
        - 長い段落が連続している
        - 章・節・項の階層構造（例：1.1、1.2、2.1など）
        - 文章が自然に流れる文書形式
        - 表や図表が文章に埋め込まれている
        - ページ全体にテキストが配置されている

        ### PowerPoint資料の特徴
        - 各ページに明確なタイトルがある
        - 箇条書き（・、●、-、数字など）が多用されている
        - 短いフレーズや文が中心
        - 図表やグラフが独立して配置されている
        - スライド的な構成（タイトル＋内容の組み合わせ）
        - 視覚的な強調（太字、色分けなど）が多い

        ### 判定のポイント
        - 文章の長さ：Word→長い段落、PowerPoint→短いフレーズ
        - 構造：Word→章節構造、PowerPoint→スライド構造
        - 箇条書きの使用頻度：PowerPointで頻繁に使用
        - タイトルの扱い：PowerPointでは各ページに明確なタイトル

        ### 出力形式
        **スコア分析:**
        Word: [1-5点] - [理由と根拠となるテキスト例]
        PowerPoint: [1-5点] - [理由と根拠となるテキスト例]

        **結論:**
        最も可能性が高いと判断される形式: [Word/PowerPoint]

        ### 出力要件
        - 各スコア（1～5）は、記述された理由と一貫性を保ってください。
        - テキスト主体／画像主体の両方のスコアが高くならないよう注意してください。
        - 最後に、最も可能性が高いカテゴリーを1つ明示してください。""",
    )
    
    chain = detection_prompt | llm
    result = chain.invoke({
        "text": merged_text, 
        "total_pages": len(texts),
        "pages_count": pages_to_analyze
    })
    
    # 結果をパース
    result_text = result.content.strip()
    
    # スコアと理由を抽出
    scores = {}
    reasoning = {}
    conclusion = ""
    
    lines = result_text.split('\n')
    current_section = ""
    
    for line in lines:
        line = line.strip()
        if "**スコア分析:**" in line:
            current_section = "scores"
            continue
        elif "**結論:**" in line:
            current_section = "conclusion"
            continue
        
        if current_section == "scores" and ":" in line and " - " in line:
            # "Word: 3 - 理由" の形式をパース
            parts = line.split(" - ", 1)
            if len(parts) == 2:
                category_score = parts[0].strip()
                reason = parts[1].strip()
                
                # カテゴリー名とスコアを分離
                if ":" in category_score:
                    category, score_str = category_score.split(":", 1)
                    category = category.strip()
                    score_str = score_str.strip()
                    
                    # スコア数値を抽出
                    import re
                    score_match = re.search(r'\d+', score_str)
                    if score_match:
                        score = int(score_match.group())
                        scores[category] = score
                        reasoning[category] = reason
        
        elif current_section == "conclusion" and line:
            if "最も可能性が高い" in line:
                conclusion = line
    
    # 判定結果をマッピング
    doc_type = "other"  # デフォルト
    
    if scores:
        # 最高スコアのカテゴリーを特定
        max_score = max(scores.values())
        max_categories = [cat for cat, score in scores.items() if score == max_score]
        
        if max_categories:
            top_category = max_categories[0]  # 複数ある場合は最初の一つ
            
            # カテゴリー名から doc_type を決定
            if "Word" in top_category:
                doc_type = "word"
            elif "PowerPoint" in top_category:
                doc_type = "powerpoint"
            elif "Excel" in top_category or "Scan" in top_category:
                doc_type = "other"
    
    # 共通のパース処理を使用
    return _parse_detection_result(result_text, len(texts), pages_to_analyze)


def _parse_detection_result(result_text: str, total_pages: int, analyzed_pages: int = None) -> tuple[str, dict]:
    """
    文書タイプ判定結果をパースする共通関数
    
    Args:
        result_text: LLMからの判定結果テキスト
        total_pages: 総ページ数
        analyzed_pages: 分析対象ページ数（Noneの場合はtotal_pagesと同じ）
        
    Returns:
        tuple[str, dict]: (判定結果, 詳細情報)
    """
    if analyzed_pages is None:
        analyzed_pages = total_pages
    
    # スコアと理由を抽出
    scores = {}
    reasoning = {}
    conclusion = ""
    
    lines = result_text.split('\n')
    current_section = ""
    
    for line in lines:
        line = line.strip()
        if "スコア分析:" in line:
            current_section = "scores"
            continue
        elif "結論:" in line:
            current_section = "conclusion"
            continue
        
        if current_section == "scores" and ":" in line and " - " in line:
            # "Word: 3 - 理由" の形式をパース
            parts = line.split(" - ", 1)
            if len(parts) == 2:
                category_score = parts[0].strip()
                reason = parts[1].strip()
                
                # カテゴリー名とスコアを分離
                if ":" in category_score:
                    category, score_str = category_score.split(":", 1)
                    category = category.strip()
                    score_str = score_str.strip()
                    
                    # スコア数値を抽出
                    import re
                    score_match = re.search(r'\d+', score_str)
                    if score_match:
                        score = int(score_match.group())
                        scores[category] = score
                        reasoning[category] = reason
        
        elif current_section == "conclusion" and line:
            if "最も可能性が高い" in line or "Word" in line or "PowerPoint" in line:
                conclusion = line
    
    # 判定結果をマッピング（Word vs PowerPoint の2択）
    doc_type = "other"  # デフォルト
    
    if scores:
        # WordとPowerPointのスコアを比較
        word_score = scores.get("Word", 0)
        ppt_score = scores.get("PowerPoint", 0)
        
        if word_score > ppt_score:
            doc_type = "word"
        elif ppt_score > word_score:
            doc_type = "powerpoint"
        # 同点の場合は"other"のまま
    elif conclusion:
        # スコアがない場合は結論テキストから直接判定
        if "PowerPoint" in conclusion or "スライド" in conclusion:
            doc_type = "powerpoint"
        elif "Word" in conclusion:
            doc_type = "word"
    
    # 詳細情報をまとめる
    detail_info = {
        "scores": scores,
        "reasoning": reasoning,
        "conclusion": conclusion,
        "total_pages": total_pages,
        "analyzed_pages": analyzed_pages
    }
    
    return doc_type, detail_info


def word_based_summarize(texts: list[str]) -> str:
    """Wordベース文書の要約処理
    
    タイトルと目次から文書の全体構造を把握し、構造ベースの要約を生成
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        str: 要約テキスト
    """
    # TODO: Wordベース文書専用の要約ロジックを実装
    # 現在は従来ロジックにフォールバック
    return traditional_summarize(texts)


def extract_powerpoint_title(texts: list[str]) -> str:
    """powerpointのタイトルを抽出する
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        str: 抽出されたタイトル
    """
    llm = Model().llm()
    
    # 最初の3ページからタイトル抽出
    pages_to_analyze = min(3, len(texts))
    sample_texts = texts[:pages_to_analyze]
    
    # ページ数に応じてサンプルテキストを準備
    if pages_to_analyze == 1:
        merged_text = f"ページ1:\n{sample_texts[0]}"
    else:
        merged_text = "\n\n".join([f"ページ{i+1}:\n{text}" for i, text in enumerate(sample_texts)])
    
    title_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下はPowerPoint資料の最初の数ページです。
        この資料の適切なタイトルを抽出してください。

        テキスト:
        {text}

        ### 抽出方針
        - 最初のページのメインタイトルを優先
        - 副題がある場合は含める
        - 組織名や日付は除外
        - タイトルを変更してはならない

        ### 出力形式
        タイトルのみを出力してください（説明や前置きは不要）""",
    )
    
    chain = title_prompt | llm
    result = chain.invoke({"text": merged_text})
    extracted_title = result.content.strip()
    return extracted_title


def extract_titles_and_score(texts: list[str], start_page: int, end_page: int) -> dict:
    """10ページずつスライドタイトルを抽出し、重要度をスコアリング
    
    Args:
        texts: PDFから抽出されたテキストのリスト
        start_page: 開始ページ（0ベース）
        end_page: 終了ページ（0ベース、inclusive）
    
    Returns:
        dict: {"slides": [{"page": int, "title": str, "score": int, "reason": str}]}
    """
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    
    class SlideInfo(BaseModel):
        page: int = Field(description="ページ番号")
        title: str = Field(description="スライドタイトル")
        score: int = Field(description="重要度スコア（1-5点）", ge=1, le=5)
        reason: str = Field(description="スコアの理由")
    
    class SlideAnalysis(BaseModel):
        slides: list[SlideInfo] = Field(description="スライド分析結果")
    
    llm = Model().llm()
    parser = PydanticOutputParser(pydantic_object=SlideAnalysis)
    
    # 指定範囲のページを取得
    page_texts = texts[start_page:end_page+1]
    content = "\n\n".join([f"--- ページ {start_page + i + 1} ---\n{text}" for i, text in enumerate(page_texts)])
    
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""以下のPowerPoint資料の各ページからスライドタイトルを抽出し、重要度を5点満点でスコアリングしてください。

        内容:
        {content}

        ### スコアリング基準（1-5点）
        5点: アジェンダ・目次・検討事項・まとめ・結論・今後の方針・スケジュール
        4点: 骨子・要点・ポイント・とりまとめ・提案・(案)・取組
        3点: 主要な論点・背景・課題
        2点: 説明・詳細・補足
        1点: その他

        各ページについて、ページ番号、タイトル、スコア、理由を抽出してください。

        ### 重要
        JSON形式で出力してください。最後の要素にはカンマを付けないでください。

        {format_instructions}""",
    )
    
    chain = prompt | llm
    
    # リトライ機能付きでJSONパースを実行
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "content": content,
                "format_instructions": parser.get_format_instructions(),
            })
            
            parsed_result = parser.parse(result.content)
            analysis_result = parsed_result.dict()
            if attempt > 0:
                logger.info(f"Successfully parsed JSON on retry {attempt} for pages {start_page+1}-{end_page+1}")
            return analysis_result
            
        except Exception as e:
            logger.warning(f"JSON parse attempt {attempt+1}/{max_retries} failed for pages {start_page+1}-{end_page+1}: {e}")
            if attempt == max_retries - 1:
                # 最後の試行でも失敗した場合
                logger.error(f"All {max_retries} attempts failed for pages {start_page+1}-{end_page+1}")
                logger.error(f"Final raw output: {result.content}")
                return {"slides": []}
            else:
                logger.info(f"Retrying JSON parsing for pages {start_page+1}-{end_page+1}...")
    
    # ここには到達しないはずだが、安全のため
    return {"slides": []}


def powerpoint_based_summarize(texts: list[str]) -> dict:
    """PowerPointベース文書の3段階要約処理
    
    3段階処理：
    1. 10ページずつタイトル抽出・スコアリング（LLM）
    2. 最高スコアスライド選択（非LLM）
    3. 要約作成（LLM）
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        dict: {"title": str, "summary": str}
    """
    llm = Model().llm()
    
    # ステップ1: タイトル抽出
    title = extract_powerpoint_title(texts)
    logger.info(f"Extracted PowerPoint title: {title}")
    
    # ステップ2: 10ページずつスライドタイトル抽出・スコアリング
    total_pages = len(texts)
    all_slides = []
    
    for start_page in range(0, total_pages, 10):
        end_page = min(start_page + 9, total_pages - 1)
        try:
            slide_analysis = extract_titles_and_score(texts, start_page, end_page)
            logger.info(f"Slide analysis for pages {start_page+1}-{end_page+1}/{total_pages}: {len(slide_analysis['slides'])} slides analyzed")
            for slide in slide_analysis['slides']:
                logger.info(f"  Page {slide['page']}: {slide['title']} (Score: {slide['score']} - {slide['reason']})")
            all_slides.extend(slide_analysis["slides"])
        except Exception as e:
            logger.warning(f"Failed to analyze slides {start_page+1}-{end_page+1}: {e}")
    
    # ステップ3: 最高スコアのスライドを選択
    if not all_slides:
        # スライドが取得できない場合は全文を使用
        merged_content = "\n\n".join([f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(texts)])
        page_info = f"全{total_pages}ページ（スライド分析失敗）"
        selected_slide_info = "分析失敗のため全ページ使用"
    else:
        # スコアでソートし、最高スコアのスライドのみを選択
        sorted_slides = sorted(all_slides, key=lambda x: x["score"], reverse=True)
        max_score = sorted_slides[0]["score"]
        top_slides = [slide for slide in sorted_slides if slide["score"] == max_score]

        logger.info(f"Selected slides: {', '.join([str(slide['page']) for slide in top_slides])}")

        # 最高スコアのスライドのテキストを取得
        selected_texts = []
        for slide in top_slides:
            page_idx = slide["page"] - 1  # 1ベースから0ベースに変換
            if 0 <= page_idx < len(texts):
                selected_texts.append(f"--- ページ {slide['page']} ({slide['title']}) ---\n{texts[page_idx]}")
        
        merged_content = "\n\n".join(selected_texts)
        page_info = f"最高スコア{max_score}点のスライド{len(top_slides)}枚（総{total_pages}ページ中）"
        selected_slide_info = f"選択されたスライド: " + ", ".join([f"ページ{s['page']}({s['title']})" for s in top_slides])
    
    # ステップ4: 要約作成
    powerpoint_summary_prompt = PromptTemplate(
        input_variables=["title", "content", "page_info", "selected_slide_info"],
        template="""以下はPowerPoint資料「{title}」の重要スライドです。

        分析対象: {page_info}
        {selected_slide_info}

        内容:
        {content}

        ### 要約作成
        以下の構成で簡潔に要約してください：
        - 資料の目的・背景
        - 主要な検討事項・論点
        - 結論・提案・今後の方向性

        ### 出力形式
        「{title}」について：[要約内容]

        ### 制約
        - 簡潔で分かりやすく
        - 提供されたスライドの内容のみ使用
        - 推測や補完は行わない""",
    )
    
    chain = powerpoint_summary_prompt | llm
    result = chain.invoke({
        "title": title,
        "content": merged_content,
        "page_info": page_info,
        "selected_slide_info": selected_slide_info
    })
    
    return {"title": title, "summary": result.content.strip()}


def traditional_summarize(texts: list[str]) -> str:
    """従来の全文要約処理"""
    llm = Model().llm()
    
    # テキストを直接ドキュメントに変換
    docs = [Document(page_content=t) for t in texts]

    # まず要約を生成し、その後JSONに変換する2段階のプロセス
    # ステップ1: テキストの要約
    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下の文章を分析し、段階的に要約を作成してください。

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
        **要約**: [作成した要約]""",
    )

    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下の要約を1つの文章にまとめてください。

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
        {text}""",
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

        logger.info(f"Processing {url}")

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
            # 文書タイプを判定
            doc_type, detection_detail = detect_document_type(texts)
            
            # 判定結果のログ出力
            if detection_detail.get("scores"):
                top_scores = sorted(detection_detail["scores"].items(), key=lambda x: x[1], reverse=True)[:3]
                score_summary = ", ".join([f"{cat}:{score}" for cat, score in top_scores])
                logger.info(f"Processing as {doc_type}-based document (scores: {score_summary})")
            else:
                logger.info(f"Processing as {doc_type}-based document")

            # タイプ別要約処理
            powerpoint_result: dict | None = None
            if doc_type == "word":
                summary_content = word_based_summarize(texts)
            elif doc_type == "powerpoint":
                powerpoint_result = powerpoint_based_summarize(texts)
                # powerpointは {title, summary} のdictを返す
                summary_content = f"「{powerpoint_result.get('title', name)}」について：{powerpoint_result.get('summary', '').strip()}"
            else:
                # Word/powerpoint以外はスキップ
                logger.info(f"Skipping non-Word/PowerPoint document: {name}")
                message = HumanMessage(
                    content=f"文書: {name}\nURL: {url}\n\n要約: (Word/powerpoint以外のためスキップ)"
                )
                return {
                    **state,
                    "messages": [message],
                    "target_report_summaries": state.get("target_report_summaries", []),
                    "target_report_index": target_report_index,
                }

            # 表紙・タイトルページの場合、対象reportのnameをタイトル情報で上書きする
            if current_index == 0 and name == "PDF":  # 最初の文書のみチェック
                if doc_type == "powerpoint" and powerpoint_result and powerpoint_result.get("title"):
                    title = powerpoint_result.get("title", "").strip()
                    if len(title) > 3:
                        # target_reportsの現在のレポートのnameを更新
                        current_report.name = title
                        logger.info(f"Updated report name with title: {title}")
            else:
                    # 「～について：」の資料名部分を抽出（非powerpoint）
                    import re
                    match = re.search(r'「([^」]+)」について[：:]', summary_content)
                    if match:
                        title = match.group(1).strip()
                        if len(title) > 3:
                            # target_reportsの現在のレポートのnameを更新
                            current_report.name = title
                            logger.info(f"Updated report name with title: {title}")

            # 既に資料名が含まれているかチェックし、含まれていない場合のみ付加
            if doc_type == "powerpoint":
                # powerpointは既に「タイトル」について：形式にしている
                summary_content_with_name = summary_content
            elif not ("について：" in summary_content or "について:" in summary_content):
                summary_content_with_name = f"「{name}」について：{summary_content}"
            else:
                summary_content_with_name = summary_content

            # ステップ2: 要約をJSONに変換するプロンプト
            json_prompt = PromptTemplate(
                input_variables=["content", "url", "name"],
                template="""以下の要約内容をJSON形式に変換してください。

                文書名: {name}
                URL: {url}
                要約: {content}

                JSON形式の出力形式：
                {format_instructions}""",
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

            # 判定結果をsummary_objに追加（カスタム属性として）
            summary_obj.document_type = doc_type
            summary_obj.detection_detail = detection_detail

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
