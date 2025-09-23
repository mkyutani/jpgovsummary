from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser

from .. import Model, State, Summary, TargetReportList, logger
from ..tools import load_pdf_as_text


def detect_document_type(texts: list[str]) -> tuple[str, str, str, dict]:
    """文書タイプを判定する
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        tuple[str, str, str, dict]: (判定結果, 判定理由, 根拠テキスト, 詳細情報)
            判定結果: "word" | "powerpoint" | "agenda" | "participants" | "other"
            判定理由: 選択されたカテゴリーの判定理由
            根拠テキスト: 判定の根拠となるテキスト
            詳細情報: {"scores": {...}, "reasoning": {...}, "conclusion": str}
    """
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    
    class CategoryAnalysis(BaseModel):
        score: int = Field(description="重要度スコア（1-5点）", ge=1, le=5)
        reason: str = Field(description="スコアの理由")
        evidence: str = Field(description="根拠テキスト例")
    
    class DocumentTypeAnalysis(BaseModel):
        word: CategoryAnalysis = Field(description="Word文書の分析")
        powerpoint: CategoryAnalysis = Field(description="PowerPoint文書の分析")
        agenda: CategoryAnalysis = Field(description="議事次第の分析")
        participants: CategoryAnalysis = Field(description="参加者一覧の分析")
        news: CategoryAnalysis = Field(description="ニュース・お知らせの分析")
        survey: CategoryAnalysis = Field(description="調査・アンケートの分析")
        other: CategoryAnalysis = Field(description="その他の分析")
        conclusion: str = Field(description="最も可能性が高いと判断される形式")
    
    llm = Model().llm()
    parser = PydanticOutputParser(pydantic_object=DocumentTypeAnalysis)
    # 最初の数ページを分析用に取得（最大5ページ）
    pages_to_analyze = min(5, len(texts))
    sample_texts = texts[:pages_to_analyze]
    logger.info(f"先頭{len(sample_texts)}ページを分析して文書タイプを判定します")

    # ページ数に応じてサンプルテキストを準備
    if pages_to_analyze == 1:
        merged_text = f"ページ1:\n{sample_texts[0]}"
    else:
        merged_text = "\n\n".join([f"ページ{i+1}:\n{text}" for i, text in enumerate(sample_texts)])
    
    # 文書判定プロンプト
    detection_prompt = PromptTemplate(
        input_variables=["text", "total_pages", "pages_count", "format_instructions"],
        template="""### 目的
分析対象のPDFから抽出されたテキストを分析し、判定カテゴリーに沿って元の文書タイプを判定してください。

### 判定カテゴリー（7分類）

**1. Word文書 (word)**
- 連続した長文や段落構造を持つ実質的な内容
- 議事録（発言内容・議論の記録）
- 詳細な説明、分析、報告書
- ガイドライン、仕様書、制度説明

**2. PowerPoint資料 (powerpoint)**
- スライド形式の実質的なプレゼンテーション内容
- 分析結果、提案、説明資料
- 図表と説明の組み合わせ
- 概要資料、サマリー資料

**3. 議事次第 (agenda)**
- 会議の議事次第、議事日程、アジェンダ
- 開催日時、場所、議題リスト
- 配付資料一覧、会議進行スケジュール
- 「議事次第」「アジェンダ」等のタイトル

**4. 参加者一覧 (participants)**
- 委員名簿、参加者リストが文書の主要内容（50%以上）
- **1ページ目が参加者一覧の場合**: 文書の主目的が参加者情報
- **2ページ目以降に参加者一覧がある場合**: 付随資料の可能性が高い → agenda として判定
- 名前、所属、役職の一覧が文書の大部分を占める
- 「委員名簿」「参加者一覧」「出席者」等のタイトルで、かつ実際に参加者情報が主要コンテンツ
- **注意**: 議事次第に参加者一覧が付随している場合は「agenda」として判定

**5. ニュース・お知らせ (news)**
- 報道発表、プレスリリース、お知らせ
- 日付、部署名、見出し、本文、問い合わせ先の構造
- 政策発表、事業案内、取り組み紹介
- 「報道発表」「プレスリリース」「お知らせ」等のタイトル
- 添付資料一覧、同時発表先の記載

**6. 調査・アンケート結果 (survey)**
- アンケート集計結果、調査結果、意識調査報告
- 表形式のデータが文書の大部分（70%以上）を占める
- 質問項目と回答データの明確な対応関係
- 「アンケート結果」「調査結果」「集計結果」「意識調査」「ヒアリングシート」等のタイトル
- 数値データ、グラフ、統計情報が主要コンテンツ
- 選択肢別回答数、回答率、集計値などの定量的データ
- 質問番号（Q1、問1、設問1等）と回答の組み合わせ
- 罫線や表組みが多用されている文書構造
- データの分析や考察よりも、集計結果の提示が主目的

**7. その他 (other)**
- 手書き文書のスキャン、複雑なレイアウト
- 図表・グラフのみで説明文が少ない（調査結果以外）
- 開催案内、事務連絡、判読困難な文書
- 上記6分類に明確に該当しない文書

### 重要な区別ポイント
- 議事録（実際の発言記録）→ word
- 議事次第（スケジュールのみ）→ agenda
- 議事次第+参加者一覧（混合文書）→ agenda（議事次第が主目的）
- 参加者一覧のみ（名簿が主目的）→ participants
- **調査・アンケート結果（質問と回答データ）→ survey**
- **単なるデータ表示（参加者一覧など）→ participants**
- **実質的な調査結果（質問と回答の対応）→ survey**
- 実質的内容の有無が重要な判断基準
- **文書の主要目的を判定基準とし、付随的な情報は無視する**
- **表の目的が重要：名簿・リスト vs 調査結果・データ分析**

### 判定のポイント
- 文章の長さ：Word→長い段落、PowerPoint→短いフレーズ
- 構造：Word→章節構造、PowerPoint→スライド構造
- 箇条書きの使用頻度：PowerPointで頻繁に使用
- タイトルの扱い：PowerPointでは各ページに明確なタイトル
- **表の内容：survey→調査データ・統計、participants→人名・組織のリスト**
- **数値の種類：survey→集計値・割合・評価点、others→連絡先・番号**
- **文書の目的：survey→調査結果の報告、others→情報の整理・提供**
- **質問形式：survey→明確な質問項目と回答、others→項目名とデータ**

### 表構造・データ文書の特別な判定基準
以下の特徴が多く見られる場合は survey カテゴリーを検討：
- 質問番号や設問番号（Q1、問1、設問1、質問1等）
- 回答選択肢（はい/いいえ、満足/不満足、1-5段階評価等）
- 集計用語（合計、平均、割合、件数、回答者数、パーセント等）
- 調査関連用語（回答、回答者、対象者、サンプル、調査期間等）
- 表形式の回答データ（数値、割合、グラフ等）
- 罫線や表組みが文書の大部分を占める構造

### 出力形式（必ず以下の形式で出力してください）
**スコア分析:**

**Word:**
**スコア**: 3
**理由**: 文書に含まれる文字情報の特徴を具体的に説明
**根拠テキスト例**: 「文書から引用した具体的なテキスト」

**PowerPoint:**
**スコア**: 5
**理由**: プレゼンテーション形式の特徴を具体的に説明
**根拠テキスト例**: 「文書から引用した具体的なテキスト」

**Agenda:**
**スコア**: 1
**理由**: 議題形式の特徴を具体的に説明
**根拠テキスト例**: 「文書から引用した具体的なテキスト」

**Participants:**
**スコア**: 2
**理由**: 参加者情報の特徴を具体的に説明
**根拠テキスト例**: 「文書から引用した具体的なテキスト」

**News:**
**スコア**: 1
**理由**: ニュース形式の特徴を具体的に説明
**根拠テキスト例**: 「文書から引用した具体的なテキスト」

**Survey:**
**スコア**: 1
**理由**: アンケート形式の特徴を具体的に説明
**根拠テキスト例**: 「文書から引用した具体的なテキスト」

**Other:**
**スコア**: 2
**理由**: その他の形式の特徴を具体的に説明
**根拠テキスト例**: 「文書から引用した具体的なテキスト」

**結論:**
最も可能性が高いと判断される形式: powerpoint

### 重要な注意事項
- 必ず7つのカテゴリー全てに「**スコア**:」「**理由**:」「**根拠テキスト例**:」を記載してください
- スコアは1-5の数字のみを記載してください
- 理由は文書の特徴を具体的に説明してください
- 根拠テキスト例は文書の内容から実際のテキストを引用してください

### 出力要件
- 各スコア（1～5）は、記述された理由と一貫性を保ってください。
- 複数カテゴリーが同じ高スコアにならないよう注意してください。
- 最後に、最も可能性が高いカテゴリーを1つ明示してください。

### 分析対象
総ページ数: {total_pages}ページ
分析対象: 最初の{pages_count}ページ

PDFテキスト:
{text}

### 出力フォーマット
以下のJSON形式で出力してください：

{format_instructions}
    """)
    
    chain = detection_prompt | llm | parser
    result = chain.invoke({
        "text": merged_text, 
        "total_pages": len(texts),
        "pages_count": pages_to_analyze,
        "format_instructions": parser.get_format_instructions()
    })
    
    # Pydanticオブジェクトから情報を抽出
    scores = {
        "Word": result.word.score,
        "PowerPoint": result.powerpoint.score,
        "Agenda": result.agenda.score,
        "Participants": result.participants.score,
        "News": result.news.score,
        "Survey": result.survey.score,
        "Other": result.other.score
    }
    
    reasoning = {
        "Word": result.word.reason,
        "PowerPoint": result.powerpoint.reason,
        "Agenda": result.agenda.reason,
        "Participants": result.participants.reason,
        "News": result.news.reason,
        "Survey": result.survey.reason,
        "Other": result.other.reason
    }
    
    evidence = {
        "Word": result.word.evidence,
        "PowerPoint": result.powerpoint.evidence,
        "Agenda": result.agenda.evidence,
        "Participants": result.participants.evidence,
        "News": result.news.evidence,
        "Survey": result.survey.evidence,
        "Other": result.other.evidence
    }
    
    conclusion = result.conclusion
    
    # 判定結果をマッピング（7カテゴリ）
    doc_type = None
    doc_reason = ""
    
    # カテゴリー名とタイプのマッピング
    category_mapping = {
        "word": "Word",
        "powerpoint": "PowerPoint", 
        "agenda": "Agenda",
        "participants": "Participants",
        "news": "News",
        "survey": "Survey",
        "other": "Other"
    }
    
    if scores:
        # 最高スコアのカテゴリーを特定
        max_score = max(scores.values())
        max_categories = [cat for cat, score in scores.items() if score == max_score]
        
        if max_categories:
            top_category = max_categories[0]  # 複数ある場合は最初の一つ
            
            # カテゴリー名から doc_type を決定
            for doc_type_key, category_name in category_mapping.items():
                if category_name in top_category:
                    doc_type = doc_type_key
                    doc_reason = reasoning.get(category_name, "不明")
                    selected_evidence = evidence.get(category_name, "不明")
                    break

    if doc_type is None:
        # スコアから文書タイプを得られなかった場合は結論テキストから直接判定
        conclusion_lower = conclusion.lower()
        for doc_type_key in category_mapping.keys():
            if doc_type_key in conclusion_lower:
                doc_type = doc_type_key
                doc_reason = "結論から判定"
                selected_evidence = "なし"
                break
    
    # 詳細情報をまとめる
    detail_info = {
        "scores": scores,
        "reasoning": reasoning,
        "conclusion": conclusion,
        "total_pages": len(texts),
        "analyzed_pages": pages_to_analyze
    }

    sorted_detail_info = sorted(detail_info["scores"].items(), key=lambda x: x[1], reverse=True)
    logger.info(f"文書タイプを推定しました: {', '.join([f'{cat}:{score}' for cat, score in sorted_detail_info])}")
    logger.info(f"{doc_type}の推定理由: {doc_reason}")
    logger.info(f"{doc_type}の根拠: {selected_evidence}")

    return doc_type, doc_reason, selected_evidence, detail_info


def extract_word_title(texts: list[str]) -> str:
    """Word文書のタイトルを抽出する
    
    Args:
        texts: PDFから抽出されたテキストのリスト
        
    Returns:
        str: 抽出されたタイトル
    """
    llm = Model().llm()
    
    # 最初の5ページを取得
    title_pages = min(5, len(texts))
    merged_text = "\n\n".join([f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(texts[:title_pages])])
    
    title_prompt = PromptTemplate(
        input_variables=["text", "pages"],
        template="""以下はWord文書の最初の{pages}ページです。文書のタイトルを抽出してください。

{text}

### タイトル抽出基準
- 表紙や冒頭に記載された文書の主タイトル
- 副題がある場合は主タイトルのみ
- 章タイトルではなく文書全体のタイトル
- 組織名や日付は除外

### 出力形式
タイトルのみを出力してください（説明や前置きは不要）
    """)
    
    chain = title_prompt | llm
    result = chain.invoke({"text": merged_text, "pages": title_pages})
    extracted_title = result.content.strip()
    return extracted_title


def extract_word_table_of_contents(texts: list[str]) -> str:
    """Word文書の目次を抽出する
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        str: 抽出された目次（構造化されたテキスト）
    """
    llm = Model().llm()
    
    # 最初の10ページから目次を抽出
    toc_pages = min(10, len(texts))
    merged_text = "\n\n".join([f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(texts[:toc_pages])])
    
    toc_prompt = PromptTemplate(
        input_variables=["text", "pages"],
        template="""以下はWord文書の最初の{pages}ページです。目次部分を抽出してください。

{text}

### 目次抽出基準
- 「目次」「Contents」などの見出しがある部分
- 章・節・項の構造を持つリスト
- ページ番号が記載されている項目リスト
- 文書の構成を示す階層的なリスト

### 抽出対象外
- 単なる箇条書き
- 図表リスト
- 参考文献リスト
- 配付資料一覧

### 出力形式
目次が見つかった場合:
```
第1章 ○○○
  1.1 ○○○
  1.2 ○○○
第2章 ○○○
  2.1 ○○○
```

目次が見つからない場合:
目次なし

### 出力要件
- 目次の階層構造を保持
- ページ番号は除外
- 簡潔で読みやすい形式
    """)
    
    chain = toc_prompt | llm
    result = chain.invoke({"text": merged_text, "pages": toc_pages})
    
    extracted_toc = result.content.strip()
    
    return extracted_toc


def create_summary_from_toc(title: str, table_of_contents: str) -> str:
    """目次から要約を作成する
    
    Args:
        title: 文書のタイトル
        table_of_contents: 抽出された目次
        
    Returns:
        str: 目次ベースの要約
    """
    llm = Model().llm()
    
    toc_summary_prompt = PromptTemplate(
        input_variables=["title", "toc"],
        template="""以下のWord文書のタイトルと目次から、文書の要約を作成してください。

### 文書タイトル
{title}

### 目次
{toc}

### 要約作成の方針
- 目次の構造から文書の全体像を把握
- 各章・節の内容を推測して論理的な流れを構築
- 文書の目的・背景・主要論点・結論を整理
- 具体的な内容は推測せず、構造に基づいた概要を記述

### 出力形式
「{title}」：
- 文書の目的・位置づけ
- 主要な検討項目・論点（目次の章構成に基づく）
- 文書の特徴・性格

### 制約
- 簡潔で分かりやすく（3-5文程度）
- 目次にない具体的内容は推測しない
- 文書の構造と論理的な流れを重視
- 「検討」「分析」「提案」等の性格を明示
    """)
    
    chain = toc_summary_prompt | llm
    result = chain.invoke({
        "title": title,
        "toc": table_of_contents
    })
    
    summary = result.content.strip()
    
    return summary


def agenda_summarize(texts: list[str]) -> dict:
    """議事次第の要約処理
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        dict: {"title": str, "summary": str}
    """
    llm = Model().llm()
    
    # 全文を結合
    merged_text = "\n\n".join([f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(texts)])
    
    # タイトル抽出
    title_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下の議事次第から会議名を抽出してください。

{text}

### 抽出基準
- 文書の主タイトルとなる会議名
- 回数や日付は除外
- 組織名は含めても可

### 出力形式
会議名のみを出力してください（説明や前置きは不要）
    """)
    
    title_chain = title_prompt | llm
    title_result = title_chain.invoke({"text": merged_text})
    title = title_result.content.strip()
    
    # 要約作成
    agenda_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下は議事次第です。基本情報を簡潔に要約してください。

{text}

### 抽出項目
- 会議名
- 開催日時
- 開催場所
- 主要議題（最大3項目）

### 出力形式
「[会議名]」の議事次第（[開催日時]、[場所]）：主要議題は[議題1]、[議題2]等

### 制約
- 1-2文で簡潔に
- 日時は「令和○年○月○日」形式で
    """)
    
    chain = agenda_prompt | llm
    result = chain.invoke({"text": merged_text})
    summary = result.content.strip()
    
    return {"title": title, "summary": summary}


def news_based_summarize(texts: list[str]) -> dict:
    """ニュース・お知らせ（プレスリリース）の要約処理
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        dict: {"title": str, "summary": str}
    """
    llm = Model().llm()
    
    # 全文を結合
    merged_text = "\n\n".join([f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(texts)])
    
    # タイトル抽出
    title_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下のプレスリリース・報道発表・お知らせから主要タイトルを抽出してください。

{text}

### 抽出基準
- 文書の主タイトル（見出し）
- 日付や部署名は除外
- 「報道発表」「お知らせ」等の接頭語は除外

### 出力形式
タイトルのみを出力してください（説明や前置きは不要）
    """)
    
    title_chain = title_prompt | llm
    title_result = title_chain.invoke({"text": merged_text})
    title = title_result.content.strip()
    
    # 要約作成
    news_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下はプレスリリース・報道発表・お知らせです。重要な情報を簡潔に要約してください。

{text}

### 抽出項目
- 発表内容・事業名
- 発表元（部署・組織名）
- 主要な内容・目的
- 開始時期・実施予定（該当する場合）
- 背景・意義（簡潔に）

### 出力形式
「[タイトル]」：[発表元]が[発表内容]について発表。[主要な内容・目的]を[時期]に実施予定。[背景・意義]

### 制約
- 2-3文で簡潔に
- 具体的な事実のみを記載
- 問い合わせ先や技術的詳細は除外
- 日付は「令和○年○月○日」形式で
    """)
    
    chain = news_prompt | llm
    result = chain.invoke({"text": merged_text})
    summary = result.content.strip()
    
    return {"title": title, "summary": summary}


def participants_summarize(texts: list[str]) -> dict:
    """参加者一覧の要約処理
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        dict: {"title": str, "summary": str}
    """
    llm = Model().llm()
    
    # 全文を結合
    merged_text = "\n\n".join([f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(texts)])
    
    # タイトル抽出
    title_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下の参加者一覧・委員名簿から委員会・会議名を抽出してください。

{text}

### 抽出基準
- 委員会・会議の正式名称
- 回数や日付は除外
- 組織名は含めても可

### 出力形式
委員会・会議名のみを出力してください（説明や前置きは不要）
    """)

    title_chain = title_prompt | llm
    title_result = title_chain.invoke({"text": merged_text})
    title = title_result.content.strip()
    
    # 要約作成
    participants_prompt = PromptTemplate(
        input_variables=["text"],
        template="""以下は参加者一覧・委員名簿です。基本情報を簡潔に要約してください。

{text}

### 抽出項目
- 委員会・会議名
- 総参加者数
- 主要メンバー（座長、委員長等の役職者）

### 出力形式
「[委員会名]」委員名簿：委員[○]名、座長は[氏名]（[所属]）

### 制約
- 1文で簡潔に
- 役職者が複数いる場合は代表者のみ
    """)
    
    chain = participants_prompt | llm
    result = chain.invoke({"text": merged_text})
    summary = result.content.strip()
    
    return {"title": title, "summary": summary}


def word_based_summarize(texts: list[str]) -> dict:
    """Wordベース文書の要約処理
    
    タイトルと目次から文書の全体構造を把握し、構造ベースの要約を生成
    
    Args:
        texts: PDFから抽出されたページ別テキストのリスト
        
    Returns:
        dict: {"title": str, "summary": str}
    """
    llm = Model().llm()
    
    # ステップ1: タイトル抽出
    title = extract_word_title(texts)
    logger.info(f"このスライドのタイトルは「{title.replace('\n', '\\n')}」です")

    # ステップ2: 目次抽出
    table_of_contents = extract_word_table_of_contents(texts)
    
    # ステップ3: 目次から要約を作成
    if table_of_contents and table_of_contents != "目次なし":
        logger.info("目次から要約を作成します")
        summary = create_summary_from_toc(title, table_of_contents)
    else:
        # 目次がない場合は従来ロジックを使用
        logger.info("目次が見つからないため、全文から要約を作成します")
        summary = traditional_summarize(texts)
    
    return {"title": title, "summary": summary}


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
タイトルのみを出力してください（説明や前置きは不要）
        """)
    
    chain = title_prompt | llm
    result = chain.invoke({"text": merged_text})
    extracted_title = result.content.strip()
    return extracted_title


def extract_titles_and_score(texts: list[str], start_page: int, end_page: int):
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
    
    # スコアリング基準：振り返りより論点を優先
    scoring_criteria = """
5点: アジェンダ・目次・検討事項・主な論点・まとめ・結論・骨子
4点: 要点・ポイント・とりまとめ・提案・(案)・取組・重要課題・今後の方針・スケジュール
3点: 振り返り・背景・課題・分析結果・戦略
2点: 説明・詳細・補足・参考資料・事例紹介
1点: その他・表紙・事務連絡
"""

    prompt = PromptTemplate(
        input_variables=["content", "format_instructions"],
        template=f"""以下のPowerPoint資料の各ページからスライドタイトルを抽出し、重要度を5点満点でスコアリングしてください。

内容:
{{content}}

### スコアリング基準（1-5点）
{scoring_criteria}

### 重要な判定ポイント
- 「主な論点」「検討事項」「今回の論点」は5点（最重要）
- 「第○回の振り返り」「前回の振り返り」は3点（中程度）
- アジェンダや目次の中に複数項目が含まれる場合、実質的な論点部分を重視
- 論点提示 > 振り返り内容の優先度で判定してください

各ページについて、ページ番号、タイトル、スコア、理由を抽出してください。

#### 出力フォーマット
以下のフォーマットで出力してください。最後の要素にはカンマを付けないでください。

{{format_instructions}}
        """)
    
    chain = prompt | llm | parser
    
    # リトライ機能付きでJSONパースを実行
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"再検索({attempt+1}回目)")
            result = chain.invoke({
                "content": content,
                "format_instructions": parser.get_format_instructions(),
            })
            
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"❌ 適切なフォーマットによる結果を得られませんでした")
                return SlideAnalysis(slides=[])
            else:
                continue
    # ここには到達しないはずだが、安全のため
    return SlideAnalysis(slides=[])


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
    logger.info(f"このスライドのタイトルは「{title.replace('\n', '\\n')}」です")
    
    # ステップ2: 指定ページ数ずつスライドタイトル抽出・スコアリング
    pages_per_batch = 20  # 一度に処理するページ数
    total_pages = len(texts)
    all_slides = []

    for start_page in range(0, total_pages, pages_per_batch):
        end_page = min(start_page + pages_per_batch - 1, total_pages - 1)
        try:
            logger.info(f"スライドタイトルからスライドの内容を推定します(ページ{start_page+1}-{end_page+1}/{total_pages})")
            slide_analysis = extract_titles_and_score(texts, start_page, end_page)
            for slide in slide_analysis.slides:
                logger.info(f"  ページ{slide.page}: {slide.title} → スコア: {slide.score} - {slide.reason}")
            all_slides.extend(slide_analysis.slides)
        except Exception as e:
            logger.warning(f"⚠️ スライド分析に失敗（ページ{start_page+1}-{end_page+1}）: {e}")
    
    # ステップ3: 最高スコアのスライドを選択
    if not all_slides:
        # スライドが取得できない場合は全文を使用
        merged_content = "\n\n".join([f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(texts)])
        page_info = f"全{total_pages}ページ（スライド分析失敗）"
        selected_slide_info = "分析失敗のため全ページ使用"

        logger.info(f"すべてのスライドを使って要約します")
    else:
        # スコアでソートし、最高スコアのスライドのみを選択
        sorted_slides = sorted(all_slides, key=lambda x: x.score, reverse=True)
        max_score = sorted_slides[0].score
        top_slides = [slide for slide in sorted_slides if slide.score == max_score]

        logger.info(f"以下のスライドを選択して要約します: {', '.join([str(slide.page) for slide in top_slides])}")

        # 最高スコアのスライドのテキストを取得
        selected_texts = []
        for slide in top_slides:
            page_idx = slide.page - 1  # 1ベースから0ベースに変換
            if 0 <= page_idx < len(texts):
                selected_texts.append(f"--- ページ {slide.page} ({slide.title}) ---\n{texts[page_idx]}")
        
        merged_content = "\n\n".join(selected_texts)
        page_info = f"最高スコア{max_score}点のスライド{len(top_slides)}枚（総{total_pages}ページ中）"
        selected_slide_info = f"選択されたスライド: " + ", ".join([f"ページ{s.page}({s.title})" for s in top_slides])
    
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
「{title}」：[要約内容]

### 制約
- 簡潔で分かりやすく
- 提供されたスライドの内容のみ使用
- 推測や補完は行わない
        """)
    
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
**要約**: [作成した要約]
        """)

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
- 「[適切なタイトル]：」の形式で出力する
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
2. 表紙情報のみの場合：「[タイトル]：」
3. 実質的内容がある場合：「[タイトル]：[要約内容]」

**手順：**
1. まず事前チェックを実行し、完全に無効かを判定
2. 完全に無効な場合は空文字列を返す
3. 表紙情報から適切なタイトルを特定
4. 本文・詳細資料に実質的な内容があるかを確認
5. 実質的な内容がない場合は「[タイトル]：」で終了
6. 実質的な内容がある場合は「[タイトル]：[内容]」を作成

**例：**
- 実質的内容がある場合：「デジタル庁個人情報保護ガイドライン」では、個人情報の適切な取り扱いについて...
- 表紙情報のみの場合：「第3回検討会資料1-2」：
- 完全に無効な場合：（空文字列）

**注意：**
- 推測や創作は一切行わず、実際に書かれている内容のみを使用する
- 表紙情報のみの場合は文書の説明や構成の説明は追加しない
- タイトル後のコロンの後に無意味な説明を追加しない

要約：
{text}]
    """)

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

    logger.info("● 文書を要約...")

    llm = Model().llm()
    parser = JsonOutputParser(pydantic_object=Summary)

    # 現在のインデックスを取得
    current_index = state.get("target_report_index", 0)
    # state に target_reports が存在しないか None の場合に備えて正規化
    target_reports = state.get("target_reports")
    if not target_reports or (hasattr(target_reports, '__len__') and len(target_reports) == 0):
        logger.info("関連文書がないため文書要約をスキップします")
        return {
            **state,
            "messages": state.get("messages", []),
            "target_report_summaries": state.get("target_report_summaries", []),
            "target_report_index": current_index,
        }

    # 初期値を設定
    summary_obj = None
    message = None
    target_report_index = current_index + 1

    try:
        if current_index >= len(target_reports):
            return state

        # 現在の文書のURLを取得
        current_report = target_reports[current_index]
        url = current_report.url
        name = current_report.name

        logger.info(f"{name}を要約します")

        # PDFを読み込んでテキストを抽出
        texts = load_pdf_as_text(url)
        if not texts:
            logger.warning(f"⚠️ PDFの読み込みに失敗しました: {url}")
            summary_obj = Summary(url=url, name=name, content="")
            
            message = AIMessage(content=f"""
## 個別文書要約結果（読み込み失敗）

**処理内容**: PDF文書の個別要約を生成
**要約タイプ**: individual_document（読み込み失敗）
**文書名**: {name}
**文書URL**: {url}
**エラー理由**: PDFファイルの読み込みに失敗
**影響**: 該当文書の内容が最終要約に含まれない可能性

**生成された要約**:
(PDFを読み込めませんでした)
""")
            return {
                **state,
                "messages": [message],
                "target_report_summaries": state.get("target_report_summaries", []),
                "target_report_index": target_report_index,
            }

        logger.info(f"{name}をテキスト化しました({len(texts)}ページ)")
        # 文書タイプを判定
        doc_type, doc_reason, evidence_text, detection_detail = detect_document_type(texts)
        
        # タイプ別要約処理
        result: dict | None = None
        if doc_type == "word":
            result = word_based_summarize(texts)
        elif doc_type == "powerpoint":
            result = powerpoint_based_summarize(texts)
        elif doc_type == "agenda":
            result = agenda_summarize(texts)
        elif doc_type == "participants":
            result = participants_summarize(texts)
        elif doc_type == "news":
            result = news_based_summarize(texts)
        else:
            # SurveyとOtherはスキップ
            logger.info(f"文書をスキップ: {name}（タイプ: {doc_type}）")
            message = HumanMessage(
                content=f"文書: {name}\nURL: {url}\n\n要約: (処理対象外のためスキップ)"
            )
            return {
                **state,
                "messages": [message],
                "target_report_summaries": state.get("target_report_summaries", []),
                "target_report_index": target_report_index,
            }

        title = result.get('title', name)
        summary = result.get('summary', '')
        # 要約内容をログに出力
        logger.info(f"この資料の要約: {summary.replace('\n', '\\n').strip()}")

        # 最初の文書でタイトルが抽出できた場合、reportのnameを更新
        if current_index == 0 and not name:
            if title and len(title) > 3:
                current_report.name = title.replace('\n', ' ').strip()
                logger.info(f"この資料の正式なタイトルは「{current_report.name}」です")

        # 直接Summaryオブジェクトを作成
        summary_obj = Summary(
            content=summary,
            url=url,
            name=title if title else name,
            document_type=doc_type,
            detection_detail=detection_detail
        )

        # 詳細説明付きメッセージを作成
        message = AIMessage(content=f"""
## 個別文書要約結果

**処理内容**: PDF文書の個別要約を生成
**要約タイプ**: individual_document（個別文書要約）
**文書名**: {name}
**文書タイプ**: {doc_type}
**文書URL**: {url}
**ページ数**: {len(texts)}ページ
**選択理由**: 会議で配布された一次資料であり、要約作成にとって重要な参照資料

**生成された要約**:
{summary_obj.content}
""")

    except Exception as e:
        logger.error(f"❌ 文書要約中にエラーが発生: {str(e)}")
        if current_index < len(target_reports):
            current_report = target_reports[current_index]
            summary_obj = Summary(url=current_report.url, name=current_report.name, content="")
        else:
            summary_obj = Summary(url="", name="", content="")

        message = AIMessage(content=f"""
## 個別文書要約結果（エラー）

**処理内容**: PDF文書の個別要約を生成
**要約タイプ**: individual_document（処理失敗）
**文書名**: {summary_obj.name}
**文書URL**: {summary_obj.url}
**エラー理由**: 文書処理中にエラーが発生
**影響**: 該当文書の詳細が最終要約に含まれない可能性

**生成された要約**:
(エラーのため要約できませんでした)
""")

        return {
            **state,
            "messages": [message],
            "target_report_summaries": state.get("target_report_summaries", []),
            "target_report_index": target_report_index,
        }

    # 既存のsummariesを取得し、新しい要約を追加
    current_summaries = state.get("target_report_summaries", [])
    new_summaries = current_summaries + ([summary_obj] if summary_obj else [])

    # 新しい状態を返す
    system_message = HumanMessage(content="PDF文書の内容を読み取り、要約を作成してください。")

    logger.info(f"✅ {summary_obj.name}の要約を作成しました")

    return {
        **state,
        "messages": [system_message, message] if message else [system_message],
        "target_report_summaries": new_summaries,
        "target_report_index": target_report_index,
    }
