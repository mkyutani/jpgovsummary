from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage

from .. import Config, Model, State, logger


def overview_generator(state: State) -> dict:
    """
    ## Overview Generator Agent

    Write a summary of the meeting based on the main content extracted by main_content_extractor.

    Args:
        state (State): The current state containing meeting information and main_content

    Returns:
        dict: A dictionary containing the generated summary message
    """
    logger.info("overview_generator")

    # main_content_extractorの結果を取得
    if "main_content" not in state:
        logger.error("main_content not found in state. overview_generator requires main_content_extractor to run first.")
        return {"overview": "エラー: メインコンテンツが抽出されていません。", "messages": []}

    main_content = state["main_content"]
    logger.info(f"Processing main_content with length: {len(main_content)}")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
あなたは政府会議資料の要約を作成する専門エージェントです。
内部で段階的に処理を行い、最終的に要約文のみを出力してください。
処理手順や中間結果は出力せず、完成した要約文のみを返してください。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
以下の手順でメインコンテンツを分析し、要約を作成してください。各手順は内部処理として行い、最終的に要約文のみを出力してください。

## 内部処理手順（出力しない）
### 手順0: 議事録判定（内部処理）
- 実際の発言記録パターンを確認：
    - 「○○委員から」「○○委員より」「○○座長から」
    - 「質問に対し」「意見として」「発言があった」
    - 「について述べた」「について説明した」
    - 「これに対して」「この点について」
- 議論の流れを示すパターンを確認：
    - 複数の発言者による対話形式
    - 質疑応答の記録
    - 意見交換の詳細
- 議事録判定の条件（すべて満たす必要あり）：
    1. 実際の発言内容が具体的に記録されている
    2. 複数の参加者による議論の流れが確認できる
    3. 質疑応答や意見交換の詳細が含まれている
    4. 他の資料を参照しなくても会議の内容が理解できる十分な情報量がある
- 単なるタイトルや議題リスト、参加者名簿は除外

### 手順1: 基本情報の特定と会議/文書判定
- ページの名称（会議、報告書、とりまとめ、案内、お知らせ、募集など）
- 会議の場合は回数（第×回）を特定
- 主催・事務局の府省庁名や審議会名
- 上位の委員会、研究会、ワーキンググループ名（該当する場合）

**会議か文書かの判定基準：**
- 会議として扱う場合：
    - ページ名に「会議」「検討会」「委員会」「審議会」「協議会」「ワーキンググループ」「研究会」「部会」「分科会」が含まれる
    - 会議の資料リストや議事録が含まれている
    - 複数の議題や検討事項が含まれている
- 文書として扱う場合：
    - 「お知らせ」「案内」「募集」「通知」「報告書」「ガイドライン」などの単発文書
    - 特定の制度や政策の説明文書
    - 調査結果やデータの公表文書

### 手順2: 内容の要約（判定結果に応じた表現）
- 議事録と判定された場合：
    - 実際の発言内容や議論の詳細を重視して包括的な要約を作成
    - 委員の具体的な発言や提案を反映
    - 質疑応答の内容や議論の流れを含める
    - 決定事項や今後の方針を明確化
    - 「[会議名]では〜が議論された」の形式で表現（会議名の前に「会議では」は付けない）
- 会議として判定された場合：
    - メインコンテンツから主要な議論内容・検討事項を抽出
    - 重要なポイントや結論を整理
    - 「[会議名]では〜が議論された」「[検討会名]では〜が検討された」の形式で表現（会議名の前に「会議では」は付けない）
- 文書として判定された場合：
    - メインコンテンツから主要な内容を抽出
    - 重要なポイントや結論を整理
    - 「文書では〜が記載されている」「報告書では〜が報告されている」の形式で表現
- 共通の除外項目：
    - 会議の委員提出資料や参考資料の名称、委員による資料提出情報は除外
    - メインコンテンツに含まれていない内容は含めない

**除外すべき情報（重要）：**
- ファイルサイズ情報（「1.2MB」「14.2MB」など）
- ソフトウェア案内（「Adobe Acrobat Reader」「PDFリーダー」など）
- ファイル形式の説明（「PDF形式」「Excel形式」など）
- ダウンロード方法や閲覧方法の案内
- ファイルを読むための技術的な注意事項や推奨環境
- ファイルの保存場所や配布に関する情報
- 会議の開催日時・日付（「令和○年○月○日」「○月○日開催」など）
- 会議の開催場所（「○○省会議室」「オンライン開催」「ハイブリッド開催」など）
- 会議の開催形式（「対面」「Web会議」「Teams会議」など）
- 委員による資料提出情報（「○○委員資料提出」「○○委員提供資料」など）
- 事務局による説明情報（「事務局説明」「事務局より説明」など）
- **会議の出席者・参加者情報（「出席者」「参加者一覧」「委員名簿」など）**
- **具体的な日時表現（「午前9時30分開始」「14:00-16:00」など）**
- **具体的な場所表現（「東京都千代田区」「○○ビル3階」など）**
- その他、文書の内容とは無関係な技術的メタデータ

### 手順3: 文章の調整
- 1文で簡潔にまとめる
- 「だ・である調」で統一
- 主語述語の関係、助詞の使い方を確認
- 重複表現や不自然な省略を修正

## 出力要件（重要）
- 上記の内部処理を経て、要約文のみを出力
- 議事録と判定した場合：要約文の最後に「[DETAILED_MINUTES_DETECTED]」を付加
- 文書と判定した場合：要約文の最後に「[DOCUMENT_PAGE_DETECTED]」を付加
- 処理手順、ステップ番号、見出し（###、##など）は一切出力しない
- 箇条書き（・、-、1.など）は使用しない
- 「概要」「要約」「ステップ」「チェック」などのラベルは出力しない
- マークダウン記法（```、**など）は使用しない
- コードブロックは使用しない

## 期待する出力例

### 会議の場合（良い例）：
教育分野の認証基盤の在り方に関する検討会（第3回）では、組織間・外部連携における認証基盤の取りまとめ案について、ユースケース整理や実装パターン、個人情報保護の留意事項などを中心に議論し、スケジュールの明確化や複数自治体での実証などの改善点を確認した。

### 文書の場合（良い例）：
デジタル社会推進のための新制度に関する報告書では、個人情報保護の強化とデータ活用の促進を両立させる制度設計について記載されており、技術的な安全管理措置と法的な規制枠組みの整備が重要な要素として示されている。[DOCUMENT_PAGE_DETECTED]

### 議事録の場合（良い例）：
教育分野の認証基盤の在り方に関する検討会（第3回）では、組織間・外部連携における認証基盤の取りまとめ案について審議し、田中委員からは実装パターンの具体化と技術仕様の詳細化について質問があり、事務局からはセキュリティ対策の強化と個人情報保護の留意事項について説明があった。佐藤座長からは複数自治体での実証実験の必要性が提案され、次回会議までにスケジュールの明確化と詳細な実装計画の策定を行うことが確認された。[DETAILED_MINUTES_DETECTED]

### 悪い例（避けるべき）：
会議では第33回保健医療福祉分野における公開鍵基盤認証局の整備と運営に関する専門家会議で...

**重要：**
- 会議名の前に「会議では」「検討会では」などは付けないこと
- 会議名そのものを主語として使用すること
    """)
    
    # メインコンテンツを明示的にLLMに渡す
    content_message = f"以下のメインコンテンツを分析して要約を作成してください：\n\n{main_content}"
    
    prompt = ChatPromptTemplate.from_messages(
        [system_prompt, assistant_prompt, MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt | llm
    
    # メインコンテンツを含むメッセージを作成
    messages = [HumanMessage(content=content_message)]
    
    result = chain.invoke({"messages": messages}, Config().get())
    logger.info(f"Overview: {result.content.replace('\n', '\\n')}")
    
    # 議事録検出フラグのチェックと処理
    meeting_minutes_detected = "[DETAILED_MINUTES_DETECTED]" in result.content
    document_page_detected = "[DOCUMENT_PAGE_DETECTED]" in result.content
    
    # フラグを除去してクリーンな要約文にする
    clean_overview = result.content
    if meeting_minutes_detected:
        clean_overview = clean_overview.replace("[DETAILED_MINUTES_DETECTED]", "").strip()
        logger.info("Meeting minutes detected")
    if document_page_detected:
        clean_overview = clean_overview.replace("[DOCUMENT_PAGE_DETECTED]", "").strip()
        logger.info("Document page detected")
    
    # 会議かどうかを判定
    # 議事録が検出されている場合は確実に会議
    # 文書フラグが検出されている場合は文書
    # どちらも検出されていない場合はデフォルトで会議として扱う
    if meeting_minutes_detected:
        is_meeting = True
    elif document_page_detected:
        is_meeting = False
    else:
        # フラグが明確でない場合はデフォルトで会議として扱う
        is_meeting = True

    # 詳細説明付きメッセージを作成
    detailed_message = AIMessage(content=f"""
## 会議概要生成結果

**処理内容**: メインコンテンツから会議の概要要約を生成
**要約タイプ**: overview（会議全体の概要）
**議事録検出**: {'有' if meeting_minutes_detected else '無'}
**文書ページ検出**: {'有' if document_page_detected else '無'}
**内容種別**: {'会議' if is_meeting else '文書'}
**入力サイズ**: {len(main_content)}文字
**出力サイズ**: {len(clean_overview)}文字

**生成された概要**:
{clean_overview}
""")

    # システムプロンプトを追加
    system_message = HumanMessage(content="会議の全体概要を生成してください。")
    
    return {
        "overview": clean_overview, 
        "messages": [system_message, detailed_message],
        "meeting_minutes_detected": meeting_minutes_detected,
        "is_meeting_page": is_meeting
    }
