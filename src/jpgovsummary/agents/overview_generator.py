from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from .. import Config, Model, State, logger


def overview_generator(state: State) -> dict:
    """
    ## Overview Generator Agent

    Write a summary of the meeting based on the input state.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the generated summary message
    """
    logger.info("overview_generator")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたは政府会議資料の要約を作成する専門エージェントです。
        内部で段階的に処理を行い、最終的に要約文のみを出力してください。
        処理手順や中間結果は出力せず、完成した要約文のみを返してください。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        以下の手順でマークダウンを分析し、要約を作成してください。各手順は内部処理として行い、最終的に要約文のみを出力してください。

        ## 内部処理手順（出力しない）
        ### 手順1: 基本情報の特定
        - ページの名称（会議、報告書、とりまとめ、案内、お知らせ、募集など）
        - 会議の場合は回数（第×回）を特定
        - 主催・事務局の府省庁名や審議会名
        - 上位の委員会、研究会、ワーキンググループ名（該当する場合）

        ### 手順2: 内容の要約
        - メインコンテンツから主要な議論内容・検討事項を抽出
        - 会議の委員提出資料や参考資料の名称は除外
        - マークダウンに含まれていない内容は含めない
        - 重要なポイントや結論を整理
        
        **除外すべき情報（重要）：**
        - ファイルサイズ情報（「1.2MB」「14.2MB」など）
        - ソフトウェア案内（「Adobe Acrobat Reader」「PDFリーダー」など）
        - ファイル形式の説明（「PDF形式」「Excel形式」など）
        - ダウンロード方法や閲覧方法の案内
        - ファイルを読むための技術的な注意事項や推奨環境
        - ファイルの保存場所や配布に関する情報
        - その他、文書の内容とは無関係な技術的メタデータ

        ### 手順3: 文章の調整
        - 1文で簡潔にまとめる
        - 「だ・である調」で統一
        - 主語述語の関係、助詞の使い方を確認
        - 重複表現や不自然な省略を修正

        ## 出力要件（重要）
        - 上記の内部処理を経て、要約文のみを出力
        - 処理手順、ステップ番号、見出し（###、##など）は一切出力しない
        - 箇条書き（・、-、1.など）は使用しない
        - 「概要」「要約」「ステップ」「チェック」などのラベルは出力しない
        - マークダウン記法（```、**など）は使用しない
        - コードブロックは使用しない

        ## 期待する出力例
        教育分野の認証基盤の在り方に関する検討会（第3回）では、組織間・外部連携における認証基盤の取りまとめ案について、ユースケース整理や実装パターン、個人情報保護の留意事項などを中心に議論し、スケジュールの明確化や複数自治体での実証などの改善点を確認した。
    """)
    prompt = ChatPromptTemplate.from_messages(
        [system_prompt, assistant_prompt, MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt | llm
    result = chain.invoke(state, Config().get())
    logger.info(f"Overview: {result.content.replace('\n', '\\n')}")
    return {"overview": result.content, "messages": [result]}
