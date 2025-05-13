from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .. import Config, Model, State, logger
from ..tools import html_loader, pdf_loader

def main_content_extractor(state: State) -> dict:
    """
    ## Main Content Extractor Agent

    Extract main content from markdown by removing headers, footers, navigation, and related sections.

    Args:
        state (State): The current state containing markdown content

    Returns:
        dict: A dictionary containing the extracted main content
    """
    logger.info("main_content_extractor")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたはマークダウンを読んでメインコンテンツを抽出する優秀なデータエンジニアです。
        ユーザから受け取ったマークダウンを解析し、ヘッダ、フッタ、ナビゲーション、関連サイトに関するセクションを取り除きます。
        メインコンテンツは、会議の議事録や報告書の本文、資料の内容などです。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        マークダウンから、メインコンテンツを抽出してください。

        ## 制約事項

        - 以下のセクションは取り除いてください：
          - ヘッダ（ページ上部のナビゲーション、ロゴ、検索ボックスなど）
          - フッタ（ページ下部の著作権表示、プライバシーポリシーなど）
          - ナビゲーション（サイドバー、パンくずリスト、メニューなど）
          - 関連サイト（外部リンク、関連ページなど）
          - 広告、バナー、通知など
          - その他の補足的な情報

        - 以下のセクションは保持してください：
          - 会議、報告書、とりまとめ、案内、お知らせ、募集など、ページの概要に関連するセクション
          - 会議の議事録や報告書の本文
          - 会議の議題や議事録の概要
          - 会議の決定事項や結論
          - その他の重要な情報

        - メインコンテンツの構造は保持してください：
          - 見出しの階層
          - リストや表
          - リンク（メインコンテンツ内のリンクは保持）
          - 強調や引用

        ## 出力形式
        - 抽出したメインコンテンツをマークダウン形式で出力してください
        - 不要なセクションを削除した後の、整理されたマークダウンを返してください
    """)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            assistant_prompt,
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | llm
    result = chain.invoke(state, Config().get())
    logger.info(f"length: {len(result.content)}")
    return { "main_content": result.content, "messages": [result] } 