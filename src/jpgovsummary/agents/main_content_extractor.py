from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage

from .. import Config, Model, State, logger
from ..tools import load_html_as_markdown
from lxml import html


def main_content_extractor(state: State) -> dict:
    """
    ## Main Content Extractor Agent

    Extract main content from markdown by removing headers, footers, navigation, and related sections.

    Args:
        state (State): The current state containing markdown content

    Returns:
        dict: A dictionary containing the extracted main content
    """
    logger.info("🟢 メインコンテンツを抽出...")

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
- もしメインコンテンツが存在しない、または抽出できない場合は、必ず「[HTML_PARSING_ERROR]」と出力してください
    """)
    prompt = ChatPromptTemplate.from_messages(
        [system_prompt, assistant_prompt, MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt | llm
    result = chain.invoke(state, Config().get())
    
    # HTMLパースエラーが検出された場合の自動修正処理
    if "[HTML_PARSING_ERROR]" in result.content:
        logger.warning("⚠️ HTMLパースエラーが検出されました。lxmlで自動修正を試みます...")
        
        # 元のURLを取得
        url = state.get("url")
        if url:
            try:
                # HTMLを再取得してlxmlで正規化
                import requests
                from io import BytesIO
                from markitdown import MarkItDown
                from lxml import etree
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=30, verify=True)
                response.raise_for_status()
                
                # lxmlでHTMLを正規化
                doc = html.fromstring(response.content)
                normalized_html = etree.tostring(doc, encoding='unicode', method='html')
                
                # 正規化されたHTMLをMarkItDownで変換
                html_bytes = normalized_html.encode('utf-8')
                html_stream = BytesIO(html_bytes)
                markdown = MarkItDown().convert(html_stream, file_extension='.html')
                
                logger.info("🔧 HTMLを正規化して再変換しました")
                
                # 修正されたマークダウンで再度メインコンテンツ抽出
                fixed_state = state.copy()
                fixed_state["messages"] = [
                    HumanMessage(content=f'会議のURLは"{url}"です。'),
                    HumanMessage(content=f"マークダウンは以下の通りです：\n\n{markdown.text_content}"),
                ]
                
                fixed_result = chain.invoke(fixed_state, Config().get())
                
                if "[HTML_PARSING_ERROR]" not in fixed_result.content:
                    logger.info("✅ HTML正規化後にメインコンテンツの抽出に成功しました")
                    result = fixed_result
                else:
                    logger.error("❌ HTML正規化後もメインコンテンツの抽出に失敗しました")
                    
            except Exception as e:
                logger.error(f"❌ HTML自動修正中にエラーが発生しました: {e}")
        else:
            logger.warning("⚠️ URLが見つからないため、HTML自動修正をスキップします")
    
    logger.info(f"メインコンテンツ: {result.content.replace('\n', '\\n').strip()}")
    logger.info(f"✅ {len(result.content)}文字のメインコンテンツを抽出しました")

    return {"main_content": result.content, "messages": [result]}
