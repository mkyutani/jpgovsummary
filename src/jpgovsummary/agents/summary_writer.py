from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .. import Config, Model, State, logger
from ..tools import html_loader, pdf_loader

def summary_writer(state: State) -> dict:
    """
    ## Summary Writer Agent

    Write a summary of the meeting based on the input state.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the generated summary message
    """
    logger.info("summary_writer")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたはメインコンテンツの内容を要約するエージェントです。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        以下の条件に沿って、メインコンテンツの内容を要約します。

        ## 条件
        - 対象となっているメインコンテンツのマークダウンを読み、メインコンテンツに含まれるセクション、資料に相当するリンクの名称やファイル名から、要約を作成する。ただし、会議の委員提出資料や参考資料の名称は要約の対象から除外する。
        - 対象となっているメインコンテンツのマークダウンから読み取れない内容は含んではならない。
        - 以下の制約に従い、これまでに得られた要約を1文で簡潔にまとめる。
        - 日本語の文章としてただしいものを出力する。

        #### 制約事項
        - 箇条書きや番号付き列挙にはしない。
        - 文末は「です・ます調」ではなく「だ・である調」とする。
        - ページの概要やメインコンテンツから読み取れない内容を含んではならない。

        ## 出力形式
        - 以下の形式を参考にして要約と対象URLを出力する。

        ### 会議ページの場合

        ```
        〇〇〇会議(第×回)では、・・・・・・について議論された。
        https://...
        ```

        ### 報告書やとりまとめのページの場合

        ```
        〇〇〇報告書は、・・・・・・。
        https://...
        ```
        ### その他の場合

        ```
        ・・・・・・。
        https://...
        ```
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
    logger.info(result.content)
    return { "summary": result.content, "messages": [result] } 