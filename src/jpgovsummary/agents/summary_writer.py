from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .. import Config, Model, State, log

def summary_writer(state: State) -> dict:
    """
    ## Summary Writer Agent

    Write a summary of the meeting based on the input state.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the generated summary message
    """
    log("summary_writer")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたは会議の内容を要約するエージェントです。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        以下の条件に沿って、会議の内容を要約します。

        ## 条件
        - 対象となっているメインコンテンツのマークダウンを読み、議事もしくは議事次第に該当するセクション、会議資料に相当するリンクのファイル名から、議事内容を作成する。ただし、委員提出資料や参考資料の名称は要約の対象から除外する。
        - 対象となっているメインコンテンツのマークダウンから読み取れない内容は含んではならない。
        - 以下の制約に従い、これまでに得られた議事内容を1文で簡潔にまとめる。

        #### 制約事項
        - 箇条書きや番号付き列挙にはしない。
        - 文末は「です・ます調」ではなく「だ・である調」とする。
        - これまでに得られた議事内容から読み取れない内容を含んではならない。

        ## 出力形式
        - 以下の形式を参考にして要約と対象URLを出力する。

        ### 会議ページの場合

        ```
        〇〇〇会議(第×回)では、・・・・・・について議論された。
        https://...
        ```

        ### 報告書ページの場合

        ```
        〇〇〇報告書は、・・・・・・。
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
    return { "summary": result.content, "messages": [result] } 