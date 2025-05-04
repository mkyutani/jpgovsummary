from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .. import Config, Model, State, log

def overview_generator(state: State) -> dict:
    """
    ## Overview Generator Agent

    Extract meeting title and number from HTML content.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the extracted overview message
    """
    log("overview_generator")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたはHTMLを読んで会議内容をまとめる優秀な書記です。
        ユーザから受け取ったHTMLを解析し、会議の名称と回数を特定します。
        会議の名称は、HTMLタイトルやog:title、H1タグなどを参照して決定します。
        また、パンくずリストや主催・事務局の情報も確認し、可能な限り詳細な会議名を特定します。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        以下の対象URLについて、会議の名称と回数(第〇回)を特定してください。

        ## 制約事項

        - 対象URLを読み、会議の名称と回数(第〇回)を特定する。
        - 会議の名称は、HTMLタイトルまたはog:titleを参照する。
        - 主催・事務局の府省庁名もしくは審議会名がわかる場合は、これを追加する。
        - パンくずリストなどから上位の委員会、研究名が読みとることができれば、これを追加する。
        - H1タグにワーキンググループやサブワーキンググループの名称があれば、これを追加する。
        - 対象URLのページに含まれていない内容を含んではならない。

        ## 出力形式
        - 以下の形式を参考にして要約と対象URLを出力する。

        ```
        〇〇〇会議(第×回)
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
    return { "meeting_title": result.content, "messages": [result] } 