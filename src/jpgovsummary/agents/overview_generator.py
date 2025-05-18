from langchain_core.prompts import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)

from .. import Config, Model, State, logger

def overview_generator(state: State) -> dict:
    """
    ## Overview Generator Agent

    Extract meeting title and number from markdown content.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the extracted overview message
    """
    logger.info("overview_generator")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたはマークダウンを読んでページの概要をまとめる優秀な書記です。
        ページの概要は、そのページを表す適切な名称、及び会議の場合はその回数を含みます。
        ページの名称は、マークダウンの中からレベルの高い見出しなどを参照して決定します。
        また、ナビゲーターや主催・事務局の情報も確認し、可能な限り詳細なページの概要を特定します。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        対象マークダウンを読んでページの概要をまとめてください。
        ページの概要は、ページの名称とページのURLから構成されます。

        ## ページの名称の例
        - 〇〇〇会議(第×回)
        - 〇〇〇報告書
        - 〇〇〇とりまとめ
        - 〇〇〇の案内
        - 〇〇〇のお知らせ
        - 〇〇〇の募集

        ## 制約事項
        - 対象マークダウンを読み、ページの概要をまとめる。
        - 対象マークダウンが会議である場合は、その会議の回数(第×回)を特定して付加する。
        - ページの概要は、マークダウンの見出しやタイトルを参照する。
        - 主催・事務局の府省庁名もしくは審議会名がわかる場合は、これを追加する。
        - ナビゲーターなどから上位の委員会、研究会の名称を読みとることができれば、これを追加する。
        - ワーキンググループやサブワーキンググループの名称があれば、これを追加する。
        - 対象マークダウンに含まれていない内容を含んではならない。

        ## 出力形式
        - 以下の形式を参考にしてページの名称とページのURLを出力する。

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
    logger.info(result.content)
    return { "overview": result.content, "messages": [result] } 