from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .. import Config, Model, State, log
from ..tools import html_loader, pdf_loader

def meeting_page_type_selector(state: State) -> dict:
    """
    ## Meeting Page Type Selector Agent

    Read a meeting page and determine the appropriate tool to use based on the file type.
    This agent decides whether to use HTML loader or PDF loader.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the tool selection message
    """
    log("meeting_page_type_selector")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたは会議のURLを読んでファイルの種類に応じたツールを特定するエージェントです。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        ユーザから受け取ったURLの拡張子を見て、ルールに沿ってツールを選定します。
        拡張子には、.pdfと.htmlがあります。

        ### ルール
        - 拡張子が.pdfである場合、PDF Loader Toolを指定します。
        - 拡張子が.htmlである場合、HTML Loader Toolを返します。
        - それ以外の場合、HTML Loader Toolを返します。
    """)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            assistant_prompt,
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | llm.bind_tools([html_loader, pdf_loader])
    result = chain.invoke(state, Config().get())
    return { "messages": [result] } 