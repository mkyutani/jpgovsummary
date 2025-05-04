from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .. import Config, Model, State, log
from ..tools.meeting_url_collector import meeting_url_collector

def base_url_generator(state: State) -> dict:
    """
    ## Base URL Generator Agent

    Generate or retrieve the base URL for a meeting based on the input state.
    This agent decides whether to use the meeting URL collector tool or skip it.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the generated or retrieved URL message
    """
    log("base_url_generator")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたは会議のURLを特定するエージェントです。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        ユーザから受け取った会議情報の種類により、以下のルールにて会議のURLを取得するツールを選定します。

        ### 会議情報を判定するルール
        - 会議のUUIDが与えられた場合、会議のURLを取得するツールを指定します。
        - 会議のURLが与えられた場合、ツールは指定せず、会議のURLをそのまま返します。
    """)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            assistant_prompt,
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | llm.bind_tools([meeting_url_collector])
    result = chain.invoke(state, Config().get())
    return { "messages": [result] } 