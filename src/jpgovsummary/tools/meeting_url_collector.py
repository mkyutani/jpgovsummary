import os
import requests
from urllib.parse import urljoin

from langchain_core.tools import tool
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .. import Config, State, log
from ..utils import get_llm

@tool
def meeting_url_collector(uuid: str) -> str:
    """
    ## Meeting URL Collector

    Collect the URL for a meeting based on the input UUID.

    Args:
        uuid (str): The UUID of the meeting

    Returns:
        str: The URL of the meeting
    """
    log("meeting_url_collector")

    llm = get_llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたは会議のURLを収集するエージェントです。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        ユーザから受け取った会議情報の種類により、以下のルールにて会議のURLを収集します。

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

def meeting_url_collector(state: State) -> dict:
    """
    ## Meeting URL Collector

    Collect the URL for a meeting based on the input state.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the collected URL message
    """
    log("meeting_url_collector")

    llm = get_llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたは会議のURLを収集するエージェントです。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        ユーザから受け取った会議情報の種類により、以下のルールにて会議のURLを収集します。

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