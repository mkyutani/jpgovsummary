from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .agent import Agent
from .. import Config, State
from ..tools import meeting_url_collector

class BaseURLGenerator(Agent):

    def think(self, state: State) -> dict:
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
        chain = prompt | self.llm.bind_tools([meeting_url_collector])
        result = chain.invoke(state, Config().get())
        return { "messages": [result] }