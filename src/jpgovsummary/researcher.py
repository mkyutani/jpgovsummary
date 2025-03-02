from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langgraph.types import Command

from .agent import Agent
from .config import Config
from .meeting_information_collector import meeting_information_collector
from .state import State

class Researcher(Agent):

    def __init__(self) -> None:
        super().__init__()

    def think(self, state: State) -> dict:
        system_prompt = SystemMessagePromptTemplate.from_template('あなたは優秀な調査員です。会議の情報を収集します。')
        assistant_prompt = AIMessagePromptTemplate.from_template('''
            ユーザから受け取った会議について、会議情報収集ツールを使って会議の情報を収集します。
        ''')
        user_prompt = HumanMessagePromptTemplate.from_template('''
            質問：{content}
        ''')
        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt,
                assistant_prompt,
                user_prompt
            ]
        )
        chain = prompt | self.llm().bind_tools(tools=[meeting_information_collector])
        result = chain.invoke(state["messages"], Config().get())
        return {'messages': [result]}