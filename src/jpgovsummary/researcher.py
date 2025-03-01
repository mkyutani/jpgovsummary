from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from .agent import Agent
from .config import Config
from .meeting_information_collector import MeetingInformationCollector
from .state import State

class Researcher(Agent):

    def __init__(self, uuid: str) -> None:
        super().__init__()
        self.tools = [MeetingInformationCollector.tool]
        self.uuid = uuid

    def think(self, state: State) -> dict:
        system_prompt = SystemMessagePromptTemplate.from_template('あなたは優秀な調査員です。会議の情報を収集します。')
        assistant_prompt = AIMessagePromptTemplate.from_template('''
            ユーザから受け取った会議の番号(UUID)について、会議情報収集ツールを使って番号にひもづく会議の情報を収集します。
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
        chain = prompt | self.llm().bind_tools(tools=self.tools)
        result = chain.invoke(state["messages"], Config().get())
        return {'messages': [result]}