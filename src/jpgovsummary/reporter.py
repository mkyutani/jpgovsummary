from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from .agent import Agent
from .config import Config
from .meeting_information_collector import meeting_information_collector
from .state import State

class Reporter(Agent):

    def __init__(self, uuid: str) -> None:
        super().__init__()
        self.tools = [meeting_information_collector]
        self.uuid = str

    def node(self, state: State) -> str:
        system_prompt = SystemMessagePromptTemplate.from_template('あなたは優秀な書記です。会議の情報を収集します。')
        assistant_prompt = AIMessagePromptTemplate.from_template('''
            以下の手順で情報を収集します。
            (1)ユーザから受け取った会議の番号(UUID)について、それにひもづく会議の情報を収集します。
            (2)会議の情報から、会議名、回数(何回目か)、開催日などを取得します。
            (3)会議のアジェンダや議事次第等から会議を要約し、要旨を作成します。

            ### 制約条件
            - 会議の回数と開催日は不明であれば記載不要。
            - 要旨がわからなければ「要旨不明」と記載。
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
        return {"messages": [result]}