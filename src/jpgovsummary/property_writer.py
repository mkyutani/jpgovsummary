import json
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)

from .agent import Agent
from .config import Config
from .state import State

class PropertyWriter(Agent):

    def __init__(self) -> None:
        super().__init__()

    def node(self, state: State) -> str:
        system_prompt = SystemMessagePromptTemplate.from_template('あなたは優秀な書記です。会議の情報を簡潔にまとめます。')
        assistant_prompt = AIMessagePromptTemplate.from_template('''
            次のJSONデータには、会議のタイトル、回数(何回目か)、開催日、会議URLが含まれていますので、それらを抽出してください。

            ### JSONデータ
            {content}

            ### 制約条件
            ・改行はおこなわず、まとめは1行にまとめる。
            ・回数と開催日がない場合は記載する必要がない。
            ・「timestamp」は開催日ではない。

            ### 出力形式
            {{
                "title": "会議タイトル",
                "number": "回数",
                "date": "開催日",
                "url": "会議URL",
                "overview": "要旨"
            }}
        ''')
        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt,
                assistant_prompt
            ]
        )

        chain = prompt | self.llm()
        result = chain.invoke(state["messages"], Config().get())
        output = json.loads(result.content)
        return {**output, 'messages': [result]}