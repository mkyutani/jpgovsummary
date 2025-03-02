import json
import re
import sys
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langgraph.types import Command

from .agent import Agent
from .config import Config
from .state import State

class PropertyFormatter(Agent):

    def __init__(self) -> None:
        super().__init__()

    def think(self, state: State) -> dict:
        system_prompt = SystemMessagePromptTemplate.from_template('あなたは優秀な書記です。会議の情報を簡潔にまとめます。')
        assistant_prompt = AIMessagePromptTemplate.from_template('''
            次のJSONデータには、会議のタイトル、回数(何回目か)、開催日、会議URLが含まれていますので、それらを抽出してください。

            ### JSONデータ
            {content}

            ### 制約条件
            ・回数と開催日がない場合は記載する必要がない。
            ・回数がなく「とりまとめ」や「報告書」などの場合は「number」に「とりまとめ」や「報告書」を記載する。
            ・「timestamp」は開催日ではない。

            ### 出力形式
            {{
                "title": "会議タイトル",
                "number": "回数",
                "date": "開催日",
                "url": "会議URL"
            }}
        ''')
        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt,
                assistant_prompt
            ]
        )

        chain = prompt | self.llm()
        result = chain.invoke(state['messages'], Config().get())

        content = result.content
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'^.*{', '{', content)
        content = re.sub(r'}.*$', '}', content)
        try:
            output = json.loads(content)
        except json.JSONDecodeError as e:
            print(f'{type(e).__name__}: {e} {content}', file=sys.stderr)
            return {**state, 'messages': [result]}

        state_update = {
            'title': output.get('title', ''),
            'number': output.get('number', ''),
            'date': output.get('date', ''),
            'url': output.get('url', ''),
            'messages': [result]
        }
        return Command(update=state_update)