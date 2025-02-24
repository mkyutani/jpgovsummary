from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .agent import Agent
from .state import State

class Chatbot(Agent):

    def __init__(self):
        super().__init__()

    def node(self, state: State) -> dict:
        print(f'State: {state}')
        return {"messages": [self.llm().invoke(state["messages"])]}

    def template__(self) -> str:
        return ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    'あなたは優秀な書記です。会議の要旨を作成します。'
                ),
                (
                    'assistant',
                    f'''
                    日本語100字程度で要約します。
                    '''
                ),
                (
                    'user',
                    '''
                    以下の書面の内容を要約してください。
# 産業構造審議会 経済産業政策新機軸部会 事業再構築小委員会（第６回）
（書面開催）

## 議事要旨

### 開催概要
事業再構築小委員会報告書（案）は、第５回事業再構築小委員会において、委員長一任
で議決された。その後、令和６年１２月２７日から令和７年１月２７日にかけて、パブリ
ックコメントを実施したところ、１８件の御意見をいただいた。
今回、パブリックコメントの結果も踏まえ、修正を行った事業再構築小委員会報告書（案）
の最終案が審議事項であるが、書面でも審議を十分に尽くせると判断し、書面審議を行う
こととした。

### 回答者一覧
神田委員長、小林委員、杉本委員、長田委員、藤原委員、三木委員、南委員、望月委員、
山田委員、山本委員

### 議題
・事業再構築小委員会報告書について

### 審議期間
令和７年２月１４日（金）～２月１８日（火）

### 審議結果
議題について書面審議を行った結果、賛成１０名、反対０名により、事業再構築小委員
会報告書は決議された。なお、一部の委員から、「ただし、より実務的に利用しやすい制
度となるよう今後詳細にかつ具体的な検討を進めていくことが重要であるものと思料す
る。」との意見があった。

## お問い合わせ先
経済産業政策局産業組織課
電話：０３－３５０１－１５１１（内線２６２１）
                '''
                )
            ]
        )