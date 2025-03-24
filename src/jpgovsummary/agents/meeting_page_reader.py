from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .agent import Agent
from .. import Config, State
from ..tools import html_loader, pdf_loader

class MeetingPageReader(Agent):

    def think(self, state: State) -> dict:
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
        chain = prompt | self.llm.bind_tools([html_loader, pdf_loader])
        result = chain.invoke(state, Config().get())
        return { "messages": [result] }