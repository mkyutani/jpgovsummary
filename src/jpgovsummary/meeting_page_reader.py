import os
import requests

from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from .agent import Agent
from .config import Config
from .state import State

class MeetingPageReader(Agent):

    def __init__(self) -> None:
        super().__init__()
        self.firecrawl_api_key = os.environ.get('FIRECRAWL_API_KEY')

    def load(self, url: str):
        loader = FireCrawlLoader(
            api_key=self.firecrawl_api_key,
            url=url,
            mode="scrape"
        )

        pages = []
        for doc in loader.lazy_load():
            pages.append(doc)

        markdown = '\n'.join([page.page_content for page in pages])

        return markdown

    def node(self, state: State) -> str:
        system_prompt = SystemMessagePromptTemplate.from_template('あなたは優秀な調査員です。会議のマークダウン文書を読み込んで概要を作成します。')
        assistant_prompt = AIMessagePromptTemplate.from_template(
            '''
            マークダウンを読んで会議の概要を作成してください。

            ### 概要作成の手順
            - 「議事」「議事次第」というセクションやリストがあれば、それをまとめる。
            - 議事次第がなければ「資料」の名前などから類推する。(参考資料は除く)

            ### 判断基準
            - 議事次第の中に固有名詞とりわけ地名等があれば積極的に記載する。
            - 最初のセクション名より前の内容は無視したほうがよい。
            - 資料名として固有名詞などがある場合は、それらの名前を積極的に記載する。
            - 半角の英数字や7bit-ASCII記号に変換できる全角文字は半角に変換する。
            - 概要は文であるため、最後は句点「。」で終える。
            - 過去の会議であるため文末は過去形または体言止め。
            - 概要の中に、開催日、及び、URLの情報は含めない。
            - 200字以内、1センテンスのみ。
            ''')
        user_prompt = HumanMessagePromptTemplate.from_template('{markdown}')
        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt,
                assistant_prompt,
                user_prompt
            ]
        )
        markdown = self.load(state['url'])
        chain = prompt | self.llm()
        result = chain.invoke({'messages': state['messages'], 'markdown': markdown}, Config().get())
        return {"messages": [result]}