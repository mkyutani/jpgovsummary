import requests
import sys

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

    def load(self, url: str) -> str:
        headers = { 'Cache-Control': 'no-cache', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36' }

        try:
            res = requests.get(url, headers=headers, timeout=10)
        except Exception as e:
            print(str(e), file=sys.stderr)
            return []

        if res.status_code >= 400:
            message = ' '.join([str(res.status_code), res.text if res.text is not None else ''])
            print(f'{message} ', file=sys.stderr)
            return []

        res.encoding = res.apparent_encoding
        text = res.text

        return text

    def think(self, state: State) -> dict:
        system_prompt = SystemMessagePromptTemplate.from_template('あなたは優秀な調査員です。会議のHTML文書を読み込んで概要を作成します。')
        assistant_prompt = AIMessagePromptTemplate.from_template(
            '''
            HTMLを読んで会議の概要を作成してください。

            ### 概要作成の手順
            - 「議事」「議事次第」というセクションやリストがあれば、それをまとめる。
            - 議事次第がなければ「資料」の名前などから類推する。(参考資料は除く)

            ### 判断基準
            - 議事次第の中に固有名詞とりわけ地名等があれば積極的に記載する。
            - 最初のセクション名より前の内容は無視したほうがよい。
            - 資料名に「(案)」や「とりまとめ」などの表現がある場合は、必ず記載する。
            - 資料名として固有名詞などがある場合は、それらの名前を積極的に記載する。
            - 「議事次第」や「資料一覧」、「名簿」などの資料については記載しない。
            - 半角の英数字や7bit-ASCII記号に変換できる全角文字は半角に変換する。
            - 概要は文であるため、最後は句点「。」で終える。
            - 過去の会議であるため文末は過去形または体言止め。
            - 概要の中に、開催日、及び、URLの情報は含めない。

            ### 出力
            - 会議の概要を200字以内、1センテンスで記述してください。
            - 会議の概要のあとに以下を記載してください：{url}

            ### 対象HTML
            {html}
            ''')
        user_prompt = HumanMessagePromptTemplate.from_template('{html}')
        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt,
                assistant_prompt,
                user_prompt
            ]
        )
        html = self.load(state['url'])
        chain = prompt | self.llm()
        result = chain.invoke({'url': state['url'], 'html': html}, Config().get())
        return {'messages': [result]}