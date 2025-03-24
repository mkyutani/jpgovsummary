from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .agent import Agent
from .. import Config, State

class SummaryWriter(Agent):

    def think(self, state: State) -> dict:
        system_prompt = SystemMessagePromptTemplate.from_template("""
            あなたは資料を読んで要約するエージェントです。
        """)
        assistant_prompt = AIMessagePromptTemplate.from_template("""
            以下のURL及びHTMLから「会議名」、「回数(第○回)」、「議事概要」を抜き出し、出力例に沿ってまとめてください。

            ### 出力例
            医療介護総合確保促進会議(第21回)では、地域医療介護総合確保基金の執行状況、及び、医療法等の一部を改正する法律案(閣議決定)等が議論された。
            https://...(URL)

            ### 概要作成の手順
            - ユーザーから与えられた情報がJSONだけで文章になっていないなど不十分な場合、ツールを使ってまず議事概要を取得する必要がある。
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
            - 会議の概要を200字以内、1センテンスで記述する。
            - 語尾は「議論された。」「報告された。」「とりまとめられた」などから適切なものを選択する。
            - 満足のできる概要を作成できなかった場合は、「概要不明」とだけ記載する。
        """)
        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt,
                assistant_prompt,
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        chain = prompt | self.llm
        result = chain.invoke(state, Config().get())
        return { "messages": [result] }