from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from .. import Config, Model, State, logger


def overview_generator(state: State) -> dict:
    """
    ## Overview Generator Agent

    Write a summary of the meeting based on the input state.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the generated summary message
    """
    logger.info("overview_generator")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたはメインコンテンツの内容を要約するエージェントです。
        以下の3つのステップで要約を作成します：
        1. ページの概要を作成：ページの名称、会議の回数、主催・事務局の情報などを特定
        2. ページの要約を作成：メインコンテンツの内容を簡潔にまとめる
        3. 要約の文章チェック：作成した要約が正しい日本語の文章になっているか確認
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        以下の3つのステップで要約を作成します。

        ## ステップ1: ページの概要を作成
        対象マークダウンを読み、以下の情報を特定します：
        - ページの名称（会議、報告書、とりまとめなど）
        - 会議の場合はその回数（第×回）
        - 主催・事務局の府省庁名や審議会名
        - 上位の委員会、研究会の名称（わかる場合）
        - ワーキンググループやサブワーキンググループの名称（ある場合）

        ### ページの名称の例
        - 〇〇〇会議(第×回)
        - 〇〇〇報告書
        - 〇〇〇とりまとめ
        - 〇〇〇の案内
        - 〇〇〇のお知らせ
        - 〇〇〇の募集

        ### 制約事項
        - 対象マークダウンを読み、ページの概要をまとめる
        - 対象マークダウンが会議である場合は、その会議の回数(第×回)を特定して付加する
        - ページの概要は、マークダウンの見出しやタイトルを参照する
        - 主催・事務局の府省庁名もしくは審議会名がわかる場合は、これを追加する
        - ナビゲーターなどから上位の委員会、研究会の名称を読みとることができれば、これを追加する
        - ワーキンググループやサブワーキンググループの名称があれば、これを追加する
        - 対象マークダウンに含まれていない内容を含んではならない

        ## ステップ2: ページの要約を作成
        以下の条件に沿って、メインコンテンツの内容を要約します：
        - メインコンテンツのマークダウンを読み、セクションや資料のリンク名称から要約を作成
        - 会議の委員提出資料や参考資料の名称は要約の対象から除外
        - メインコンテンツから読み取れない内容は含まない
        - 1文で簡潔にまとめる

        ## ステップ3: 要約の文章チェック
        作成した要約が以下の条件を満たしているか確認し、必要に応じて修正します：
        - 文末は「だ・である調」になっているか
        - 箇条書きや番号付き列挙になっていないか
        - 主語と述語の関係が正しいか
        - 助詞の使い方が適切か
        - 文の接続が自然か
        - 重複表現がないか
        - 不自然な省略がないか

        #### 制約事項
        - 箇条書きや番号付き列挙にはしない
        - 文末は「です・ます調」ではなく「だ・である調」とする
        - ページの概要やメインコンテンツから読み取れない内容を含まない

        ## 出力形式
        以下の形式で要約を出力します：

        ### 会議ページの場合
        ```
        〇〇〇会議(第×回)・・・・・・について議論された。
        ```

        ### 報告書やとりまとめのページの場合
        ```
        〇〇〇報告書・・・・・・。
        ```

        ### その他の場合
        ```
        ・・・・・・。
        ```
    """)
    prompt = ChatPromptTemplate.from_messages(
        [system_prompt, assistant_prompt, MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt | llm
    result = chain.invoke(state, Config().get())
    logger.info(result.content)
    return {"overview": result.content, "messages": [result]}
