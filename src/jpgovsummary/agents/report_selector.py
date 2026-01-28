from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

from .. import Config, Model, ScoredReportList, State, TargetReportList, logger


def report_selector(state: State) -> State:
    """Select reports to be used for summarization."""
    logger.info("🟢 資料を選択...")

    llm = Model().llm()
    parser = JsonOutputParser(pydantic_object=ScoredReportList)
    system_prompt = SystemMessagePromptTemplate.from_template("""
あなたは要約の精度を向上させるために、どの資料を追加で読むべきかを判断する優秀なデータエンジニアです。
要約と候補資料の情報を分析し、より精緻な要約を作成するために必要な資料を選択します。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
より精緻な要約を作成するために、追加でどの資料を読むべきかを判断してください。

## ステップ1: 資料の評価
各候補資料について、以下の基準に基づいて5段階評価で評定してください：

5: 必須 - 要約の精度向上に不可欠な資料
4: 重要 - 要約の精度向上に重要な資料
3: 有用 - 要約の精度向上に役立つ資料
2: 参考 - 要約の精度向上に多少参考になる資料
1: 不要 - 要約の精度向上に不要な資料

評価の際は以下の点を考慮してください：
- 要約との関連性
- 資料の内容の重要度
- 資料の種類（議事録、報告書、資料など）
- 資料の時系列的な位置づけ
- 同じ資料に概要と本文がある場合、概要を優先し、本文のスコアを1段階下げてください
- 「委員名簿」「座席表」「資料や会議の公開方法」に関する資料は要約の精度向上に役立ちません
- 個別委員の資料（「委員提出資料」「委員意見」「○○委員提出」など）は、会議全体の要約において優先度が低いため、スコアを2段階下げてください
- 会社や社団の資料（「株式会社○○提出資料」「一般社団法人○○意見書」など）は、会議全体の要約において優先度が低いため、スコアを1段階下げてください

## ステップ2: 議事次第優先度の調整（重要）
**必須チェック**: 実質的な内容を含む資料（説明資料、報告書、議事録、配布資料など）が存在する場合、議事次第やアジェンダ単体を最高スコア（5点）にしてはいけません。

**調整ルール**:
- 実質的な内容を含む資料がある場合、議事次第・アジェンダのスコアを4点以下に制限する
- または、実質的な内容を含む資料も同じく最高スコア（5点）にする
- 議事次第しか存在しない場合のみ、議事次第を5点とする

**参考資料の優先度調整（重要）**:
- 「資料」（参考資料以外）と「参考資料」の両方が存在する場合、「参考資料」のスコアを4点以下に制限する
- 「参考資料」のみが存在する場合は、この制限を適用しない

**実質的な内容を含む資料の例**:
- 事務局説明資料、配布資料、参考資料
- 議事録、会議録
- 報告書、調査結果
- 提案書、検討資料

## 入力情報
1. ページの要約: {overview}
2. 候補資料:
{candidate_reports}

#### 出力フォーマット
すべての資料について評価を行い、以下のフォーマットで出力してください：

{format_instructions}

## 制約事項
- すべての資料について評価を行ってください
- 出力は必ずJSON形式にしてください
- 出力は必ず上記の形式に従ってください
- 各資料について、評価点（score）と具体的な理由（reason）を必ず記述してください
- 評価点は1から5の整数で記述してください
- 同じ資料に概要と本文がある場合、概要を優先し、本文のスコアを1段階下げてください
- **ステップ2の議事次第優先度調整を必ず実行してください**
- **「資料」（参考資料以外）と「参考資料」の両方が存在する場合、「参考資料」のスコアを4点以下に制限してください**
- **資料名に「委員提出」「委員意見」「○○委員提出資料」などが含まれる個別委員の資料は、スコアを2段階下げてください**
- **資料名に「株式会社」「一般社団法人」「一般財団法人」などが含まれる会社や社団の資料は、スコアを1段階下げてください**
    """)
    prompt = ChatPromptTemplate.from_messages([system_prompt, assistant_prompt])
    chain = prompt | llm | parser
    result = chain.invoke(
        {**state, "format_instructions": parser.get_format_instructions()}, Config().get()
    )

    reports = result["reports"]
    if not reports or len(reports) == 0:
        logger.info("関連資料が選択されませんでした")
        reports = []
        target_reports = []
        highest_score = 0  # Set default value when no reports found
    else:
        reports = sorted(reports, key=lambda x: x["score"], reverse=True)
        for report in reports:
            logger.info(f"{report['score']} {report['name']} {report['url']} {report['reason']}")

        # 最高評価の資料をtarget_reportsに設定（プロンプトの指示に従う）
        highest_score = reports[0]["score"]
        target_reports = [r for r in reports if r["score"] == highest_score]
        logger.info(f"{len(target_reports)}件の資料が最高スコア{highest_score}です")

    # 簡潔な結果メッセージを作成
    system_message = HumanMessage(content="要約の精度向上のために、どの資料を追加で読むべきかを判断してください。")
    result_message = AIMessage(content=f"""
## 文書選択結果

**処理内容**: 候補文書から要約精度向上に必要な文書を選択
**選択文書数**: {len(target_reports)}件
**最高スコア**: {highest_score}点
**選択文書**: {', '.join([r['name'] for r in target_reports])}
""")

    logger.info(f"✅ {len(target_reports)}件の資料を選択しました: {', '.join([r['name'] for r in target_reports])}")

    return {
        **state,
        "scored_reports": ScoredReportList(reports=reports),
        "target_reports": TargetReportList(reports=target_reports),
        "target_report_index": 0,
        "messages": [system_message, result_message]
    }
