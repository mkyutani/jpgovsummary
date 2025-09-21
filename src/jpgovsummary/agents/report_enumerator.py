from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
import urllib.parse

from .. import CandidateReportList, Config, Model, State, logger


def report_enumerator(state: State) -> State:
    """
    ## Report Enumerator Agent

    Extract document URLs and their names from main content markdown.
    This agent identifies and lists all document links and their corresponding names in the main content markdown.

    Args:
        state (State): The current state containing main content markdown

    Returns:
        State: The updated state with extracted document information
    """
    logger.info("📋 関連資料を列挙...")

    llm = Model().llm()
    parser = JsonOutputParser(pydantic_object=CandidateReportList)
    system_prompt = SystemMessagePromptTemplate.from_template("""
あなたはメインコンテンツのマークダウンを読んでリンクを列挙し、そのリンクが要約に対する関連資料であるか否かを判断する優秀なデータエンジニアです。
リンクが関連資料であるか否かの判断には、メインコンテンツのマークダウンの構造とコンテキストを注意深く分析します。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
以下の手順で処理を行ってください。

## 処理手順
Step 1. 指定されたメインコンテンツのマークダウンに含まれるすべてのリンクを抽出し、リンク先のURLとリンク先のテキストを取得します。
Step 2. 取得した各リンクについて、以下の判断基準に照らし合わせて関連資料であるか否かを判断します。
Step 3. 取得したすべてのリンクについて、リンク先のURLとリンク先のテキスト、及び、判断結果と判断理由を記述します。

### Step 1. リンク先のURLとリンク先のテキストの取得
メインコンテンツのマークダウンのリンクをすべて抽出し、リンク先のURLとリンク先のテキストを取得します。
すべてのリンクを漏れなく抽出してください。ただし、最初に読み込んだマークダウンには存在するがメインコンテンツではないリンクは出力に含める必要はありません。
抽出したリンクは必ず出力に含めてください。

### Step 2. 関連資料であるか否かの判断
取得した各リンクについて、以下の判断基準に照らし合わせて関連資料であるか否かを判断します。
各リンクについて、必ず以下の5つの基準を順番に確認してください：

1. リンク先がの関連資料であるか
   - 会議の議事録、報告書、目次、索引、構成員一覧、資料などであれば関連資料
   - 報告書やとりまとめの本文や概要であれば関連資料
   - 案内、お知らせ、募集などの本文であれば関連資料
   - その他、会議、報告書、とりまとめ、案内、お知らせ、募集など、ページの概要に関連する資料

2. リンク先が会議の資料や補足資料であるか
   - 会議で使用された資料、補足資料であれば関連資料
   - 参考資料、追加資料なども関連資料
   - その他、会議、報告書、とりまとめ、案内、お知らせ、募集など、ページの概要に関連する資料や補足資料

3. リンク先が一般的な資料でないか
   - プライバシーポリシー、サイトマップ、youtube、adobe、NDL Warp(国立国会図書館インターネット資料収集保存事業)などは関連資料ではない
   - 動画ファイル（mp4、avi、mov、wmvなど）や動画配信サービス（YouTube、Vimeoなど）のリンクは関連資料ではない
   - 一般的な案内、お知らせなども関連資料ではない
   - その他、ページの概要との関連が乏しい資料

4. ページのヘッダ、フッタ、メニュー、パンくずリストに含まれるリンクでないか
   - ページの上部、下部、サイドバーなどに配置されたリンクは関連資料ではない
   - ナビゲーション用のリンクは関連資料ではない

#### 判断の注意点
- リンクのテキストとURLの両方を確認し、コンテキストを考慮して判断してください
- リンクの階層構造や位置関係も判断の参考にしてください
- 判断理由は具体的に記述し、なぜそのリンクが資料として適切か/不適切かを説明してください
- 不確かな場合は、より厳密な判断をしてください

### Step 3. 出力
取得したすべてのリンクについて、以下の情報を記述します：
- リンク先のURL
- リンク先のテキスト
- 関連資料であるか否かの判断結果
- 判断理由

#### 出力の注意点
- すべてのリンクを漏れなく出力してください
- リンクが相対的なパスである場合は、ベースURL（{url}）と組み合わせて完全なURLに変換してください
  例：
  - 相対パス: "/documents/report.pdf" 
  - ベースURL: "https://example.gov.jp/meeting/"
  - 完全URL: "https://example.gov.jp/documents/report.pdf"
  - 相対パス: "../files/data.pdf"
  - ベースURL: "https://example.gov.jp/meeting/page/"
  - 完全URL: "https://example.gov.jp/meeting/files/data.pdf"
- 判断結果がtrue/falseに関わらず、すべてのリンクについて判断理由を記述してください

#### 出力フォーマット
以下のフォーマットで出力してください：

{format_instructions}
    """)
    prompt = ChatPromptTemplate.from_messages(
        [system_prompt, assistant_prompt, MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt | llm | parser
    
    # リトライ機能付きでJSONパースを実行
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = chain.invoke(
                {**state, "format_instructions": parser.get_format_instructions()}, Config().get()
            )
            if attempt > 0:
                logger.info(f"✅ JSON解析成功（{attempt}回目）")
            break
            
        except Exception as e:
            logger.warning(f"⚠️ JSON解析失敗（{attempt+1}/{max_retries}回目）: {e}")
            if attempt == max_retries - 1:
                # 最後の試行でも失敗した場合
                logger.error(f"❌ 全{max_retries}回の試行が失敗")
                logger.error(f"❌ 最終エラー: {e}")
                # フォールバック: 空のレポートリストを返す
                result = {"reports": []}
            else:
                logger.info(f"🔄 JSON解析をリトライ中...")
                continue

    reports = result["reports"]
    if not reports or len(reports) == 0:
        logger.info("📄 関連資料が見つかりませんでした")
        reports = []
    else:
        # Python側でURL正規化を実行（確実な相対パス変換）
        base_url = state.get("url", "")
        if base_url:
            for report in reports:
                original_url = report["url"]
                normalized_url = urllib.parse.urljoin(base_url, original_url)
                if original_url != normalized_url:
                    logger.info(f"🔗 URL正規化: {original_url} -> {normalized_url}")
                report["url"] = normalized_url

        reports = sorted(reports, key=lambda x: x["is_document"], reverse=True)
        for report in reports:
            logger.info(
                f"{'o' if report['is_document'] else 'x'} {report['name']} {report['url']} {report['reason']}"
            )

        reports = [report for report in result["reports"] if report["is_document"]]

    # 簡潔な結果メッセージを作成
    system_message = HumanMessage(content="文書URLとその名前をマークダウンから抽出し、関連性を判定してください。")
    result_message = AIMessage(content=f"""
## 候補文書列挙結果

**処理内容**: マークダウンから候補文書を抽出・判定
**発見文書数**: {len(reports)}件
**発見文書**: {', '.join([r['name'] for r in reports[:3]])}{'...' if len(reports) > 3 else ''}
""")

    return {
        **state, 
        "candidate_reports": CandidateReportList(reports=reports),
        "messages": [system_message, result_message]
    }
