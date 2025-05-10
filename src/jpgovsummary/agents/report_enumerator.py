from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)
from langchain_core.output_parsers import JsonOutputParser

from .. import Config, Model, Report, ReportList, State, log

def report_enumerator(state: State) -> State:
    """
    ## Report Enumerator Agent

    Extract document URLs and their names from HTML content.
    This agent identifies and lists all document links and their corresponding names in the HTML page.

    Args:
        state (State): The current state containing HTML content

    Returns:
        State: The updated state with extracted document information
    """
    log("report_enumerator")

    llm = Model().llm()
    parser = JsonOutputParser(pydantic_object=ReportList)
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたはHTMLを読んで報告書の内容をまとめる優秀な書記です。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        指定されたHTMLから条件にあうリンクを特定し正確なURLにしてください。
        回答形式に指定した項目以外は回答に含めないでください。

        ## 選定条件

        - aタグである
        - ページのヘッダ、フッタ、メニュー、パンくずリストではない
        - youtube、adobe、NDL Warp(国立国会図書館インターネット資料収集保存事業)のリンクではない
        - このページの趣旨である会議や報告の関連資料であり、一般的な資料ではない

        {format_instructions}
    """)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            assistant_prompt,
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | llm | parser
    result = chain.invoke(
        {
            **state,
            "format_instructions": parser.get_format_instructions()
        },
        Config().get()
    )
    return { **state, "reports": result["reports"] } 