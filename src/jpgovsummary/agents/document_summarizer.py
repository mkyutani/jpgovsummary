import logging
from typing import Dict, Any
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from .. import Model, TargetReportList, State, Summary, logger
from ..tools import load_pdf_as_text

def document_summarizer(state: State) -> State:
    """PDF文書を要約するエージェント"""

    logger.info("document_summarizer")

    llm = Model().llm()
    parser = JsonOutputParser(pydantic_object=Summary)

    # 現在のインデックスを取得
    current_index = state.get("target_report_index", 0)
    target_reports = state.get("target_reports", TargetReportList(reports=[]))
    
    # 初期値を設定
    summary_obj = None
    message = None
    target_report_index = current_index + 1

    try:
        if current_index >= len(target_reports):
            logger.info("All documents have been summarized")
            return state

        # 現在の文書のURLを取得
        current_report = target_reports[current_index]
        url = current_report.url
        name = current_report.name

        logger.info(f"Processing: {name} {url}")
        
        # PDFを読み込んでテキストを抽出
        texts = load_pdf_as_text(url)
        if not texts:
            logger.warning(f"Failed to load PDF: {url}")
            summary_obj = Summary(url=url, name=name, content="")
            message = HumanMessage(content=f"文書: {name}\nURL: {url}\n\n要約: (PDFを読み込めませんでした)")
        else:
            # テキストを分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            )
            docs = [Document(page_content=t) for t in texts]

            # まず要約を生成し、その後JSONに変換する2段階のプロセス
            # ステップ1: テキストの要約
            map_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                以下の文章を要約してください。
                重要な情報を漏らさないように、かつ簡潔にまとめてください。

                文章：
                {text}
                """
            )

            combine_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                以下の要約を1つの文章にまとめてください。
                重複を避け、論理的な流れを保ちながら、重要な情報をすべて含めてください。

                要約：
                {text}
                """
            )

            # 要約を生成
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=True
            )

            summary_result = chain.invoke(docs)
            summary_content = summary_result["output_text"]
            
            # ステップ2: 要約をJSONに変換するプロンプト
            json_prompt = PromptTemplate(
                input_variables=["content", "url", "name"],
                template="""
                以下の要約内容をJSON形式に変換してください。

                文書名: {name}
                URL: {url}
                要約: {content}

                JSON形式の出力形式：
                {format_instructions}
                """
            )

            # JSON形式に変換
            json_chain = json_prompt | llm
            json_result = json_chain.invoke({
                "content": summary_content,
                "url": url,
                "name": name,
                "format_instructions": parser.get_format_instructions()
            })

            # JSONのパースを行う
            # エラーは外側のtry-exceptでキャッチされる
            summary_obj = parser.parse(json_result.content)
            print(summary_obj)

            # メッセージも作成
            message = HumanMessage(content=f"文書: {name}\nURL: {url}\n\n要約:\n{json_result.content}")

    except Exception as e:
        logger.error(f"Error occurred while summarizing document: {str(e)}")
        if current_index < len(target_reports):
            current_report = target_reports[current_index]
            summary_obj = Summary(
                url=current_report.url, 
                name=current_report.name, 
                content=""
            )
        else:
            summary_obj = Summary(url="", name="", content="")

        message = HumanMessage(content=f"文書: {summary_obj.name}\nURL: {summary_obj.url}\n\n要約: (エラーのため要約できませんでした)")

    # 既存のsummariesを取得し、新しい要約を追加
    current_summaries = state.get("target_report_summaries", [])
    new_summaries = current_summaries + ([summary_obj] if summary_obj else [])

    # 新しい状態を返す
    return {
        **state,
        "messages": [message] if message else [],
        "target_report_summaries": new_summaries,
        "target_report_index": target_report_index
    } 