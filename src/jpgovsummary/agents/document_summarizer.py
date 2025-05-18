import logging
from typing import Dict, Any
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from .. import Model, ScoredReportList, State, logger
from ..tools import load_pdf_as_text

def document_summarizer(state: State) -> State:
    """PDF文書を要約するエージェント"""

    logger.info("document_summarizer")

    llm = Model().llm()

    try:
        # 現在のインデックスを取得
        current_index = state.get("target_report_index", 0)
        target_reports = state.get("target_reports", ScoredReportList(reports=[]))

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
            return {**state, "target_report_index": current_index + 1}

        # テキストを分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        docs = [Document(page_content=t) for t in texts]

        # 要約チェーンの設定
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

        # LLMChainを作成して要約を生成
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
        )
        
        # 要約を実行
        summary = chain.invoke(docs)

        # 要約をHumanMessageとして保存（messages用）
        message = HumanMessage(content=f"URL: {url}\n\n要約:\n{summary['output_text']}")
        
        # 新しいstateを作成
        new_summaries = state.get("target_report_summaries", []) + [summary["output_text"]]
        
        return {
            **state,
            "messages": [message],
            "target_report_summaries": new_summaries,
            "target_report_index": current_index + 1
        }

    except Exception as e:
        logger.error(f"Error occurred while summarizing document: {str(e)}")
        current_index = state.get("target_report_index", 0)
        return {**state, "target_report_index": current_index + 1} 