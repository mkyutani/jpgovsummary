import requests
from typing import List
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from PyPDF2 import PdfReader
from io import BytesIO

from .. import logger

def load_pdf_as_text(url: str) -> List[str]:
    """
    PDFファイルをダウンロードしてテキストを抽出する

    Args:
        url (str): PDFファイルのURL

    Returns:
        List[str]: 抽出されたテキストのリスト（ページごと）
    """
    try:
        # PDFファイルをダウンロード
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # PDFを読み込んでテキストを抽出
        pdf_file = BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        texts = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

        return texts

    except Exception as e:
        logger.error(f"PDFファイルの読み込み中にエラーが発生しました: {str(e)}")
        raise

@tool
def pdf_loader(state: dict) -> dict:
    """
    PDFファイルをダウンロードしてテキストを抽出するツール

    Args:
        state (dict): 現在の状態

    Returns:
        dict: 更新された状態
    """
    try:
        # 最後のメッセージからURLを取得
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            raise ValueError("最後のメッセージがHumanMessageではありません")
        
        url = last_message.content
        if not url.startswith("http"):
            raise ValueError("URLが指定されていません")

        # PDFファイルをダウンロードしてテキストを抽出
        texts = load_pdf_as_text(url)

        # 結果をメッセージとして追加
        return {
            **state,
            "messages": [
                *state["messages"],
                HumanMessage(content=f"PDFファイルのテキストは以下の通りです：\n\n{texts[0]}")
            ]
        }

    except Exception as e:
        logger.error(f"PDFファイルの読み込み中にエラーが発生しました: {str(e)}")
        return state