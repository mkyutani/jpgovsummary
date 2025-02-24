from langchain_core.tools import tool
import requests

@tool
def meeting_page_reader(url: str) -> str:
    '''
    ## 会議ページ読み取りツール

    指定されたURLにアクセスして会議ページ全体のHTMLデータを読みこみます。

    Returns:
        str: 会議ページ全体のHTMLデータ
    '''

    headers = {}
    res = None
    try:
        print(url)
        res = requests.get(url, headers=headers)
    except Exception as e:
        print({'error': str(e)})
        return {'error': str(e)}

    if res.status_code >= 400:
        message = ' '.join([str(res.status_code), res.text if res.text is not None else ''])
        print({'error': message})
        return {'error': message}

    print(res.text)
    return {'content': res.text}