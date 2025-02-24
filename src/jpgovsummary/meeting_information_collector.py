import json
import os
from urllib.parse import urljoin
from langchain_core.tools import tool
import requests

@tool
def meeting_information_collector(uuid: str) -> str:
    '''
    ## 会議情報収集ツール

    Sitewatcherを利用して会議の情報を収集するためのツールです。

    Returns:
        str: 会議の情報
    '''

    api = os.environ.get('SW2_SERVER') or 'http://localhost:18085'
    headers = { 'Cache-Control': 'no-cache' }
    query = urljoin(api, f'/api/v1/resources/{uuid}')

    res = None
    try:
        res = requests.get(query, headers=headers)
    except Exception as e:
        return {'error': str(e)}

    if res.status_code >= 400:
        message = ' '.join([str(res.status_code), res.text if res.text is not None else ''])
        return {'error': message}

    resource = json.loads(res.text)
    data = {}
    for kv in resource['kv']:
        data.update({kv['key']: kv['value']})
    return data