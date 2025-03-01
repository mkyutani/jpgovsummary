import json
import os
import sys
from urllib.parse import urljoin
from langchain_core.tools import tool
import requests

from .agent import Agent

class MeetingInformationCollector(Agent):

    def __init__(self) -> None:
        super().__init__()

    @tool
    def tool(uuid: str) -> dict:
        '''
        ## 会議情報収集ツール

        Sitewatcherを利用して会議のUUIDから会議の情報を収集するためのツールです。

        Args:
            uuid (str): 会議のUUID

        Returns:
            str: 会議の情報
        '''

        print('meeting_information_collector', file=sys.stderr)

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

        return res.text