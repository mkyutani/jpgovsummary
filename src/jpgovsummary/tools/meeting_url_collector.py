import os
import requests
from urllib.parse import urljoin

from langchain_core.tools import tool

from .. import log

@tool
def meeting_url_collector(uuid: str) -> str:
    """
    ## 会議URL取得ツール

    Sitewatcherを利用して会議のUUIDから会議のURLを取得するツールです。

    Args:
        uuid (str): 会議のUUID

    Returns:
        str: 会議のURL
    """

    log("meeting_url_collector")

    api = os.environ.get("SW2_SERVER") or "http://localhost:18085"
    headers = { "Cache-Control": "no-cache" }
    query = urljoin(api, f"/api/v1/resources/{uuid}")

    res = None
    try:
        res = requests.get(query, headers=headers)
    except Exception as e:
        return {"error": str(e)}

    if res.status_code >= 400:
        message = " ".join([str(res.status_code), res.text if res.text is not None else ""])
        return {"error": message}

    data = res.json()
    result = data["resource_uri"]

    log(result)

    return result