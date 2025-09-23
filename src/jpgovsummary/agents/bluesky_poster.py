import sys
import asyncio
import os
import json
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from .. import Model, State, logger


def bluesky_poster(state: State) -> State:
    """
    Human reviewerの後にBlueskyへの投稿を確認・実行するエージェント
    """
    logger.info("● Blueskyに投稿...")
    
    # 最終要約とURLを取得
    final_summary = state.get("final_review_summary") or state.get("final_summary", "")
    url = state.get("url", "")
    
    if not final_summary:
        logger.warning("⚠️ Bluesky投稿用の最終要約がありません")
        state["bluesky_post_completed"] = True
        return state
    
    try:
        # 投稿内容をフォーマット
        post_content = _format_bluesky_content(final_summary, url)
        
        # ユーザーに投稿意思を確認
        if _ask_user_for_bluesky_posting(final_summary, url, post_content):
            # MCPClientを使ってBluesky投稿を実行
            post_result = asyncio.run(_post_to_bluesky_via_mcp(post_content))
            
            if post_result["success"]:
                # AT URIがあれば抽出してログに含める
                uri = "None"
                if post_result.get("result"):
                    try:
                        result_data = json.loads(str(post_result["result"]))
                        if result_data.get("status") == "success" and "data" in result_data:
                            uri = result_data["data"].get("uri")
                    except (json.JSONDecodeError, KeyError):
                        pass
                state["bluesky_post_completed"] = True
                state["bluesky_post_content"] = post_content
                state["bluesky_post_requested"] = True
                if post_result.get("result"):
                    state["bluesky_post_response"] = str(post_result["result"])
            else:
                logger.error(f"❌ Bluesky投稿に失敗しました: {post_result['error']}")
                state["bluesky_post_completed"] = True
                state["bluesky_post_requested"] = True
        else:
            state["bluesky_post_completed"] = True
            state["bluesky_post_requested"] = False
            
    except Exception as e:
        logger.error(f"❌ Bluesky投稿で想定しないエラーが発生しました: {type(e).__name__}: {str(e)}")
        state["bluesky_post_completed"] = True
        
    return state


def _parse_ssky_response(result_str: str) -> dict:
    """
    ssky mcp-serverのレスポンスを解析して成功/失敗を判定
    """
    def _extract_uri(parsed: dict) -> str:
        """URIを抽出するヘルパー関数"""
        uri = parsed.get("uri")
        if not uri and "data" in parsed and isinstance(parsed["data"], list) and len(parsed["data"]) > 0:
            uri = parsed["data"][0].get("uri")
        return uri

    def _success_response(parsed: dict, default_msg: str = "Posted successfully") -> dict:
        """成功レスポンスを生成するヘルパー関数"""
        return {
            "success": True,
            "uri": _extract_uri(parsed),
            "message": parsed.get("message", default_msg)
        }

    def _error_response(message: str) -> dict:
        """エラーレスポンスを生成するヘルパー関数"""
        return {"success": False, "message": message}

    try:
        parsed = json.loads(result_str)
        
        # HTTPステータスコードベースの判定（最優先）
        http_code = parsed.get("http_code")
        if http_code is not None:
            return _success_response(parsed) if 200 <= http_code < 300 else _error_response(parsed.get("message", f"HTTP error {http_code}"))
        
        # statusフィールドベースの判定
        status = parsed.get("status")
        if status in ["success", "ok"]:
            return _success_response(parsed)
        elif status in ["error", "failure"]:
            return _error_response(parsed.get("message", "Failed to post"))
        
        # 後方互換性：URIの存在チェック
        if parsed.get("uri"):
            return _success_response(parsed)
        
        return _error_response(f"Unknown response format: {result_str}")
        
    except json.JSONDecodeError:
        # JSONではない場合、文字列パターンマッチング
        success_indicators = ["posted successfully", "successfully posted", "posted to bluesky", "post has been", "successfully sent", "message posted"]
        return {"success": True, "message": "Posted successfully"} if any(indicator in result_str.lower() for indicator in success_indicators) else _error_response(f"Unable to parse response: {result_str}")


async def _post_to_bluesky_via_mcp(content: str) -> dict:
    """
    MultiServerMCPClientを使用してLangGraph Agent経由でBlueskyに投稿
    """
    try:
        
        # 環境変数からSSKY_USERを取得
        ssky_user = os.getenv("SSKY_USER")
        if not ssky_user:
            error_msg = "SSKY_USER environment variable not set. Format: 'USER:PASSWORD'"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "content": content,
                "result": None,
                "error": error_msg
            }
        
        # MCPクライアントを初期化
        client = MultiServerMCPClient({
            "ssky": {
                "command": "docker",
                "args": [
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    f"SSKY_USER={ssky_user}",
                    "ghcr.io/simpleskyclient/ssky-mcp"
                ],
                "transport": "stdio",
            }
        })
        
        # MCPツールを取得
        try:
            tools = await client.get_tools()
        except Exception as e:
            logger.error(f"❌ MCPツール取得に失敗しました: {str(e)}")
            return {
                "success": False,
                "content": content,
                "result": None,
                "error": f"Failed to get MCP tools: {str(e)}"
            }
        
        # LangGraphエージェントを作成（実際にツールを実行する）
        llm = Model().llm()
        agent = create_react_agent(llm, tools)
        
        # メッセージを作成して投稿を依頼（JSONフォーマットを指定）
        message = f"Please post the following content to Bluesky using output_format='json': '{content}'"
        
        try:
            # エージェントを実行（ツールが実際に実行される）
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=message)]
            })
            
            # 結果を解析
            if "messages" in result:
                last_message = result["messages"][-1]
                response_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                # ツール呼び出しをチェック（messagesからも確認）
                actual_tool_result = None
                tool_used = None
                
                # messagesからツール呼び出しを検出
                for msg in result["messages"]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call.get('name') == 'ssky_post':
                                tool_used = tool_call['name']
                                break
                    if hasattr(msg, 'content') and isinstance(msg.content, str) and 'ssky_post' in msg.content:
                        # ToolMessageからの結果を取得
                        if hasattr(msg, 'name') and msg.name == 'ssky_post':
                            actual_tool_result = msg.content
                
                # intermediate_stepsからも確認
                if "intermediate_steps" in result and result["intermediate_steps"]:
                    for step in result["intermediate_steps"]:
                        if isinstance(step, tuple) and len(step) == 2:
                            action, observation = step
                            if hasattr(action, 'tool') and action.tool == "ssky_post":
                                actual_tool_result = observation
                                tool_used = action.tool
                                break
                
                if tool_used:
                    logger.info(f"{tool_used}を使用します")
                
                # 実際のツール結果がある場合はそれを優先、なければエージェントレスポンスを使用
                result_to_check = actual_tool_result if actual_tool_result is not None else response_content
                
                # エラーパターンの判定（シンプル版：HTTPステータスコードベース）
                result_str = str(result_to_check)
                
                # "Error: 4xx" または "Error: 5xx" パターンをチェック
                error_pattern = re.compile(r'Error:\s*[45]\d\d')
                is_error = bool(error_pattern.search(result_str)) or "Command timed out" in result_str
                
                if is_error:
                    return {
                        "success": False,
                        "content": content,
                        "result": None,
                        "error": str(result_to_check)
                    }
                else:
                    # 成功判定 - 新しいssky mcp-serverのJSONフォーマットに対応
                    result_str = str(result_to_check)
                    
                    # 新しい成功判定ロジックを使用
                    parsed_result = _parse_ssky_response(result_str)
                    
                    if parsed_result["success"]:
                        return {
                            "success": True,
                            "content": content,
                            "result": result_str,
                            "error": None
                        }
                    else:
                        # ツールが実行されていて、エラーでない場合は成功とみなす（フォールバック）
                        if actual_tool_result is not None:
                            logger.info("✅ ツール実行に成功しました")
                            return {
                                "success": True,
                                "content": content,
                                "result": result_str,
                                "error": None
                            }
                        
                        # 曖昧な場合
                        logger.warning(f"⚠️ 結果が曖昧で成功/失敗を判定できません: {result_str}")
                        return {
                            "success": False,
                            "content": content,
                            "result": None,
                            "error": parsed_result["message"]
                        }
            else:
                return {
                    "success": False,
                    "content": content,
                    "result": None,
                    "error": "No response from agent"
                }
                    
        except Exception as e:
            logger.error(f"❌ エージェント実行に失敗: {str(e)}")
            return {
                "success": False,
                "content": content,
                "result": None,
                "error": f"Failed to execute agent: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"❌ MCP経由Bluesky投稿に失敗: {str(e)}", exc_info=True)
        return {
            "success": False,
            "content": content,
            "result": None,
            "error": str(e)
        }


def _format_bluesky_content(summary: str, url: str) -> str:
    """
    Bluesky投稿用のコンテンツをフォーマット
    ローカルファイルのURLの場合は付加しない
    """
    # URLがWebのURLかどうかを判定
    if url and (url.startswith('http://') or url.startswith('https://')):
        return f"{summary}\n{url}"
    else:
        # ローカルファイルパスの場合はURLを付加しない
        return summary


def _ask_user_for_bluesky_posting(summary: str, url: str, post_content: str) -> bool:
    """
    ユーザーにBluesky投稿の意思を確認（シンプル版）
    ^C: false (キャンセル), ^D: false (no)
    """
    # シンプルなY/n確認
    while True:
        try:
            response = _safe_input("Post to Bluesky? (Y/n): ").strip()

            if response == "" or response.lower()[0] == "y":
                return True
            elif response.lower()[0] == "n":
                return False
        except KeyboardInterrupt:
            # ^C: キャンセル (false)
            return False
        except EOFError:
            # ^D: no として処理
            return False


def _safe_input(prompt: str, default: str = "?") -> str:
    """Safely get user input with Unicode error handling"""
    try:
        return input(prompt).strip()
    except UnicodeDecodeError as e:
        logger.error(f"❌ 文字エンコーディングエラーが発生: {e}")
        return default
    except (EOFError, KeyboardInterrupt):
        print("")
        raise 