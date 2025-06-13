import sys
import asyncio
import os
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
    logger.info("bluesky_poster")
    
    # 最終要約とURLを取得
    final_summary = state.get("final_review_summary") or state.get("final_summary", "")
    url = state.get("url", "")
    
    if not final_summary:
        logger.warning("No final summary available for Bluesky posting")
        state["bluesky_post_completed"] = True
        return state
    
    try:
        # 投稿内容をフォーマット
        post_content = _format_bluesky_content(final_summary, url)
        
        # ユーザーに投稿意思を確認
        if _ask_user_for_bluesky_posting(final_summary, url, post_content):
            print("\n📤 Posting to Bluesky...")
            
            # MCPClientを使ってBluesky投稿を実行
            post_result = asyncio.run(_post_to_bluesky_via_mcp(post_content))
            
            if post_result["success"]:
                print("✅ Successfully posted to Bluesky!")
                state["bluesky_post_completed"] = True
                state["bluesky_post_content"] = post_content
                state["bluesky_post_requested"] = True
                # URIが取得できる場合は保存
                if post_result.get("result"):
                    state["bluesky_post_uri"] = str(post_result["result"])
            else:
                print(f"❌ Failed to post to Bluesky: {post_result['error']}")
                state["bluesky_post_completed"] = True
                state["bluesky_post_requested"] = True
        else:
            print("❌ Bluesky posting cancelled by user.")
            state["bluesky_post_completed"] = True
            state["bluesky_post_requested"] = False
            
    except Exception as e:
        logger.error(f"Error in bluesky_poster: {str(e)}")
        print(f"❌ Error during Bluesky posting: {str(e)}")
        state["bluesky_post_completed"] = True
        
    return state


async def _post_to_bluesky_via_mcp(content: str) -> dict:
    """
    MultiServerMCPClientを使用してLangGraph Agent経由でBlueskyに投稿
    """
    try:
        
        # 環境変数からSSKY_USERを取得
        ssky_user = os.getenv("SSKY_USER")
        if not ssky_user:
            error_msg = "SSKY_USER environment variable not set. Format: 'USER:PASSWORD'"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
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
            logger.error(f"Failed to get MCP tools: {str(e)}")
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
                    logger.info(f"Selected tool: {tool_used}")
                
                # 実際のツール結果がある場合はそれを優先、なければエージェントレスポンスを使用
                result_to_check = actual_tool_result if actual_tool_result is not None else response_content
                
                # エラーパターンの判定（シンプル版：HTTPステータスコードベース）
                result_str = str(result_to_check)
                
                # "Error: 4xx" または "Error: 5xx" パターンをチェック
                import re
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
                    # 成功判定 - JSONレスポンスや投稿成功の兆候を確認
                    result_str = str(result_to_check)
                    
                    # JSONレスポンスの場合（成功の場合）
                    if '"author"' in result_str and '"uri"' in result_str:
                        return {
                            "success": True,
                            "content": content,
                            "result": result_str,
                            "error": None
                        }
                    
                    # その他の成功パターン
                    success_indicators = [
                        "posted successfully", "successfully posted", "posted to bluesky",
                        "post has been", "successfully sent", "message posted"
                    ]
                    
                    if any(indicator in result_str.lower() for indicator in success_indicators):
                        return {
                            "success": True,
                            "content": content,
                            "result": result_str,
                            "error": None
                        }
                    
                    # ツールが実行されていて、エラーでない場合は成功とみなす
                    if actual_tool_result is not None:
                        logger.info("Tool was executed and no error patterns detected, assuming success")
                        return {
                            "success": True,
                            "content": content,
                            "result": result_str,
                            "error": None
                        }
                    
                    # 曖昧な場合
                    logger.warning(f"Ambiguous result, cannot determine success/failure: {result_str}")
                    return {
                        "success": False,
                        "content": content,
                        "result": None,
                        "error": f"Unclear posting result: {result_str}"
                    }
            else:
                return {
                    "success": False,
                    "content": content,
                    "result": None,
                    "error": "No response from agent"
                }
                    
        except Exception as e:
            logger.error(f"Failed to execute agent: {str(e)}")
            return {
                "success": False,
                "content": content,
                "result": None,
                "error": f"Failed to execute agent: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Bluesky posting via MCP failed: {str(e)}", exc_info=True)
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
    """
    # シンプルなY/n確認
    while True:
        try:
            response = _safe_input("Blueskyに投稿しますか？ (Y/n): ").strip()
            
            # デフォルトはYes（Enterのみでも投稿）
            if response == "" or response.lower() in ['y', 'yes']:
                return True
            elif response.lower() in ['n', 'no']:
                return False
            else:
                print("❌ Y/y/yes または N/n/no で回答してください。")
                
        except (KeyboardInterrupt, EOFError):
            return False


def _safe_input(prompt: str, default: str = "") -> str:
    """Safely get user input with Unicode error handling"""
    try:
        return input(prompt).strip()
    except UnicodeDecodeError as e:
        print(f"❌ 文字エンコーディングエラーが発生しました: {e}")
        print("💡 入力に使用できない文字が含まれています。")
        return default
    except (EOFError, KeyboardInterrupt):
        # Re-raise these as they should be handled by the main loop
        raise 