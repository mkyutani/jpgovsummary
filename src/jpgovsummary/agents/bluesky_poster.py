import sys
import asyncio
import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient

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
    MultiServerMCPClientを使用してBlueskyに投稿
    """
    try:
        logger.info("Initializing MCP client for Bluesky posting")
        
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
        
        logger.info("Getting MCP tools")
        # MCPツールを取得
        try:
            tools = await client.get_tools()
            logger.info(f"Retrieved {len(tools)} MCP tools")
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {str(e)}")
            return {
                "success": False,
                "content": content,
                "result": None,
                "error": f"Failed to get MCP tools: {str(e)}"
            }
        
        # ssky_postツールを直接呼び出し
        logger.info("Finding ssky_post tool")
        ssky_post_tool = None
        for tool in tools:
            if tool.name == "ssky_post":
                ssky_post_tool = tool
                break
        
        if not ssky_post_tool:
            error_msg = "ssky_post tool not found in available tools"
            logger.error(error_msg)
            return {
                "success": False,
                "content": content,
                "result": None,
                "error": error_msg
            }
        
        # ツールを直接実行
        logger.info(f"Calling ssky_post tool with content: {content[:100]}...")
        try:
            result = await ssky_post_tool.ainvoke({
                "message": content,
                "dry_run": False,
                "output_format": "text"
            })
            logger.info(f"ssky_post result: {result}")
            
            # 結果を文字列として取得
            result_str = str(result) if result else ""
            
            # 成功判定（エラーメッセージがない場合は成功とみなす）
            if "error" not in result_str.lower() and "failed" not in result_str.lower():
                return {
                    "success": True,
                    "content": content,
                    "result": result_str,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "content": content,
                    "result": None,
                    "error": result_str
                }
                
        except Exception as e:
            logger.error(f"Failed to call ssky_post tool: {str(e)}")
            return {
                "success": False,
                "content": content,
                "result": None,
                "error": f"Failed to call ssky_post tool: {str(e)}"
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