import json
import os
import subprocess

from .. import State, logger


def bluesky_poster(state: State) -> State:
    """
    Human reviewerの後にBlueskyへの投稿を確認・実行するエージェント
    """
    logger.info("🟢 Blueskyに投稿...")

    # 最終要約とURLを取得
    final_summary = state.get("final_review_summary") or state.get("final_summary", "")
    url = state.get("url", "")
    batch = state.get("batch", False)

    if not final_summary:
        logger.warning("⚠️ Bluesky投稿用の最終要約がありません")
        state["bluesky_post_completed"] = True
        return state

    try:
        # 投稿内容をフォーマット
        post_content = _format_bluesky_content(final_summary, url)

        # ユーザーに投稿意思を確認
        if _ask_user_for_bluesky_posting(final_summary, url, post_content, batch):
            # sskyコマンドを直接実行してBluesky投稿
            post_result = _post_to_bluesky_via_ssky(post_content)

            if post_result["success"]:
                logger.info("✅ Blueskyへの投稿に成功しました")
                if post_result.get("uri"):
                    logger.debug(f"URI: {post_result['uri']}")
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


def _post_to_bluesky_via_ssky(content: str) -> dict:
    """
    sskyコマンドを直接実行してBlueskyに投稿
    """
    # 環境変数からSSKY_USERを取得
    ssky_user = os.getenv("SSKY_USER")
    if not ssky_user:
        error_msg = "SSKY_USER environment variable not set. Format: 'USER:PASSWORD'"
        logger.error(f"❌ {error_msg}")
        return {"success": False, "content": content, "result": None, "error": error_msg}

    try:
        # ssky post コマンドを直接実行
        result = subprocess.run(
            ["ssky", "post", "--json", content],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # 成功時はJSONレスポンスをパース
            try:
                response_data = json.loads(result.stdout)
                uri = response_data.get("uri")
                return {
                    "success": True,
                    "content": content,
                    "result": result.stdout,
                    "uri": uri,
                    "error": None
                }
            except json.JSONDecodeError:
                # JSONパース失敗でも終了コード0なら成功とみなす
                return {
                    "success": True,
                    "content": content,
                    "result": result.stdout,
                    "uri": None,
                    "error": None
                }
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            logger.error(f"❌ sskyコマンドが失敗しました: {error_msg}")
            return {"success": False, "content": content, "result": None, "error": error_msg}

    except subprocess.TimeoutExpired:
        error_msg = "sskyコマンドがタイムアウトしました (30秒)"
        logger.error(f"❌ {error_msg}")
        return {"success": False, "content": content, "result": None, "error": error_msg}
    except Exception as e:
        error_msg = f"sskyコマンド実行エラー: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {"success": False, "content": content, "result": None, "error": error_msg}


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


def _ask_user_for_bluesky_posting(summary: str, url: str, post_content: str, batch: bool = False) -> bool:
    """
    ユーザーにBluesky投稿の意思を確認（シンプル版）
    ^C: false (キャンセル), ^D: true (yes)
    バッチモードの場合は自動的にtrueを返す
    """
    # バッチモードの場合は自動承認
    if batch:
        logger.info("バッチモードのため自動的にBlueskyに投稿します")
        return True

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
            # ^D: yes として処理
            return True


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
