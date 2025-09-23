import logging
import sys

# ロガーの設定
logger = logging.getLogger("jpgovsummary")
logger.setLevel(logging.INFO)

# ハンドラーの設定
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)

# デフォルトフォーマッター（batchモード用）
batch_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# シンプルフォーマッター（interactiveモード用）
interactive_formatter = logging.Formatter("%(message)s")

# デフォルトはinteractiveモード
handler.setFormatter(interactive_formatter)

# ハンドラーの追加
logger.addHandler(handler)



class ColorFormatter(logging.Formatter):
    """シンプルな色付きログフォーマッター"""
    
    def format(self, record):
        msg = super().format(record)
        
        # ログレベル別・絵文字別の色付け
        if record.levelno == logging.WARNING:
            return f"\033[33m{msg}\033[0m"  # 黄
        elif record.levelno == logging.ERROR:
            return f"\033[31m{msg}\033[0m"  # 赤
        elif record.levelno == logging.CRITICAL:
            return f"\033[31m{msg}\033[0m"  # 赤
        elif "●" in msg:
            return f"\n\n\033[32m{msg}\033[0m"  # 緑
        elif any(emoji in msg for emoji in ["✅", "🔍", "📄", "🔗", "🔄", "💬"]):
            return msg  # デフォルト色
        else:
            return f"\033[90m{msg}\033[0m"  # 薄いグレー


def supports_color() -> bool:
    """カラーサポートの確認（常にTrue）"""
    return True


def set_batch_mode(batch: bool = False):
    """ログ出力形式をbatchモードの有無で切り替える"""
    if batch:
        handler.setFormatter(batch_formatter)
    else:
        handler.setFormatter(ColorFormatter("%(message)s"))
