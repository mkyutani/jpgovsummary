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

def set_batch_mode(batch: bool = False):
    """ログ出力形式をbatchモードの有無で切り替える"""
    if batch:
        handler.setFormatter(batch_formatter)
    else:
        handler.setFormatter(interactive_formatter)
