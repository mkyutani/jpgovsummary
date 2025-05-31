import logging
import sys

# ロガーの設定
logger = logging.getLogger("jpgovsummary")
logger.setLevel(logging.INFO)

# ハンドラーの設定
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)

# フォーマッターの設定
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# ハンドラーの追加
logger.addHandler(handler)
