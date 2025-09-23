import logging
import os
import sys
from typing import Dict, Optional



def parse_ls_colors(ls_colors: Optional[str] = None) -> Dict[str, str]:
    """
    LS_COLORS環境変数をパースしてディクショナリに変換
    
    Args:
        ls_colors: LS_COLORS文字列（Noneの場合は環境変数から取得）
    
    Returns:
        キーと色コードのディクショナリ
    """
    if ls_colors is None:
        ls_colors = os.environ.get('LS_COLORS', '')
    
    if not ls_colors:
        return get_default_colors()
    
    colors = {}
    # コロンで分割してkey=value形式をパース
    for item in ls_colors.split(':'):
        if '=' in item:
            key, value = item.split('=', 1)
            colors[key] = value
        elif item:  # 拡張子など
            colors[item] = '01;31'  # デフォルト色
    
    # デフォルト値でフォールバック
    default_colors = get_default_colors()
    for key, value in default_colors.items():
        if key not in colors:
            colors[key] = value
    
    return colors


def get_default_colors() -> Dict[str, str]:
    """デフォルトの色設定"""
    return {
        'rs': '0',           # リセット
        'di': '01;34',       # ディレクトリ（青・太字）
        'ln': '01;36',       # シンボリックリンク（シアン・太字）
        'ex': '01;32',       # 実行可能（緑・太字）
        'fi': '',            # 通常ファイル（デフォルト）
        'or': '40;31;01',    # 孤立リンク（赤背景・赤・太字）
        'mi': '40;31;01',    # 存在しないファイル（赤背景・赤・太字）
        'so': '01;35',       # ソケット（マゼンタ・太字）
        'pi': '40;33',       # 名前付きパイプ（黄背景）
        'bd': '40;33;01',    # ブロックデバイス（黄背景・太字）
        'cd': '40;33;01',    # キャラクターデバイス（黄背景・太字）
        '*~': '00;90',       # バックアップファイル（薄いグレー） - デフォルトINFO用
    }


def color_code_to_ansi(color_code: str) -> str:
    """
    LS_COLORSの色コード（例：01;34）をANSIエスケープシーケンスに変換
    
    Args:
        color_code: LS_COLORSの色コード（例：'01;34'）
    
    Returns:
        ANSIエスケープシーケンス（例：'\033[01;34m'）
    """
    if not color_code:
        return ''
    return f'\033[{color_code}m'


def get_reset_code() -> str:
    """リセットコードを取得"""
    return '\033[0m'


class LSColorFormatter(logging.Formatter):
    """LS_COLORS対応のカラーフォーマッター"""
    
    def __init__(self, fmt=None):
        super().__init__(fmt)
        self.colors = parse_ls_colors()
        
        # ログレベルとLS_COLORSキーのマッピング
        self.level_mapping = {
            logging.DEBUG: 'fi',      # 通常ファイル（デフォルト）
            logging.INFO: 'fi',       # 通常ファイル（デフォルト）
            logging.WARNING: 'so',    # ソケット（紫）
            logging.ERROR: 'or',      # 孤立リンク（赤背景）
            logging.CRITICAL: 'mi',   # 存在しないファイル（赤背景・点滅）
        }
        
        # 進行状況表示用の特別な絵文字（緑色・改行付き）
        self.progress_emoji = '🟢'
    
    def format(self, record):
        msg = super().format(record)
        
        # ログレベルベースの色選択
        color_key = self.level_mapping.get(record.levelno, 'fi')
        
        # 進行状況絵文字の特別処理（改行付きだが色はデフォルト）
        if self.progress_emoji in msg:
            prefix = "\n\n"
            # 色はデフォルト（fi）のまま、薄いグレーにしない
        else:
            prefix = ""
            # 絵文字がない通常のINFOメッセージのみ薄いグレーに
            has_emoji = any(emoji in msg for emoji in ['✅', '🔍', '📄', '🔗', '🔄', '💬'])
            if not has_emoji and record.levelno == logging.INFO:
                color_key = '*~'  # バックアップファイル色（薄いグレー）
        
        # 色コードを適用
        color_code = self.colors.get(color_key, '')
        if color_code:
            ansi_color = color_code_to_ansi(color_code)
            reset = get_reset_code()
            return f"{prefix}{ansi_color}{msg}{reset}"
        else:
            return f"{prefix}{msg}"


def supports_color() -> bool:
    """カラーサポートの確認（常にTrue）"""
    return True


def set_batch_mode(batch: bool = False):
    """ログ出力形式をbatchモードの有無で切り替える"""
    if batch:
        handler.setFormatter(batch_formatter)
    else:
        handler.setFormatter(LSColorFormatter("%(message)s"))


# ロガーの設定
logger = logging.getLogger("jpgovsummary")
logger.setLevel(logging.INFO)

# ハンドラーの設定
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)

# デフォルトフォーマッター（batchモード用）
batch_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# デフォルトはLS_COLORS対応のinteractiveモード
handler.setFormatter(LSColorFormatter("%(message)s"))

# ハンドラーの追加
logger.addHandler(handler)
