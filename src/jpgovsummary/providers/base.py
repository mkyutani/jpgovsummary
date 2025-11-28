from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class BaseLLMProvider(ABC):
    """LLMプロバイダーの基底クラス"""

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """
        LangChain互換のChatモデルを返す

        Returns:
            BaseChatModel: プロバイダー固有のChatモデルインスタンス
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        プロバイダー固有の設定を検証する

        Raises:
            ValueError: 必要な環境変数が設定されていない場合
        """
        pass
