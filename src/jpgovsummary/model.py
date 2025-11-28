import os

from langchain_core.language_models import BaseChatModel

from .logger import logger
from .providers import get_provider


class Model:
    model = None
    provider_name = None

    @classmethod
    def initialize(cls, model=None) -> None:
        if cls.model is None:
            cls.provider_name = os.environ.get("LLM_PROVIDER", "openai").lower()

            if model is None:
                # 環境変数からモデル名を取得（プロバイダーに応じて変わる）
                if cls.provider_name == "openai":
                    model_name = os.environ.get("OPENAI_MODEL_NAME")
                    if not model_name:
                        raise ValueError("OPENAI_MODEL_NAME environment variable not set")
                elif cls.provider_name == "anthropic":
                    model_name = os.environ.get("ANTHROPIC_MODEL_NAME")
                    if not model_name:
                        raise ValueError("ANTHROPIC_MODEL_NAME environment variable not set")
                elif cls.provider_name == "gemini":
                    model_name = os.environ.get("GEMINI_MODEL_NAME")
                    if not model_name:
                        raise ValueError("GEMINI_MODEL_NAME environment variable not set")
                elif cls.provider_name == "ollama":
                    model_name = os.environ.get("OLLAMA_MODEL_NAME")
                    if not model_name:
                        raise ValueError("OLLAMA_MODEL_NAME environment variable not set")
                else:
                    raise ValueError(f"Unknown LLM provider: {cls.provider_name}")
                cls.model = model_name
            else:
                cls.model = model

            logger.info(f"🤖 プロバイダー {cls.provider_name} でモデル {cls.model} を使用")

    def __init__(self, model=None) -> None:
        if Model.model is None:
            Model.initialize(model)
        self.model = Model.model
        self.provider_name = Model.provider_name

    def llm(self) -> BaseChatModel:
        """
        環境変数で指定されたプロバイダーのLLMインスタンスを返す

        Returns:
            BaseChatModel: プロバイダー固有のChatモデルインスタンス
        """
        provider = get_provider(self.model)
        return provider.get_llm()
