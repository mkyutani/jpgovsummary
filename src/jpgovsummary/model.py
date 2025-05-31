import os

from langchain_openai import ChatOpenAI

from .logger import logger


class Model:
    model = None

    @classmethod
    def initialize(cls, model=None) -> None:
        if cls.model is None:
            if model is None:
                model_name = os.environ.get("OPENAI_MODEL_NAME")
                if not model_name:
                    raise ValueError("OPENAI_MODEL_NAME environment variable not set")
                cls.model = model_name
            else:
                cls.model = model
            logger.info(f"Use model {cls.model}")

    def __init__(self, model=None) -> None:
        if Model.model is None:
            Model.initialize(model)
        self.model = Model.model

    def llm(self) -> ChatOpenAI:
        """
        Get a ChatOpenAI instance with the model specified in the environment variable.

        Returns:
            ChatOpenAI: A configured ChatOpenAI instance.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return ChatOpenAI(api_key=api_key, model=self.model)
