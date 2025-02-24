import os

from langchain_openai import ChatOpenAI
from openai import OpenAI

class Agent:

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    client = None

    @classmethod
    def initialize(cls) -> None:
        if cls.client is None:
            cls.client = OpenAI(api_key=cls.openai_api_key)

    @classmethod
    def get(cls) -> any:
        return cls.client

    def __init__(self) -> None:
        Agent.initialize()

    def llm(self) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-4o-mini")

    def node(self, placeholder: dict) -> str:
        chain = self.prompt() | ChatOpenAI(model="gpt-4o-mini").with_structured_output(self.structure())
        result = chain.invoke(placeholder)
        return result