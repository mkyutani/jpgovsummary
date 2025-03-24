import os
import sys

from langchain_core.messages.ai import AIMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from .. import log, State

class Agent:

    openai_api_key = os.environ.get("OPENAI_API_KEY")
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

    @property
    def llm(self) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-4o-mini")

    def think(self) -> dict:
        raise NotImplementedError()

    def node(self, state: State) -> dict:
        log(self.__class__.__name__)
        return self.think(state)