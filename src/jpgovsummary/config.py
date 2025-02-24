from langchain_core.runnables import RunnableConfig

class Config:

    config = None

    @classmethod
    def initialize(cls, uuid=None) -> None:
        if cls.config is None and uuid is not None:
            cls.config = {'configurable': {'thread_id': uuid}}

    def __init__(self, uuid=None) -> None:
        Config.initialize(uuid)

    def get(self) -> RunnableConfig:
        return self.config