from langchain_core.runnables import RunnableConfig

class Config:

    config = None

    @classmethod
    def initialize(cls, id=None) -> None:
        if cls.config is None and id is not None:
            cls.config = {"configurable": {"thread_id": id}}

    def __init__(self, id=None) -> None:
        Config.initialize(id)

    def get(self) -> RunnableConfig:
        return self.config