from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

from .. import Config, Model, State, log

def summary_writer(state: State) -> dict:
    """
    ## Summary Writer Agent

    Write a summary of the meeting based on the input state.

    Args:
        state (State): The current state containing meeting information

    Returns:
        dict: A dictionary containing the generated summary message
    """
    log("summary_writer")

    llm = Model().llm()
    system_prompt = SystemMessagePromptTemplate.from_template("""
        あなたは会議の内容を要約するエージェントです。
    """)
    assistant_prompt = AIMessagePromptTemplate.from_template("""
        会議の内容を要約します。

        ### 要約のルール
        - 会議の内容を簡潔に要約します。
        - 重要なポイントを箇条書きで示します。
        - 会議の結論や決定事項があれば、それを明確に示します。
    """)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            assistant_prompt,
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | llm
    result = chain.invoke(state, Config().get())
    return { "messages": [result] } 