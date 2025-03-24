import sys
from langchain_core.messages.ai import AIMessage

def log(target: any) -> None:
    if type(target) is str:
        print(f"> {target}", file=sys.stderr)
    else:
        result = target
        message = None
        if result is not None:
            if hasattr(result, "content") and result.content is not None and len(result.content) > 0:
                message = result.content
            elif type(result) is AIMessage:
                if hasattr(result, "additional_kwargs") and result.additional_kwargs is not None and "tool_calls" in result.additional_kwargs:
                    tool_calls = result.additional_kwargs["tool_calls"]
                    messages = []
                    for tool_call in tool_calls:
                        if tool_call["type"] == "function":
                            id = tool_call["id"]
                            name = tool_call["function"]["name"]
                            arguments = tool_call["function"]["arguments"]
                            messages.append(f"Function {name}: {arguments}")
                    message = "\n".join(messages)
        if message is None:
            print(f"> No message found: {result}", file=sys.stderr)
        else:
            print(f"> {message}", file=sys.stderr)