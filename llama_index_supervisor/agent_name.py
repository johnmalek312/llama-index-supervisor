import re
from typing import Literal

from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
NAME_PATTERN = re.compile(r"<name>(.*?)</name>", re.DOTALL)
CONTENT_PATTERN = re.compile(r"<content>(.*?)</content>", re.DOTALL)

AgentNameMode = Literal["inline"]

def add_inline_agent_name(message: ChatMessage) -> ChatMessage:
    """Add name and content XML tags to the message content.

    Examples:

        >>> add_inline_agent_name(AIMessage(content="Hello", name="assistant"))
        AIMessage(content="<name>assistant</name><content>Hello</content>", name="assistant")

        >>> add_inline_agent_name(AIMessage(content=[{"type": "text", "text": "Hello"}], name="assistant"))
        AIMessage(content=[{"type": "text", "text": "<name>assistant</name><content>Hello</content>"}], name="assistant")
    """
    if not isinstance(message, ChatMessage):
        return message
    formatted_message = message.model_copy()
    formatted_message.content = (
        f"<name>{message.name}</name><content>{formatted_message.content}</content>"
    )
    return formatted_message


def remove_inline_agent_name(message: ChatMessage) -> ChatMessage:
    """Remove explicit name and content XML tags from the AI message content.

    Examples:

        >>> remove_inline_agent_name(AIMessage(content="<name>assistant</name><content>Hello</content>", name="assistant"))
        AIMessage(content="Hello", name="assistant")

        >>> remove_inline_agent_name(AIMessage(content=[{"type": "text", "text": "<name>assistant</name><content>Hello</content>"}], name="assistant"))
        AIMessage(content=[{"type": "text", "text": "Hello"}], name="assistant")
    """
    if not isinstance(message, ChatMessage):
        return message


    content = message.content

    name_match: re.Match | None = NAME_PATTERN.search(content)
    content_match: re.Match | None = CONTENT_PATTERN.search(content)
    if not name_match or not content_match:
        return message

    if name_match.group(1) != message.name:
        return message

    parsed_content = content_match.group(1)
    parsed_message = message.model_copy()
    parsed_message.content = parsed_content
    return parsed_message
