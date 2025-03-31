import re
import uuid
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.llms import ChatMessage, MessageRole


WHITESPACE_RE = re.compile(r"\s+")

HANDOFF_TOOL_DESCRIPTION = (
    "Transfers the task to {agent_name}. This agent is responsible for {agent_description}. "
    "Use this tool when the task falls within the agent’s expertise or when delegation is necessary for better task execution. "
    "Provide a clear reason for the transfer and describe the task in detail to ensure smooth handover."
)

def _normalize_agent_name(agent_name: str) -> str:
    """Normalize an agent name to be used inside the tool name."""
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def create_handoff_tool(agent) -> FunctionTool:
    """Create a tool that can handoff control to the requested agent.

    Args:
        agent_name: The name of the agent to handoff control to, i.e.
            the name of the agent node in the multi-agent graph.
            Agent names should be simple, clear and unique, preferably in snake_case,
            although you are only limited to the names accepted by LangGraph
            nodes as well as the tool names accepted by LLM providers
            (the tool name will look like this: `transfer_to_<agent_name>`).
    """
    tool_name = f"transfer_to_{_normalize_agent_name(agent.name)}"

    def handoff_to_agent(ctx: Context, task: str, reason: str) -> str:
        return  # filler function

    return FunctionTool.from_defaults(fn=handoff_to_agent, name=tool_name, description=HANDOFF_TOOL_DESCRIPTION.format(
        agent_name=agent.name, agent_description=agent.description))


def create_handoff_back_messages(
    agent_name: str, supervisor_name: str
) -> tuple[ChatMessage]:
    """Create a pair of (AIMessage, ToolMessage) to add to the message history when returning control to the supervisor."""
    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
    tool_name = f"transfer_back_to_{_normalize_agent_name(supervisor_name)}"
    tool_calls = [
        {
            "id": tool_call_id,
            "function": {"name": tool_name, "arguments": "{}"},
            "type": "function",
        }
    ]
    return (
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"Transferring back to {supervisor_name}",
            additional_kwargs={"tool_calls": tool_calls},
            name=agent_name,
        ),
        ChatMessage(
            tool_call_id=tool_call_id,
            role="tool",
            content=f"Successfully transferred back to {supervisor_name}",
            additional_kwargs={
                "tool_call_id": tool_call_id,
                "name": tool_name,
            },
        ),
    )
