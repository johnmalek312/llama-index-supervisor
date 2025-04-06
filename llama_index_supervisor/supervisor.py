from typing import Any

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from .events import InputEvent, StreamEvent, ToolCallEvent
from llama_index.core.agent.workflow import BaseWorkflowAgent
from .handoff import (
    _normalize_agent_name,
    create_handoff_tool,
    create_handoff_back_messages,
)
from .agent_name import add_inline_agent_name
import json
import re
DEFAULT_SYSTEM_PROMPT = (
    "You are a {agent_name}. You will be responsible for managing the workflow of other agents and tools. "
    "You will receive user input and delegate tasks to the appropriate agents or tools. "
    "You will also handle the responses from the agents and tools, and provide feedback to the user. "
)

DEFAULT_DESCRIPTION = (
    "This agent is responsible for managing the following agents and tools. "
    "Agents: {agents}. "
    "Tools: {tools}. "
    )

TREE_STRUCTURE_PROMPT = (
    "The following is a tree structure of the agents and tools in the workflow. "
    "The tree structure is as follows:\n\n{tree_structure}\n\n"
)
## This is an event driven workflow agent, functions that are decorated with @step return an event that is passed to the next step in the workflow.
## So the moment an event is returned, the workflow manager will pick it up and pass it to the next step which takes the event as input.
class Supervisor(Workflow):
    def __init__(
        self,
        llm: FunctionCallingLLM,
        agents: set[BaseWorkflowAgent | Workflow] = [],
        tools: list[BaseTool] = [],
        name: str = "supervisor",
        system_prompt: str | None = None,
        add_handoff_back_messages: bool = True,
        output_mode: str = "full_history",
        description: str | None = DEFAULT_DESCRIPTION,
        add_tree_structure: bool = False,
        name_addition: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Supervisor agent.
        Args:
            llm: The LLM to use for the supervisor agent.
            agents: A list of agents to manage.
            tools: A list of tools to use.
            name: The name of the supervisor agent. (sent to LLM as part of the system prompt)
            system_prompt: The system prompt to use for the supervisor agent. (defaults to DEFAULT_SYSTEM_PROMPT)
            add_handoff_back_messages: Whether to add handoff back messages to the chat history.
            output_mode: The output mode for the supervisor agent. Can be either 'full_history' or 'last_message'.
            description: The description of the supervisor agent. (sent to LLM as part of function description if this supervisor is used as an agent by another supervisor)
            add_tree_structure: Whether to add a tree structure to the context to give llm more context about the agents and tools.
            name_addition: Whether to add the name of the agent that the message belongs to in the message. (defaults to True)
        """
        super().__init__(*args, **kwargs)
        self.validate_agents(agents)
        assert (
            llm.metadata.is_function_calling_model
        ), "Supervisor only supports function calling LLMs"
        assert output_mode in [
            "full_history",
            "last_message",
        ], "output_mode must be either 'full_history' or 'last_message'"
        assert (
            len(agents) + len(tools) > 0
        ), "At least one agent or tool must be provided"

        # Initialize core attributes
        self.name = name
        self.llm = llm
        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT.format(agent_name=name)
        if isinstance(system_prompt, str):
            self.system_prompt = [ChatMessage(role="system", content=system_prompt)]
        elif isinstance(system_prompt, list[str]):
            self.system_prompt = [
                ChatMessage(role="system", content=sp) for sp in system_prompt
            ]
        elif isinstance(system_prompt, ChatMessage):
            self.system_prompt = [system_prompt]
        elif isinstance(system_prompt, list[ChatMessage]):
            self.system_prompt = system_prompt
        self.add_handoff_back_messages = add_handoff_back_messages
        self.output_mode = output_mode

        # Initialize tools and agents
        self.tools = tools or []
        self.agents = agents

        # Setup agents and tools
        self._setup_agents()
        self._setup_tools()

        # Set up the description
        self.description = description or DEFAULT_DESCRIPTION.format(
            agents=", ".join([name for name in self.agent_names]),
            tools=", ".join([name for name in self.tools_by_name.keys()]),
        )
        self.name_addition = name_addition
        self.add_tree_structure = add_tree_structure        
        self.tree_structure = {}
        if add_tree_structure:
            self.tree_dict = self._build_agent_tool_tree(self)
    
    def _build_agent_tool_tree(self, entity: BaseWorkflowAgent | Workflow) -> dict[str, Any]:
        """
        Recursively build the tree structure of agents and tools.

        Args:
            entity: The current agent or supervisor instance.

        Returns:
            A dictionary representing the tree structure, potentially wrapped
            with the top-level agent's name if entity is self.
        """
        subtree: dict[str, Any] = {} # Renamed 'tree' to 'subtree' for clarity

        # Get tools directly associated with the current entity (excluding agent handoff tools for supervisors)
        entity_tools = []
        if hasattr(entity, 'tools'):
            # Ensure agent_tools exists before trying to access it, default to empty list
            agent_tools = getattr(entity, 'agent_tools', [])
            # Ensure metadata and name exist before accessing
            agent_tool_names = set(
                getattr(getattr(at, 'metadata', None), 'name', None)
                for at in agent_tools
            )
            agent_tool_names.discard(None) # Remove None if metadata/name was missing

            entity_tools = [
                getattr(getattr(tool, 'metadata', None), 'name', None)
                for tool in entity.tools
                if hasattr(tool, 'metadata') and (
                    not isinstance(entity, Supervisor) or
                    getattr(getattr(tool, 'metadata', None), 'name', None) not in agent_tool_names
                )
            ]
            # Filter out None names if metadata or name was missing
            entity_tools = [name for name in entity_tools if name]

        if entity_tools:
            subtree["tools"] = sorted(entity_tools) # Sort for consistent output

        # Get agents associated with the current entity
        if hasattr(entity, 'agents') and entity.agents:
            subtree["agents"] = {}
            for agent in entity.agents:
                # Check if agent has a name attribute before using it as a key
                agent_name = getattr(agent, 'name', None)
                if agent_name:
                    # Recursively build the tree for sub-agents
                    # Use self._build_agent_tool_tree for the recursive call
                    subtree["agents"][agent_name] = self._build_agent_tool_tree(agent)

        # If the entity being processed is the top-level 'self', wrap the result
        if entity is self:
            # Ensure self has a name, provide a default if not
            self_name = getattr(self, 'name', 'root_agent') # Use a default name if needed
            return {self_name: subtree}
        else:
            # Otherwise, return the subtree directly for recursive calls
            return subtree
        

    def _setup_agents(self) -> None:
        """Register and initialize all agents."""

        self.agent_names = set()
        self.agents_by_name = {}

        for agent in self.agents:
            normalized_name = _normalize_agent_name(agent.name)
            if normalized_name in self.agents_by_name:
                raise ValueError(
                    f"Duplicate agent name found: {normalized_name}. Agent names must be unique."
                )
            self.agent_names.add(normalized_name)
            self.agents_by_name[normalized_name] = agent
    def validate_agents(self, agents):
        for agent in agents:
            assert hasattr(agent, "description"), f"Agent {agent} is missing 'description'"
            assert hasattr(agent, "name"), f"Agent {agent} is missing 'name'"
            assert hasattr(agent, "run") and callable(getattr(agent, "run")), f"Agent {agent} is missing 'run()' method"

    def _setup_tools(self) -> None:
        """Create tools for agents and register all tools."""
        # Create agent handoff tools
        self.agent_tools = [create_handoff_tool(agent) for agent in self.agents]
        self.tools.extend(self.agent_tools)

        # Create lookup dictionaries
        self.tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        self.agents_by_tool_name = {
            tool.metadata.get_name(): tool for tool in self.agent_tools
        }

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        """Prepare chat history from user input."""

        # Get or create memory
        memory: ChatMemoryBuffer = await ctx.get(
            "memory", default=ChatMemoryBuffer.from_defaults(llm=self.llm)
        )
        user_input = ev.get("input", default=None)
        assert len(memory.get_all()) > 0 or user_input, "Memory input cannot be empty."
        if self.add_tree_structure:
            # Add tree structure to memory
            await memory.aput(
                ChatMessage(role="system", content=TREE_STRUCTURE_PROMPT.format(tree_structure=json.dumps(self.tree_dict, indent=2)))
            )
        # Add user input to memory
        if user_input:
            await memory.aput(ChatMessage(role="user", content=user_input))
            # Update context
            await ctx.set("memory", memory)
        input_messages = memory.get()
        return InputEvent(input=input_messages)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """Process input through LLM and handle streaming response."""

        chat_history = ev.input

        # Stream response from LLM
        response = await self._get_llm_response(ctx, chat_history)

        # Save the final response
        memory = await ctx.get("memory")

        # Check for tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        if not tool_calls:
            message = message.model_copy(deep=True)
        else:
            message = response.message
        add_inline_agent_name(message, self.name)
        await memory.aput(message)
        await ctx.set("memory", memory)

        if not tool_calls:
            return StopEvent(result=response)

        return ToolCallEvent(tool_calls=tool_calls)

    async def _get_llm_response(self, ctx: Context, chat_history):
        """Get streaming response from LLM."""
        response_stream = await self.llm.astream_chat_with_tools(
            self.tools, chat_history=self.system_prompt + chat_history
        )
        response = None
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        return response

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        """Handle tool calls and agent handoffs."""
        tool_calls = ev.tool_calls

        # Split agent handoffs from regular tool calls
        agent_handoffs, regular_tools = self._split_tool_calls(tool_calls)

        # Process all tool calls
        tool_msgs = []
        await self._process_regular_tools(regular_tools, tool_msgs)
        await self._process_agent_handoffs(ctx, agent_handoffs, tool_msgs)

        # Update memory and return input event
        await self._update_memory(ctx, tool_msgs)
        return await self._get_input_event(ctx)

    def _split_tool_calls(self, tool_calls):
        """Split tool calls into agent handoffs and regular tools."""
        agent_handoffs = [
            tc
            for tc in tool_calls
            if any(tc.tool_name == at.metadata.name for at in self.agent_tools)
        ]
        regular_tools = [tc for tc in tool_calls if tc not in agent_handoffs]
        return agent_handoffs, regular_tools

    async def _process_regular_tools(self, regular_tools, tool_msgs: list) -> None:
        """Process regular tool calls."""
        for tool_call in regular_tools:
            tool_name = tool_call.tool_name

            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool_name,
            }

            if not (tool := self.tools_by_name.get(tool_name)):

                tool_msgs.append(
                    self._create_tool_error_message(
                        f"Tool {tool_name} does not exist", additional_kwargs
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)

                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:

                tool_msgs.append(
                    self._create_tool_error_message(
                        f"Encountered error in tool call: {e}", additional_kwargs
                    )
                )

    def _create_tool_error_message(
        self, content: str, kwargs: dict[str, Any]
    ) -> ChatMessage:
        """Create a tool error message."""
        return ChatMessage(
            role="tool",
            content=content,
            additional_kwargs=kwargs,
        )

    async def _process_agent_handoffs(
        self, ctx: Context, agent_handoffs, tool_msgs: list
    ) -> None:
        """Process agent handoff tool calls."""
        if len(agent_handoffs) > 1:
            # Multiple handoffs - return error

            handoff_names = [h.tool_name for h in agent_handoffs]

            for handoff in agent_handoffs:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Multiple agent handoff tools selected: {', '.join(handoff_names)} - please select only one.",
                        additional_kwargs={
                            "tool_call_id": handoff.tool_id,
                            "name": handoff.tool_name,
                        },
                    )
                )
        elif len(agent_handoffs) == 1:
            # Process single handoff
            await self._process_agent_handoff(ctx, agent_handoffs[0], tool_msgs)

    async def _process_agent_handoff(
        self, ctx: Context, handoff, tool_msgs: list
    ) -> None:
        """Process a single agent handoff."""
        handoff_agent = handoff.tool_name.removeprefix("transfer_to_")

        agent = self.agents_by_name.get(handoff_agent)
        if not agent:

            tool_msgs.append(
                ChatMessage(
                    role="tool",
                    content=f"Agent {handoff.tool_name} does not exist",
                    additional_kwargs={
                        "tool_call_id": handoff.tool_id,
                        "name": handoff.tool_name,
                    },
                )
            )
            return

        # Extract handoff parameters
        parameters = handoff.tool_kwargs
        task = parameters.get("task")
        reason = parameters.get("reason")

        # Add success message
        tool_msgs.append(
            ChatMessage(
                role="tool",
                content=f"Successfully transferred to {agent.name}. Your task is: `{task}`, reason: `{reason}`",
                additional_kwargs={
                    "tool_call_id": handoff.tool_id,
                    "name": handoff.tool_name,
                },
            )
        )
        # this adds tool_msgs to the memory and clears tool_msgs
        await self._update_memory(ctx, tool_msgs)

        # Run the agent
        await self._run_agent(ctx, agent)
        # Add handoff back messages if needed
        if self.add_handoff_back_messages:
            handoff_messages = create_handoff_back_messages(
                agent_name=agent.name, supervisor_name=self.name
            )
            tool_msgs.extend(handoff_messages)
            # this adds handoff_messages to the memory
            await self._update_memory(ctx, tool_msgs)

    async def _run_agent(
        self, ctx: Context, agent: BaseWorkflowAgent | Workflow
    ) -> None:
        """Run an agent with the current context."""

        new_ctx = Context(agent)

        memory: ChatMemoryBuffer = await ctx.get("memory")
        new_memory = memory.model_copy()

        # Create a new chat_store instance (assuming it has a copy method or constructor)
        new_memory.chat_store = memory.chat_store.model_copy()  # or manually copy if needed
        # Shallow copy the lists in store
        new_memory.chat_store.store = {key: value[:] for key, value in memory.chat_store.store.items()}

        await new_ctx.set("memory", new_memory) 

        # Run the agent
        await agent.run(ctx=new_ctx, chat_history=new_memory.get())
        new_memory = await new_ctx.get("memory")
        if self.name_addition:
            self._add_name_to_messages(new_memory.get_all(), agent, start_range=len(memory.get_all()))

        # Update supervisor memory with agent's memory
        if self.output_mode == "full_history":
            memo: ChatMemoryBuffer = new_memory
            await ctx.set("memory", memo)
        elif self.output_mode == "last_message":
            memo: ChatMemoryBuffer = await new_ctx.get("memory")
            last_message = memo.get_all()[-1]
            await memory.aput(last_message)
    def _add_name_to_messages(self, messages: list[ChatMessage], agent: BaseWorkflowAgent, start_range: int) -> None:
        """Add agent name to messages."""
        for message in messages[start_range:]:
            # handle none message.content inside the if statement
            if message.role == "assistant" and  not re.search(r"<name>.*?</name><content>.*?</content>", message.content or ""):
                add_inline_agent_name(message, agent.name)

    async def _update_memory(self, ctx: Context, messages: list[ChatMessage]) -> None:
        """Update memory with the provided messages."""
        if not messages:
            return
        memory = await ctx.get("memory")
        for msg in messages:
            await memory.aput(msg)
        messages.clear()  # Empty the list after processing
        await ctx.set("memory", memory)

    async def _get_input_event(self, ctx: Context) -> InputEvent:
        """Get an input event from the current memory."""
        memory = await ctx.get("memory")
        return InputEvent(input=memory.get())
