"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    \"\"\"Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    \"\"\"

    is_last_step: IsLastStep = field(default=False)
    \"\"\"
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    \"\"\"

    next: Optional[str] = field(default=None)
    \"\"\"
    Indicates the next node to route to in the hierarchical agent system.
    Used by the supervisor agent to route requests to specialized agents.
    \"\"\"

    active_agent: Optional[str] = field(default=None)
    \"\"\"
    Tracks which specialized agent is currently handling the request.
    \"\"\"

    feature_requests: List[Dict[str, Any]] = field(default_factory=list)
    \"\"\"
    Stores documented feature requests that have been processed by the Feature Request agent.
    Each feature request is a dictionary with details about the request.
    \"\"\"

    user_name: Optional[str] = field(default=None)
    \"\"\"
    Stores the user's name retrieved from the session.
    \"\"\"

    user_uid: Optional[str] = field(default=None)
    \"\"\"
    Stores the unique user ID retrieved or generated from the session.
    \"\"\"
    
    agent_status: Optional[str] = field(default=None)
    \"\"\"
    Stores the current status of the agent for UI display purposes.
    Possible values include: thinking, routing, gathering_more_info, etc.
    \"\"\"
