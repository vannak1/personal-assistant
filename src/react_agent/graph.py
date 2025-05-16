# graph.py
"""
Main entry point for defining the agent graph.
This file now initializes and exposes the enhanced multi-agent system.
The 'graph' variable exposed by this module is the primary runnable graph.
"""

# Essential imports for the new system
from react_agent.main import create_personal_assistant
from react_agent.configuration import Configuration # To be exposed
from react_agent.state import State, InputState     # To be exposed
from react_agent.tools import TOOLS                 # To be exposed

# For persistence, replace InMemorySaver with your actual checkpointer if needed
# e.g., if you are using Postgres, you'd initialize your Postgres checkpointer here.
# The original asyncio error mentioned 'langgraph_runtime_postgres',
# so you likely have a specific checkpointer for that.
from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.store.memory import InMemoryStore # if InMemoryStore is also used directly

# --- Initialize the Enhanced Graph System ---

# If you have a specific checkpointer (e.g., for PostgreSQL), initialize it here.
# For example:
# from your_postgres_setup import get_postgres_checkpointer
# checkpointer = get_postgres_checkpointer()
# store = get_postgres_store() # if applicable

# For demonstration, using InMemorySaver as per the defaults in main.py
# You MUST replace this with your actual checkpointer setup if not using in-memory.
checkpointer = InMemorySaver()
# store = InMemoryStore() # If your create_personal_assistant expects a store explicitly

# The `create_personal_assistant` function returns a tuple: (supervisor_system, memory_manager)
# The `supervisor_system` is the compiled, runnable LangGraph.
# The `memory_manager` is also returned if you need to interact with it directly outside the graph.
supervisor_system, memory_manager = create_personal_assistant(
    checkpointer=checkpointer
    # store=store # Pass store if your setup requires it
)

# The LangGraph runner typically looks for a variable named 'graph'.
graph = supervisor_system

print("Enhanced graph (supervisor_system) is now assigned to the 'graph' variable.")

# The function below can be kept if you need a way to get both components,
# or if it's referenced elsewhere. Otherwise, it might be redundant if
# the top-level 'graph' variable is already the enhanced one.
def create_enhanced_graph(with_human_in_loop: bool = True):
    """
    Helper function to create the enhanced personal assistant system.
    Returns the supervisor system (runnable graph) and the memory manager.
    """
    # This will re-initialize. Consider if `supervisor_system` and `memory_manager`
    # defined above should be returned directly if configuration is the same.
    # For simplicity now, it calls the main factory function.
    # Ensure you pass the correct checkpointer here as well.
    temp_checkpointer = InMemorySaver() # Replace with actual checkpointer logic
    return create_personal_assistant(
        checkpointer=temp_checkpointer,
        with_human_in_loop=with_human_in_loop
    )

# Expose State, InputState, Configuration, and TOOLS if your runner/framework
# expects to import them directly from this graph.py module.
__all__ = [
    "graph",
    "State",
    "InputState",
    "Configuration",
    "TOOLS",
    "create_enhanced_graph",
    "memory_manager" # Optionally expose memory_manager
]
```**Key changes in `graph.py`**:
*   Removed the old agent definitions (`personal_assistant_agent`, `features_agent`, etc.) and the old `StateGraph` builder logic.
*   Now, it directly calls `create_personal_assistant` (from `react_agent.main`) to get the new `supervisor_system` and assigns it to the `graph` variable.
*   It includes placeholders for you to insert your actual checkpointer (and store, if applicable), especially if you are using PostgreSQL as hinted by your original error log. Using `InMemorySaver` is a default/fallback.
*   It explicitly lists exports like `State`, `InputState`, `Configuration`, and `TOOLS` as they might be expected by the system loading the graph.

**3. Adjust `__init__.py` (Optional but Recommended)**

Your `react_agent/__init__.py` imports several things from `graph.py`. Since `graph.py` has changed significantly, you should update these imports.

**File: `react_agent/__init__.py` (Suggestion for update)**
```python
"""React Agent multi-agent personal assistant system.

This module defines a custom reasoning and action agent graph.
It implements a multi-agent system with a supervisor, personal assistant, and specialists,
featuring differentiated memory types, handoff mechanisms, optimized tool calling,
and human-in-the-loop capabilities.
"""

from react_agent.configuration import Configuration
from react_agent.state import State, InputState # Assuming these are still central
from react_agent.tools import TOOLS, search_tool, manage_user_session_tool

# Import enhanced components
from react_agent.memory import (
    PrimaryAgentMemory,
    SpecializedAgentMemory,
    UserProfile,
    SessionContext,
    TaskContext,
    create_memory_for_agent
)

from react_agent.handoff import (
    create_handoff_tool,
    create_handoff_tools,
    extract_relevant_context
)

from react_agent.tool_registry import (
    ToolRegistry,
    create_tool_registry,
    TOOL_REGISTRY
)

from react_agent.memory_manager import MemoryManager

from react_agent.supervisor import create_supervisor_system

from react_agent.human_loop import (
    add_human_in_the_loop,
    get_default_approval_configuration
)

from react_agent.main import (
    create_personal_assistant,
    process_user_query
)

# Updated imports from the refactored graph.py
# Only import what's truly necessary to be exposed at this top level from graph.py
from react_agent.graph import (
    graph,  # This is now the enhanced supervisor_system
    create_enhanced_graph, # If still needed
    # The old individual agent functions (personal_assistant_agent, etc.)
    # are no longer defined in the new graph.py, so they are removed here.
)

__all__ = [
    # Configuration
    "Configuration",
    
    # State
    "State",
    "InputState",
    
    # Memory
    "PrimaryAgentMemory",
    "SpecializedAgentMemory",
    "UserProfile",
    "SessionContext",
    "TaskContext",
    "create_memory_for_agent",
    
    # Tools
    "TOOLS",
    "search_tool",
    "manage_user_session_tool",
    
    # Handoff
    "create_handoff_tool",
    "create_handoff_tools",
    "extract_relevant_context",
    
    # Tool registry
    "ToolRegistry",
    "create_tool_registry",
    "TOOL_REGISTRY",
    
    # Memory manager
    "MemoryManager",
    
    # Supervisor
    "create_supervisor_system",
    
    # Human-in-the-loop
    "add_human_in_the_loop",
    "get_default_approval_configuration",
    
    # Main exports
    "create_personal_assistant", # From main.py
    "process_user_query",      # From main.py
    
    # From graph.py
    "graph",
    "create_enhanced_graph"
]