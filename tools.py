from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import json


@tool
async def tool_or_not(state):
    prompt = """
        Consider user input, history of your conversation and tools, think about what you want to do next:
        User input: {state.current_input}
        Tools you can use: {[": ".join((t.name, t.description)) for t in tools]}
        Decide:
        1. A tool is required
        2. If a tool is required, what is its name
        3. With what parameters should that tool be called
        Return in format {{"thought": "thought process", "tool_required": true/false, "tool": "tool name", "tool_input": "parameters"}}
    """

    response = await llm.ainvoke(prompt)
    result = json.loads(response)
    return AgentState(
        **state.dict(),
        thought=result["thought"],
        selected_tool=result.get("tool"),
        tool_input=result.get("tool_input"),
        status="NEED_TOOL" if result["need_tool"] else "GENERATE_RESPONSE",
    )
