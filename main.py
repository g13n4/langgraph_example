import json
from typing import Annotated, List, Optional

from const import LOCAL_MODEL_NAME, PSQL_CONNECTION_URI
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    filter_messages,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.graph import END, START, StateGraph, add_messages

from db_query import get_city_location
from distance_calculator import calculate_route
from pydantic import BaseModel


llm_model = ChatOllama(model=LOCAL_MODEL_NAME)
system_prompt = (
    " You are a helpful travel guide assistance whose are of expertise is SEA."
)

db_vector = PGVector(
    OllamaEmbeddings(model=LOCAL_MODEL_NAME), connection=PSQL_CONNECTION_URI
)

class State(BaseModel):
    messages: Annotated[list, add_messages]

    cities_to_visit: List[str]
    recommended_places: List[Document]
    home_city: str

    needed_advice: bool = False

    current_input: str = ""

    tool_pick_thought: str = ""

    selected_tool: Optional[str] = None
    tool_input: str = ""
    tool_output: str = ""
    status: str = "CONTINUE"

@tool
async def give_advice(state: State) -> State:
    """
    Provide user with a travel destination advice using last input and context.
    The advice should be about a city in SEA region
    """

    generalized_context = await llm_model.ainvoke(
        """
        You are a copyrighter in a travel agency. Your job is to extract the most important bits about preferences of a person regarding travel.
        Your text will be later use for vector similarity search.
        """
        + "\n\n"
        + "\n".join(filter_messages(state["messages"], include_types="human"))
    )

    docs = db_vector.similarity_search(query=generalized_context.content, k=3)
    if docs:
        message = (
            """
            I can recommend you these places:
            """
            # We can use ids for links. Here they are useless
            + "\n"
            + "\n".join(
                f"{doc['metadata']['country']}, {doc['metadata']['city']}: {doc['metadata']['id']}"
                for doc in docs
            )
        )
    else:
        message = await llm_model.ainvoke(
            """
            You are a helpful travel agency worker. Recommend destinations for travel using provided information as base for your assessment.\n
            """
            + generalized_context.content
        )
        message = message.content

    print(message)

    state.messages.append(AIMessage(message))

    return state


@tool
async def calculate_perfect_route(state: State) -> State:
    """
    Calculates the most optimal route for travel using TSP algorithm and provide appropriate route if proper context was provided during conversation.
    """

    response = await llm_model.ainvoke(
        """
        You are a copyrighter in a travel agency. Find cities that a person wants to travel to and include the home city if it's mentioned. 
        If you sure about in which country that city is you can add that information.
        Return in format {{"city": "travel destination", "country": "_ if you are not sure where the city is located and the name of the country if you are absolutely sure", "home_city": true/false}}
        """
        + "\n\n"
        + "\n".join(filter_messages(state["messages"], include_types="human"))
    )

    result = json.loads(response.content)

    location_map = {}
    no_locations = []
    home_city = None
    for item in result:
        city = item["city"]
        country = item["country"] if item["country"] != "_" else None

        db_output = get_city_location(city_name=city, country=country) or get_city_location(
            city_name=city, country=country, regex=True
        )
        if db_output is None:
            key = f"{city}, {country}"
            no_locations.append(key)
        else:
            lat, long, db_country_name = db_output
            if db_country_name != country:
                country = db_country_name

            key = f"{city}, {country}" if country else city
            location_map[key] = (lat, long)

        if item["home_city"]:
            home_city = key

    if location_map:
        distance, db_output, routes_names_in_optimal_order = calculate_route(
            location_map.keys(), location_map.values(), 0.1
        )

        output = [name for name in location_map.keys()]
        # offset list so the home city will be the first place to leave
        offset_index = routes_names_in_optimal_order.index(home_city)
        if offset_index:
            output = output[offset_index:] + output[:offset_index]

        response = await llm_model.ainvoke(
            """
        You are a helpful assistant in a travel agency. Create a traveling schedule considering the guide of traveling provided. 
        Don't change the traveling information order. Only provide traveling guidance.
        """
            + (
                ""
                if output[0] != home_city
                else " The person will be traveling from your home country"
            )
            + "\n\n"
            + "\n".join([f"{idx}. {name}" for idx, name in enumerate(output)])
        )

    else:
        response = await llm_model.ainvoke(
            """
            You are a helpful assistant in a travel agency. Create a traveling schedule considering the guide of traveling context provided. 
            """
            + "\n\n"
            + "\n".join(filter_messages(state["messages"], include_types="human"))
        )

    print(response.content)
    state.messages.append(AIMessage(response.content))

    return state


tools = [give_advice, calculate_perfect_route]


# We use a simple prompt based tool picker instead of using in-memory vector based one

async def tool_or_not(state: State) -> State:
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

    response = await llm_model.ainvoke(prompt)
    result = json.loads(response.content)
    return State(
        **state.model_dump(),
        selected_tool=result.get("tool"),
        tool_input=result.get("tool_input"),
        status="NEED_TOOL" if result["need_tool"] else "CONTINUE",
    )


async def execute_tool(state: State) -> State:
    """Execute tool call"""
    use_tool = next((t for t in tools if t.name == state.selected_tool), None)
    if not use_tool:
        return await State(
            **state.model_dump(),
            status="ERROR",
            selected_tool=None,
            tool_input="",
            tool_output="",
        )
    try:
        state = await use_tool.func.ainvoke(state.tool_input)
        return State(
            **state.model_dump(),
            status="CONTINUE",
            selected_tool=None,
            tool_input="",
            tool_output="",
        )
    except Exception as e:
        return await State(
            **state.model_dump(),
            status="ERROR",
            selected_tool=None,
            tool_input="",
            tool_output="",
        )


async def chatbot(state: State):
    # your test input
    human_prompt = input()

    # readd system prompt to ensure it was not reset previously
    answer = await llm_model.ainvoke([SystemMessage(system_prompt), HumanMessage(human_prompt)])

    # let reducer take care of appending the message
    return {"messages": [answer]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tool_or_not", tool_or_not)
builder.add_node("execute_tool", execute_tool)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "tool_or_not")
builder.add_conditional_edges(
    "tool_or_not",  lambda s: "execute_tool" if s.status != "CONTINUE" else "chatbot"
)
builder.add_conditional_edges(
    "execute_tool", lambda s: "chatbot" if s.status == "CONTINUE" else END)

graph = builder.compile()
graph.name = "TravelHelper"
