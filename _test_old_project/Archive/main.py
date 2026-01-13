from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import save_to_file


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOllama(model="llama3.2")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

system_prompt = """
            You are a helpful research assistant that will help in generating a research paper. Answer the questions using necessory tools.
            wrap the output in this format and provide no other text\n
"""

tools = [save_to_file]

agent = create_agent(model=llm, system_prompt=system_prompt, tools=tools, debug=True)

raw_response = agent.invoke(
    {
        "query": "Find capital of france and save it to a filename 'capital.txt'",
    }
)
print(raw_response)
