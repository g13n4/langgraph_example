import os
import pathlib

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from const import PLACES_TO_VISIT, LOCAL_MODEL_NAME

load_dotenv()

COMPANY_NAME = os.getenv("TravelWithUs")
SAVE_PATH = pathlib.Path("./mock_data")

llm_model = ChatOllama(model=LOCAL_MODEL_NAME)


SYSTEM_PROMPT = """\
You are a creative writer in a travel agency called TravelWithUs. 
Your task is to create short travel guides about cities in South-East Asia that will advertise local food, nature and places to visit and how TravelWithUs can help you with that. 
A travel guide should be about 350 symbols long.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        (
            "human",
            "Write a short ad guide about {city}, {country} mentioning how {company_name} can help a tourist to get the most out of the trip.",
        ),
    ]
)

model_obj = prompt_template | llm_model


def save_story_to_file(country: str, city: str, text: str) -> None:
    file_name = f"{country}-{city}.txt"
    save_path = SAVE_PATH / file_name
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"Ad guide for {city}, {country} saved to {save_path}")
    
    except IOError as e:
        raise f"Error saving file for {city}, {country}\n{e}"

    return None


for country, cities in PLACES_TO_VISIT.items():
    for city in cities:
        output = model_obj.invoke(
            {
                "city": city,
                "country": country,
                "company_name": COMPANY_NAME,
            }
        )
        save_story_to_file(country=country, city=city, text=output.content)