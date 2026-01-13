from dotenv import load_dotenv

import os


load_dotenv()


LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME")


POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_URI = os.getenv("POSTGRES_URI")

PSQL_CONNECTION_URI = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_URI}/{POSTGRES_DB}"


PLACES_TO_VISIT = {
    "Thailand": ["Bangkok", "Chiang Mai", "Phuket"],
    "Vietnam:": ["Hoi An", "Ha Long Bay", "Da Nang"],
    "Indonesia": ["Bali", "Lombok"],
    "Philippines": ["El Nido", "Boracay"],
    "Malaysia": ["Kuala Lumpur", "Borneo", "Langkawi"],
    "Singapore": ["Singapore"],
    "Cambodia": ["Siem Reap", "Koh Rong"],
}
