from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import pathlib
import re

import os
import uuid
from const import LOCAL_MODEL_NAME, PSQL_CONNECTION_URI

SAVE_PATH = pathlib.Path("./mock_data")

documents = list()
for file_name in os.scandir(SAVE_PATH):
    country, city, *_ = re.split(r"[.\-]", file_name.name)
    with open(file_name, "r", encoding="utf-8") as file:
        documents.append(
        Document(
            page_content=file.read(),
            metadata={
                "city": city,
                "country": country,
            }
        )
        )

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
documents = text_splitter.split_documents(documents)

db_vector = PGVector(OllamaEmbeddings(model=LOCAL_MODEL_NAME), connection=PSQL_CONNECTION_URI)

ids = [str(uuid.uuid4()) for _ in documents]
db_vector.add_documents(documents, ids=ids)

assert(len(ids)==len(db_vector.get_by_ids(ids)))
