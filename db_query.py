from sqlalchemy import create_engine, MetaData, Table, text
from dotenv import load_dotenv
from const import PSQL_CONNECTION_URI
from decimal import Decimal
import pathlib


load_dotenv()

SAVE_PATH = pathlib.Path("./mock_data")

engine = create_engine(PSQL_CONNECTION_URI)

cities = Table("cities", MetaData(), autoload_with=engine)


def get_city_location(
    city_name: str, country: str | None = None, regex: bool = False
) -> tuple[Decimal, Decimal, str] | None:
    with engine.connect() as db_connection:
        use_regex = "OR translations LIKE {city_name}" if regex else ""
        use_country = "AND country = {country}" if country else ""

        query = text(
            f"SELECT latitude, longitude, country FROM cities WHERE name = '{city_name}' {use_regex} {use_country} ORDER BY population ASC LIMIT 1"
        )
        try:
            data = next(iter(db_connection.execute(query)))
            data = (float(data[0]), float(data[1]), data[2])
        except StopIteration:
            data = None

        return data
