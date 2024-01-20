from typing import Any, Coroutine

from pandas import DataFrame
from app.execute_database import get_sql_data_as_dataframe


async def get_data_sql(table_name, place=None) -> DataFrame:
    """
    get dataframe data from db
    :param table_name:
    :param place: to filter place condition
    :return: data from a table as dataframe type
    """
    script = f"Select * from {table_name}"
    if place is not None:
        script += f" where place='{place}'"

    return await get_sql_data_as_dataframe(script)
