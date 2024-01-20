from app.app_config import CONFIG
import pandas as pd
from pandas import DataFrame
import psycopg2
import logging


async def execute_database(script) -> str:
    con = psycopg2.connect(
        database=CONFIG.DATABASE,
        user=CONFIG.USER,
        password=CONFIG.PASSWORD,
        host=CONFIG.HOST,
        port=CONFIG.PORT
    )
    cursor_obj = con.cursor()
    cursor_obj.execute(script)
    con.commit()
    cursor_obj.close()
    con.close()
    return "successfully"


async def get_sql_data_as_dataframe(script) -> DataFrame:
    con = psycopg2.connect(
        database=CONFIG.DATABASE,
        user=CONFIG.USER,
        password=CONFIG.PASSWORD,
        host=CONFIG.HOST,
        port=CONFIG.PORT
    )
    df = pd.read_sql_query(script, con=con)
    con.close()

    return df
