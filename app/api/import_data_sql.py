from pydantic import BaseModel
from fastapi import APIRouter
import pandas as pd

from app.execute_database import execute_database
router = APIRouter()


class ImportSQLData(BaseModel):
    table_name: str
    file_address: str


class OutImport(BaseModel):
    message: str


@router.post("/import_data_sql")
async def import_data_sql(request: ImportSQLData) -> OutImport:
    """
    insert column data from csv into DB if the date (and place) is not exist
    :param request:
    :return: message insert successfully
    """
    df = pd.read_csv(r'./data/date_feature.csv')
    if request.table_name in ['weather_feature']:
        df['insert_script'] = df.apply(lambda x: __get_insert_script(x, request, has_place=True), axis=1)
    elif request.table_name in ['energy_data']:
        df['insert_script'] = df.apply(lambda x: __get_insert_script(x, request, True, True), axis=1)
    else:
        df['insert_script'] = df.apply(lambda x: __get_insert_script(x, request), axis=1)
    script = f"DO\
                $$\
                BEGIN "
    for s in df['insert_script']:
        script += s

    script += " END\
                $$"
    out_message = await execute_database(script)

    df2 = pd.read_csv(r'./data/energy_data.csv')
    if request.table_name in ['weather_feature']:
        df2['insert_script'] = df2.apply(lambda x: __get_insert_script(x, request, has_place=True), axis=1)
    elif request.table_name in ['energy_data']:
        df2['insert_script'] = df2.apply(lambda x: __get_insert_script(x, request, True, True), axis=1)
    else:
        df2['insert_script'] = df2.apply(lambda x: __get_insert_script(x, request), axis=1)
    script2 = f"DO\
                $$\
                BEGIN "
    for s in df2['insert_script']:
        script2 += s

    script2 += " END\
                $$"
    out_message2 = await execute_database(script2)

    df3 = pd.read_csv(r'./data/weather_feature.csv')
    if request.table_name in ['weather_feature']:
        df3['insert_script'] = df3.apply(lambda x: __get_insert_script(x, request, has_place=True), axis=1)
    elif request.table_name in ['energy_data']:
        df3['insert_script'] = df3.apply(lambda x: __get_insert_script(x, request, True, True), axis=1)
    else:
        df3['insert_script'] = df3.apply(lambda x: __get_insert_script(x, request), axis=1)
    script3 = f"DO\
                $$\
                BEGIN "
    for s in df3['insert_script']:
        script3 += s

    script3 += " END\
                $$"
    out_message3 = await execute_database(script3)

    return OutImport(
        message=out_message,
        message2=out_message2,
        message3=out_message3,
    )


def __get_insert_script(x, request, has_place=False, has_time_hr=False):
    cols_name = ''
    cols_value = ''
    for ind in x.index:
        cols_name += ind + ','
        cols_value += f"'{x[ind]}',"

    cols_name = cols_name[:-1]
    cols_value = cols_value[:-1]
    condition_script = f"dt='{x['dt']}'"

    if has_place:
        condition_script += f" and place='{x['place']}'"
    if has_time_hr:
        condition_script += f" and time_hr='{x['time_hr']}'"

    script = f"IF NOT EXISTS (SELECT *\
                          FROM {request.table_name}\
                          WHERE {condition_script}) THEN\
                  INSERT INTO {request.table_name}({cols_name})\
                  VALUES ({cols_value});\
               END IF;"
    return script

