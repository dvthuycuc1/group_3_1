from app.execute_database import execute_database


async def save_dataframe_to_db(df, table_name, col_value_name: list) ->str:
    df['insert_script'] = df.apply(lambda x: __get_insert_script(x, table_name, col_value_name), axis=1)
    script = f"DO\
                    $$\
                    BEGIN "
    for s in df['insert_script']:
        script += s

    script += " END\
                    $$"
    print(script)
    out_message = await execute_database(script)
    return out_message


def __get_insert_script(x, table_name, col_value_name):
    cols_name = ''
    cols_value = ''
    condition_val = ''
    update_val = ''
    for ind in x.index:
        if ind not in col_value_name:
            condition_val += f"{ind}='{x[ind]}' and "
        else:
            update_val += f"{ind} = '{x[ind]}', "

        cols_name += ind + ','
        cols_value += f"'{x[ind]}',"

    cols_name = cols_name[:-1]
    cols_value = cols_value[:-1]
    condition_val = condition_val[:-4]
    update_val = update_val[:-2]
    script = f"IF NOT EXISTS (SELECT *\
                          FROM {table_name}\
                          WHERE {condition_val}) THEN\
                  INSERT INTO {table_name}({cols_name})\
                  VALUES ({cols_value});\
                ELSE\
                    UPDATE {table_name} SET {update_val}\
                    WHERE {condition_val};\
               END IF;"
    return script