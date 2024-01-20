from pydantic import BaseModel
from fastapi import APIRouter
import pandas as pd
from pandas import DataFrame
import numpy as np
import tensorflow as tf

from app.code.get_data import get_data_sql
from app.code.save_data import save_dataframe_to_db
from app.app_config import QUANTILES, QUANTILES_COL

router = APIRouter()


class PredictEnergyData(BaseModel):
    place: str


class OutPredictEnergy(BaseModel):
    message: str


@router.post("/predict_energy_data")
async def predict_energy_data(request: PredictEnergyData) -> OutPredictEnergy:
    """
    predict energy data and insert data into db
    :param request:
    :return: message insert successfully
    """
    df_energy = await get_data_sql('energy_data', request.place)
    df_date = await get_data_sql('date_feature')
    df_weather = await get_data_sql('weather_feature')

    # Data Transformation
    # replace missing value by median value
    df_energy['price_actual'].fillna(df_energy['price_actual'].median(), inplace=True)
    df_energy = df_energy.groupby(['time_hr', 'place'], as_index=False, group_keys=False).apply(__get_previous_data)

    #Excluding first 7 days
    df_energy = df_energy[~pd.isna(df_energy['price_7'])]

    #Normalize month_num data by sin function
    df_date['month_sin'] = df_date['month_num'].apply(lambda x: round(np.sin(2 * np.pi * x / 12), 3))
    # normalize data by normal distribution
    y_std = df_energy['price_actual'].mean()
    y_mean = df_energy['price_actual'].std()
    df_energy['price_actual_norm'] = (df_energy['price_actual'] - y_mean) / y_std
    for i in range(1, 8):
        df_energy[f'price_{i}_norm'] = (df_energy[f'price_{i}'] - y_mean) / y_std

    # left join weather and date
    df_energy = df_energy.merge(df_weather, how='left', on=['dt', 'place'])
    df_energy = df_energy.merge(df_date, how='left', on=['dt'])

    # Check correlation
    corr_cols = ['price_actual_norm', 'price_1_norm', 'price_2_norm', 'price_3_norm', 'price_4_norm', 'price_5_norm',
                 'price_6_norm', 'price_7_norm', 'temp_avg', 'temp_min', 'temp_max', 'month_sin', 'is_weekend']

    df_cor = df_energy[corr_cols].copy()
    df_cor_result = pd.DataFrame(
        np.abs(df_cor.corr(method='pearson')['price_actual_norm']).sort_values(ascending=False),
        columns=['price_actual_norm']).reset_index()
    df_cor_result = df_cor_result.rename({'price_actual_norm': 'corr_value', 'index': 'attr'}, axis=1)
    df_cor_result['place'] = request.place
    df_cor_result['corr_value'] = round(df_cor_result['corr_value'], 2)
    await save_dataframe_to_db(df_cor_result, 'energy_corr_data', ['corr_value'])
    # select attributes has corr >= 0.5
    corr_cols = df_cor_result[
        (df_cor_result['corr_value'] >= 0.5) & (df_cor_result['corr_value'] != 1)]['attr'].to_list()

    # NN prediction
    df_energy = df_energy.groupby(['time_hr', 'place'], as_index=False, group_keys=False).apply(
        __nn_predict_quantile_value, x_cols=corr_cols, y_col=['price_actual_norm'])

    df_energy = df_energy[[
        'dt', 'time_hr', 'place', 'price_actual', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']]
    for quantile in QUANTILES_COL:
        df_energy[quantile] = round(df_energy[quantile] * y_std + y_mean, 2)

    # adjust quantile prediction value
    df_energy[QUANTILES_COL] = df_energy.apply(lambda x: __sort_quantile_values(x), axis=1, result_type="expand")
    await save_dataframe_to_db(df_energy, 'energy_predict_data', ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9'])

    print(df_energy)
    return OutPredictEnergy(
        message='done'
    )


def __get_previous_data(df):
    for i in range(1, 8):
        df[f'price_{i}'] = df['price_actual'].shift(i)
    return df


# Define quantile loss function
def __quantile_loss(quantile):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))

    return loss


def __nn_model(len_input):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(len_input,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model


def __nn_predict_quantile_value(df, x_cols, y_col) -> DataFrame:
    # split data to training and testing
    n = round(len(df) * 0.8)
    df_train = df[:n].copy()
    df_test = df[n:].copy()

    X = df[x_cols].copy()
    X_train = df_train[x_cols]
    y_train = df_train[y_col]

    for i, quantile in enumerate(QUANTILES):
        model = __nn_model(len(x_cols))
        model.compile(optimizer='adam', loss=__quantile_loss(quantile), metrics=['mae'])
        model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)
        df[f"q{str(quantile)[-1]}"] = model.predict(X)

    return df


def __sort_quantile_values(x):
    val = []
    for q in QUANTILES_COL:
        val.append(x[q])
    val.sort()

    return val
