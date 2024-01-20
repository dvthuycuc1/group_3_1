from pandas import DataFrame
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf

QUANTILES = [0.1, 0.5,  0.9]
QUANTILES_COL = ['q1', 'q5', 'q9']

def predict_my_data(df: DataFrame, date_col: str, val_col:str):
    df = df[[date_col, val_col]]
    df.sort_values(by=[date_col], inplace=True)
    df[date_col] = pd.to_datetime(df[date_col])

    #Handling missing value
    df[val_col].fillna(df[val_col].median(), inplace=True)

    y_std = df[val_col].mean()
    y_mean = df[val_col].std()

    #Add next date as new row
    df.loc[len(df.index)] = [max(df[date_col])+ datetime.timedelta(days=1), np.nan]

    df[f"{val_col}_norm"] = (df[val_col] - y_mean) / y_std

    #Get last 7 days value
    for i in range(1, 8):
        df[f'price_{str(i)}_norm'] = df[f"{val_col}_norm"].shift(i)

    # Excluding first 7 days
    df = df[~pd.isna(df['price_7_norm'])]

    corr_cols = [f"{val_col}_norm", 'price_1_norm', 'price_2_norm', 'price_3_norm', 'price_4_norm', 'price_5_norm',
                 'price_6_norm', 'price_7_norm']
    df_cor = df[corr_cols].copy()
    df_cor_result = pd.DataFrame(
        np.abs(df_cor.corr(method='pearson')[f"{val_col}_norm"]).sort_values(ascending=False),
        columns=[f"{val_col}_norm"]).reset_index()
    df_cor_result = df_cor_result.rename({f"{val_col}_norm": 'corr_value', 'index': 'attr'}, axis=1)
    df_cor_result['corr_value'] = round(df_cor_result['corr_value'], 2)

    corr_cols = df_cor_result[
        (df_cor_result['corr_value'] >= 0.5) & (df_cor_result['corr_value'] != 1)]['attr'].to_list()

    df_predict = __nn_predict_quantile_value(df, corr_cols, f"{val_col}_norm")

    for quantile in QUANTILES_COL:
        df[quantile] = round(df[quantile] * y_std + y_mean, 2)

    # adjust quantile prediction value
    df[QUANTILES_COL] = df.apply(lambda x: __sort_quantile_values(x), axis=1, result_type="expand")

    return df_predict, df_cor_result


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

    X = df[x_cols].copy().values.astype(np.float32)
    X_train = df_train[x_cols].values.astype(np.float32)
    y_train = df_train[y_col].values.astype(np.float32)

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