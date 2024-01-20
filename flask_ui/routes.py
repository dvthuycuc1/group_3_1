from flask import render_template, request
from datetime import datetime
from flask_ui import app
from app.code.get_data import get_data_sql
import pandas as pd
import numpy as np
from app.app_config import QUANTILES_COL
from app.code.predict_my_data import predict_my_data


@app.route("/", methods=['GET', 'POST']) #router to go different pages, allows write a function return the information will be shown
#"/": root page of website, homepage
@app.route("/home", methods=['GET', 'POST'])
#has 2 links to get to this page
async def home():
    place = "italy"
    time_hr = "0"
    if request.method == 'POST':
        place = request.form.get("place")
        time_hr = request.form.get("time_hr")

    df_input = await get_data_sql('energy_data', place)
    df_corr = await get_data_sql('energy_corr_data', place)
    df_output = await get_data_sql('energy_predict_data', place)

    df_output.drop('id', axis=1, inplace=True)
    df_output[QUANTILES_COL] = round(df_output[QUANTILES_COL],2)
    df_input['dt'] = df_input['dt'].apply(lambda x: datetime.combine(x, datetime.min.time()))
    df_input['dt'] = df_input['dt'].apply(lambda x: int(x.timestamp() * 1000))


    id_var_cols = [col for col in df_output.columns if col not in QUANTILES_COL]
    df_out_trans = df_output.copy().melt(id_vars=id_var_cols, var_name="quantiles", value_name="price_predict")
    df_out_trans = df_out_trans[df_out_trans['dt']==max(df_out_trans['dt'])]
    date_hr_pred = []
    hr_cols = []
    for hr in range(0, 24):
        hr_cols.append('h' + str(hr))
        date_hr_pred.append(df_out_trans[df_out_trans['time_hr']==str(hr)]['price_predict'].to_list())

    df_output_cov = df_output.copy()
    df_output_cov['dt'] = df_output_cov['dt'].apply(lambda x: datetime.combine(x, datetime.min.time()))
    df_output_cov['dt'] = df_output_cov['dt'].apply(lambda x: int(x.timestamp() * 1000))

    # Calculate MAE, RMSE
    mae = round(np.abs(df_output[df_output['time_hr']==time_hr]['price_actual'] - df_output[df_output['time_hr']==time_hr]['q5']).mean(),2)
    rmse = round(np.sqrt((pow(df_output[df_output['time_hr']==time_hr]['price_actual'] - df_output[df_output['time_hr']==time_hr]['q5'], 2).sum()) / len(df_output[df_output['time_hr']==time_hr])),2)

    return render_template('home.html', selected_place=place, time_hr=time_hr,
                           energy_col=['dt','price_actual'], energy_data=list(df_output_cov[df_output_cov['time_hr']==time_hr][['dt','price_actual']].values.tolist()),
                           histogram_data=df_input[df_input['time_hr']==time_hr]['price_actual'].to_list(),
                           attr_values=df_corr['attr'].tolist(), corr_value=df_corr['corr_value'].tolist(),
                           table_pred_cols=df_output.columns.values, table_pred_data= [df_output[df_output['dt']==max(df_output['dt'])].to_html(classes='data', index=False, justify='center')],
                           hr_cols=hr_cols, date_hr_pred=date_hr_pred, date_str=str(max(df_output['dt'])),
                           range_data=list(df_output[df_output['time_hr']==time_hr][['q1', 'q9']].values.tolist()), range_y_data=df_output[df_output['time_hr']==time_hr]['price_actual'].values.tolist(),
                           predict_data=list((df_output_cov[df_output_cov['time_hr']==time_hr][['dt','q5']].values.tolist())),
                           mae=mae, rmse=rmse)


def get_csv_headers(file_path):
    # Function to get the headers from a CSV file
    with open(file_path, 'r') as csv_file:
        headers = csv_file.readline().strip().split(',')
    return headers

@app.route('/test', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('test.html')

    if request.method == 'POST' and 'file' in request.files and request.form.get("datecolumn") is None:
        # Check if the POST request has the file part
        file = request.files['file']
        print(file)
        # Save the uploaded file temporarily
        file_path = f"temp/{file.filename}"
        file.save(file_path)

        #overwrite file_path
        handle = open("temp/file_name.txt", "w")
        handle.seek(0)
        handle.write(file_path)
        handle.truncate()
        handle.close()
        # Get CSV headers for the dropdown
        csv_headers = get_csv_headers(file_path)

        # You can use the file_path and csv_headers as needed (e.g., process, validate, etc.)
        # Here, we are just printing the file name and headers
        print(f"Received file: {file.filename}")
        print(f"CSV Headers: {csv_headers}")
        # Optionally, you can return a response or redirect to another page
        return render_template('test.html', success='File uploaded successfully', headers=csv_headers)

    if request.method == 'POST' and request.form['button']=="predictiondata":
        f = open("temp/file_name.txt", "r")
        file_path = f.read()
        csv_headers = get_csv_headers(file_path)

        date_col = request.form.get("datecolumn")
        val_col = request.form.get("valuecolumn")
        df = pd.read_csv(file_path)
        df, df_corr = predict_my_data(df, date_col, val_col)

        df_date_converted = df.copy()
        df_date_converted = df_date_converted
        df_date_converted[date_col] = df_date_converted[date_col].apply(lambda x: datetime.combine(x, datetime.min.time()))
        df_date_converted[date_col] = df_date_converted[date_col].apply(lambda x: int(x.timestamp() * 1000))

        df_out_trans = df[~pd.isna(df_date_converted[val_col])][[date_col,'q1','q5','q9']].copy().melt(id_vars=[date_col], var_name="quantiles", value_name="price_predict")
        df_out_trans = df_out_trans[df_out_trans[date_col] == max(df_out_trans[date_col])]
        df_out_trans['quantiles'] = df_out_trans['quantiles'].str[-1].astype('int')
        pred_val = list(df_out_trans['price_predict'].to_list())
        print(df_out_trans)

        # Calculate MAE, RMSE
        mae = round(np.abs(df[val_col] - df['q5']).mean(), 2)
        rmse = round(np.sqrt((pow(df[val_col] - df['q5'], 2).sum()) / len(df)), 2)

        #Table setup
        df_table = df[df[date_col]==max(df[date_col])][[date_col, val_col, 'q1', 'q5', 'q9']].copy()

    return render_template('test.html', success='File uploaded successfully', headers=csv_headers,
                           date_col=date_col, val_col=val_col,
                           energy_data=list(df_date_converted[~pd.isna(df_date_converted[val_col])][[date_col,val_col]].values.tolist()),
                           histogram_data=df_date_converted[~pd.isna(df_date_converted[val_col])][val_col].to_list(),
                           attr_values=df_corr['attr'].tolist(), corr_value=df_corr['corr_value'].tolist(),
                           hr_cols=[val_col], date_hr_pred=pred_val, date_str=str(max(df[date_col])),
                           start_point=min(df_date_converted[date_col]),range_data=list(df_date_converted[['q1', 'q9']].values.tolist()), range_y_data=df_date_converted[~pd.isna(df_date_converted[val_col])][val_col].values.tolist(),
                           predict_data=list((df_date_converted[[date_col, 'q5']].values.tolist())),
                           mae=mae, rmse=rmse,
                           table_pred_cols=df_table.columns.values, table_pred_data= [df_table.to_html(classes='data', index=False, justify='center')],)

