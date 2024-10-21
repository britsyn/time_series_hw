import json

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def add_increasing_column(df, value_column, new_column_name='направление'):
    df[new_column_name] = (df[value_column].shift(-1) > df[value_column]).astype(int)
    return df

def prepoc(df):
    df = df.iloc[::-1]
    try:
        df["выход"] = df["выход"].str.replace(',', '.').astype(float)
    except:
        pass
    df['дата'] = pd.to_datetime(df['дата'], dayfirst=True)
    df['направление'] = df['направление'].map(dict(л=1, ш=-1))
    df['day_of_week'] = df['дата'].dt.dayofweek
    df['month'] = df['дата'].dt.month
    df['year'] = df['дата'].dt.year
    return df

df=prepoc(pd.read_csv("Данные.csv"))
df_test = prepoc(pd.read_csv("Данные_2.csv"))


arima_model = ARIMA(df['выход'], exog=df[['day_of_week', 'month', 'year']], order=(5, 1, 3))
arima_result = arima_model.fit()

df_test['выход'] = arima_result.forecast(
    steps=df_test.shape[0],
    exog=df_test[['day_of_week', 'month', 'year']]
).values
add_increasing_column(df_test, 'выход', new_column_name='направление')

forecast_value = df_test['выход'].tolist()
with open('forecast_value.json', 'w') as file:
    json.dump(forecast_value, file)

forecast_class = df_test['направление'].tolist()
with open('forecast_class.json', 'w') as file:
    json.dump(forecast_class, file)

