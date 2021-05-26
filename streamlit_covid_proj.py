"""
To train this model, in your terminal:
> streamlit run streamlit_covid_proj.py
"""

import joblib
import streamlit as st
import pandas as pd
import datetime
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df_filt = pd.read_csv('covid_cases_vaccine_data.csv')
df_filt['date'] = pd.to_datetime(df_filt['date']).dt.strftime('%m-%d-%Y')
df_filt.set_index('date', inplace=True)

locations = df_filt.location.unique()
mapping = dict(zip(locations, range(len(locations))))
df_filt['location'].replace(mapping, inplace=True)

# Create title and sidebar
st.title("Covid Cases Tracking App")
st.sidebar.title("Location and Days Input")

location = st.sidebar.selectbox('Select location:', locations, index=51)
date = st.sidebar.date_input("Select date for prediction:", min_value=datetime.date.today())
days = (date - date.today()).days


def forecast_model(location, feature, days_out):
    loc = mapping.get(location)

    X = df_filt.loc[df_filt['location'] == loc][feature]
    p, d, q = (1, 2, 2)

    model = ARIMA(X, order=(p, d, q), freq='D')
    model_fit = model.fit()
    model_fit.forecast(days_out).index = pd.to_datetime(model_fit.forecast(days_out).index)
    return model_fit.forecast(days_out)


def aggregate_state_forecast(location, days_out):
    li = []
    forecast_features = df_filt.columns.to_list()
    forecast_features.remove('new_infections')
    for feature in forecast_features:
        data = forecast_model(location, feature, days_out)
        data.rename(feature)
        li.append(data)
        forecasted_vac_df = pd.concat(li, axis=1, ignore_index=True)

    # Renaming columns
    mapping = dict(zip(forecast_features, range(len(forecast_features))))
    mapping = dict(zip(mapping.values(), mapping.keys()))
    forecasted_vac_df.rename(columns=mapping, inplace=True)

    return forecasted_vac_df


def forecast_new_infections(location, days):
    forecast_state_vac_df = aggregate_state_forecast(location, days)
    prediction_model = joblib.load('models/' + location + '_lr_model.joblib')
    return pd.Series(prediction_model.predict(forecast_state_vac_df), index=forecast_state_vac_df.index)


def plot_forecast(location, days):
    if days == 0: days = 1
    state = mapping.get(location)
    prediction_model = joblib.load('models/' + location + '_lr_model.joblib')
    pred = prediction_model.predict(df_filt.loc[df_filt['location'] == state].drop(['new_infections'], axis=1))
    df_filt.index = pd.to_datetime(df_filt.index)
    date_index = df_filt.loc[df_filt['location'] == state]['new_infections'].index.unique()
    pred_series = pd.Series(pred, index=date_index)



    fig = plt.figure(figsize=(25, 12))

    plt.plot(pred_series, color='blue')
    plt.plot(df_filt.loc[df_filt['location'] == state]['new_infections'], color='red', label='actual')
    plt.plot(forecast_new_infections(location, days), color='blue', label='prediction')

    plt.xticks(rotation='vertical')
    plt.title("Predicting New Infections for the next " +str(days)+ " days in " +location+ " Using Vaccine Data", fontsize=20)
    plt.ylabel("Number of New Infections", fontsize=15)
    plt.xlabel("Date")
    plt.legend()
    plt.show()
    return fig


st.write('\n\n')
fig = plot_forecast(location, days)
st.pyplot(fig)

