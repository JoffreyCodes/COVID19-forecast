"""
To train this model, in your terminal:
> python train_covid_models.py
"""

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

df_filt = pd.read_csv(r'.\covid_cases_vaccine_data.csv')
df_filt['date'] = pd.to_datetime(df_filt['date']).dt.strftime('%m-%d-%Y')
df_filt.set_index('date', inplace=True)

locations = df_filt.location.unique()
map_loc = dict(zip(locations, range(len(locations))))
df_filt['location'].replace(map_loc, inplace=True)


def get_lr_model(location):
    loc = map_loc.get(location)

    X = df_filt.loc[df_filt['location'] == loc].drop(['new_infections'], axis=1)
    y = df_filt.loc[df_filt['location'] == loc]['new_infections']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression(normalize=True).fit(X_train, y_train)
    joblib.dump(model, 'models/' + location + '_lr_model.joblib')


print("Generating prediction and forecasting models.")
print("wait...")
for location in locations:
    get_lr_model(location)
print("Finished generating models.")