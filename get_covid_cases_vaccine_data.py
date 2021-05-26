"""
To get up-to-date data, run in your terminal:
> python get_covid_cases_vaccine_data.py
"""
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

NEW_INFECTIONS_PERIOD = 7

# Get COVID-19 Vaccine Data
VAC_DATA = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv'
print("Retrieving vaccine data from " + VAC_DATA)
vac_df = pd.read_csv(VAC_DATA)


# Get COVID-19 Cases Data
URL = 'https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports_us'
RAW_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/{}'
print("Retrieving data from " + URL)
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
# get information regarding csv files
csv_soup = soup.find_all(href=re.compile("csv"))


# extract file names
file_list = []
for title in csv_soup:
    # csv files are mapped to 'title'
    csv_title = title.get('title')
    file_list.append(csv_title)

# get dates from the file names in the dataset folder
dates = [s.strip('.csv') for s in file_list]

# set start date of data retrieval
start_date = datetime.strptime('01-12-2021', '%m-%d-%Y')   # start date of vaccine data collection

# create list of dates that range from the start_date to the current most date in dataset folder
mod_dates = [i for i in dates if start_date <= datetime.strptime(i, '%m-%d-%Y')]

# create list of raw_URL file path and append '.csv' to the date
mod_file_list = [RAW_URL.format(i + '.csv') for i in mod_dates]

print('loading..')
# place data into a list (seems faster this way compared to directly to DF)
li=[]
i=0
for file in mod_file_list:
    df = pd.read_csv(file)
    df['date'] = mod_dates[i]
    li.append(df)
    i +=1
cases_df = pd.concat(li, axis=0, ignore_index=True)

# Merge Vaccine data and Cases data into single dataframe
cases_df.rename(columns={'Province_State':'location'}, inplace=True)
vac_df['date']= pd.to_datetime(vac_df['date']).dt.strftime('%m-%d-%Y')
df = pd.merge(cases_df, vac_df, on=['location', 'date'], how='inner')


print("Processing Data")
# Data Preprocessing
df['new_infections'] = df.groupby("location")['Confirmed'].diff(periods=NEW_INFECTIONS_PERIOD)

# fill in first 7 dates from each location with the maximum value found in a location's 'new infection'
df.loc[df['new_infections'].isnull(), 'new_infections'] = df['location'].map(df.groupby('location')['new_infections'].max())
df.set_index('date',inplace=True)
location = df.location.unique()

df.drop(['Recovered', 'Active', 'FIPS'], inplace=True, axis=1)
df.interpolate(method='linear', limit_direction='backward', inplace=True)
cor = df.corr()

#Correlation with output variable
cor_target = cor["new_infections"]

# relevant_features = cor_target[((cor_target>0.15) & (cor_target<0.95))]
relevant_features = cor_target[cor_target<-0.1]
relevant_features_list = relevant_features.index.tolist()
relevant_features_list.extend((['new_infections','location']))

df_filt = df.reindex(relevant_features_list, axis=1)

df_filt.to_csv('covid_cases_vaccine_data.csv')

print("Finished generating data set")