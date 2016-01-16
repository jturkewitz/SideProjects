# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 00:52:14 2015

@author: Jared
"""

import pandas as pd
import numpy as np
import sys
#import xgboost as xgb
import datetime as datetime
#import scipy.sparse
#import pickle
import timeit
#%%
tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'

tic=timeit.default_timer()

#%%
berlin_weather = pd.read_csv('Weather/BadenWürttemberg.csv',delimiter=';',header=0)
def get_weather(input_str,state):
    print(state)
    input_df = pd.read_csv(input_str,delimiter=';',header=0)
    input_df = input_df[['Date','Precipitationmm','Mean_TemperatureC','Mean_Wind_SpeedKm_h']]
    precip_name = 'precip_'+state
    temp_name = 'temperature_'+state
    wind_name = 'wind_'+state
    input_df = input_df.rename(columns={'Precipitationmm':precip_name,
                                        'Mean_TemperatureC':temp_name,
                                        'Mean_Wind_SpeedKm_h':wind_name})
    return input_df


#weather_df = get_weather('Weather/BadenWürttemberg.csv','BW')
weather_df = get_weather('Weather/SachsenAnhalt.csv','ST')
#%%
state_list = ['BadenWürttemberg','Bremen','Niedersachsen','Sachsen','Bayern',
              'Hamburg','NordrheinWestfalen','SachsenAnhalt','Berlin','Hessen',
              'RheinlandPfalz','SchleswigHolstein','Brandenburg','MecklenburgVorpommern',
              'Saarland','Thüringen']
#state_keys = ['HE', 'TH', 'NW', 'BE', 'SN', 'SH', 'HB,NI', 'BY', 'BW', 'RP', 'ST','HH']
abbrev_list = ['BW','HB','NI','SN','BY','HH','NW','ST','BE','HE','RP','SH','BB','MV','SL','TH']
abbrev_dict = dict(zip(state_list,abbrev_list))
df_dict = {}
for state in state_list:
    weather_input_file = 'Weather/'+state+'.csv'
    abbrev = abbrev_dict[state]
    df_dict[state] = get_weather(weather_input_file,state=abbrev)
#%%
i = 0
for key in df_dict:
    i=i+1
    if(i == 1):
        result_df = df_dict[key]
        continue
    result_df = pd.merge(df_dict[key],result_df,on='Date',how='left')
#result_df.to_csv('weather.csv',index=False)
toc=timeit.default_timer()
print('Total Time',toc - tic0)
#%%