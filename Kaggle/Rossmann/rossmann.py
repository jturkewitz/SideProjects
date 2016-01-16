# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:46:15 2015

@author: Jared
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from sklearn import cross_validation
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
#from sklearn import preprocessing
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV,Lasso,ElasticNetCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer
#import xgboost as xgb
import datetime as datetime
import xgboost as xgb
import operator

import scipy.stats as sp_stats
#import scipy.sparse
#import pickle
import timeit
#import sklearn_mod.linear_model as sk_mod_lin
#from sklearn_mod.linear_model import LassoCV as LassoCV_mod

tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'

tic=timeit.default_timer()
#%%
store_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Rossmann/store.csv', header=0)
train_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Rossmann/train.csv', header=0,low_memory=False)
test_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Rossmann/test.csv', header=0,low_memory=False)


test_df.fillna(0,inplace=True)
train_df['Id'] = train_df.index
missing_stores = list(set(train_df.Store.unique()) - set(test_df.Store.unique()))
missing_stores.sort()
#%%
holidays = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Rossmann/Holidays.csv', header=0)
holidays['Date'] = pd.to_datetime(holidays['Date'])
beginning = pd.to_datetime('2013-01-01 00:00:00')
holidays['date_int'] = holidays['Date'].map(lambda x: (x - beginning).days)
holiday_set = set(holidays['date_int'].unique())
#%%
store_states = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Rossmann/store_states.csv', header=0)
labels, levels = pd.factorize(store_states.State)
store_states['state_id'] = labels
store_states_dict = dict(zip(store_states['State'],store_states['state_id']))
#%%
google_trends = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Rossmann/daily_google_trends.csv', header=0)
google_trends['Date'] = pd.to_datetime(google_trends.Day)
google_trends.drop(['Unnamed: 0','Day'],axis=1,inplace=True)
google_trends = google_trends.rename(columns={'daily_rossmann_DE': 'g_trends'})
#%%
weather = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Rossmann/weather.csv', header=0)
weather['Date'] = pd.to_datetime(weather.Date)
#%%
train_df = pd.merge(train_df,store_states[['Store','state_id']],on = ['Store'],how='left')
test_df = pd.merge(test_df,store_states[['Store','state_id']],on = ['Store'],how='left')
#%%
#%%
#use_cv = True
use_cv = False
#%%
store_df = pd.merge(store_df,store_states[['Store','state_id']],on = ['Store'],how='left')
#%%
store_df['Promo2SinceWeek'].fillna(1,inplace=True)
store_df['PromoInterval'].fillna('none',inplace=True)

def convert_strings_to_ints(input_df,col_name,output_col_name):
    labels, levels = pd.factorize(input_df[col_name])
    input_df[output_col_name] = labels
    output_dict = dict(zip(input_df[col_name],input_df[output_col_name]))
    return (output_dict,input_df)
(promo2_interval_dict,store_df) = convert_strings_to_ints(store_df,'PromoInterval','promo2_interval_hash')
store_df['Promo2SinceYear'].fillna(2050,inplace=True)
store_df['Promo2SinceYear'] = store_df['Promo2SinceYear'].astype(int)

store_df['CompetitionOpenSinceYear'].fillna(2050,inplace=True)
store_df['CompetitionOpenSinceYear'] = store_df['CompetitionOpenSinceYear'].astype(int)
store_df['CompetitionOpenSinceMonth'].fillna(1,inplace=True)
store_df['CompetitionOpenSinceMonth'] = store_df['CompetitionOpenSinceMonth'].astype(int)

store_df['CompetitionDistance'].fillna(-1,inplace=True)
bins = [-1.0,0,50,100,150,200,240,300.400,600,1000,2000,4000,8000,16000,100000000]
store_df['comp_distance_binned'] = np.digitize(store_df['CompetitionDistance'], bins, right=True)

beginning = pd.to_datetime('2013-01-01 00:00:00')

def get_promo2_start(row):
    year = row['Promo2SinceYear']
    week = row['Promo2SinceWeek']
    day = (week - 1) * 7
    return pd.to_datetime(datetime.datetime(year, 1, 1) + datetime.timedelta(day))
store_df['p2_start'] = store_df.apply(lambda row: get_promo2_start(row),axis=1)
store_df['p2_start'] = store_df['p2_start'].map(lambda x: (x - beginning).days)
#train_df = pd.merge(train_df,store_df[['Store','p2_start']], on = ['Store'],how='left')
#test_df = pd.merge(test_df,store_df[['Store','p2_start']], on = ['Store'],how='left')

def get_comp_start(row):
    year = row['CompetitionOpenSinceYear']
    month = row['CompetitionOpenSinceMonth']
    date_string = str(year)+'-'+str(month)+'-01'
    return pd.to_datetime(date_string)
store_df['comp_start'] = store_df.apply(lambda row: get_comp_start(row),axis=1)
store_df['comp_start'] = store_df['comp_start'].map(lambda x: (x - beginning).days)

def get_assortment(x):
    if x == 'a':
        return 0
    elif x == 'b':
        return 1
    else:
        return 2
def get_store_type(x):
    if x == 'a':
        return 0
    elif x == 'b':
        return 1
    elif x == 'c':
        return 2
    else:
        return 3
store_df['assortment'] = store_df['Assortment'].map(lambda x: get_assortment(x))
store_df['store_type'] = store_df['StoreType'].map(lambda x: get_store_type(x))

col_list = ['Store','comp_start','p2_start','assortment','store_type','comp_distance_binned',
            'promo2_interval_hash','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
#%%
train_df = pd.merge(train_df,store_df[col_list],
                    on = ['Store'],how='left')
test_df = pd.merge(test_df,store_df[col_list],
                    on = ['Store'],how='left')
#%%
tic=timeit.default_timer()
sun_thur_set = set([7,1,2,3,4,])
tue_sat_set = set([2,3,4,5,6])
wed_sat_set = set([3,4,5,6])
sun_fri_set = set([7,1,2,3,4,5])

summer_set = set([6,7,8,9])

def get_quarter(x):
    if x in [1,2,3]:
        return 1
    elif x in [4,5,6]:
        return 2
    elif x in [7,8,9]:
        return 3
    elif x in [10,11,12]:
        return 4
    else:
        return 0
def get_state_holiday(x):
    if x == '0':
        return 0
    elif x == 'a':
        return 1
    elif x == 'b':
        return 2
    elif x == 'c':
        return 3

month_end_set = set([28,29,30,31])
month_start_set = set([1,2,3,4])

month_p2_1_set = set([1,4,7,10])
month_p2_2_set = set([2,5,8,11])
month_p2_3_set = set([3,6,9,12])
def generate_features(input_df):
    input_df['Date'] = pd.to_datetime(input_df['Date'])
    #convoluted, but a bit faster way to get date features
    dts = input_df['Date'].drop_duplicates()
    week = dts.map(lambda x: x.week)
    year = dts.map(lambda x: x.year)
    month = dts.map(lambda x: x.month)
    day = dts.map(lambda x: x.day)
    day_of_year = dts.map(lambda x: x.dayofyear)
    is_month_end = dts.map(lambda x: x.is_month_end.astype(int))

    beginning = pd.to_datetime('2013-01-01 00:00:00')
    date_int = dts.map(lambda x: (x - beginning).days)

    potential_hol = date_int.map(lambda x: 1 if x in holiday_set else 0)

    w_dict = dict(zip(dts, week))
    y_dict = dict(zip(dts, year))
    m_dict = dict(zip(dts, month))
    d_dict = dict(zip(dts, day))
    d_o_y_dict = dict(zip(dts, day_of_year))
    is_month_end_dict = dict(zip(dts, is_month_end))
    d_int_dict = dict(zip(dts, date_int))
    hol_pot_dict = dict(zip(dts, potential_hol))

    input_df['week'] = input_df['Date'].map(w_dict)
    input_df['year'] = input_df['Date'].map(y_dict)
    input_df['month'] = input_df['Date'].map(m_dict)
    input_df['day'] = input_df['Date'].map(d_dict)
    input_df['day_of_year'] = input_df['Date'].map(d_o_y_dict)
    input_df['is_month_end'] = input_df['Date'].map(is_month_end_dict)
    input_df['date_int'] = input_df['Date'].map(d_int_dict)
    input_df['Potential_hol'] = input_df['Date'].map(hol_pot_dict)

    input_df['p2_active'] = input_df['p2_start'] - input_df['date_int']
    input_df['p2_active'] = input_df['p2_active'].map(lambda x: 2 if x <= 0 else 0)
    no_promo2_change_cond = input_df['p2_start'] <= 0
    input_df['p2_active'][no_promo2_change_cond] = 1

    input_df['is_promo2_start_month'] = 0
#    p2_0_cond = input_df['promo2_interval_hash'] == 0
    p2_1_cond = input_df['promo2_interval_hash'] == 1
    p2_2_cond = input_df['promo2_interval_hash'] == 2
    p2_3_cond = input_df['promo2_interval_hash'] == 3

    p2_cond = input_df['p2_active'] >= 1
    input_df['promo2_1_month'] = input_df['month'].map(lambda x: x in month_p2_1_set)
    input_df['promo2_2_month'] = input_df['month'].map(lambda x: x in month_p2_2_set)
    input_df['promo2_3_month'] = input_df['month'].map(lambda x: x in month_p2_3_set)
    month1_cond = input_df['promo2_1_month'] == 1
    month2_cond = input_df['promo2_2_month'] == 1
    month3_cond = input_df['promo2_3_month'] == 1
    input_df['is_promo2_start_month'][(month1_cond & p2_1_cond & p2_cond)] = 1
    input_df['is_promo2_start_month'][(month2_cond & p2_2_cond & p2_cond)] = 1
    input_df['is_promo2_start_month'][(month3_cond & p2_3_cond & p2_cond)] = 1


    year_2013 = input_df['year'] == 2013
    year_2014 = input_df['year'] == 2014
    year_2015 = input_df['year'] == 2015
    input_df['days_from_easter'] = input_df['day_of_year']
    input_df['days_from_easter'][year_2013] = input_df['days_from_easter'][year_2013] - 90
    input_df['days_from_easter'][year_2014] = input_df['days_from_easter'][year_2014] - 110
    input_df['days_from_easter'][year_2015] = input_df['days_from_easter'][year_2015] - 95

#    input_df['57_days_from_easter'] = input_df['days_from_easter'].map(lambda x: 1 if x == 57 else 0)

    input_df['days_from_first_monday'] = input_df['day_of_year']
    input_df['days_from_first_monday'][year_2013] = input_df['days_from_first_monday'][year_2013] - 7
    input_df['days_from_first_monday'][year_2014] = input_df['days_from_first_monday'][year_2014] - 6
#    input_df['days_from_first_monday'][year_2014] = input_df['days_from_first_monday'][year_2014] - 13
    input_df['days_from_first_monday'][year_2015] = input_df['days_from_first_monday'][year_2015] - 5

    input_df['days_from_first_monday_adj'] = input_df['day_of_year']
    input_df['days_from_first_monday_adj'][year_2013] = input_df['days_from_first_monday_adj'][year_2013] - 7
    input_df['days_from_first_monday_adj'][year_2014] = input_df['days_from_first_monday_adj'][year_2014] - 13
    input_df['days_from_first_monday_adj'][year_2015] = input_df['days_from_first_monday_adj'][year_2015] - 12

    input_df['summer'] = input_df['month'].map(lambda x: 1 if x in summer_set else 0)

    bins = [-1.0,200,400,600,800,1000]
    input_df['date_int_binned'] = np.digitize(input_df['date_int'], bins, right=True)


    input_df['is_weekend'] = input_df['DayOfWeek'].map(lambda x: 1 if x > 5 else 0)
    input_df['is_monday'] = input_df['DayOfWeek'].map(lambda x: 1 if x == 1 else 0)
    input_df['is_tuesday'] = input_df['DayOfWeek'].map(lambda x: 1 if x == 2 else 0)
    input_df['is_wednesday'] = input_df['DayOfWeek'].map(lambda x: 1 if x == 3 else 0)
    input_df['is_thursday'] = input_df['DayOfWeek'].map(lambda x: 1 if x == 4 else 0)
    input_df['is_friday'] = input_df['DayOfWeek'].map(lambda x: 1 if x == 5 else 0)
    input_df['is_saturday'] = input_df['DayOfWeek'].map(lambda x: 1 if x == 6 else 0)
    input_df['is_sunday'] = input_df['DayOfWeek'].map(lambda x: 1 if x == 7 else 0)
    input_df['weekday_no_promo'] = (~input_df['Promo'].astype(bool) &
                                    ~input_df['is_weekend'].astype(bool)).astype(int)
    input_df['is_monday_and_promo'] = (input_df['Promo'].astype(bool) &
                                    input_df['is_monday'].astype(bool)).astype(int)
    input_df['is_tuesday_and_promo'] = (input_df['Promo'].astype(bool) &
                                    input_df['is_tuesday'].astype(bool)).astype(int)
    input_df['is_wednesday_and_promo'] = (input_df['Promo'].astype(bool) &
                                    input_df['is_wednesday'].astype(bool)).astype(int)
    input_df['is_thursday_and_promo'] = (input_df['Promo'].astype(bool) &
                                    input_df['is_thursday'].astype(bool)).astype(int)
    input_df['is_friday_and_promo'] = (input_df['Promo'].astype(bool) &
                                    input_df['is_friday'].astype(bool)).astype(int)
    input_df['quarter'] = input_df['month'].map(lambda x: get_quarter(x))

    input_df['month_start'] = input_df['day'].map(lambda x: x in month_start_set).astype(int)
    input_df['month_end'] = input_df['day'].map(lambda x: x in month_end_set).astype(int)

    input_df['is_monday_and_month_end'] = (input_df['month_end'].astype(bool) &
                                    input_df['is_monday'].astype(bool)).astype(int)
    input_df['is_tuesday_and_month_end'] = (input_df['month_end'].astype(bool) &
                                    input_df['is_tuesday'].astype(bool)).astype(int)
    input_df['is_wednesday_and_month_end'] = (input_df['month_end'].astype(bool) &
                                    input_df['is_wednesday'].astype(bool)).astype(int)
    input_df['is_thursday_and_month_end'] = (input_df['month_end'].astype(bool) &
                                    input_df['is_thursday'].astype(bool)).astype(int)
    input_df['is_friday_and_month_end'] = (input_df['month_end'].astype(bool) &
                                    input_df['is_friday'].astype(bool)).astype(int)

    input_df['is_saturday_and_month_end'] = (input_df['month_end'].astype(bool) &
                                input_df['is_saturday'].astype(bool)).astype(int)
    input_df['is_sunday_and_month_end'] = (input_df['month_end'].astype(bool) &
                                input_df['is_sunday'].astype(bool)).astype(int)

    input_df['is_monday_and_month_start'] = (input_df['month_start'].astype(bool) &
                                    input_df['is_monday'].astype(bool)).astype(int)
    input_df['is_tuesday_and_month_start'] = (input_df['month_start'].astype(bool) &
                                    input_df['is_tuesday'].astype(bool)).astype(int)
    input_df['is_wednesday_and_month_start'] = (input_df['month_start'].astype(bool) &
                                    input_df['is_wednesday'].astype(bool)).astype(int)
    input_df['is_thursday_and_month_start'] = (input_df['month_start'].astype(bool) &
                                    input_df['is_thursday'].astype(bool)).astype(int)
    input_df['is_friday_and_month_start'] = (input_df['month_start'].astype(bool) &
                                    input_df['is_friday'].astype(bool)).astype(int)
    input_df['is_saturday_and_month_start'] = (input_df['month_start'].astype(bool) &
                                    input_df['is_saturday'].astype(bool)).astype(int)
    input_df['is_sunday_and_month_start'] = (input_df['month_start'].astype(bool) &
                                    input_df['is_sunday'].astype(bool)).astype(int)

    input_df['sunday_month'] = input_df['month']
    sunday_cond = input_df['is_sunday'] == 0
    input_df['sunday_month'][sunday_cond] = 0
#    input_df['c_eve'] = input_df['day_of_year'].map(lambda x: 1 if x == 358 else 0)
#    input_df['dec_31'] = input_df['day_of_year'].map(lambda x: 1 if x == 365 else 0)

    pot_hol = input_df['Potential_hol'] == 1
    is_not_open = input_df['Open'] == 0
    is_not_hol = input_df['StateHoliday'] == '0'
    is_not_sunday = input_df['DayOfWeek'] != 7
    cond = (pot_hol & is_not_open & is_not_hol & is_not_sunday)
    input_df['all_a'] = 'a'
    input_df['StateHoliday'][cond] = input_df['all_a'][cond]

    input_df['st_hol'] = input_df['StateHoliday'].map(lambda x: get_state_holiday(x))

    input_df['is_st_hol'] = input_df['st_hol'].map(lambda x: 1 if x >= 1 else 0)
    input_df['sun_thur'] = input_df['DayOfWeek'].map(lambda x: 1 if x in sun_thur_set else 0)
    input_df['sun_fri'] = input_df['DayOfWeek'].map(lambda x: 1 if x in sun_fri_set else 0)
    input_df['tue_sat'] = input_df['DayOfWeek'].map(lambda x: 1 if x in tue_sat_set else 0)
    input_df['wed_sat'] = input_df['DayOfWeek'].map(lambda x: 1 if x in wed_sat_set else 0)




    input_df['comp_time_to_start'] = input_df['comp_start'] - input_df['date_int']
    input_df['comp_active'] = input_df['comp_time_to_start'].map(lambda x: 2 if x <= 0 else 0)
    no_comp_change_cond = input_df['comp_start'] <= 0
    input_df['comp_active'][no_comp_change_cond] = 1

    input_df['is_comp_starting_month'] = ((input_df['comp_time_to_start'] <= 0) &
                                            (input_df['comp_time_to_start'] >= -31)).astype(int)
    input_df['is_comp_after_14_days'] = (input_df['comp_time_to_start'] <= -14).astype(int)

    input_df['Open_Or_Holiday'] = input_df['Open'] + input_df['is_st_hol']
    input_df['Open_Or_Holiday'] = input_df['Open_Or_Holiday'].map(lambda x: 1 if x > 0 else 0)
    input_df.drop(['all_a','p2_start','comp_start','StateHoliday'],axis=1,inplace=True)
    return input_df

train_df = generate_features(train_df)
test_df = generate_features(test_df)

toc=timeit.default_timer()
print('Date Time',toc - tic)
#%%
tic=timeit.default_timer()
train_df = pd.merge(train_df,google_trends,on = ['Date'],how='left')
test_df = pd.merge(test_df,google_trends,on = ['Date'],how='left')

state_keys = ['HE', 'TH', 'NW', 'BE', 'SN', 'SH', 'HB,NI', 'BY', 'BW', 'RP', 'ST','HH']
state_columns = ['daily_rossmann_DE-HE','daily_rossmann_DE-TH', 'daily_rossmann_DE-NW',
                 'daily_rossmann_DE-BE', 'daily_rossmann_DE-SN','daily_rossmann_DE-SH',
                 'daily_rossmann_DE-NI','daily_rossmann_DE-BY','daily_rossmann_DE-BW',
                 'daily_rossmann_DE-RP','daily_rossmann_DE-ST','daily_rossmann_DE-HH']
state_columns_dict = dict(zip(state_keys,state_columns))
def get_state_gt(input_df):
    input_df['gt_st'] = 0
    for key in store_states_dict:
        state_cond = input_df['state_id'] == store_states_dict[key]
        state_col = state_columns_dict[key]
        input_df['gt_st'][state_cond] = input_df[state_col][state_cond]
    input_df.drop(state_columns,axis=1,inplace=True)
    return input_df
train_df = get_state_gt(train_df)
test_df = get_state_gt(test_df)
toc=timeit.default_timer()
print('Merging gt Time',toc - tic)
#%%
tic=timeit.default_timer()
train_df = pd.merge(train_df,weather,on = ['Date'],how='left')
test_df = pd.merge(test_df,weather,on = ['Date'],how='left')

weather_columns = weather.columns.values
weather_columns = np.delete(weather_columns, [0])

state_keys = ['HE', 'TH', 'NW', 'BE', 'SN', 'SH', 'HB,NI', 'BY', 'BW', 'RP', 'ST','HH']
state_precip = ['precip_HE','precip_TH','precip_NW','precip_BE','precip_SN','precip_SH',
                'precip_NI','precip_BY','precip_BW','precip_RP','precip_ST','precip_HH']
state_temp = ['temperature_HE','temperature_TH','temperature_NW',
              'temperature_BE','temperature_SN','temperature_SH',
              'temperature_NI','temperature_BY','temperature_BW',
              'temperature_RP','temperature_ST','temperature_HH']
state_wind = ['wind_HE','wind_TH','wind_NW','wind_BE','wind_SN','wind_SH',
                'wind_NI','wind_BY','wind_BW','wind_RP','wind_ST','wind_HH']
state_precip_dict = dict(zip(state_keys,state_precip))
state_temp_dict = dict(zip(state_keys,state_temp))
state_wind_dict = dict(zip(state_keys,state_wind))
def get_state_weather(input_df):
    input_df['precip'] = 0
    input_df['temperature'] = 0
    input_df['wind'] = 0
    for key in store_states_dict:
        state_cond = input_df['state_id'] == store_states_dict[key]
        state_precip_col = state_precip_dict[key]
        state_temp_col = state_temp_dict[key]
        state_wind_col = state_wind_dict[key]
        input_df['precip'][state_cond] = input_df[state_precip_col][state_cond]
        input_df['temperature'][state_cond] = input_df[state_temp_col][state_cond]
        input_df['wind'][state_cond] = input_df[state_wind_col][state_cond]
    input_df.drop(weather_columns,axis=1,inplace=True)
    return input_df
train_df = get_state_weather(train_df)
test_df = get_state_weather(test_df)
toc=timeit.default_timer()
print('Merging weather Time',toc - tic)
#%%
def get_weather_levels(input_df):
    input_df['high_rain'] = input_df['precip'].map(lambda x: 1 if x >= 5 else 0)
    input_df['low_rain'] = input_df['precip'].map(lambda x: 1 if (x >= 1 and x < 5) else 0)
    input_df['high_wind'] = input_df['wind'].map(lambda x: 1 if x >= 20 else 0)
    input_df['low_wind'] = input_df['wind'].map(lambda x: 1 if (x >= 10 and x < 20) else 0)
    input_df['is_cold'] = input_df['temperature'].map(lambda x: 1 if x <= 0 else 0)
    input_df['is_hot'] = input_df['temperature'].map(lambda x: 1 if x >= 23 else 0)
    return input_df
train_df = get_weather_levels(train_df)
test_df = get_weather_levels(test_df)
#%%
bins = [0,30,40,50,60,70,100]
train_df['g_trends_binned'] = np.digitize(train_df['g_trends'], bins, right = True)
test_df['g_trends_binned'] = np.digitize(test_df['g_trends'], bins, right = True)

train_df['gt_st_binned'] = np.digitize(train_df['gt_st'], bins, right = True)
test_df['gt_st_binned'] = np.digitize(test_df['gt_st'], bins, right = True)
#%%

#%%
tic=timeit.default_timer()

def generate_shifted_features(input_df):
#note that index is in reverse order of date
    STORES_LIST = []
    for name,group in input_df.groupby('Store'):
        store = group
        store = store.sort_values('date_int')

        store['is_hol_prev'] = store['is_st_hol'].shift(1)
        store['is_hol_next'] = store['is_st_hol'].shift(-1)

        store['shifted_down_promo'] = store['Promo'].shift(1)
        store['shifted_up_promo'] = store['Promo'].shift(-1)

        store['shifted_down2_promo'] = store['Promo'].shift(2)
        store['shifted_up2_promo'] = store['Promo'].shift(-2)

        store['shifted_down3_promo'] = store['Promo'].shift(3)
        store['shifted_up3_promo'] = store['Promo'].shift(-3)

        store['is_promo_past_week'] = store['Promo'].shift(7)
        store['is_promo_next_week'] = store['Promo'].shift(-7)

        store['is_school_hol_prev_day'] = store['SchoolHoliday'].shift(1)
        store['is_school_hol_next_day'] = store['SchoolHoliday'].shift(-1)

        store['is_school_hol_prev_week'] = store['SchoolHoliday'].shift(7)
        store['is_school_hol_next_week'] = store['SchoolHoliday'].shift(-7)

        store.fillna(0,inplace=True) #default promo/holiday to 0
        sc_hol_cond = store['SchoolHoliday'] == 1
        sc_hol_prev_cond = store['is_school_hol_prev_day'] == 1
        sc_hol_next_cond = store['is_school_hol_next_day'] == 1
        store['isolated_school_hol'] = 0
        iso_cond = (sc_hol_cond & ~sc_hol_prev_cond & ~sc_hol_next_cond)
        store['isolated_school_hol'][iso_cond] = 1

        store['shifted_down_open'] = store['Open_Or_Holiday'].shift(1)
        store['shifted_up_open'] = store['Open_Or_Holiday'].shift(-1)

        store['shifted_down2_open'] = store['Open_Or_Holiday'].shift(2)
        store['shifted_up2_open'] = store['Open_Or_Holiday'].shift(-2)

        store['shifted_down3_open'] = store['Open_Or_Holiday'].shift(3)
        store['shifted_up3_open'] = store['Open_Or_Holiday'].shift(-3)

        store['is_open_1_week_ago'] = store['Open_Or_Holiday'].shift(7)
        store['is_open_in_1_week'] = store['Open_Or_Holiday'].shift(-7)

        store['is_open_2_weeks_ago'] = store['Open_Or_Holiday'].shift(14)
        store['is_open_in_2_weeks'] = store['Open_Or_Holiday'].shift(-14)

        store.fillna(1,inplace=True) #open usually 1

        store['is_promo_next_weekday'] = 0
        store['is_promo_prev_weekday'] = 0

        store['is_open_next_workday'] = 1
        store['is_open_prev_workday'] = 1

        store['is_open_next2_workday'] = 1
        store['is_open_prev2_workday'] = 1

        cond_sun_thur = store['sun_thur'] == 1
        cond_wed_sat = store['wed_sat'] == 1
        cond_tue_sat = store['tue_sat'] == 1
        cond_sun_fri = store['sun_fri'] == 1
        cond_tue = store['DayOfWeek'] == 2
        cond_fri = store['DayOfWeek'] == 5
        cond_sat = store['DayOfWeek'] == 6
        cond_mon = store['DayOfWeek'] == 1
        cond_sun = store['DayOfWeek'] == 7

        cond_not_open = store['Open'] == 0

        summer_cond = store['summer'] == 1
        store['sc_hol_mod'] = store['SchoolHoliday']
        store['sc_hol_mod'][cond_sun & sc_hol_next_cond] = 1
        store['sc_hol_mod'][cond_sat & sc_hol_prev_cond] = 1
        store['sc_hol_mod'][~summer_cond] = 0


        store['is_promo_next_weekday'][cond_sun_thur] = store['shifted_up_promo'][cond_sun_thur]
        store['is_promo_next_weekday'][cond_sat] = store['shifted_up2_promo'][cond_sat]
        store['is_promo_next_weekday'][cond_fri] = store['shifted_up3_promo'][cond_fri]

        store['is_promo_prev_weekday'][cond_tue_sat] = store['shifted_down_promo'][cond_tue_sat]
        store['is_promo_prev_weekday'][cond_sun] = store['shifted_down2_promo'][cond_sun]
        store['is_promo_prev_weekday'][cond_mon] = store['shifted_down3_promo'][cond_mon]

#        store['is_open_next_workday'][cond_sun_fri] = store['shifted_up_open'][cond_sun_fri]
#        store['is_open_next_workday'][cond_sat] = store['shifted_up2_open'][cond_sat]
#
#        store['is_open_prev_workday'][cond_tue_sat] = store['shifted_down_open'][cond_tue_sat]
#        store['is_open_prev_workday'][cond_sun] = store['shifted_down_open'][cond_sun]
#        store['is_open_prev_workday'][cond_mon] = store['shifted_down2_open'][cond_mon]
#
#        store['is_open_next2_workday'][cond_sun_thur] = store['shifted_up2_open'][cond_sun_thur]
#        store['is_open_next2_workday'][cond_sat] = store['shifted_up3_open'][cond_sat]
#        store['is_open_next2_workday'][cond_fri] = store['shifted_up3_open'][cond_fri]
#
#        store['is_open_prev2_workday'][cond_wed_sat] = store['shifted_down2_open'][cond_wed_sat]
#        store['is_open_prev2_workday'][cond_sun] = store['shifted_down2_open'][cond_sun]
#        store['is_open_prev2_workday'][cond_mon] = store['shifted_down3_open'][cond_mon]
#        store['is_open_prev2_workday'][cond_tue] = store['shifted_down3_open'][cond_tue]

        store['is_open_next_workday'] = store['shifted_up_open']
        store['is_open_next2_workday'] = store['shifted_up2_open']
        store['is_open_next3_workday'] = store['shifted_up3_open']
        store['is_open_prev_workday'] = store['shifted_down_open']
        store['is_open_prev2_workday'] = store['shifted_down2_open']
        store['is_open_prev3_workday'] = store['shifted_down3_open']


        store['is_closed_next_consec_workdays'] = (~store['is_open_next_workday'].astype(bool) &
                                ~store['is_open_next2_workday'].astype(bool) &
                                ~store['is_open_next3_workday'].astype(bool)).astype(int)
        store['is_closed_prev_consec_workdays'] = (~store['is_open_prev_workday'].astype(bool) &
                                ~store['is_open_prev2_workday'].astype(bool) &
                                ~store['is_open_prev3_workday'].astype(bool)).astype(int)
#        store['is_closed_prev_consec_workdays'] = (~store['is_open_prev_workday'].astype(bool) &
#                                ~store['is_open_prev2_workday'].astype(bool)).astype(int)

        store['is_closed_next_consec_workdays'][cond_not_open] = 0
        store['is_closed_prev_consec_workdays'][cond_not_open] = 0

        store['is_closed_shift1'] = store['is_closed_next_consec_workdays'].shift(-1)
        store['is_closed_shift2'] = store['is_closed_next_consec_workdays'].shift(-2)
        store['is_closed_shift3'] = store['is_closed_next_consec_workdays'].shift(-3)
        store['is_closed_shift4'] = store['is_closed_next_consec_workdays'].shift(-4)
        store['is_closed_shift5'] = store['is_closed_next_consec_workdays'].shift(-5)
        store['is_closed_shift6'] = store['is_closed_next_consec_workdays'].shift(-6)
        store['is_closed_shift7'] = store['is_closed_next_consec_workdays'].shift(-7)
        store['is_closed_shift8'] = store['is_closed_next_consec_workdays'].shift(-8)
        store['is_closed_shift9'] = store['is_closed_next_consec_workdays'].shift(-9)
        store['is_closed_shift10'] = store['is_closed_next_consec_workdays'].shift(-10)
        store['is_closed_shift11'] = store['is_closed_next_consec_workdays'].shift(-11)
        store['is_closed_shift12'] = store['is_closed_next_consec_workdays'].shift(-12)
        store['is_closed_shift13'] = store['is_closed_next_consec_workdays'].shift(-13)
        store['is_closed_shift14'] = store['is_closed_next_consec_workdays'].shift(-14)

        store['was_closed_prev_shift1'] = store['is_closed_prev_consec_workdays'].shift(1)
        store['was_closed_prev_shift2'] = store['is_closed_prev_consec_workdays'].shift(2)
        store['was_closed_prev_shift3'] = store['is_closed_prev_consec_workdays'].shift(3)
        store['was_closed_prev_shift4'] = store['is_closed_prev_consec_workdays'].shift(4)
        store['was_closed_prev_shift5'] = store['is_closed_prev_consec_workdays'].shift(5)
        store.fillna(0,inplace=True)

        store.drop(['sun_thur','wed_sat','tue_sat','sun_fri'],axis=1,inplace=True)
        store.drop(['shifted_up_promo','shifted_up2_promo','shifted_up3_promo'],axis=1,inplace=True)
        store.drop(['shifted_down_promo','shifted_down2_promo','shifted_down3_promo'],axis=1,inplace=True)
        store.drop(['shifted_up_open','shifted_up2_open','shifted_up3_open'],axis=1,inplace=True)
        store.drop(['shifted_down_open','shifted_down2_open','shifted_down3_open'],axis=1,inplace=True)
        STORES_LIST.append(store)
    return pd.concat(STORES_LIST)

test_df['Sales'] = 'dummy_sales'
train_df = train_df.drop(['Customers'],axis=1)
test_df['Open'] = test_df['Open'].fillna(0)
combined_df = pd.concat([train_df, test_df], axis=0)
combined_df = generate_shifted_features(combined_df)
train_shifted_df = combined_df.loc[combined_df.date_int <= 941]
test_shifted_df = combined_df.loc[combined_df.date_int > 941]

toc=timeit.default_timer()
print('Shifted Dates Time',toc - tic)
#%%
train_shifted_df['Sales'] = train_shifted_df['Sales'].astype(float)
train_df = train_shifted_df
test_df = test_shifted_df
#%%
def get_sc_hol_features(input_df):
    input_df['is_saturday_and_sc_mod'] = (input_df['sc_hol_mod'].astype(bool) &
                                    input_df['is_saturday'].astype(bool)).astype(int)
    input_df['is_sunday_and_sc_hol'] = (input_df['SchoolHoliday'].astype(bool) &
                                    input_df['is_sunday'].astype(bool)).astype(int)
    return input_df
train_df = get_sc_hol_features(train_df)
test_df = get_sc_hol_features(test_df)
#%%
tic=timeit.default_timer()
combined = train_df[['Store','date_int','year','Promo','DayOfWeek']].append(
                test_df[['Store','date_int','year','Promo','DayOfWeek']])
store_1_combined = combined.groupby('Store').get_group(1)
store_1_combined.sort_values(by = 'date_int',inplace=True)
promo_number1 = 0
promo_number2 = 0
promo_number3 = 0
promo_number_dict = {}
for index,row in store_1_combined.iterrows():
    is_promo = row.Promo
    year = row.year

    date_int = row.date_int
    is_monday = (row.DayOfWeek == 1)
    if(is_monday & is_promo & (year == 2013)):
        promo_number1 += 1
    elif(is_monday & is_promo & (year == 2014)):
        promo_number2 += 1
    elif(is_monday & is_promo & (year == 2015)):
        promo_number3 += 1

    if(year == 2013):
        promo_number_dict[date_int] = promo_number1
    elif(year == 2014):
        promo_number_dict[date_int] = promo_number2
    else:
        promo_number_dict[date_int] = promo_number3

def get_promo_number(input_df):
    input_df['promo_number'] = input_df['date_int'].map(promo_number_dict)
    promo_cond = input_df['Promo'] == 1
    input_df['promo_number_and_is_promo'] = input_df['promo_number']
    input_df['promo_number_and_is_promo'][~promo_cond] = 0
    input_df['promo_number_and_is_not_promo'] = input_df['promo_number']
    input_df['promo_number_and_is_not_promo'][promo_cond] = 0

    input_df['days_from_easter_and_is_promo'] = input_df['days_from_easter']
    input_df['days_from_easter_and_is_promo'][~promo_cond] = 0
    input_df['days_from_easter_and_is_not_promo'] = input_df['days_from_easter']
    input_df['days_from_easter_and_is_not_promo'][promo_cond] = 0

    return input_df

#train_df['promo_number'] = train_df['date_int'].map(promo_number_dict)
#test_df['promo_number'] = test_df['date_int'].map(promo_number_dict)
train_df = get_promo_number(train_df)
test_df = get_promo_number(test_df)

toc=timeit.default_timer()
print('Promo Number Time',toc - tic)
#%%

#%%
zero_sales_df = train_df.loc[(train_df.Sales == 0) & (train_df.Open == 1)]
#%%
train_open_df = train_df.loc[train_df.Sales > 0]
train_open_df['Sales_Log'] = np.log(train_open_df['Sales'])




#trans_const = 2000.0
#train_open_df['Sales_Transform'] = train_open_df['Sales_Log'] - trans_const / train_open_df['Sales']

#train_open_df = train_open_df.drop(['Customers'],axis=1)
#test_df['Open'] = test_df['Open'].fillna(0)
low_sales_df = train_open_df.loc[train_open_df.Sales <= 1000]
high_sales_df = train_open_df.loc[train_open_df.Sales >= 30000]

train_open_df = train_open_df.loc[train_open_df.month != 12]
#train_open_df = train_open_df.loc[train_open_df.month >= 5] ##TODO testing
#train_open_df = train_open_df.loc[(train_open_df.month != 2) & (train_open_df.day != 11)]
train_open_df = train_open_df.loc[(train_open_df.is_st_hol == 0) |
                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
train_open_df = train_open_df.loc[(train_open_df.is_hol_prev == 0) |
                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
train_open_df = train_open_df.loc[(train_open_df.is_hol_next == 0) |
                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
train_open_df = train_open_df.loc[(train_open_df.is_hol_next == 0) |
                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
train_open_df = train_open_df.loc[(train_open_df.Potential_hol == 0) |
                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]

train_open_df = train_open_df.loc[train_open_df.day_of_year != 365]
train_open_df = train_open_df.loc[train_open_df.day_of_year != 358]

#TODO testing
#train_open_df = train_open_df.loc[train_open_df.day_of_year != 180]
#train_open_df = train_open_df.loc[train_open_df.day_of_year != 181]

#train_open_df = train_open_df.loc[(train_open_df.month != 1) | (train_open_df.year != 2013)]
#train_open_df = train_open_df.loc[(train_open_df.month != 2) | (train_open_df.day != 25)]

#train_open_df = train_open_df.loc[(train_open_df.month != 10) | (train_open_df.day != 28) | (train_open_df.year != 2013)]
#train_open_df = train_open_df.loc[(train_open_df.month != 7) | (train_open_df.day != 25) | (train_open_df.year != 2015)]

#train_open_df = train_open_df.loc[train_open_df.month != 11]

train_open_df = train_open_df.loc[~((train_open_df.precip >= 4) & (train_open_df.temperature <= 0))]
train_open_df = train_open_df.loc[train_open_df.days_from_easter != -55]
train_open_df = train_open_df.loc[train_open_df.days_from_easter != -52]
train_open_df = train_open_df.loc[train_open_df.days_from_easter != -48]
train_open_df = train_open_df.loc[train_open_df.days_from_easter != -47]
train_open_df = train_open_df.loc[train_open_df.days_from_easter != -49]
train_open_df = train_open_df.loc[train_open_df.days_from_easter != -50]
train_open_df = train_open_df.loc[train_open_df.days_from_easter != -51]

train_open_df = train_open_df.loc[train_open_df.days_from_easter != -62]

train_open_df = train_open_df.loc[train_open_df.days_from_easter != 57]

train_open_df = train_open_df.loc[abs(train_open_df.days_from_easter) > 14]
#train_open_df = train_open_df.loc[abs(train_open_df.days_from_easter) > 21]


#%%
#train_open_df = train_open_df.loc[(train_open_df['is_closed_prev_consec_workdays'] == 0) |
#                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
#train_open_df = train_open_df.loc[(train_open_df['was_closed_prev_shift1'] == 0) |
#                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
#train_open_df = train_open_df.loc[(train_open_df['was_closed_prev_shift2'] == 0) |
#                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
#train_open_df = train_open_df.loc[(train_open_df['was_closed_prev_shift3'] == 0) |
#                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
#train_open_df = train_open_df.loc[(train_open_df['was_closed_prev_shift4'] == 0) |
#                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
#train_open_df = train_open_df.loc[(train_open_df['was_closed_prev_shift5'] == 0) |
#                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
#
#train_open_df = train_open_df.loc[(train_open_df['is_closed_next_consec_workdays'] == 0) |
#                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
#train_open_df = train_open_df.loc[(train_open_df['is_closed_shift1'] == 0) |
#                                  ((train_open_df.month == 8) | (train_open_df.month == 9))]
#%%
tic=timeit.default_timer()
def generate_dummies(input_df):
    categoricals_df = input_df[['DayOfWeek','year','month','quarter',
                                'promo_number','day','date_int_binned','gt_st_binned',
                                ]]
    categoricals_df = categoricals_df.rename(columns={'DayOfWeek':'Weekday'})
    categoricals_df['Weekday'] = categoricals_df['Weekday'].map(str)
    categoricals_df['year'] = categoricals_df['year'].map(str)
    categoricals_df['month'] = categoricals_df['month'].map(str)
    categoricals_df['quarter'] = categoricals_df['quarter'].map(str)
    categoricals_df['day'] = categoricals_df['day'].map(str)
    categoricals_df['date_int_binned'] = categoricals_df['date_int_binned'].map(str)
    categoricals_df['promo_number'] = categoricals_df['promo_number'].map(str)
    categoricals_df['gt_st_binned'] = categoricals_df['gt_st_binned'].map(str)
#    categoricals_df['week'] = categoricals_df['week'].map(str)
    dummies = pd.get_dummies(categoricals_df,dummy_na = False)

    input_df = pd.concat([input_df,dummies],axis=1,join='inner')
    return input_df
test_df['Sales'] = 'dummy_sales'
test_df['Sales_Log'] = 'dummy_sales'

if (use_cv):
#    train = train_open_df.loc[train_open_df.date_int <= 850]
#    test = train_open_df.loc[train_open_df.date_int > 850]
#    combined_df = pd.concat([train, cv_df], axis=0)
    combined_df = generate_dummies(train_open_df)
    train = combined_df.loc[combined_df.date_int <= 850]
    test = combined_df.loc[combined_df.date_int > 850]
#    train = combined_df.loc[(combined_df.day_of_year < 213) | (combined_df.day_of_year > 260) |
#                            (combined_df.year != 2014)]
#    test = combined_df.loc[(combined_df.day_of_year >= 213) & (combined_df.day_of_year <= 260) &
#                            (combined_df.year == 2014)]
else:
    combined_df = pd.concat([train_open_df, test_df], axis=0)
    combined_df = generate_dummies(combined_df)
    train = combined_df.loc[combined_df['Sales'] != 'dummy_sales']
    test = combined_df.loc[combined_df['Sales'] == 'dummy_sales']

train['Sales_Log'] = train['Sales_Log'].astype(float)
train['Sales'] = train['Sales'].astype(float)
#train['Sales_Transform'] = train['Sales_Transform'].astype(float)
toc=timeit.default_timer()
print('Dummies Time',toc - tic)
#%%
train = train.loc[(train['is_closed_prev_consec_workdays'] == 0)]
train = train.loc[(train['was_closed_prev_shift1'] == 0)]
train = train.loc[(train['was_closed_prev_shift2'] == 0)]
train = train.loc[(train['was_closed_prev_shift3'] == 0)]
train = train.loc[(train['was_closed_prev_shift4'] == 0)]
train = train.loc[(train['was_closed_prev_shift5'] == 0)]

train = train.loc[(train['is_closed_next_consec_workdays'] == 0)]
train = train.loc[(train['is_closed_shift1'] == 0)]
train = train.loc[(train['is_closed_shift2'] == 0)]
train = train.loc[(train['is_closed_shift3'] == 0)]
train = train.loc[(train['is_closed_shift4'] == 0)]
train = train.loc[(train['is_closed_shift5'] == 0)]
train = train.loc[(train['is_closed_shift6'] == 0)]
train = train.loc[(train['is_closed_shift7'] == 0)]
train = train.loc[(train['is_closed_shift8'] == 0)]
train = train.loc[(train['is_closed_shift9'] == 0)]
train = train.loc[(train['is_closed_shift10'] == 0)]
train = train.loc[(train['is_closed_shift11'] == 0)]
train = train.loc[(train['is_closed_shift12'] == 0)]
train = train.loc[(train['is_closed_shift13'] == 0)]
train = train.loc[(train['is_closed_shift14'] == 0)]
#train = train.loc[train_open_df.month >= 5] ##TODO testing
#%%

def rmspe(y_true, y_pred):
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true )))
    return rmspe
def rmspe_exp(y_true, y_pred):
    y_true = np.exp(y_true)
    y_pred = np.exp(y_pred)
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return rmspe

def rmspe_exp_xg(y_pred, y_true):
    y_true = y_true.get_label()
    y_true = np.exp(y_true)
    y_pred = np.exp(y_pred)
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return "rmspe", rmspe

def rmspe_xgb(y_pred,y_true):
    y_true = y_true.get_label()
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return "rmspe", rmspe

rmspe_exp_scorer = make_scorer(rmspe_exp,greater_is_better=False)
rmspe_scorer = make_scorer(rmspe,greater_is_better=False)
#%%
tic=timeit.default_timer()
train_stores = train_df.groupby('Store')
train_dates = train_df.groupby('Date')
train_dates_mean = train_dates.mean()
test_stores = test_df.groupby('Store')
test_dates = test_df.groupby('Date')
test_dates_mean = test_dates.mean()
store_1 = train_stores.get_group(1)

store_8_comp = train_stores.get_group(8)
store_10_comp = train_stores.get_group(10)
store_25_comp = train_stores.get_group(25)
store_102_comp = train_stores.get_group(102)
store_103_comp = train_stores.get_group(103)
store_120_comp = train_stores.get_group(120)
store_125_comp = train_stores.get_group(125)
store_175_comp = train_stores.get_group(175)
store_183_comp = train_stores.get_group(183)
store_227_comp = train_stores.get_group(227)
store_238_comp = train_stores.get_group(238)
store_269_comp = train_stores.get_group(269)
store_274_comp = train_stores.get_group(274)
store_286_comp = train_stores.get_group(286)
store_326_comp = train_stores.get_group(326)
store_339_comp = train_stores.get_group(339)
store_383_comp = train_stores.get_group(383)
store_415_comp = train_stores.get_group(415)
store_470_comp = train_stores.get_group(470)
store_530_comp = train_stores.get_group(530)
store_550_comp = train_stores.get_group(550)
store_586_comp = train_stores.get_group(586)
store_622_comp = train_stores.get_group(622)
store_548_comp = train_stores.get_group(548)
store_644_comp = train_stores.get_group(644)
store_652_comp = train_stores.get_group(652)
store_663_comp = train_stores.get_group(663)
store_676_comp = train_stores.get_group(676)
store_685_comp = train_stores.get_group(685)
store_711_comp = train_stores.get_group(711)
store_731_comp = train_stores.get_group(731)
store_782_comp = train_stores.get_group(782)
store_803_comp = train_stores.get_group(803)
store_809_comp = train_stores.get_group(809)
store_831_comp = train_stores.get_group(831)
store_837_comp = train_stores.get_group(837)
store_848_comp = train_stores.get_group(848)
store_882_comp = train_stores.get_group(882)
store_897_comp = train_stores.get_group(897)
store_901_comp = train_stores.get_group(901)
store_902_comp = train_stores.get_group(902)
store_909_comp = train_stores.get_group(909)
store_917_comp = train_stores.get_group(917)
store_930_comp = train_stores.get_group(930)
store_863_comp = train_stores.get_group(863)
store_877_comp = train_stores.get_group(877)
store_947_comp = train_stores.get_group(947)
store_950_comp = train_stores.get_group(950)
store_956_comp = train_stores.get_group(956)
store_983_comp = train_stores.get_group(983)
store_1039_comp = train_stores.get_group(1039)
store_1072_comp = train_stores.get_group(1072)
store_1093_comp = train_stores.get_group(1093)
store_1003_comp = train_stores.get_group(1003)
#store_909_comp = train_df.groupby('Store').get_group(909)

train_df_mean = train_stores.mean()
test_df_mean = test_stores.mean()

test_store_622 = test_stores.get_group(274)
test_store_622 = test_stores.get_group(622)
test_store_703 = test_stores.get_group(703)
test_store_879 = test_stores.get_group(879)
test_store_1097 = test_stores.get_group(1097)
test_store_877 = test_stores.get_group(877)
toc=timeit.default_timer()
print('Calculating Means',toc - tic)
#%%
#train_open_df_norm = train_open_df

train_sunday = train.loc[(train.DayOfWeek == 7) & (train.Open == 1)]

store_means_train = train.groupby('Store')['Sales'].mean().to_dict()
store_means_train_sat = train.loc[train.DayOfWeek == 6].groupby('Store')['Sales'].mean().to_dict()
store_means_train_sun = train_sunday.loc[train_sunday.DayOfWeek == 7].groupby('Store')['Sales'].mean().to_dict()
train['Store_Mean'] = train['Store'].map(lambda x: store_means_train[x])

train['Store_Mean_Sat'] = train['Store'].map(lambda x: store_means_train_sat[x])
train_sunday['Store_Mean_Sun'] = train_sunday['Store'].map(lambda x: store_means_train_sun[x])
train['Sales_Norm'] = train.Sales / train.Store_Mean
train['Sales_Norm_Sat'] = train.Sales / train.Store_Mean_Sat
train_sunday['Sales_Norm_Sun'] = train_sunday.Sales / train_sunday.Store_Mean_Sun

low_train_store_sales = train.loc[(train['Sales_Norm'] <= 0.5) & (train.DayOfWeek != 7) & (train.DayOfWeek != 6)]
low_train_store_sales_summer = train.loc[(train['Sales_Norm'] <= 0.5) & ((train.month == 8) | (train.month == 9))]
low_train_store_sales_sat = train.loc[(train['Sales_Norm_Sat'] <= 0.5) & ((train.month == 8) | (train.month == 9)) & (train.DayOfWeek == 6)]
low_train_store_sales_sun = train_sunday.loc[(train_sunday['Sales_Norm_Sun'] <= 0.5) & ((train_sunday.month == 8) | (train_sunday.month == 9)) & (train_sunday.DayOfWeek == 7)]
#low_store_sales = train_open_df_norm.loc[(train_open_df_norm['Sales_Norm'] <= 0.6) &
#                                         (train_open_df_norm.Promo == 1)]
high_store_sales_no_promo = train.loc[(train['Sales_Norm'] >= 2.0) & (train.Promo == 0)]
high_store_sales_promo = train.loc[(train['Sales_Norm'] >= 3.5) & (train.Promo == 1)]
#%%
#TODO testing
train = train.loc[(train['Sales_Norm'] < 2.0) | (train.Promo == 0)]
train = train.loc[(train['Sales_Norm'] < 3.5) | (train.Promo == 1)]
#%%
tic=timeit.default_timer()
missing_stores_set = set(missing_stores)
train['is_missing_store'] = train.Store.map(lambda x: 1 if x in missing_stores_set else 0)
train_reduced = train.loc[train['is_missing_store'] == 0]
low_train_store_sales_reduced = train_reduced.loc[(train_reduced['Sales_Norm'] <= 0.5) &
                                                   ((train_reduced.month == 8) | (train_reduced.month == 9))]
if(use_cv):
    test['is_missing_store'] = test.Store.map(lambda x: 1 if x in missing_stores_set else 0)
    test_reduced = test.loc[test['is_missing_store'] == 0]
else:
    test_reduced = test
    test_reduced = test_reduced.set_index(['Id'],drop=False)
toc=timeit.default_timer()
print('Getting reduced train',toc-tic)
#%%
#175 8-26
#809 9-22
#685 9-02
#831 8-26,8-27
#837 8-26
#407 9-03
#188 8-06
#636 9-18
#575 9-18
#385 8-21
#897 9-04
#279 9-05
#244 8-22

#120 8-17 222
#663 8-17

#%%
tic=timeit.default_timer()
#
store_values = train_reduced.groupby(['Store']).Sales.mean()
store_values = store_values.reset_index()
store_values.rename(columns={'Sales':'sales_mean'},inplace=True)
train_reduced = pd.merge(train_reduced,store_values, on = ['Store'],how='left')
test_reduced = pd.merge(test_reduced,store_values, on = ['Store'],how='left')

na_val = train_reduced.Sales.mean()
test_reduced['sales_mean'].fillna(na_val,inplace=True)

train_reduced = train_reduced.set_index(['Id'],drop=False)
test_reduced = test_reduced.set_index(['Id'],drop=False)
toc=timeit.default_timer()
#%%
#march_3_2014_df = train_df.loc[train_df.date_int == 426]
#%%
#clean monday, 8/08 are missed as holidays sometimes
#may30 is chorpus christie
#also date after clean monday
not_open_df = train.loc[(train['is_open_next_workday'] == 0) & (train['is_closed_next_consec_workdays'] == 0)]
not_open_consec_df = train.loc[train['is_closed_next_consec_workdays'] == 1]
#%%
train_open_df_norm = train_open_df
#train_open_df_norm = train
store_means = train_open_df_norm.groupby('Store')['Sales'].mean().to_dict()
store_means_comp_inactive = train_open_df_norm.loc[train_open_df_norm.comp_active != 2].groupby('Store')['Sales'].mean().to_dict()
train_open_df_norm['Store_Mean'] = train_open_df_norm['Store'].map(lambda x: store_means[x])
train_open_df_norm['Sales_Norm'] = train_open_df_norm.Sales / train_open_df_norm.Store_Mean

low_store_sales = train_open_df_norm.loc[train_open_df_norm['Sales_Norm'] <= 0.3]
#low_store_sales = train_open_df_norm.loc[(train_open_df_norm['Sales_Norm'] <= 0.6) &
#                                         (train_open_df_norm.Promo == 1)]
high_store_sales = train_open_df_norm.loc[train_open_df_norm['Sales_Norm'] >= 3]
sunday_sc_hol_sales = train_open_df_norm.loc[(train_open_df_norm['SchoolHoliday'] == 1)
                                        & (train_open_df_norm['DayOfWeek'] == 7)]

train_day_group = train_open_df_norm.groupby('day')
train_day_means = train_day_group['Sales_Norm'].mean()

#june 30
train_day_of_year_group = train_open_df_norm.groupby(['day_of_year','DayOfWeek'])
train_day_of_year_means = train_day_of_year_group['Sales_Norm'].mean()

train_days_from_easter_group = train_open_df_norm.groupby(['DayOfWeek','days_from_easter'])
train_days_from_easter = train_days_from_easter_group['Sales_Norm'].mean()

train_days_from_monday_group = train_open_df_norm.groupby('days_from_first_monday')
train_days_from_monday = train_days_from_monday_group['Sales_Norm'].mean()

train_store_group = train_open_df_norm.groupby(['Store','Promo'])
train_store_means = train_store_group['Sales_Norm'].mean()

train_week_group = train_open_df_norm.groupby('week')
train_week_means = train_week_group['Sales_Norm'].mean()

train_promo_number_group = train_open_df_norm.groupby(['Promo','promo_number'])
train_promo_number_means = train_promo_number_group['Sales_Norm'].mean()

train_promo_next_week_group = train_open_df_norm.groupby(['Promo','is_promo_next_week'])
train_promo_next_week_means = train_promo_next_week_group['Sales_Norm'].mean()

train_promo_past_week_group = train_open_df_norm.groupby(['Promo','is_promo_past_week'])
train_promo_past_week_means = train_promo_past_week_group['Sales_Norm'].mean()

train_promo_group = train_open_df_norm.groupby(['Promo','DayOfWeek'])
train_promo_means = train_promo_group['Sales_Norm'].mean()

train_year_group = train_open_df_norm.groupby(['Store','year'])
train_year_means = train_year_group['Sales_Norm'].mean()

train_quarter_group = train_open_df_norm.groupby(['quarter','DayOfWeek'])
train_quarter_means = train_quarter_group['Sales_Norm'].mean()

train_month_end_group = train_open_df_norm.groupby(['month_end','DayOfWeek'])
train_month_end_means = train_month_end_group['Sales_Norm'].mean()
train_month_start_group = train_open_df_norm.groupby(['month_start','DayOfWeek'])
train_month_start_means = train_month_start_group['Sales_Norm'].mean()

train_weekday_group = train_open_df_norm.groupby(['Promo','month_start','DayOfWeek'])
train_weekday_means = train_weekday_group['Sales_Norm'].mean()

train_promo_next_group = train_open_df_norm.groupby(['is_promo_next_weekday','DayOfWeek','Promo'])
train_promo_next_means = train_promo_next_group['Sales_Norm'].mean()

train_promo_prev_group = train_open_df_norm.groupby(['is_promo_prev_weekday','DayOfWeek','Promo'])
train_promo_prev_means = train_promo_prev_group['Sales_Norm'].mean()

train_is_open_in_2_weeks_group = train_open_df_norm.groupby(['is_open_in_2_weeks','is_open_in_1_week'])
train_is_open_in_2_weeks_means = train_is_open_in_2_weeks_group['Sales_Norm'].mean()

train_is_open_next_workday_group = train_open_df_norm.groupby(['is_open_next_workday'])
train_is_open_next_workday_means = train_is_open_next_workday_group['Sales_Norm'].mean()

train_is_closed_next_2workdays_group = train_open_df_norm.groupby(['Store','is_closed_next_consec_workdays'])
train_is_closed_next_2workdays_means = train_is_closed_next_2workdays_group['Sales_Norm'].mean()

train_is_closed_group = train_open_df_norm.groupby(['is_closed_next_consec_workdays','is_closed_shift1',
                                                    'is_closed_shift2','is_closed_shift3','is_closed_shift4',
                                                    'is_closed_shift5','is_closed_shift6',
                                                    'is_closed_shift7','is_closed_shift8','is_closed_shift9',
                                                    'is_closed_shift10','is_closed_shift11',
                                                    'is_closed_shift12','is_closed_shift13','is_closed_shift14'])
train_is_closed_means = train_is_closed_group['Sales_Norm'].mean()

train_was_closed_prev_group = train_open_df_norm.groupby(['is_closed_prev_consec_workdays','was_closed_prev_shift1',
                                                          'was_closed_prev_shift2','was_closed_prev_shift3',
                                                          'was_closed_prev_shift4','was_closed_prev_shift5'
                                                          ])
train_was_closed_prev_means = train_was_closed_prev_group['Sales_Norm'].mean()

train_is_closed_prev_2workdays_group = train_open_df_norm.groupby(['is_closed_prev_consec_workdays',
                                                                   'was_closed_prev_shift1','was_closed_prev_shift2','was_closed_prev_shift3'])
train_is_closed_prev_2workdays_means = train_is_closed_prev_2workdays_group['Sales_Norm'].mean()

train_hol_next_group = train_open_df_norm.groupby('is_hol_next')
train_hol_next_means = train_hol_next_group['Sales_Norm'].mean()

train_hol_prev_group = train_open_df_norm.groupby('is_hol_prev')
train_hol_prev_means = train_hol_prev_group['Sales_Norm'].mean()

train_gt_group = train_open_df_norm.groupby('g_trends_binned')
train_gt_means = train_gt_group['Sales_Norm'].mean()

train_gt_st_group = train_open_df_norm.groupby('gt_st_binned')
train_gt_st_means = train_gt_st_group['Sales_Norm'].mean()

train_high_rain_group = train_open_df_norm.groupby(['high_rain','high_wind'])
train_high_rain_means = train_high_rain_group['Sales_Norm'].mean()

train_low_rain_group = train_open_df_norm.groupby('low_rain')
train_low_rain_means = train_low_rain_group['Sales_Norm'].mean()

train_high_wind_group = train_open_df_norm.groupby('high_wind')
train_high_wind_means = train_high_wind_group['Sales_Norm'].mean()

train_low_wind_group = train_open_df_norm.groupby('low_wind')
train_low_wind_means = train_low_wind_group['Sales_Norm'].mean()

train_is_cold_group = train_open_df_norm.groupby('is_cold')
train_is_cold_means = train_is_cold_group['Sales_Norm'].mean()

train_is_hot_group = train_open_df_norm.groupby('is_hot')
train_is_hot_means = train_is_hot_group['Sales_Norm'].mean()

train_pot_hol_group = train_open_df_norm.groupby('Potential_hol')
train_pot_hol_means = train_pot_hol_group['Sales_Norm'].mean()

train_date_int_group = train_open_df_norm.groupby('date_int')
train_date_int_means = train_date_int_group['Sales_Norm'].mean()

train_month_group = train_open_df_norm.groupby(['Store','year','month'])
train_month_means = train_month_group['Sales_Norm'].mean()
train_month_weekday_group = train_open_df_norm.groupby(['summer','DayOfWeek'])
train_month_weekday_means = train_month_weekday_group['Sales_Norm'].mean()

train_sc_hol_group = train_open_df_norm.groupby(['sc_hol_mod','DayOfWeek'])
train_sc_hol_means = train_sc_hol_group['Sales_Norm'].mean()

train_iso_sc_hol_group = train_open_df_norm.groupby(['isolated_school_hol','DayOfWeek'])
train_iso_sc_hol_means = train_iso_sc_hol_group['Sales_Norm'].mean()

train_p2_start_month_group = train_open_df_norm.groupby(['is_promo2_start_month'])
train_p2_start_month_means = train_p2_start_month_group['Sales_Norm'].mean()

train_comp_active_group = train_open_df_norm.groupby(['Store','comp_active'])

def get_store_mean_comp_inactive(x):
    try:
        return store_means_comp_inactive[x]
    except:
        return store_means[x]
train_open_df_norm['Store_Mean_Comp_Inactive'] = train_open_df_norm['Store'].map(lambda x: get_store_mean_comp_inactive(x))
train_open_df_norm['Sales_Norm_Comp'] = train_open_df_norm.Sales / train_open_df_norm.Store_Mean_Comp_Inactive
train_comp_active_group = train_open_df_norm.groupby(['comp_active','comp_distance_binned'])
train_comp_series = train_comp_active_group['Sales_Norm_Comp'].mean()
train_comp_active_means = train_comp_series

train_promo2_active_group = train_open_df_norm.groupby(['Store','p2_active'])
train_promo2_active_means = train_promo2_active_group['Sales_Norm'].mean()

#train_day_of_year_group = train_open_df_norm.groupby(['day_of_year'])
#train_day_of_year_means = train_day_of_year_group['Sales_Norm'].mean()
train_st_hol_group = train_open_df_norm.groupby('is_st_hol')
train_st_hol_means = train_st_hol_group['Sales_Norm'].mean()
#%%

#%%
test_store1 = test_df.groupby('Store').get_group(1)
test_store274 = test_df.groupby('Store').get_group(274)
#%%
#tic=timeit.default_timer()
#
##store_days_from_easter = train.groupby(['state_id','days_from_easter']).Sales_Log.mean()
##store_days_from_easter = store_days_from_easter.reset_index()
##store_days_from_easter.rename(columns={'Sales_Log':'days_from_easter_sales_log'},inplace=True)
##
#store_values = train_reduced.groupby(['Store']).Sales.mean()
#store_values = store_values.reset_index()
#store_values.rename(columns={'Sales':'sales_mean'},inplace=True)
###
###store_months = train.groupby(['Store','month']).Sales_Log.mean()
###store_months = store_months.reset_index()
###store_months.rename(columns={'Sales_Log':'month_sales_log'},inplace=True)
###
###store_week_num = train.groupby(['Store','week']).Sales_Log.mean()
###store_week_num = store_week_num.reset_index()
###store_week_num.rename(columns={'Sales_Log':'week_num_sales_log'},inplace=True)
###
###store_day_num = train.groupby(['Store','day']).Sales_Log.mean()
###store_day_num = store_day_num.reset_index()
###store_day_num.rename(columns={'Sales_Log':'day_num_sales_log'},inplace=True)
###
##train = pd.merge(train,store_weeks, on = ['Store','DayOfWeek'],how='left')
##test = pd.merge(test,store_weeks, on = ['Store','DayOfWeek'],how='left')
###
###train = pd.merge(train,store_months, on = ['Store','month'],how='left')
###test = pd.merge(test,store_months, on = ['Store','month'],how='left')
###
###train = pd.merge(train,store_week_num, on = ['Store','week'],how='left')
###test = pd.merge(test,store_week_num, on = ['Store','week'],how='left')
###
###train = pd.merge(train,store_day_num, on = ['Store','day'],how='left')
###test = pd.merge(test,store_day_num, on = ['Store','day'],how='left')
##
##train = pd.merge(train,store_days_from_easter, on = ['state_id','days_from_easter'],how='left')
##test = pd.merge(test,store_days_from_easter, on = ['state_id','days_from_easter'],how='left')
#train_reduced = pd.merge(train_reduced,store_values, on = ['Store'],how='left')
#test_reduced = pd.merge(test_reduced,store_values, on = ['Store'],how='left')
##
#na_val = train_reduced.Sales.mean()
#test_reduced['sales_mean'].fillna(na_val,inplace=True)
#toc=timeit.default_timer()
#print('Imputing Averages Time',toc-tic)
#test_null = test.loc[test.days_from_easter_sales_log.isnull() == True]
#null_cond = test.days_from_easter_sales_log.isnull()
#test['days_from_easter_sales_log'][null_cond] = test['weekday_sales_log'][null_cond]
#%%
days_list = ['day_1', 'day_10', 'day_11', 'day_12', 'day_13',
       'day_14', 'day_15', 'day_16', 'day_17', 'day_18', 'day_19', 'day_2',
       'day_20', 'day_21', 'day_22', 'day_23', 'day_24', 'day_25', 'day_26',
       'day_27', 'day_28', 'day_29', 'day_3', 'day_30', 'day_31', 'day_4',
       'day_5', 'day_6', 'day_7', 'day_8', 'day_9']
promo_numbers = ['promo_number_0',
       'promo_number_1', 'promo_number_10', 'promo_number_11',
       'promo_number_12', 'promo_number_13', 'promo_number_14',
       'promo_number_15', 'promo_number_16', 'promo_number_17',
       'promo_number_18', 'promo_number_19', 'promo_number_2',
       'promo_number_20', 'promo_number_21', 'promo_number_22',
       'promo_number_23', 'promo_number_24', 'promo_number_25',
       'promo_number_26', 'promo_number_3', 'promo_number_4',
       'promo_number_5', 'promo_number_6', 'promo_number_7',
       'promo_number_8', 'promo_number_9']
sunday_months = [       'sunday_month_0', 'sunday_month_1', 'sunday_month_10',
       'sunday_month_11', 'sunday_month_2', 'sunday_month_3',
       'sunday_month_4', 'sunday_month_5', 'sunday_month_6',
       'sunday_month_7', 'sunday_month_8', 'sunday_month_9']
gt_st_binned_list = ['gt_st_binned_1', 'gt_st_binned_2', 'gt_st_binned_3',
       'gt_st_binned_4', 'gt_st_binned_5', 'gt_st_binned_6']

#days_list = ['day_1', 'day_2','day_26',
#       'day_27', 'day_28', 'day_29', 'day_3', 'day_30', 'day_31', 'day_4',
#       'day_5']
#%%
test_stores = test.groupby('Store')
train_stores_comp = train_df.groupby('Store')
test_lb_stores = set(test_df.Store.unique())
test_stores_list = test.Store.unique()
i = 0
partial_close_list = []
for name,group in test_stores:
#for name in range(1,200):
    if name not in test_stores_list:
        continue
    if name not in test_lb_stores:
        continue
    results_df = pd.DataFrame()
    train_store_df = train_stores_comp.get_group(name)
    if (train_store_df.shape[0] < 800):
        i = i+1
        partial_close_list.append(name)
print(i)
#%%
tic=timeit.default_timer()
random.seed(4)
#RES = {}
result_series = pd.Series()
rmspe_total = 0
spe_total = 0
#p.figure()
num_test = 0

test_stores_list = test_reduced.Store.unique()
columns_list = [
                'SchoolHoliday',
                'sc_hol_mod',
                'is_saturday_and_sc_mod',
                'is_sunday_and_sc_hol',
                'isolated_school_hol',
                'is_school_hol_next_day','is_school_hol_prev_day',
                'is_school_hol_next_week','is_school_hol_prev_week',
#                'Promo',
                'weekday_no_promo',
#                'is_weekend',
                'p2_active',
#                'is_promo2_start_month',
#                'comp_active',
                'is_comp_after_14_days',
#                'is_comp_starting_month',
                'is_st_hol',
#                'Potential_hol',
                'Weekday_1','Weekday_2','Weekday_3','Weekday_4','Weekday_5','Weekday_6','Weekday_7',
                'is_monday_and_promo','is_tuesday_and_promo',
                'is_wednesday_and_promo','is_thursday_and_promo','is_friday_and_promo',
                'is_monday_and_month_end','is_monday_and_month_start',
                'is_tuesday_and_month_end','is_tuesday_and_month_start',
                'is_wednesday_and_month_end','is_wednesday_and_month_start',
                'is_thursday_and_month_end','is_thursday_and_month_start',
                'is_friday_and_month_end','is_friday_and_month_start',
                'is_saturday_and_month_end','is_saturday_and_month_start',
                'is_sunday_and_month_end','is_sunday_and_month_start',
                'year_2013','year_2014','year_2015',
                'month_1',
                'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
                'month_8', 'month_9','month_10',
                'month_11',
#                'month_12',
                'date_int_binned_1','date_int_binned_2', 'date_int_binned_3',
                'date_int_binned_4','date_int_binned_5',
#                '57_days_from_easter',
#                'summer',

#                'is_open_1_week_ago','is_open_2_weeks_ago',
#                'is_open_in_1_week','is_open_in_2_weeks',

#                'is_open_next_workday','is_open_prev_workday',
#                'is_open_next2_workday','is_open_prev2_workday',
#                'is_closed_next_consec_workdays','is_closed_prev_consec_workdays',
#                'days_from_easter_sales_log',
                'month_start',
                'month_end',
                'is_month_end',
                'quarter_1','quarter_2','quarter_3','quarter_4',
                'is_hol_prev','is_hol_next',
                'is_promo_next_weekday','is_promo_prev_weekday',
                'is_promo_next_week','is_promo_past_week',

                'g_trends','gt_st',
                'is_cold',
                'is_hot',
                'high_rain',
                'low_rain',
                'high_wind',
                'low_wind'
                ]
columns_list = columns_list + days_list
#columns_list = columns_list + sunday_months
#columns_list = columns_list + gt_st_binned_list
#columns_list = columns_list + promo_numbers

test_stores = test_reduced.groupby('Store')
train_stores = train_reduced.groupby('Store')
test_lb_stores = set(test_df.Store.unique())
for name,group in test_stores:
#for name in range(1,200):
    if name not in test_stores_list:
        continue
    if name not in test_lb_stores:
        continue
    results_df = pd.DataFrame()
    train_store_df = train_stores.get_group(name)
    test_store_df = test_stores.get_group(name)
    cv_l = cross_validation.KFold(len(train_store_df), n_folds=5, shuffle=True,random_state = 4)
    regr = LassoCV(cv=cv_l, n_jobs = 1,max_iter=5000,tol=0.002)
#    regr = ElasticNetCV(cv=cv_l, n_jobs = 1,max_iter=100,tol=0.01,n_alphas=10,
#                        l1_ratio = [0.1, .9, .95, 0.98, .99, 1],selection='random')
#    regr = Lasso()
#    regr = RandomizedSearchCV(Lasso(max_iter=10000,tol=0.01),{'alpha':sp_rand(0,0.001)},
#                              verbose=0, n_jobs=1, cv = 2, scoring='mean_squared_error', n_iter = 10)

    X_train = train_store_df[columns_list]
    X_test = test_store_df[columns_list]
    test_sales = test_store_df['Sales']
    train_sales = train_store_df['Sales']
    test_sales_log = test_store_df['Sales_Log']
    train_sales_log = train_store_df['Sales_Log']

    train_data = X_train.values
    test_data = X_test.values
    regr = regr.fit( train_data, train_sales_log )
#    regr = regr.fit( train_data, train_sales )
    pred = regr.predict(test_data)
    pred = np.exp(pred)
    pred = np.maximum(pred, 0.)
    pred_train = regr.predict(train_data)
    pred_train = np.exp(pred_train)
    pred_train = np.maximum(pred_train, 0.)

#    test_series = pd.Series(pred, test_store_df['date_int'])
    test_series = pd.Series(pred, test_store_df.Id)

    if(use_cv):
        rmspe_store = rmspe(test_sales,pred)
        spe = (((test_sales - pred) / test_sales) ** 2).sum()
        rmspe_total = rmspe_total + rmspe_store
        spe_total = spe_total + spe

#    plt.scatter(df_test.index,test_data[0::,0] - prediction)
#    plt.xlabel('date')
#    plt.xlim(0,1050)
#    plt.ylabel('truth - pred')
#    plt.title(store + ' ' + item)
#    pic_name = 'Images/Residuals/' + store + '_' + item + '.png'
#    p.savefig(pic_name, bbox_inches='tight')
#    p.clf()
#    p.cla()
    num_test = num_test + len(test_data)
#    results_df = test_series
    result_series = result_series.append(test_series)
#    RES[name] = results_df

#1020 8 bin 2
#304,137,944,1053, 7 bin 10
#131 7 bin 9
#718 6 bin 10
#269 6 bin 3
#550 6 bin 2

#902 5 bin 8
#770 4 bin 2
#1044 4 bin 6
if(not use_cv):
    store_1020_cond = test_reduced['Store'] == 1020
    store_304_cond = test_reduced['Store'] == 304
    store_137_cond = test_reduced['Store'] == 137
    store_944_cond = test_reduced['Store'] == 944
    store_1053_cond = test_reduced['Store'] == 1053
    store_131_cond = test_reduced['Store'] == 131
    store_718_cond = test_reduced['Store'] == 718
    store_269_cond = test_reduced['Store'] == 269
    store_550_cond = test_reduced['Store'] == 550

    result_series[store_1020_cond] = result_series[store_1020_cond] * 0.84

    result_series[store_304_cond] = result_series[store_304_cond] * 0.95
    result_series[store_137_cond] = result_series[store_137_cond] * 0.95
    result_series[store_944_cond] = result_series[store_944_cond] * 0.95
    result_series[store_1053_cond] = result_series[store_1053_cond] * 0.95
    result_series[store_131_cond] = result_series[store_131_cond] * 0.93

    result_series[store_718_cond] = result_series[store_718_cond] * 0.97

    result_series[store_269_cond] = result_series[store_269_cond] * 0.95
    result_series[store_550_cond] = result_series[store_550_cond] * 0.95

closed_next_cond = test_reduced['is_closed_next_consec_workdays'] == 1
closed_prev_cond = test_reduced['is_closed_prev_consec_workdays'] == 1
closed_prev_shift1_cond = test_reduced['was_closed_prev_shift1'] == 1
closed_prev_shift2_cond = test_reduced['was_closed_prev_shift2'] == 1
closed_prev_shift3_cond = test_reduced['was_closed_prev_shift3'] == 1
closed_prev_shift4_cond = test_reduced['was_closed_prev_shift4'] == 1
closed_prev_shift5_cond = test_reduced['was_closed_prev_shift5'] == 1

closed_next_shift1_cond = test_reduced['is_closed_shift1'] == 1
closed_next_shift2_cond = test_reduced['is_closed_shift2'] == 1
closed_next_shift3_cond = test_reduced['is_closed_shift3'] == 1
closed_next_shift4_cond = test_reduced['is_closed_shift4'] == 1
closed_next_shift5_cond = test_reduced['is_closed_shift5'] == 1
closed_next_shift6_cond = test_reduced['is_closed_shift6'] == 1
closed_next_shift7_cond = test_reduced['is_closed_shift7'] == 1
closed_next_shift8_cond = test_reduced['is_closed_shift8'] == 1
closed_next_shift9_cond = test_reduced['is_closed_shift9'] == 1
closed_next_shift10_cond = test_reduced['is_closed_shift10'] == 1
closed_next_shift11_cond = test_reduced['is_closed_shift11'] == 1
closed_next_shift12_cond = test_reduced['is_closed_shift12'] == 1
closed_next_shift13_cond = test_reduced['is_closed_shift13'] == 1
closed_next_shift14_cond = test_reduced['is_closed_shift14'] == 1

result_series[closed_next_cond] = result_series[closed_next_cond] * 0.3
result_series[closed_prev_cond] = result_series[closed_prev_cond] * 1.7
result_series[closed_prev_shift1_cond] = result_series[closed_prev_shift1_cond] * 1.4
result_series[closed_prev_shift2_cond] = result_series[closed_prev_shift2_cond] * 1.3
result_series[closed_prev_shift3_cond] = result_series[closed_prev_shift3_cond] * 1.2
result_series[closed_prev_shift4_cond] = result_series[closed_prev_shift4_cond] * 1.1
result_series[closed_prev_shift5_cond] = result_series[closed_prev_shift5_cond] * 1.05

result_series[closed_next_shift1_cond] = result_series[closed_next_shift1_cond] * 0.9
result_series[closed_next_shift2_cond] = result_series[closed_next_shift2_cond] * 0.9
result_series[closed_next_shift3_cond] = result_series[closed_next_shift3_cond] * 0.9
result_series[closed_next_shift4_cond] = result_series[closed_next_shift4_cond] * 0.9
result_series[closed_next_shift5_cond] = result_series[closed_next_shift5_cond] * 1.1
result_series[closed_next_shift6_cond] = result_series[closed_next_shift6_cond] * 1.1
result_series[closed_next_shift7_cond] = result_series[closed_next_shift7_cond] * 1.1
result_series[closed_next_shift8_cond] = result_series[closed_next_shift8_cond] * 1.1
result_series[closed_next_shift9_cond] = result_series[closed_next_shift9_cond] * 1.1
result_series[closed_next_shift10_cond] = result_series[closed_next_shift10_cond] * 1.1
result_series[closed_next_shift11_cond] = result_series[closed_next_shift11_cond] * 1.1
result_series[closed_next_shift12_cond] = result_series[closed_next_shift12_cond] * 0.9
result_series[closed_next_shift13_cond] = result_series[closed_next_shift13_cond] * 0.9
result_series[closed_next_shift14_cond] = result_series[closed_next_shift14_cond] * 0.9

#175 8-26 231
#809 9-22
#685 9-02 238
#831 8-26,8-27 231-232
#837 8-26 231
#407 9-03 239
#188 8-06 211
#636 9-18 254
#575 9-18 254
#385 8-21 226
#897 9-04 240
#279 9-05 241
#244 8-22 227

#120 8-17 222
#663 8-17

s_175_dates = ((test_reduced['Store'] == 175) & ((test_reduced['days_from_first_monday'] == 231)
                | (test_reduced['days_from_first_monday'] == 238)))
s_685_dates = ((test_reduced['Store'] == 685) & ((test_reduced['days_from_first_monday'] == 238)
                | (test_reduced['days_from_first_monday'] == 245)))
s_831_dates = ((test_reduced['Store'] == 831) & ((test_reduced['days_from_first_monday'] == 231)
                | (test_reduced['days_from_first_monday'] == 238) | (test_reduced['days_from_first_monday'] == 232)
                | (test_reduced['days_from_first_monday'] == 239)))
s_837_dates = ((test_reduced['Store'] == 837) & ((test_reduced['days_from_first_monday'] == 231)
                | (test_reduced['days_from_first_monday'] == 238)))
s_407_dates = ((test_reduced['Store'] == 407) & ((test_reduced['days_from_first_monday'] == 239)
                | (test_reduced['days_from_first_monday'] == 246)))
s_188_dates = ((test_reduced['Store'] == 188) & ((test_reduced['days_from_first_monday'] == 211)
                | (test_reduced['days_from_first_monday'] == 218)))
s_636_dates = ((test_reduced['Store'] == 636) & ((test_reduced['days_from_first_monday'] == 254)
                | (test_reduced['days_from_first_monday'] == 261)))
s_575_dates = ((test_reduced['Store'] == 575) & ((test_reduced['days_from_first_monday'] == 254)
                | (test_reduced['days_from_first_monday'] == 261)))
s_385_dates = ((test_reduced['Store'] == 385) & ((test_reduced['days_from_first_monday'] == 226)
                | (test_reduced['days_from_first_monday'] == 233)))
s_897_dates = ((test_reduced['Store'] == 897) & ((test_reduced['days_from_first_monday'] == 240)
                | (test_reduced['days_from_first_monday'] == 247)))
s_279_dates = ((test_reduced['Store'] == 279) & ((test_reduced['days_from_first_monday'] == 241)
                | (test_reduced['days_from_first_monday'] == 248)))
s_244_dates = ((test_reduced['Store'] == 244) & ((test_reduced['days_from_first_monday'] == 227)
                | (test_reduced['days_from_first_monday'] == 234)))
s_120_dates = ((test_reduced['Store'] == 120) & ((test_reduced['days_from_first_monday'] == 222)
                | (test_reduced['days_from_first_monday'] == 229)))
s_663_dates = ((test_reduced['Store'] == 663) & ((test_reduced['days_from_first_monday'] == 222)
                | (test_reduced['days_from_first_monday'] == 229)))
result_series[s_175_dates] = result_series[s_175_dates] * 0.5
result_series[s_685_dates] = result_series[s_685_dates] * 0.5
result_series[s_831_dates] = result_series[s_831_dates] * 0.5
result_series[s_837_dates] = result_series[s_837_dates] * 0.5
result_series[s_407_dates] = result_series[s_407_dates] * 0.5
result_series[s_188_dates] = result_series[s_188_dates] * 0.5
result_series[s_636_dates] = result_series[s_636_dates] * 0.5
result_series[s_575_dates] = result_series[s_575_dates] * 0.5
result_series[s_385_dates] = result_series[s_385_dates] * 0.5
result_series[s_897_dates] = result_series[s_897_dates] * 0.5
result_series[s_279_dates] = result_series[s_279_dates] * 0.5
result_series[s_244_dates] = result_series[s_244_dates] * 0.5
result_series[s_120_dates] = result_series[s_120_dates] * 0.5
result_series[s_663_dates] = result_series[s_663_dates] * 0.5


if(use_cv):
#    print('rmspe_test_total: ',rmspe_total)
#    print('spe_total:',spe_total)
    print('num_test:',num_test)
    print('rmspe:',(spe_total / num_test)**0.5)

    sales_cv_lasso_series = pd.Series(test_reduced['Sales'],index=test_reduced['Id'],name='Sales')
    pred_sales_lasso_cv_series = pd.Series(result_series ,name='Pred_Sales')
    results_lasso_df_cv = pd.concat([sales_cv_lasso_series,pred_sales_lasso_cv_series],axis=1)
    results_lasso_df_cv['diff'] = results_lasso_df_cv['Sales'] - results_lasso_df_cv['Pred_Sales']
    results_lasso_df_cv['pe'] = np.abs(results_lasso_df_cv['diff'] / results_lasso_df_cv['Sales'])
    results_lasso_df_cv['date'] = test_reduced['Date']
    results_lasso_df_cv['Weekday'] = test_reduced['DayOfWeek']
    results_lasso_df_cv['id'] = test_reduced['Id']
    results_lasso_df_cv['store'] = test_reduced['Store']
    print('rmpse mod',np.sqrt(np.mean(np.square(results_lasso_df_cv['pe']))))
else:
    print('Not using cv')
toc=timeit.default_timer()
print('Linear Reg Time',toc - tic)
#%%
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
#%%
best_score = 0
best_params = {}
def train_xgb(features, train_xgb, test_xgb, params, num_trees, do_grid_search = False, random_seed = 4):
    tic=timeit.default_timer()
    print("Train a XGBoost model")
    random.seed(random_seed)
    np.random.seed(random_seed)

#    X_train, X_watch = cross_validation.train_test_split(train_xgb, test_size=0.001,random_state=random_seed)
#    X_train, X_watch = cross_validation.train_test_split(train_xgb, test_size=0.01,random_state=random_seed)
    #bit of a kludge
    if(random_seed <= 5):
        X_train = train_xgb.loc[(train_xgb.date_int < 146) | (train_xgb.date_int > 166)]
        X_watch = train_xgb.loc[(train_xgb.date_int >= 146) & (train_xgb.date_int <= 166)]
    else:
        X_train = train_xgb.loc[(train_xgb.date_int < 125) | (train_xgb.date_int > 145)]
        X_watch = train_xgb.loc[(train_xgb.date_int >= 125) & (train_xgb.date_int <= 145)]
    train_data = X_train[features].values
#    train_sales_log = X_train['Sales_Log'].values
    train_sales = X_train['Sales'].values / 10000
    watch_data = X_watch[features].values
#    watch_sales_log = X_watch['Sales_Log'].values
    watch_sales = X_watch['Sales'].values / 10000

    test_data = test_xgb[features]

#    dtrain = xgb.DMatrix(train_data, train_sales_log)
#    dwatch = xgb.DMatrix(watch_data, watch_sales_log)

    dtrain_custom = xgb.DMatrix(train_data, train_sales)
    dwatch_custom = xgb.DMatrix(watch_data, watch_sales)

    dtest = xgb.DMatrix(test_data.values)

#    watchlist = [(dwatch, 'watch'), (dtrain, 'train')]
#    watchlist_custom = [(dwatch_custom, 'watch'), (dtrain_custom, 'train')]
    watchlist_custom = [(dtrain_custom, 'train'),(dwatch_custom, 'watch')]
    if(do_grid_search):
        print('Random search cv')
        gbm_search = xgb.XGBRegressor()
        num_features = len(features)
#        clf = RandomizedSearchCV(gbm_search,
#                                 {'max_depth': sp_randint(1,num_features), 'learning_rate':sp_rand(0,0.5),
#                                  'subsample':sp_rand(0.7,0.3),
#                                  'colsample_bytree':sp_rand(0.4,0.6),'seed':[4],
#                                  'gamma':sp_rand(0,3), 'max_delta_step':sp_rand(0,3),
#                                  'n_estimators': [50,100,200,500,1000,2000]},
#                                  verbose=10, n_jobs=1, cv = 2, scoring=rmspe_exp_scorer, n_iter = 100,
#                                  refit=False)
#        clf.fit(train_data_sample, train_sales_log_sample)
#        print('best clf score',clf.best_score_)
#        print('best params:', clf.best_params_)
#
        clf = RandomizedSearchCV(gbm_search,
                                 {'max_depth': sp_randint(4,num_features - 4), 'learning_rate':sp_rand(0,0.4),
                                  'objective':['reg:percent_linear'],
                                  'subsample':sp_rand(0.8,0.2),
                                  'colsample_bytree':sp_rand(0.3,0.7),'seed':[4],
#                                  'gamma':sp_rand(0,0.2),'min_child_weight':sp_randint(1,100),
                                  'gamma':sp_rand(0,0.5),'min_child_weight':sp_randint(10,150),
                                  'max_delta_step':sp_rand(0,20),
                                  'n_estimators': [250,300,500,1000]},
                                  verbose=10, n_jobs=1, cv = 3, scoring=rmspe_scorer, n_iter = 30,
                                  refit=False)
        clf.fit(train_data, train_sales)
        print('best clf score',clf.best_score_)
        print('best params:', clf.best_params_)
        toc=timeit.default_timer()
        print('Grid search time',toc - tic)
        return(clf.best_params_,clf.best_score_)
#        best_score = clf.best_score_
#        best_params = clf.best_params_

#    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist,
#                    early_stopping_rounds=50, feval=rmspe_exp_xg, verbose_eval=True)
#    gbm_custom = xgb.train(params, dtrain_custom, num_trees,
#                           evals=watchlist_custom, early_stopping_rounds=50, verbose_eval=True)
    gbm_custom = xgb.train(params, dtrain_custom, num_trees,
                           evals=watchlist_custom, early_stopping_rounds=100, verbose_eval=True)

    create_feature_map(features)

#    imp_dict = gbm.get_fscore(fmap='xgb.fmap')
#    imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1),reverse=True)
#    print('{0:<20} {1:>5}'.format('Feature','Importance'))
#    print("--------------------------------------")
#    for i in imp_dict:
#        print ('{0:20} {1:5.0f}'.format(i[0], i[1]))

    imp_custom_dict = gbm_custom.get_fscore(fmap='xgb.fmap')
    imp_custom_dict = sorted(imp_custom_dict.items(), key=operator.itemgetter(1),reverse=True)
    print('{0:<20} {1:>5}'.format('Feature','Custom Imp'))
    print("--------------------------------------")
    for i in imp_custom_dict:
        print ('{0:20} {1:5.0f}'.format(i[0], i[1]))


    print("Validating")
#    train_preds = gbm.predict(dwatch,ntree_limit=gbm.best_iteration)
#    indices = train_preds < 0
#    train_preds[indices] = 0
#    error = rmspe(X_watch['Sales'].values,np.exp(train_preds) )

    train_custom_preds = gbm_custom.predict(dwatch_custom,ntree_limit=gbm_custom.best_iteration)
    indices = train_custom_preds < 0
    train_custom_preds[indices] = 0
    error_custom = rmspe(watch_sales,train_custom_preds)

#    print('rmpse train', error)
    print('rmpse custom train', error_custom)
    print('best tree',gbm_custom.best_iteration)
#    test_preds = gbm.predict(dtest,ntree_limit=gbm.best_iteration)
#    indices = test_preds < 0
#    test_preds[indices] = 0

    test_custom_preds = gbm_custom.predict(dtest,ntree_limit=gbm_custom.best_iteration)
    indices = test_custom_preds < 0
    test_custom_preds[indices] = 0

#    xgb_sales = pd.Series(np.exp(test_preds) , index=test_reduced['Id'])
    xgb_custom_sales = pd.Series(test_custom_preds * 10000, index=test_xgb['Id'],name='Custom_Pred_Sales')

    if(not use_cv):
        #use different parameters as xgb probably better at extracting this type of info
        #very dangerous process
        xgb_custom_sales[store_1020_cond] = xgb_custom_sales[store_1020_cond] * 0.84
        xgb_custom_sales[store_304_cond] = xgb_custom_sales[store_304_cond] * 0.96
        xgb_custom_sales[store_137_cond] = xgb_custom_sales[store_137_cond] * 0.96
        xgb_custom_sales[store_944_cond] = xgb_custom_sales[store_944_cond] * 0.96
        xgb_custom_sales[store_1053_cond] = xgb_custom_sales[store_1053_cond] * 0.96
        xgb_custom_sales[store_131_cond] = xgb_custom_sales[store_131_cond] * 0.95
        xgb_custom_sales[store_718_cond] = xgb_custom_sales[store_718_cond] * 0.98
        xgb_custom_sales[store_269_cond] = xgb_custom_sales[store_269_cond] * 0.95
        xgb_custom_sales[store_550_cond] = xgb_custom_sales[store_550_cond] * 0.95


    xgb_custom_sales[closed_next_cond] = xgb_custom_sales[closed_next_cond] * 0.3
    xgb_custom_sales[closed_prev_cond] = xgb_custom_sales[closed_prev_cond] * 1.7
    xgb_custom_sales[closed_prev_shift1_cond] = xgb_custom_sales[closed_prev_shift1_cond] * 1.4
    xgb_custom_sales[closed_prev_shift2_cond] = xgb_custom_sales[closed_prev_shift2_cond] * 1.3
    xgb_custom_sales[closed_prev_shift3_cond] = xgb_custom_sales[closed_prev_shift3_cond] * 1.2
    xgb_custom_sales[closed_prev_shift4_cond] = xgb_custom_sales[closed_prev_shift4_cond] * 1.1
    xgb_custom_sales[closed_prev_shift5_cond] = xgb_custom_sales[closed_prev_shift5_cond] * 1.05


    xgb_custom_sales[closed_next_shift1_cond] = xgb_custom_sales[closed_next_shift1_cond] * 0.9
    xgb_custom_sales[closed_next_shift2_cond] = xgb_custom_sales[closed_next_shift2_cond] * 0.9
    xgb_custom_sales[closed_next_shift3_cond] = xgb_custom_sales[closed_next_shift3_cond] * 0.9
    xgb_custom_sales[closed_next_shift4_cond] = xgb_custom_sales[closed_next_shift4_cond] * 0.9
    xgb_custom_sales[closed_next_shift5_cond] = xgb_custom_sales[closed_next_shift5_cond] * 1.1
    xgb_custom_sales[closed_next_shift6_cond] = xgb_custom_sales[closed_next_shift6_cond] * 1.1
    xgb_custom_sales[closed_next_shift7_cond] = xgb_custom_sales[closed_next_shift7_cond] * 1.1
    xgb_custom_sales[closed_next_shift8_cond] = xgb_custom_sales[closed_next_shift8_cond] * 1.1
    xgb_custom_sales[closed_next_shift9_cond] = xgb_custom_sales[closed_next_shift9_cond] * 1.1
    xgb_custom_sales[closed_next_shift10_cond] = xgb_custom_sales[closed_next_shift10_cond] * 1.1
    xgb_custom_sales[closed_next_shift11_cond] = xgb_custom_sales[closed_next_shift11_cond] * 1.1
    xgb_custom_sales[closed_next_shift12_cond] = xgb_custom_sales[closed_next_shift12_cond] * 0.9
    xgb_custom_sales[closed_next_shift13_cond] = xgb_custom_sales[closed_next_shift13_cond] * 0.9
    xgb_custom_sales[closed_next_shift14_cond] = xgb_custom_sales[closed_next_shift14_cond] * 0.9

    xgb_custom_sales[s_175_dates] = xgb_custom_sales[s_175_dates] * 0.5
    xgb_custom_sales[s_685_dates] = xgb_custom_sales[s_685_dates] * 0.5
    xgb_custom_sales[s_831_dates] = xgb_custom_sales[s_831_dates] * 0.5
    xgb_custom_sales[s_837_dates] = xgb_custom_sales[s_837_dates] * 0.5
    xgb_custom_sales[s_407_dates] = xgb_custom_sales[s_407_dates] * 0.5
    xgb_custom_sales[s_188_dates] = xgb_custom_sales[s_188_dates] * 0.5
    xgb_custom_sales[s_636_dates] = xgb_custom_sales[s_636_dates] * 0.5
    xgb_custom_sales[s_575_dates] = xgb_custom_sales[s_575_dates] * 0.5
    xgb_custom_sales[s_385_dates] = xgb_custom_sales[s_385_dates] * 0.5
    xgb_custom_sales[s_897_dates] = xgb_custom_sales[s_897_dates] * 0.5
    xgb_custom_sales[s_279_dates] = xgb_custom_sales[s_279_dates] * 0.5
    xgb_custom_sales[s_244_dates] = xgb_custom_sales[s_244_dates] * 0.5
    xgb_custom_sales[s_120_dates] = xgb_custom_sales[s_120_dates] * 0.5
    xgb_custom_sales[s_663_dates] = xgb_custom_sales[s_663_dates] * 0.5

    if(use_cv):
#        error_cv = rmspe(test_reduced['Sales'].values,np.exp(test_preds) )
        error_custom_cv = rmspe(test_xgb['Sales'].values,xgb_custom_sales.values )

        sales_cv_series = pd.Series(test_xgb['Sales'],index=test_xgb['Id'],name='Sales')
#        sales_cv_series = xgb_custom_sales
#        pred_sales_cv_series = pd.Series(np.exp(test_preds) ,index=test_reduced['Id'],name='Pred_Sales')
        pred_sales_custom_cv_series = xgb_custom_sales
        results_df_cv = pd.concat([sales_cv_series,pred_sales_custom_cv_series],axis=1)
        results_df_cv['custom_diff'] = results_df_cv['Sales'] - results_df_cv['Custom_Pred_Sales']
        results_df_cv['custom_pe'] = np.abs(results_df_cv['custom_diff'] / results_df_cv['Sales'])
        results_df_cv['id'] = test_xgb['Id']
        results_df_cv['store'] = test_xgb['Store']
        results_df_cv['date'] = test_xgb['Date']
        results_df_cv['is_closed_next_consec_workdays'] = test_xgb['is_closed_next_consec_workdays']
        results_df_cv['is_closed_prev_consec_workdays'] = test_xgb['is_closed_prev_consec_workdays']
        print('rmspe_custom_cv',error_custom_cv)
    else:
        results_df_cv = xgb_custom_sales
        print('submission run xgb')

    toc=timeit.default_timer()
    print('Xgb Time',toc - tic)

    return (xgb_custom_sales,results_df_cv)
#%%
features_xgb = [
                'SchoolHoliday',
                'Store',
                'sales_mean',
                'store_type',
                'assortment',
                'state_id',
                'comp_distance_binned',

                'Promo',
                'promo2_interval_hash',
                'is_promo2_start_month',
                'weekday_no_promo',

                'is_weekend',
                'p2_active',
                'comp_active',
                'is_comp_after_14_days',
                'is_comp_starting_month',
#                'CompetitionOpenSinceMonth',
#                'CompetitionOpenSinceYear',
                'is_st_hol',
                'DayOfWeek',
                'is_hol_next',
                'is_hol_prev',
                'is_promo_next_weekday',
                'is_promo_prev_weekday',
                'isolated_school_hol','is_school_hol_next_day','is_school_hol_prev_day',
                'is_school_hol_next_week','is_school_hol_prev_week',
                'sc_hol_mod',
                'is_promo_next_week','is_promo_past_week',
                'month_start',
                'month_end',
                'month',
                'day',
                'is_month_end',

#                'day_of_year',
#                'days_from_easter',
#                'days_from_easter_and_is_promo',
#                'days_from_easter_and_is_not_promo',
#                'days_from_first_monday',
#                'days_from_first_monday_adj',
#                'week',
                'year',
                'date_int_binned',
                'quarter',

                'is_monday_and_promo','is_tuesday_and_promo',
                'promo_number',
                'promo_number_and_is_not_promo',
                'promo_number_and_is_promo',

                'g_trends_binned',
                'gt_st_binned',
                'is_cold','is_hot','high_rain','high_wind','low_wind'
                ]
features_xgb_2 = [
                'SchoolHoliday',
                'Store',
                'sales_mean',
                'store_type',
                'assortment',
                'state_id',
                'comp_distance_binned',

                'Promo',
                'weekday_no_promo',

                'is_weekend',
                'p2_active',
                'is_promo2_start_month',
                'promo2_interval_hash',
                'comp_active',
                'is_comp_after_14_days',
                'is_comp_starting_month',
#                'CompetitionOpenSinceMonth',
#                'CompetitionOpenSinceYear',
                'is_st_hol',
                'DayOfWeek',
                'is_hol_next',
                'is_hol_prev',
                'is_promo_next_weekday',
                'is_promo_prev_weekday',
                'isolated_school_hol','is_school_hol_next_day','is_school_hol_prev_day',
                'is_school_hol_next_week','is_school_hol_prev_week',
                'sc_hol_mod',
                'is_promo_next_week','is_promo_past_week',
                'month_start',
                'month_end',
                'month',
                'day',
                'is_month_end',

#                'day_of_year',
#                'days_from_easter',
#                'days_from_easter_and_is_promo',
#                'days_from_easter_and_is_not_promo',
                'days_from_first_monday',
#                'days_from_first_monday_adj',
                'week',
                'year',
                'date_int_binned',
                'quarter',

                'is_monday_and_promo','is_tuesday_and_promo',
                'promo_number',
#                'promo_number_and_is_not_promo',
#                'promo_number_and_is_promo',

                'g_trends_binned',
                'gt_st_binned',
                'is_cold','is_hot','high_rain','high_wind','low_wind'
                ]

#best clf score -0.118353622384
#params = {'max_depth': 7, 'colsample_bytree': 0.776121549306617, 'seed': 4,
#'subsample': 0.9481394103498237, 'learning_rate': 0.18246069813524618}
#num_trees = 2
#num_trees = 2000

#best clf score -0.119236843273
#params_custom = {'learning_rate': 0.25928916445937944, 'objective': 'reg:percent_linear',
#              'colsample_bytree': 0.6444274621771185, 'gamma': 0.0817140255333002,
#              'max_delta_step': 6.970785574378006, 'seed': 4, 'min_child_weight': 18,
#              'subsample': 0.8131187671984621, 'max_depth': 10, 'n_estimators': 500,
#              'eval_metric':'percent_rmse'}
#num_trees_custom = 500

#best clf score -0.124578796846
#params_custom = {'learning_rate': 0.16872702207314647, 'objective': 'reg:percent_linear',
#'colsample_bytree': 0.5903602974027777, 'gamma': 0.09483486503696913,
#'max_delta_step': 3.753684341676533, 'seed': 4, 'min_child_weight': 71,
#'subsample': 0.5693756013325375, 'max_depth': 12, 'n_estimators': 1000,
#'eval_metric':'percent_rmse'}
#num_trees_custom = 2

#best clf score -0.119135951788
#params_promo = {'learning_rate': 0.400497139190924, 'objective': 'reg:percent_linear',
#              'colsample_bytree': 0.702412608655453, 'gamma': 0.06623490061822455,
#              'max_delta_step': 8.454436490941132, 'seed': 4, 'min_child_weight': 40,
#              'subsample': 0.8347549367124647, 'max_depth': 8,
#              'n_estimators': 2000,'eval_metric':'percent_rmse'}
#num_trees_promo = 2
##Grid search time 21887.490466607007
##best clf score -0.119050518638
#params_no_promo = {'learning_rate': 0.10957254891222724, 'objective': 'reg:percent_linear',
#'colsample_bytree': 0.5664210032330448, 'gamma': 0.016541522362571292,
#'max_delta_step': 9.862247233300526, 'seed': 4, 'min_child_weight': 39,
# 'subsample': 0.9375311735670065, 'max_depth': 11, 'n_estimators': 500,
# 'eval_metric':'percent_rmse'}
#num_trees_no_promo = 5
#Grid search time 27225.864518926013

#best clf score -0.116959166908
#best clf score -0.115396108995 #no day_of_year, week, quarter
#params_custom = {'learning_rate': 0.1966233641832778, 'objective': 'reg:percent_linear',
#              'colsample_bytree': 0.9293604242446425, 'gamma': 0.06784498164482378,
#              'max_delta_step': 2.7036548223830223, 'seed': 4, 'min_child_weight': 24,
#              'subsample': 0.9316735611659178, 'max_depth': 9, 'n_estimators': 2000,
#              'eval_metric':'percent_rmse'}
#num_trees_custom = 500

#best clf score -0.113441233402 #no day_of_year, week, quarter
#params_custom = {'learning_rate': 0.2524299281651614, 'objective': 'reg:percent_linear',
#'colsample_bytree': 0.8313941326160552, 'gamma': 0.06407292995252596,
#'max_delta_step': 0.9067456114497519, 'seed': 4, 'min_child_weight': 35,
# 'subsample': 0.9443512905182059, 'max_depth': 8, 'n_estimators': 3000,
# 'eval_metric':'percent_rmse'}
#num_trees_custom = 50

#best clf score -0.11580466262
#params_custom = {'learning_rate': 0.1728594429729196, 'objective': 'reg:percent_linear',
#              'colsample_bytree': 0.6444274621771185, 'gamma': 0.03268561021332008,
#              'max_delta_step': 10.456178361567009, 'seed': 4, 'min_child_weight': 18,
#              'subsample': 0.8110154010878567, 'max_depth': 13, 'n_estimators': 500,
#              'eval_metric':'percent_rmse'}
##num_trees_custom = 500
#num_trees_custom = 5
#params_custom = {'learning_rate':0.3749585064384586, 'objective':'reg:percent_linear',
#                 'colsample_bytree':0.6268398243306916,
#                 'gamma':0.11094817258851171, 'max_delta_step':13.43251749084219,
#                 'seed':4, 'min_child_weight':58, 'subsample':0.9243619745714855,
#                 'max_depth':12, 'n_estimators':1000,'eval_metric':'percent_rmse'}
#num_trees_custom = 1000
#params_custom = {'learning_rate':0.3584003663597155, 'objective':'reg:percent_linear',
# 'colsample_bytree':0.9917003385876564, 'gamma':0.03276844828093974,
# 'max_delta_step':0.13479146501332473, 'seed':4, 'min_child_weight':57,
# 'subsample':0.8575657068391231, 'max_depth':10, 'n_estimators':250}
#num_trees_custom = 25
#num_trees_custom = 250


#best clf score -0.116262822477
#params_promo = {'learning_rate': 0.3833178618861572, 'objective': 'reg:percent_linear',
# 'colsample_bytree': 0.7131355833448485, 'gamma': 0.03802599911847631,
# 'max_delta_step': 6.422324721633005, 'seed': 4, 'min_child_weight': 14,
# 'subsample': 0.9924939876942519, 'max_depth': 9, 'n_estimators': 250,
# 'eval_metric':'percent_rmse'}
#num_trees_promo = 250
#
##best clf score -0.117149391243
#params_no_promo = {'learning_rate': 0.11610897807873186, 'objective': 'reg:percent_linear', 'colsample_bytree': 0.6143969610585116,
#'gamma': 0.08520355204506883, 'max_delta_step': 0.31994759252331473, 'seed': 4,
#'min_child_weight': 25, 'subsample': 0.8995770324216559, 'max_depth': 16,
#'n_estimators': 500,'eval_metric':'percent_rmse'}
#num_trees_no_promo = 500

#best clf score -0.103143908157
#params_custom = {'learning_rate': 0.08697211711004758, 'objective': 'reg:percent_linear',
#              'colsample_bytree': 0.5223239147866283, 'gamma': 0.01903961559026457,
#              'max_delta_step': 12.10524821818627, 'seed': 4, 'min_child_weight': 98,
#              'subsample': 0.626027124953123, 'max_depth': 31, 'n_estimators': 2000,
#              'eval_metric':'percent_rmse'}
#num_trees_custom = 2000
#params_custom = {'learning_rate':0.24029708855110773, 'objective':'reg:percent_linear',
#                 'colsample_bytree':0.5865827107479741, 'gamma':0.014989717402621406,
#                 'max_delta_step':2.5195827557760575, 'seed':4,
#                 'min_child_weight':99, 'subsample':0.9750286335803768,
#                 'max_depth':19, 'n_estimators':300,'eval_metric':'percent_rmse'}
#num_trees_custom = 30

#best clf score -0.103240454992
params_fast = {'learning_rate': 0.2389335775731437, 'objective': 'reg:percent_linear',
                 'colsample_bytree': 0.99, 'subsample': 0.93,
                 'n_estimators': 500, 'max_depth': 13, 'min_child_weight': 90,
                 'seed': 5, 'gamma': 0.2, 'max_delta_step': 0.5,
                 'eval_metric':'percent_rmse'}
num_trees_fast = 1000
#does well
#params_custom = {'learning_rate': 0.03, 'objective': 'reg:percent_linear',
#                 'colsample_bytree': 0.9, 'subsample': 0.8,
#                 'n_estimators': 500, 'max_depth': 10, 'min_child_weight': 1,
#                 'seed': 4, 'gamma': 0.01, 'max_delta_step': 0,
#                 'eval_metric':'percent_rmse'}
#num_trees_custom = 5

params_custom = {'learning_rate': 0.02, 'objective': 'reg:percent_linear',
                 'colsample_bytree': 0.95, 'subsample': 0.9,
                 'n_estimators': 500, 'max_depth': 13, 'min_child_weight': 1,
                 'seed': 5, 'gamma': 0.03, 'max_delta_step': 0,
                 'eval_metric':'percent_rmse'}
#params_custom = {'learning_rate': 0.03, 'objective': 'reg:percent_linear',
#                 'colsample_bytree': 0.9, 'subsample': 0.9,
#                 'n_estimators': 500, 'max_depth': 15, 'min_child_weight': 2,
#                 'seed': 5, 'gamma': 0.02, 'max_delta_step': 0,
#                 'eval_metric':'percent_rmse'}
#params_custom = {'learning_rate': 0.03, 'objective': 'reg:percent_linear',
#                 'colsample_bytree': 0.9, 'subsample': 0.9,
#                 'n_estimators': 500, 'max_depth': 15, 'min_child_weight': 2,
#                 'seed': 5, 'gamma': 0.02, 'max_delta_step': 0,
#                 'eval_metric':'percent_rmse','num_parallel_tree':10}
num_trees_custom = 5000


train_reduced_promo = train_reduced.loc[train_reduced.Promo == 1]
train_reduced_no_promo = train_reduced.loc[train_reduced.Promo == 0]

test_reduced_promo = test_reduced.loc[test_reduced.Promo == 1]
test_reduced_no_promo = test_reduced.loc[test_reduced.Promo == 0]

(xgb_custom_sales,results_df_cv) = train_xgb(features = features_xgb, train_xgb = train_reduced,
 test_xgb = test_reduced, params = params_custom, num_trees = num_trees_custom,
 do_grid_search = False, random_seed = 5)

(xgb_custom_sales_2,results_df_cv_2) = train_xgb(features = features_xgb_2, train_xgb = train_reduced,
 test_xgb = test_reduced, params = params_custom, num_trees = num_trees_custom,
 do_grid_search = False, random_seed = 6)







#(xgb_custom_sales_fast,results_df_fast_cv) = train_xgb(features = features_xgb, train_xgb = train_reduced,
# test_xgb = test_reduced, params = params_fast, num_trees = num_trees_fast,
# do_grid_search = False, random_seed = 5)


#(xgb_custom_sales,results_df_cv) = train_xgb(features = features_xgb, train_xgb = train_reduced,
# test_xgb = test_reduced, params = params_custom, num_trees = num_trees_custom,
# do_grid_search = False, random_seed = 4)
#(xgb_custom_sales,results_df_cv) = train_xgb(features = features_xgb, train_xgb = train_reduced,
# test_xgb = test_reduced, params = params_custom, num_trees = num_trees_custom,
# do_grid_search = False, random_seed = 4)


#(best_params,best_score) = train_xgb(features = features_xgb, train_xgb = train_reduced,
# test_xgb = test_reduced, params = params_custom, num_trees = num_trees_custom,
# do_grid_search = True, random_seed = 4)

#(xgb_custom_sales_promo,results_df_cv_promo) = train_xgb(features = features_xgb, train_xgb = train_reduced_promo,
# test_xgb = test_reduced_promo, params = params_custom, num_trees = num_trees_custom,
# do_grid_search = False, random_seed = 4)
#(xgb_custom_sales_no_promo,results_df_cv_no_promo) = train_xgb(features = features_xgb, train_xgb = train_reduced_no_promo,
# test_xgb = test_reduced_no_promo, params = params_custom, num_trees = num_trees_custom,
# do_grid_search = False, random_seed = 4)
#

#(best_params_no_promo,best_score_no_promo) = train_xgb(features = features_xgb, train_xgb = train_reduced_no_promo,
# test_xgb = test_reduced_no_promo, params = params_custom, num_trees = num_trees_custom,
# do_grid_search = True, random_seed = 4)
#(best_params_promo,best_score_promo) = train_xgb(features = features_xgb, train_xgb = train_reduced_promo,
# test_xgb = test_reduced_promo, params = params_custom, num_trees = num_trees_custom,
# do_grid_search = True, random_seed = 4)

#%%
features_xgb_3 = [
                'SchoolHoliday',
                'Store',
                'sales_mean',
                'store_type',
                'assortment',
                'state_id',
                'comp_distance_binned',

                'Promo',
#                'weekday_no_promo',

                'is_weekend',
                'p2_active',
                'is_promo2_start_month',
                'promo2_interval_hash',
                'comp_active',
                'is_comp_after_14_days',
                'is_comp_starting_month',
                'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear',
#                'is_st_hol',
                'DayOfWeek',
#                'is_hol_next',
#                'is_hol_prev',
                'is_promo_next_weekday',
                'is_promo_prev_weekday',
                'isolated_school_hol','is_school_hol_next_day','is_school_hol_prev_day',
                'is_school_hol_next_week','is_school_hol_prev_week',
                'sc_hol_mod',
                'is_promo_next_week','is_promo_past_week',
                'month_start',
                'month_end',
                'month',
                'day',
#                'is_month_end',

                'day_of_year',
                'days_from_easter',
#                'days_from_easter_and_is_promo',
#                'days_from_easter_and_is_not_promo',
                'days_from_first_monday',
#                'days_from_first_monday_adj',
                'week',
                'year',
                'date_int_binned',
                'quarter',
                'is_monday_and_promo','is_tuesday_and_promo',
#                'promo_number',
#                'promo_number_and_is_not_promo',
#                'promo_number_and_is_promo',
                'is_cold',
                'is_hot',
                'high_rain',
#                'high_wind','low_wind'
                ]
#best clf score -0.103143908157
#params_custom = {'learning_rate': 0.08697211711004758, 'objective': 'reg:percent_linear',
#              'colsample_bytree': 0.5223239147866283, 'gamma': 0.01903961559026457,
#              'max_delta_step': 12.10524821818627, 'seed': 4, 'min_child_weight': 98,
#              'subsample': 0.626027124953123, 'max_depth': 31, 'n_estimators': 2000,
#              'eval_metric':'percent_rmse'}
params_custom_3 = {'learning_rate': 0.08, 'objective': 'reg:percent_linear',
                 'colsample_bytree': 0.5, 'subsample': 0.63,
                 'n_estimators': 500, 'max_depth': 20, 'min_child_weight': 120,
                 'seed': 5, 'gamma': 0.04, 'max_delta_step': 50,
                 'eval_metric':'percent_rmse'}
num_trees_custom_3 = 5000

(xgb_custom_sales_3,results_df_cv_3) = train_xgb(features = features_xgb_3, train_xgb = train_reduced,
 test_xgb = test_reduced, params = params_custom_3, num_trees = num_trees_custom_3,
 do_grid_search = False, random_seed = 5)

#%%
if(use_cv):
#    results_df_cv = results_df_cv_no_promo.append(results_df_cv_promo)
    results_df_cv = results_df_cv
    results_df_cv['state_id'] = test_reduced['state_id']
    results_df_cv['Weekday'] = test_reduced['DayOfWeek']
    results_df_cv['promo'] = test_reduced['Promo']
    results_lasso_df_cv['promo'] = test_reduced['Promo']
    results_date_mean = results_df_cv.groupby('date').mean()
    results_store_mean = results_df_cv.groupby('store').mean()
    results_lasso_date_mean = results_lasso_df_cv.groupby('date').mean()
    results_lasso_store_mean = results_lasso_df_cv.groupby('store').mean()
#%%
#tic=timeit.default_timer()
#sales_train_series = pd.Series(watch_sales * 10000,name='Sales')
#pred_sales_train_series = pd.Series(train_custom_preds * 10000 ,name='Custom_Pred_Sales')
#results_train_df = pd.concat([sales_train_series,pred_sales_train_series],axis=1)
#results_train_df['custom_diff'] = results_train_df['Sales'] - results_train_df['Custom_Pred_Sales']
#results_train_df['custom_pe'] = np.abs(results_train_df['custom_diff'] / results_train_df['Sales'])
#
#toc=timeit.default_timer()
#print('Xgb Custom Objective Time',toc - tic)
#%%
#sales_cv_series = pd.Series(test['Sales'],index=test['Id'],name='Sales')
#pred_sales_cv_series = pd.Series(np.exp(test_preds) ,index=test['Id'],name='Pred_Sales')
#results_df_cv = pd.concat([sales_cv_series,pred_sales_cv_series],axis=1)
#%%

lasso_sales = result_series
#xgb_sales = pd.concat([xgb_custom_sales_promo,xgb_custom_sales_no_promo])

xgb_sales = xgb_custom_sales


#combined_sales = (0.5*xgb_sales + 0.5*lasso_sales)
#combined_sales = (0.5*xgb_custom_sales_3 + 0.5*lasso_sales)
#combined_sales = (0.1*xgb_sales + 0.39*lasso_sales + 0.5*xgb_custom_sales_2)
#combined_sales = (0.2*xgb_sales + 0.29*lasso_sales + 0.5*xgb_custom_sales_2 + 0.0*xgb_custom_sales_3)
#combined_sales = (0.15*xgb_sales + 0.395*lasso_sales + 0.4*xgb_custom_sales_2 + 0.05*xgb_custom_sales_3)
combined_sales = (0.25*xgb_sales + 0.295*lasso_sales + 0.4*xgb_custom_sales_2 + 0.05*xgb_custom_sales_3)
#combined_sales = (0.99*lasso_sales)
#combined_sales = (1.0*xgb_custom_sales_3)
#combined_sales = (1.0*lasso_sales)

xgb_sales_log = np.log(xgb_sales)
xgb_custom_sales_2_log = np.log(xgb_custom_sales_2)
lasso_sales_log = np.log(lasso_sales)

#combined_sales = np.exp((0.2 * xgb_sales_log + 0.2 * xgb_custom_sales_2_log +
#                            0.6*lasso_sales_log))

#combined_sales[store_1020_cond] = combined_sales[store_1020_cond] * 1.25
#combined_sales[store_304_cond] = combined_sales[store_304_cond] * 0.95
#combined_sales[store_137_cond] = combined_sales[store_137_cond] * 0.95
#combined_sales[store_944_cond] = combined_sales[store_944_cond] * 0.95
#combined_sales[store_1053_cond] = combined_sales[store_1053_cond] * 0.95
#combined_sales[store_131_cond] = combined_sales[store_131_cond] * 0.95
#combined_sales[store_718_cond] = combined_sales[store_718_cond] * 0.95
#combined_sales[store_269_cond] = combined_sales[store_269_cond] * 0.8
#combined_sales[store_550_cond] = combined_sales[store_550_cond] * 0.8

#902 5 bin 8
#770 4 bin 2
#1044 4 bin 6
#if(not use_cv):
#    store_902_cond = test_reduced['Store'] == 902
#    store_770_cond = test_reduced['Store'] == 770
#    store_1044_cond = test_reduced['Store'] == 1044
#    store_339_cond = test_reduced['Store'] == 339
#
#    combined_sales[store_902_cond] = combined_sales[store_902_cond] * 0.94
#    combined_sales[store_770_cond] = combined_sales[store_770_cond] * 0.86
#    combined_sales[store_1044_cond] = combined_sales[store_1044_cond] * 0.93
#    combined_sales[store_339_cond] = combined_sales[store_339_cond] * 0.9

#combined_sales[s_175_dates] = combined_sales[s_175_dates] * 0.5
#combined_sales[s_685_dates] = combined_sales[s_685_dates] * 0.5
#combined_sales[s_831_dates] = combined_sales[s_831_dates] * 0.5
#combined_sales[s_837_dates] = combined_sales[s_837_dates] * 0.5
#combined_sales[s_407_dates] = combined_sales[s_407_dates] * 0.5
#combined_sales[s_188_dates] = combined_sales[s_188_dates] * 0.5
#combined_sales[s_636_dates] = combined_sales[s_636_dates] * 0.5
#combined_sales[s_575_dates] = combined_sales[s_575_dates] * 0.5
#combined_sales[s_385_dates] = combined_sales[s_385_dates] * 0.5
#combined_sales[s_897_dates] = combined_sales[s_897_dates] * 0.5
#combined_sales[s_279_dates] = combined_sales[s_279_dates] * 0.5
#combined_sales[s_244_dates] = combined_sales[s_244_dates] * 0.5
#combined_sales[s_120_dates] = combined_sales[s_120_dates] * 0.5
#combined_sales[s_663_dates] = combined_sales[s_663_dates] * 0.5


combined_sales = combined_sales.sort_index()
test_reduced = test_reduced.sort_index()

if(use_cv):
    actual_sales = test_reduced.Sales
    cv_combined_rmpse = rmspe(actual_sales,combined_sales)
    print('lasso xgb combined',cv_combined_rmpse)

if(not use_cv):
    combined_sales.set_value(6202,500) #unusual value, set low to be conservative, store
    #probably closed
#    submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_preds)})
    submission = pd.DataFrame({"Id": test_reduced["Id"], "Sales": combined_sales.values})
    submission.to_csv("xgboost_submission.csv", index=False)
    print('xgb submission created')
#%%
#def get_stats(input_df,group_name):
#    group = input_df.groupby(group_name)
##    train_means = train_open_group.apply(np.mean)
#    return group.Sales.agg([np.mean,sp_stats.sem,np.std])
#train_stats = get_stats(train_open_df,'Store')
#train_stats_promo_0 = get_stats(train_open_df.loc[train_open_df.Promo == 0],'Store')
#train_stats_promo_1 = get_stats(train_open_df.loc[train_open_df.Promo == 1],'Store')
#p.clf()
#p.cla()
#plt.rcParams['figure.figsize'] = 10, 6
#plt.rcParams.update({'font.size': 22})
#plt.errorbar(train_stats.index,train_stats_promo_0['mean'],train_stats_promo_0['sem'],
#             fmt='o',label = 'Promo = 0')
#plt.errorbar(train_stats.index,train_stats_promo_1['mean'],train_stats_promo_1['sem'],
#             fmt='o',label = 'Promo = 1')
#plt.xlabel('Store')
#plt.ylabel('Mean Sales')
#plt.title('Mean Sales For Each Store')
#p.legend(numpoints = 1)
#x1,x2,y1,y2 = plt.axis()
#plt.axis((x1,x2,y1,y2+5000))
##plt.axis((200,400,y1,y2+5000))
#xticks, xticklabels = plt.xticks()
#xmin = (xticks[0] - xticks[1])/4.
#xmax = (xticks[-1] - xticks[-2])/2.
##plt.xlim(xmin, xmax)
#plt.xlim(0, 200)
#plt.xticks(xticks)
#
#pic_name = 'Images/'
#pic_name = pic_name + 'store_mean_sales.png'
#p.savefig(pic_name, bbox_inches='tight')

#%%
# Make Submission


#if(not use_cv):
#    result = pd.DataFrame({'Id': result_series.index.values, 'Sales': result_series.values}).set_index('Id')
#    result = result.sort_index()
#    result.to_csv('rossman_lasso.csv')
#    print('submission created')

#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)