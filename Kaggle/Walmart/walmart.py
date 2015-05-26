# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:25:55 2015

@author: Jared
"""
#notes: weekday could well be significant
#stores are closed on Christmas, store 30 asks for a prediction on Christmas,
#can use 0 for this
#very large gains probable from considering if an item is out of stock
#526917
#spyder doesn't like a dict of dataframes, becomes very slow with this
#item 68 good candidate for weather effect
#item 93! actually seems like weather matters! -nvm this item gets out of stock
#i005 store 37 has ridiculous values - unreal?

import csv as csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm, cross_validation
from sklearn.metrics import mean_squared_error
import sklearn.metrics as skm
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
#from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from datetime import datetime
import timeit

tic0=timeit.default_timer()
weather_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Walmart/weather.csv', header=0)
key_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Walmart/key.csv', header=0)
train_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Walmart/train.csv', header=0)
test_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Walmart/test.csv', header=0)
test3_df = test_df.drop('item_nbr', axis=1).drop_duplicates()
del test_df
#train_df = train_df[train_df.date != '2013-12-25']
toc=timeit.default_timer()
print('Load Time',toc - tic0)
#sample_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Walmart/sampleSubmission.csv', header=0)
#%%

#%%
#id_list = sample_df['id'].values
key_dict = key_df.set_index('store_nbr').to_dict()
key_list = key_df['station_nbr'].values.tolist()

#bad practice in general, but whatever should be ok
item_list = ('i001', 'i002', 'i003', 'i004', 'i005', 'i006', 'i007', 'i008',
       'i009', 'i010', 'i011', 'i012', 'i013', 'i014', 'i015', 'i016',
       'i017', 'i018', 'i019', 'i020', 'i021', 'i022', 'i023', 'i024',
       'i025', 'i026', 'i027', 'i028', 'i029', 'i030', 'i031', 'i032',
       'i033', 'i034', 'i035', 'i036', 'i037', 'i038', 'i039', 'i040',
       'i041', 'i042', 'i043', 'i044', 'i045', 'i046', 'i047', 'i048',
       'i049', 'i050', 'i051', 'i052', 'i053', 'i054', 'i055', 'i056',
       'i057', 'i058', 'i059', 'i060', 'i061', 'i062', 'i063', 'i064',
       'i065', 'i066', 'i067', 'i068', 'i069', 'i070', 'i071', 'i072',
       'i073', 'i074', 'i075', 'i076', 'i077', 'i078', 'i079', 'i080',
       'i081', 'i082', 'i083', 'i084', 'i085', 'i086', 'i087', 'i088',
       'i089', 'i090', 'i091', 'i092', 'i093', 'i094', 'i095', 'i096',
       'i097', 'i098', 'i099', 'i100', 'i101', 'i102', 'i103', 'i104',
       'i105', 'i106', 'i107', 'i108', 'i109', 'i110', 'i111')
#note that store 35 is missing!
store_list = ('s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09',
       's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18',
       's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27',
       's28', 's29', 's30', 's31', 's32', 's33', 's34', 's36',
       's37', 's38', 's39', 's40', 's41', 's42', 's43', 's44', 's45')
#%%
date_format = '%Y-%m-%d'
def get_days_opened(date):
    date_format = '%Y-%m-%d'
    d0 = datetime.strptime('2012-1-1', date_format)
    d1 = datetime.strptime(date, date_format)
    delta = d1 - d0
    return delta.days
get_days_opened_vec = np.vectorize(get_days_opened, otypes=[np.int])
def get_day(date):
    date_format = '%Y-%m-%d'
    d1 = datetime.strptime(date, date_format)
    return d1.day
get_day_vec = np.vectorize(get_day, otypes=[np.int])
def get_date_from_days(num_days):
    date_format = '%Y-%m-%d'
    d0 = pd.to_datetime('2012-1-1',date_format)
    enddate = d0 + pd.DateOffset(days=num_days)
    return enddate.strftime(date_format)

def get_high_weather(precip):
    if(precip == 'M' or precip == '  T'):
        return 0
    elif(float(precip) >= 0.5):
        return 1
    else:
        return 0
def get_precip(precip):
    if(precip == 'M' or precip == '  T'):
        return 0
    else:
        return np.float(precip)
def get_temp(temp):
    if(temp == 'M' ):
        return 60
    else:
        return np.float(temp)
    store_df['fall'] = store_df['month'].map(lambda x: 1 if (9 <= x < 12) else 0)
    store_df['winter'] = store_df['month'].map(lambda x: 1 if (x==12 or x==1 or x==2) else 0)
    store_df['spring'] = store_df['month'].map(lambda x: 1 if (3 <= x < 6) else 0)
    store_df['summer'] = store_df['month'].map(lambda x: 1 if (6 <= x < 9) else 0)
def get_season(month):
    if(month==12 or month==1 or month==2):
        return 0
    elif(3 <= month < 6):
        return 1
    elif(6 <= month < 9):
        return 2
    elif(9 <=  month < 12):
        return 3
    else:
        return -1
#%%
weather_df['high_precip'] = 0
weather_df['preciptotal'] = weather_df['preciptotal'].map(lambda x: get_precip(x))
#weather_df['high_precip'] = weather_df['preciptotal'].map(lambda x: 1 if x >= 0.75 else 0)
weather_df['high_precip'] = weather_df['preciptotal'].map(lambda x: 1 if (x >= 0.75) else 0)
weather_df['snowfall'] = weather_df['snowfall'].map(lambda x: get_precip(x))
#weather_df['high_snow'] = weather_df['snowfall'].map(lambda x: 1 if x >= 1.5 else 0)
weather_df['high_snow'] = weather_df['snowfall'].map(lambda x: 1 if x >= 1.5 else 0)
weather_df['avgspeed'] = weather_df['avgspeed'].map(lambda x: get_precip(x))
weather_df['windy'] = weather_df['avgspeed'].map(lambda x: 1 if x >= 18 else 0)
weather_df['temp_missing'] = weather_df['tavg'].map(lambda x: 1 if x == 'M' else 0)
weather_df['tavg'] = weather_df['tavg'].map(lambda x: get_temp(x))
weather_df['hot'] = weather_df['tavg'].map(lambda x: 1 if x >= 80 else 0)
weather_df['cold'] = weather_df['tavg'].map(lambda x: 1 if x <= 32 else 0)
weather_df['frigid'] = weather_df['tavg'].map(lambda x: 1 if x <= 15 else 0)
weather_df['thunder'] = weather_df['codesum'].map(lambda x: 1 if 'TS' in x else 0)
weather_df['snowcode'] = weather_df['codesum'].map(lambda x: 1 if 'SN' in x else 0)
weather_df['raincode'] = weather_df['codesum'].map(lambda x: 1 if 'RA' in x else 0)
#%%
# first goal is to rearrange data so it is easier to handle and visualize
# make a new table for each store, with a column for each item number (with
#units sold as the value)
#may also want to look at this list organized by item as opposed to store -
#for now lets just make it like that

tic=timeit.default_timer()
drop_list = ['date', 'weekday', 'precip', 'snow', 'wind', 'month', 'high_precip']
STORES_DF = {}
REDUCED = {}
drop_lists_dict = {}
date_format = '%Y-%m-%d'
unique_dates = train_df.date.unique()
unique_dates_int = get_days_opened_vec(unique_dates)
date_series = pd.Series(unique_dates, index=unique_dates_int)
weekday_series = date_series.map(lambda x: pd.to_datetime(x,date_format).weekday())
month_series = date_series.map(lambda x: pd.to_datetime(x,date_format).month)
num_of_days = len(unique_dates_int)
#%%
precip_date = []
snow_date = []
for date in unique_dates:
    precip_date.append(weather_df[weather_df['date'] == date]['preciptotal'].sum())
    snow_date.append(weather_df[weather_df['date'] == date]['snowfall'].sum())
all_precip_series = pd.Series(precip_date, index=unique_dates_int)
all_snow_series = pd.Series(snow_date, index=unique_dates_int)
all_high_precip = all_precip_series.map(lambda x: 1 if x >= 3.0 else 0)
all_high_snow = all_snow_series.map(lambda x: 1 if x >= 2.5 else 0)
#%%
for store in train_df.store_nbr.unique():
    if (store == 35):
        continue
    temp_df = train_df[train_df.store_nbr == store]
    store_df = pd.DataFrame()
    temp_weather_df = weather_df[weather_df.station_nbr == key_list[store - 1]]
    temp_weather_dates = get_days_opened_vec(temp_weather_df['date'])
    temp_weather_precip_series = pd.Series(temp_weather_df['preciptotal'].values,
                                           index = temp_weather_dates)
    temp_weather_high_precip_series = pd.Series(temp_weather_df['high_precip'].values,
                                           index = temp_weather_dates)
    temp_weather_snow_series = pd.Series(temp_weather_df['snowfall'].values,
                                           index = temp_weather_dates)
    temp_weather_high_snow_series = pd.Series(temp_weather_df['high_snow'].values,
                                           index = temp_weather_dates)
    temp_weather_wind_series = pd.Series(temp_weather_df['avgspeed'].values,
                                           index = temp_weather_dates)
    temp_weather_windy_series = pd.Series(temp_weather_df['windy'].values,
                                           index = temp_weather_dates)
    temp_weather_temp_series = pd.Series(temp_weather_df['tavg'].values,
                                           index = temp_weather_dates)
    temp_missing_series = pd.Series(temp_weather_df['temp_missing'].values,
                                           index = temp_weather_dates)
    temp_weather_hot_series = pd.Series(temp_weather_df['hot'].values,
                                           index = temp_weather_dates)
    temp_weather_cold_series = pd.Series(temp_weather_df['cold'].values,
                                           index = temp_weather_dates)
    temp_weather_frigid_series = pd.Series(temp_weather_df['frigid'].values,
                                           index = temp_weather_dates)
    temp_weather_thunder_series = pd.Series(temp_weather_df['thunder'].values,
                                           index = temp_weather_dates)
    temp_weather_snowcode_series = pd.Series(temp_weather_df['snowcode'].values,
                                           index = temp_weather_dates)
    temp_weather_raincode_series = pd.Series(temp_weather_df['raincode'].values,
                                           index = temp_weather_dates)
    store_df['date'] = date_series
    store_df['beginning'] = store_df.index.map(lambda x: 1 if x <= 365 else 0)
    store_df['day'] = store_df['date'].map(lambda x: get_day(x))
    store_df['early'] = store_df['day'].map(lambda x: 1 if x <= 6 else 0)
    store_df['late'] = store_df['day'].map(lambda x: 1 if x >= 26 else 0)
    store_df['middle'] = store_df['day'].map(lambda x: 1 if 11 <= x <= 14 else 0)
    store_df['weekday'] = weekday_series
    store_df['weekend'] = store_df['weekday'].map(lambda x: 1 if (4 <= x <= 6) else 0)
    store_df['sunday'] = store_df['weekday'].map(lambda x: 1 if x == 6 else 0)
    store_df['month'] = month_series
    store_df['quarter_start'] = store_df['month'].map(lambda x: \
                            1 if (x == 1 or x == 4 or x == 7 or x ==10) else 0)
    store_df['quarter_start_early'] = store_df['quarter_start'] * store_df['early']
    store_df['fall'] = store_df['month'].map(lambda x: 1 if (9 <= x < 12) else 0)
    store_df['winter'] = store_df['month'].map(lambda x: 1 if (x==12 or x==1 or x==2) else 0)
    store_df['spring'] = store_df['month'].map(lambda x: 1 if (3 <= x < 6) else 0)
    store_df['summer'] = store_df['month'].map(lambda x: 1 if (6 <= x < 9) else 0)
    store_df['season'] = store_df['month'].map(lambda x: get_season(x))
    store_df['thanksgiving'] = \
        store_df['date'].map(lambda x: 1 if (x == '2012-11-22' or x == '2013-11-28') else 0)
    store_df['halloween'] = store_df['date'].map(lambda x: 1 if \
        (x == '2012-10-31' or x == '2013-10-31' or x == '2014-10-31') else 0)
    store_df['easter'] = store_df['date'].map(lambda x: 1 if \
        (x == '2012-04-08' or x == '2013-03-31' or x == '2014-04-20') else 0)
    store_df['precip'] = temp_weather_precip_series
    store_df['raincode'] = temp_weather_raincode_series
    store_df['high_precip'] = temp_weather_high_precip_series
    store_df['thunder'] = temp_weather_thunder_series
    store_df['all_snow'] = all_snow_series
    store_df['all_precip'] = all_precip_series
    store_df['all_high_snow'] = all_high_snow
    store_df['all_high_precip'] = all_high_precip

    store_df['snow'] = temp_weather_snow_series
    store_df['snowcode'] = temp_weather_snowcode_series
    store_df['high_snow'] = temp_weather_high_snow_series
    store_df['wind'] = temp_weather_wind_series
    store_df['windy'] = temp_weather_windy_series
    store_df['temp'] = temp_weather_temp_series
    store_df['temp_missing'] = temp_missing_series
    for i in range(12):
        month_temp_ave = store_df[store_df['month'] == i+1]['temp'].mean()
        store_df.loc[store_df[(store_df['temp_missing'] == 1) & (store_df['month'] == i+1)].index, 'temp'] = month_temp_ave

    temp_diff = store_df['temp'].diff()
    temp_diff[0] = 0
    store_df['temp_diff'] = temp_diff
    store_df['temp_change_hot'] = store_df['temp_diff'].map(lambda x: 1 if x >= 10 else 0)
    store_df['temp_change_cold'] = store_df['temp_diff'].map(lambda x: 1 if x <= -10 else 0)

    store_df['hot'] = temp_weather_hot_series
    store_df['cold'] = temp_weather_cold_series
    store_df['frigid'] = temp_weather_frigid_series
    #store_df['code'] = temp_weather_code_series

    res_date_before_rain_array = np.zeros(num_of_days)
    res_date_after_rain_array = np.zeros(num_of_days)
    res_date_before_snow_array = np.zeros(num_of_days)
    res_date_after_snow_array = np.zeros(num_of_days)
    store_index = store_df.index
    for i in range(num_of_days-1):
#        res_date_before_rain_array[i] = store_df.ix[store_df.index[i+1]]['high_precip']
#        res_date_before_snow_array[i] = store_df.ix[store_df.index[i+1]]['high_snow']
        res_date_before_rain_array[i] = store_df.ix[store_df.index[i+1]]['raincode']
        res_date_before_snow_array[i] = store_df.ix[store_df.index[i+1]]['snowcode']
    date_before_series = pd.Series(res_date_before_rain_array, unique_dates_int)
    for i in range(1,num_of_days):
#        res_date_after_rain_array[i] = store_df.ix[store_df.index[i-1]]['high_precip']
#        res_date_after_snow_array[i] = store_df.ix[store_df.index[i-1]]['high_snow']
        res_date_after_rain_array[i] = store_df.ix[store_df.index[i-1]]['raincode']
        res_date_after_snow_array[i] = store_df.ix[store_df.index[i-1]]['snowcode']
    date_before_series = pd.Series(res_date_before_rain_array, unique_dates_int)
    date_after_series = pd.Series(res_date_after_rain_array, unique_dates_int)
    date_before_snow_series = pd.Series(res_date_before_snow_array, unique_dates_int)
    date_after_snow_series = pd.Series(res_date_after_snow_array, unique_dates_int)
    store_df['precip_next_day'] = date_before_series
    store_df['precip_previous_day'] = date_after_series
    store_df['snow_next_day'] = date_before_snow_series
    store_df['snow_previous_day'] = date_after_snow_series

    dates_index = get_days_opened_vec(temp_df['date'].unique())
    for item in temp_df.item_nbr.unique():
        temp2_df = temp_df[temp_df.item_nbr == item]
        i_name = 'i' + str(item).zfill(3)
        s = pd.Series(temp2_df['units'].values, index=dates_index)
        s = np.log(s+1)
        store_df[i_name] = s
    store_name = 's' + str(store).zfill(2)

    #dangerous, but values of 5000 seem unlikely
    if (store_name == 's37'):
        store_df.ix[319,'i005'] = 5.0
        store_df.ix[690,'i005'] = 5.0

    for item in item_list:
        res_last_array = np.zeros(num_of_days)
        res_next_array = np.zeros(num_of_days)
        res_last50_array = np.zeros(num_of_days)
        res_next50_array = np.zeros(num_of_days)
#        last5 = np.array((0, 0, 0, 0, 0))
        if(store_df[item].sum() == 0):
            test_series = pd.Series(res_last_array, unique_dates_int)
            stock_name = 'in_stock_'+item
            store_df[stock_name] = test_series
            continue
        x = store_df[item].values
        last7 = np.array((bool(x[0]), bool(x[1]), bool(x[2]), bool(x[3]),
                          bool(x[4]), bool(x[5]),bool(x[6])))
#        next7 = np.array((bool(x[-1]), bool(x[-2]), bool(x[-3]), bool(x[-4]),
#                  bool(x[-5]), bool(x[-6]),bool(x[-7])))
        next7 = np.array((bool(x[-1]), bool(x[-2]), bool(x[-3]), bool(x[-4]),
                  bool(x[-5]), bool(0), bool(0)))
        #dangerous for end or beginning values of zero, but probably not a big deal
        last50 = np.zeros(30)
        next50 = np.zeros(30)
        for i in range(num_of_days):
            val = x[i]
            if(np.isnan(val)):
                if(not all(last7 == 0)):
                    res_last_array[i] = 1
            elif(not all(last7 == 0) or (val != 0)):
                last7 = np.delete(np.append(last7,int(bool(val))),0)
                res_last_array[i] = 1
            else:
                last7 = np.delete(np.append(last7,int(bool(val))),0)

            if(np.isnan(val)):
                if(not all(last50 == 0)):
                    res_last50_array[i] = 1
            elif(not all(last50 == 0) or (val != 0)):
                last50 = np.delete(np.append(last50,int(bool(val))),0)
                res_last50_array[i] = 1
            else:
                last50 = np.delete(np.append(last50,int(bool(val))),0)


            #not sure if wise, but require also current value nan or nonzero when looking forward
            val_next = x[-(i+1)]
            if(np.isnan(val_next)):
                if(not all(next7 == 0)):
                    res_next_array[-(i+1)] = 1
            elif((not all(next7 == 0)) or (val_next != 0)):
                next7 = np.delete(np.append(next7,int(bool(val_next))),0)
                res_next_array[-(i+1)] = 1
            else:
                next7 = np.delete(np.append(next7,int(bool(val_next))),0)

            if(np.isnan(val_next)):
                if(not all(next50 == 0)):
                    res_next50_array[-(i+1)] = 1
            elif((not all(next50 == 0)) or (val_next != 0)):
                next50 = np.delete(np.append(next50,int(bool(val_next))),0)
                res_next50_array[-(i+1)] = 1
            else:
                next50 = np.delete(np.append(next50,int(bool(val_next))),0)

#        test_series = pd.Series(res_array, unique_dates_int)
#        test_next_series = pd.Series(res_next_array, unique_dates_int)
        combined_res_array = np.zeros(num_of_days)
        for i in range(len(res_last_array)):
            #if next 50 or last 50 are zero, assume it is zero (even if last or next7 are nonzero
            if(res_last50_array[i] == 0 or res_next50_array[i] == 0):
                continue
            if(res_last_array[i] or res_next_array[i]):
                combined_res_array[i] = 1

        test_combined_series = pd.Series(combined_res_array, unique_dates_int)
        stock_name = 'in_stock_'+item
        store_df[stock_name] = test_combined_series
    drop3_list = []
    temp_store_df = store_df[store_df.date != '2013-12-25']
    for item in item_list:
        res_last_array = np.zeros(num_of_days)
        res_next_array = np.zeros(num_of_days)
        res_last5w_array = np.zeros(num_of_days)
        res_next5w_array = np.zeros(num_of_days)
        res_weekday_ave = np.zeros(num_of_days)
        in_stock_str = 'in_stock_'+item
        last_week_name = 'last_week_'+item
        next_week_name = 'next_week_'+item
        last_5weeks_name = 'last5_weeks_'+item
        next_5weeks_name = 'next5_weeks_'+item
        weekday_mean_name = 'weekday_ave_'+item
        month_mean_name = 'month_ave_'+item
        season_mean_name = 'season_ave_'+item
        if(store_df[in_stock_str].sum() == 0):
            drop3_list.extend((in_stock_str,item,last_week_name, next_week_name,
                               last_5weeks_name,next_5weeks_name,
                               weekday_mean_name, month_mean_name,season_mean_name))
            zero_series = pd.Series(res_last_array, unique_dates_int)
            store_df[last_week_name] = zero_series
            store_df[next_week_name] = zero_series
            store_df[last_5weeks_name] = zero_series
            store_df[next_5weeks_name] = zero_series
            store_df[weekday_mean_name] = zero_series
            store_df[month_mean_name] = zero_series
            store_df[season_mean_name] = zero_series
            continue
        #in_stock_df = store_df[store_df[in_stock_str] == 1]
        in_stock_df = temp_store_df[temp_store_df[in_stock_str] == 1]
        in_stock_df = in_stock_df[(in_stock_df['easter'] == 0) & (in_stock_df['thanksgiving'] == 0)]
        in_stock_mean = in_stock_df[item].mean()
        in_stock_weekday_mean_list = []
        in_stock_month_mean_list = []
        in_stock_season_mean_list = []
        #NOTE, can probably make this faster, by getting a weekday df independent of items
        for i in range(7):
            in_stock_weekday_mean_list.append(in_stock_df[in_stock_df['weekday'] == i][item].mean())
        for i in range(12):
            temp_month_df = in_stock_df[in_stock_df['month'] == i+1][item]
            if(temp_month_df.dropna().empty):
#                in_stock_month_mean_list.append(0)
                in_stock_month_mean_list.append(in_stock_mean)
            else:
                in_stock_month_mean_list.append(temp_month_df.mean())
        for i in range(4):
            temp_season_df = in_stock_df[in_stock_df['season'] == i][item]
            if(temp_season_df.dropna().empty):
#                in_stock_season_mean_list.append(0)
                in_stock_season_mean_list.append(in_stock_mean)
            else:
                in_stock_season_mean_list.append(temp_season_df.mean())

        last_week = np.ones(7)*in_stock_mean
        next_week = np.ones(7)*in_stock_mean
#        last_five_weeks = np.ones(35)*in_stock_mean
#        next_five_weeks = np.ones(35)*in_stock_mean
        last_five_weeks = np.ones(21)*in_stock_mean
        next_five_weeks = np.ones(21)*in_stock_mean
        x = store_df[item].values
        for i in range(num_of_days):
            res_last_array[i] = last_week.mean()
            res_next_array[-(i+1)] = next_week.mean()
            res_last5w_array[i] = last_five_weeks.mean()
            res_next5w_array[-(i+1)] = next_five_weeks.mean()
            val = x[i]
            val_next = x[-(i+1)]
            if(not np.isnan(val)):
                last_week = np.delete(np.append(last_week,val),0)
                if(store_df[in_stock_str].iloc[i]):
                    last_five_weeks = np.delete(np.append(last_five_weeks,val),0)
            if(not np.isnan(val_next)):
                next_week = np.delete(np.append(next_week,val_next),0)
                if(store_df[in_stock_str].iloc[-(i+1)]):
                    next_five_weeks = np.delete(np.append(next_five_weeks,val_next),0)
        last_series = pd.Series(res_last_array, unique_dates_int)
        next_series = pd.Series(res_next_array, unique_dates_int)
        last5w_series = pd.Series(res_last5w_array, unique_dates_int)
        next5w_series = pd.Series(res_next5w_array, unique_dates_int)
        store_df[last_week_name] = last_series
        store_df[next_week_name] = next_series
        store_df[last_5weeks_name] = last5w_series
        store_df[next_5weeks_name] = next5w_series
        store_df[weekday_mean_name] = store_df['weekday'].map(lambda x: in_stock_weekday_mean_list[x])
        store_df[month_mean_name] = store_df['month'].map(lambda x: in_stock_month_mean_list[x-1])
        store_df[season_mean_name] = store_df['season'].map(lambda x: in_stock_season_mean_list[x])
    #store_df = store_df[store_df.date != '2013-12-25']

#    drop3_list = []
#    for item in item_list:
#        in_stock_str = 'in_stock_'+item
#        if(store_df[in_stock_str].sum() == 0):
#            drop3_list.extend((in_stock_str,item))
#    store_reduced = store_df.drop(drop3_list, axis=1)
#    REDUCED[store_name] = store_reduced
    drop_lists_dict[store_name] = drop3_list
    STORES_DF[store_name] = store_df
toc=timeit.default_timer()
print('Time',toc - tic)
train_panel = pd.Panel.from_dict(STORES_DF, orient='minor')
#free up some memory, spyder freezes when dicts of df in the variable explorer
del STORES_DF
del temp_df
#%%
#not really necessary, but whatever
del train_df
#del weather_df
#%%
#for exploratory purposes only
store01_df = train_panel.minor_xs('s01').drop(drop_lists_dict['s01'],axis=1)
store02_df = train_panel.minor_xs('s02').drop(drop_lists_dict['s02'],axis=1)
store03_df = train_panel.minor_xs('s03').drop(drop_lists_dict['s03'],axis=1)
store04_df = train_panel.minor_xs('s04').drop(drop_lists_dict['s04'],axis=1)
store05_df = train_panel.minor_xs('s05').drop(drop_lists_dict['s05'],axis=1)
store06_df = train_panel.minor_xs('s06').drop(drop_lists_dict['s06'],axis=1)
store10_df = train_panel.minor_xs('s10').drop(drop_lists_dict['s10'],axis=1)
store12_df = train_panel.minor_xs('s12').drop(drop_lists_dict['s12'],axis=1)
store14_df = train_panel.minor_xs('s14').drop(drop_lists_dict['s14'],axis=1)
store15_df = train_panel.minor_xs('s15').drop(drop_lists_dict['s15'],axis=1)
store16_df = train_panel.minor_xs('s16').drop(drop_lists_dict['s16'],axis=1)
store19_df = train_panel.minor_xs('s19').drop(drop_lists_dict['s19'],axis=1)
store20_df = train_panel.minor_xs('s20').drop(drop_lists_dict['s20'],axis=1)
store24_df = train_panel.minor_xs('s24').drop(drop_lists_dict['s24'],axis=1)
store29_df = train_panel.minor_xs('s29').drop(drop_lists_dict['s29'],axis=1)
store30_df = train_panel.minor_xs('s30').drop(drop_lists_dict['s30'],axis=1)
store32_df = train_panel.minor_xs('s32').drop(drop_lists_dict['s32'],axis=1)
store36_df = train_panel.minor_xs('s36').drop(drop_lists_dict['s36'],axis=1)
store37_df = train_panel.minor_xs('s37').drop(drop_lists_dict['s37'],axis=1)
store43_df = train_panel.minor_xs('s43').drop(drop_lists_dict['s43'],axis=1)
item9_df = train_panel['i009']
#%%
def make_store_hist(item, df, store):
    p.clf()
    p.cla()
    df[item].hist(bins=100, alpha = 0.5, label = item)
    plt.xlabel(item)
    plt.ylabel('Days')
    plt.title(store)
    pic_name = 'Images/Hists/' + store + '/'
    pic_name = pic_name + item + '_hist.png'
    p.savefig(pic_name, bbox_inches='tight')

def make_store_graph(item, df, store):
    p.clf()
    p.cla()
    df[item].plot(label = item)
    plt.xlabel('Date')
    plt.ylabel(item)
    plt.title(store)
    pic_name = 'Images/Hists/' + store + '/'
    pic_name = pic_name + item + '_graph.png'
    p.savefig(pic_name, bbox_inches='tight')
#p.figure()
#for store in train_panel.minor_axis:
#    for item in train_panel.minor_xs(store).drop('date', axis = 1).columns.values:
#        if(train_panel.minor_xs(store)[item].mean() > 0):
#            make_store_hist(item, train_panel.minor_xs(store).drop('date',
#                            axis = 1), store)
#            make_store_graph(item, train_panel.minor_xs(store).drop('date',
#                             axis = 1), store)

#%%
def make_item_graph(item, df):
    p.clf()
    p.cla()
    plt.plot(df.mean(), 'bo')
    plt.xlabel('Store')
    plt.ylabel(item + ' Mean')
    plt.title(item)
    pic_name = 'Images/Items/' + item + '/'
    pic_name = pic_name + item + '_stores.png'
    p.savefig(pic_name, bbox_inches='tight')
def make_items_panel(item, panel, subset_string, differ_string):
    p.clf()
    p.cla()
    in_stock_str = 'in_stock_'+item
#    plt.plot(panel[item][(panel[in_stock_str] == 1) & (panel['high_precip'] == 1)].mean(),'oc', label = 'high')
#    plt.plot(panel[item][(panel[in_stock_str] == 1) & (panel['high_precip'] == 0)].mean(),'or', label = 'normal')
    x = []
    for i in range(1,len(train_panel[item].columns)+1):
        if (i <= 34):
            x.append(i)
        else:
            x.append(i+1)
    #string = 'high_precip'
    error_high_precip = panel[item][(panel[in_stock_str] == 1) & (panel[subset_string] >= 0) &
                                (panel[differ_string] == 1)].sem()
    error_low_precip = panel[item][(panel[in_stock_str] == 1) & (panel[subset_string] >= 0)
                                & (panel[differ_string] == 0)].sem()
    plt.errorbar(x,panel[item][(panel[in_stock_str] == 1) & (panel[subset_string] >= 0)
                                & (panel[differ_string] == 1)].mean(),
                 yerr = error_high_precip, fmt='o', color='c', label = 'high')
    plt.errorbar(x,panel[item][(panel[in_stock_str] == 1) & (panel[subset_string] >= 0)
                                & (panel[differ_string] == 0)].mean(),
                 yerr = error_low_precip, fmt='o', color='r', label = 'normal')
    plt.xlabel('Store')
    plt.ylabel(item + ' Mean')
    plt.title(item)
    p.legend(numpoints = 1)
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,y2))
    xticks, xticklabels = plt.xticks()
    # shift half a step to the left
    # x0 - (x1 - x0) / 2 = (3 * x0 - x1) / 2
    xmin = (3*xticks[0] - xticks[1])/2.
    # shaft half a step to the right
    xmax = (3*xticks[-1] - xticks[-2])/2.
    plt.xlim(xmin, xmax)
    plt.xticks(xticks)
    pic_name = 'Images/Items/' + item + '/'
    pic_name = pic_name + item + '_stores.png'
    p.savefig(pic_name, bbox_inches='tight')
#%%
#p.figure()
#for item in train_panel.items.drop(drop_list):
#    make_item_graph(item, train_panel[item])
#for item in train_panel.items.drop(drop_list):
    #make_item_from_panel_graph(item, train_weekday_panel_list)
    #make_item_month_panel(item, train_month_panel_list)
#%%
#for item in item_list:
#    make_items_panel(item, train_panel, 'summer','snowcode')
#%%
#
#dates_sum = []
#for i in train_panel.major_axis:
#    temp_date_df = train_panel.major_xs(i)
#    date = temp_date_df.loc['s01','date']
#    date_int = get_days_opened(date)
#    this_sum = 0
#    for item in item_list:
#        if(item == 'i093'):
#            continue
#        if(item != 'i009'):
#            continue
#
#        this_sum = this_sum + temp_date_df[item].mean()
##    if(this_sum <= 1.0):
##        print(date,this_sum)
##    if(this_sum <= 7.5):
##        print(date,this_sum)
#    dates_sum.append(this_sum)
#plt.plot( unique_dates_int,dates_sum,linestyle='None', color='blue', marker='o')
#%%

#%%
test3_df['weekday'] = test3_df['date'].map(lambda x: pd.to_datetime(x,date_format).weekday())
test3_df['date_int'] = test3_df['date'].map(lambda x: get_days_opened(x))
#%%
tic=timeit.default_timer()
random.seed(4)
RES = {}
rmse_total = 0
se_total = 0
store_list2 = ['s43']
#for store in store_list2:
#p.figure()
#5,9,16,37,45
num_items = 0
num_test = 0
for store in store_list:
    results_df = pd.DataFrame()
    store_df = train_panel.minor_xs(store)
    temp_store = store_df.dropna()
    temp_store = temp_store[temp_store.date != '2013-12-25']
    for item in item_list:
        #item 093 irrelavant always out of stock in test set
        if(item == 'i093'):
            continue
        in_stock_str = 'in_stock_'+item
        last_week_name = 'last_week_'+item
        next_week_name = 'next_week_'+item
        last_5weeks_name = 'last5_weeks_'+item
        next_5weeks_name = 'next5_weeks_'+item
        weekday_mean_name = 'weekday_ave_'+item
        month_mean_name = 'month_ave_'+item
        season_mean_name = 'season_ave_'+item
        if(temp_store[in_stock_str].sum() == 0):
            continue
        store_df_reduced = temp_store[temp_store[in_stock_str] == 1]
        rows = random.sample(range(len(store_df_reduced.index)),
                             int(len(store_df_reduced.index)/5)+1)
        df_test = store_df_reduced.ix[store_df_reduced.index[rows]]
        df_train = store_df_reduced.drop(store_df_reduced.index[rows])
        #to use full training data uncomment next line
#        df_train = store_df_reduced
#        regr = LinearRegression()
        cv_l = cross_validation.KFold(len(df_train), n_folds=10, shuffle=True,
                                      random_state = 1)
        regr = LassoCV(cv=cv_l, n_jobs = 2)
#        regr = RidgeCV(cv=cv_l)
        columns_list = [item,
                        weekday_mean_name,
                        last_week_name,
                        next_week_name,
#                        season_mean_name,
                        last_5weeks_name,
                        next_5weeks_name,
                        month_mean_name,
#                        'sunday',
#                        'quarter_start_early',
#                        'winter',
#                        'summer',
#                        'fall',
#                        'weekend',
#                        'high_precip',
#                        'raincode',
#                        'thunder',
#                        'precip_next_day',
#                        'precip_previous_day',
                        'high_snow',
#                        'snowcode',
#                        'snow_next_day',
#                        'snow_previous_day',
#                        'windy',
#                        'temp_change_cold',
#                        'halloween',
#                        'beginning',
                        'easter',
                        'thanksgiving',
                        'early',
                        'late',
#                        'middle',
#                        'temp_change_hot',
#                        'hot',
#                        'frigid',
#                        'all_high_snow',
#                        'all_high_precip',
                        'cold'
                        ]
        X_total = store_df[columns_list]
        X_train = df_train[columns_list]
        X_test = df_test[columns_list]

        total_data = X_total.values
        train_data = X_train.values
        test_data = X_test.values
        regr = regr.fit( train_data[0::,1::], train_data[0::,0] )
        #print(regr.alpha_,store,item)
        prediction = regr.predict(test_data[0::,1::])
        prediction = np.maximum(prediction, 0.)
        prediction_total = regr.predict(total_data[0::,1::])
        prediction_total = np.maximum(prediction_total, 0.)
        total_series = pd.Series(prediction_total, unique_dates_int)

        rmse = np.sqrt(((test_data[0::,0] - prediction) ** 2).mean())
        se = ((test_data[0::,0] - prediction) ** 2).sum()
#        print(rmse,store,item)
        rmse_total = rmse_total + rmse
        se_total = se_total + se

#        plt.scatter(df_test.index,test_data[0::,0] - prediction)
#        plt.xlabel('date')
#        plt.xlim(0,1050)
#        plt.ylabel('truth - pred')
#        plt.title(store + ' ' + item)
#        pic_name = 'Images/Residuals/' + store + '_' + item + '.png'
#        p.savefig(pic_name, bbox_inches='tight')
#        p.clf()
#        p.cla()
        num_items = num_items + 1
        num_test = num_test + len(test_data[0::,0])
        results_df['date'] = store_df['date']
        results_df[item] = store_df[item]
        res_name = 'res_'+item
        results_df[res_name] = total_series
        results_df[in_stock_str] = store_df[in_stock_str]
        results_df[res_name] = results_df[res_name] * store_df[in_stock_str]
    RES[store] = results_df
print('rmse_test_total: ',rmse_total)
print('se_total:',se_total)
print('num_items',num_items,'len_of_test',num_test)
print('Average rmse:',rmse_total / num_items )
print('Average se:', se_total / num_test)
results_store1 = RES['s01']
results_store2 = RES['s02']
results_store3 = RES['s03']
toc=timeit.default_timer()
print('Time',toc - tic)
#%%
#rmse_total = 0
#for item in set(item_list).intersection(results_store1.columns.values):
#    res_name = 'res_'+item
#    rmse = np.sqrt(((results_store1[item] - results_store1[res_name]) ** 2).mean(axis=1))
#    rmse_total = rmse_total + rmse
#print(rmse_total)
#%%
tic=timeit.default_timer()
pred = np.zeros(526917)
dates = [None] * 526917
i = 0
num_nonzero = 0
for st in test3_df.store_nbr.unique():
    temp_df = test3_df[test3_df.store_nbr == st]
    store = 's' + str(st).zfill(2)
#    print(store,len(temp_df.index))
    for index in temp_df.index.values:
        date = temp_df.get_value(index,'date')
        date_int = temp_df.get_value(index,'date_int')
        weekday = temp_df.get_value(index,'weekday')
        train_store = train_panel.minor_xs(store)
        result_store = RES[store]
        row = train_store.ix[date_int]
        results_row = result_store.ix[date_int]
        for it in range(len(item_list)):
            data_str = str(st) + '_' + str(it+1) + '_' + date
            item = item_list[it]
            in_stock_str = 'in_stock_'+item
            res_name = 'res_'+item_list[it]
            weekday_mean_name = 'weekday_ave_'+item
            if(row[in_stock_str] == 0):
                dates[i] = data_str
                pred[i] = 0
            elif(date == '2013-12-25'):
                dates[i] = data_str
                pred[i] = 0
            else:
                dates[i] = data_str
                num_nonzero = num_nonzero + 1
                if(item == 'i093'):
                    print(data_str)
#                if(store == 's45'):
#                    print(data_str)
#                pred[i] = np.exp(row[weekday_mean_name]) - 1
                pred[i] = np.exp(results_row[res_name]) - 1
            i=i+1
print('number of nonzero predictions',num_nonzero)
print ('Predicting...')
predictions_file = open('/Users/Jared/DataAnalysis/Kaggle/Walmart/walmart.csv',
                        'wt')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['id','units'])
open_file_object.writerows(zip(dates,pred))
predictions_file.close()
print ('Done.')
toc=timeit.default_timer()
print('Time',toc - tic)
toc=timeit.default_timer()
print('Total Time',toc - tic0)
