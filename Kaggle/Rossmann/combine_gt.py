# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:28:36 2015

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
def collapse_weekly(weekly_df, col_name = 'adj_factor'):
    weekly_df['row'] = range(len(weekly_df))
    starts = weekly_df[['Week_Start', col_name, 'row']].rename(columns={'Week_Start': 'date'})
    ends = weekly_df[['Week_End', col_name, 'row']].rename(columns={'Week_End':'date'})
    df_collapsed = pd.concat([starts, ends])
    df_collapsed = df_collapsed.set_index('row', append=True)
    df_collapsed.sort_index()

    df_collapsed = df_collapsed.groupby(level=[0,1]).apply(lambda x:
        x.set_index('date').resample('D').fillna(method='pad'))

    df_collapsed = df_collapsed.reset_index(level=1, drop=True)
    df_collapsed = df_collapsed.reset_index(level=0)

    df_collapsed['Day'] = df_collapsed.index
    df_collapsed.sort('Day',inplace=True)
    return df_collapsed

def combine_trends(daily_file_path = 'daily_input.csv', weekly_file_path = 'weekly_input.csv',output_path = 'GoogleTrends/output.csv',
                   geo_name = ''):
    trends_daily = pd.read_csv(daily_file_path, header=0)
    trends_weekly = pd.read_csv(weekly_file_path, header=0)
    trends_daily['Day'] = pd.to_datetime(trends_daily['Day'])

    trends_weekly['Week_Start'] = trends_weekly['Week'].map(lambda x: x.split(" - ",1)[0])
    trends_weekly['Week_End'] = trends_weekly['Week'].map(lambda x: x.split(" - ",1)[1])
    trends_weekly['Week_Start'] = pd.to_datetime(trends_weekly['Week_Start'])
    trends_weekly['Week_End'] = pd.to_datetime(trends_weekly['Week_End'])
    trends_weekly = trends_weekly.rename(columns={'rossmann': 'weekly_rossmann'})

    ##correct zero or missing daily values by imputing the weekly value
    collapsed_weekly = collapse_weekly(trends_weekly,col_name = 'weekly_rossmann')
    trends_daily_temp = pd.merge(trends_daily,collapsed_weekly,how='right',left_on = 'Day',
                      right_on = 'Day')
    zero_daily_data = trends_daily_temp.rossmann == 0
    missing_daily_data = trends_daily_temp.rossmann.isnull()
    missing_or_zero = missing_daily_data | zero_daily_data
    trends_daily_temp.rossmann[missing_or_zero] = trends_daily_temp.weekly_rossmann[missing_or_zero]
    trends_daily = trends_daily_temp[['Day','rossmann']]
    trends = pd.merge(trends_daily,trends_weekly,how='left',left_on = 'Day',
                      right_on = 'Week_Start')
    trends.dropna(inplace=True)
    trends['adj_factor'] = trends['weekly_rossmann'] / trends['rossmann']

    collapsed_adj_factor = collapse_weekly(trends)

    trends_daily.sort('Day',inplace=True)
    combined = pd.merge(trends_daily,collapsed_adj_factor,
                        on = ['Day'],how='left')
    combined.sort('Day')
    combined.dropna(inplace=True)
    geo_key = 'daily_rossmann_'+geo_name
    combined[geo_key] = combined['rossmann'] * combined['adj_factor']
    combined[geo_key] = combined[geo_key] / combined[geo_key].max() * 100
    combined = combined[['Day',geo_key]]

    return combined

#gt_de = combine_trends('GoogleTrends/gt_rossmann_DE/daily_combined.csv',
#                       'GoogleTrends/gt_rossmann_DE/weekly.csv',geo_name = 'DE')
#%%
geo_list = ['DE', 'DE-BW','DE-HH', 'DE-SH','DE-BY','DE-NI', 'DE-SN', 'DE-HB', 'DE-NW',
            'DE-ST','DE-BE', 'DE-HE', 'DE-RP', 'DE-TH']
df_dict = {}
for geo in geo_list:
    daily_input_file = 'GoogleTrends/gt_rossmann_'+geo+'/daily_combined.csv'
    weekly_input_file = 'GoogleTrends/gt_rossmann_'+geo+'/weekly.csv'
    df_dict[geo] = combine_trends(daily_input_file,weekly_input_file,geo_name=geo)
#%%
i = 0
for key in df_dict:
    i=i+1
    if(i == 1):
        result_df = df_dict[key]
        continue
    result_df = pd.merge(df_dict[key],result_df,on='Day',how='left')
#reorder columns so DE is second column
result_df = result_df[['Day','daily_rossmann_DE', 'daily_rossmann_DE-SN', 'daily_rossmann_DE-RP',
       'daily_rossmann_DE-BE', 'daily_rossmann_DE-HB', 'daily_rossmann_DE-BW',
        'daily_rossmann_DE-ST', 'daily_rossmann_DE-NI',
       'daily_rossmann_DE-SH', 'daily_rossmann_DE-TH', 'daily_rossmann_DE-HE',
       'daily_rossmann_DE-HH', 'daily_rossmann_DE-BY', 'daily_rossmann_DE-NW']]
result_df.to_csv('daily_google_trends.csv')
#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)





#%%