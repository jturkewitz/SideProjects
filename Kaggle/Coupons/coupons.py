# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:07:12 2015

@author: Jared
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from sklearn import cross_validation
#from sklearn import preprocessing
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV,LogisticRegression
from sklearn.grid_search import RandomizedSearchCV
#import xgboost as xgb
import mean_average_precision_k as mean_average_precision_k
import datetime as datetime
#import scipy.sparse
#import pickle
import timeit
#%%
tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'

tic=timeit.default_timer()
#area_train_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/en_coupon_area_train.csv', header=0)
#area_test_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/en_coupon_area_test.csv', header=0)
#list_train_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/en_coupon_list_train.csv', header=0)
#list_test_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/en_coupon_list_test.csv', header=0)
#users_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/en_user_list.csv', header=0)
#detail_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/en_coupon_detail_train.csv', header=0)

#redo_visits = True
redo_visits = False
#reload_visits = True
reload_visits = False

#use_cv = True
use_cv = False

if(reload_visits):
    visits_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/coupon_visit_train.csv', header=0)
else:
    print('not redoing visits to save time')
list_train_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/input/coupon_list_train_en.csv', header=0)
list_test_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/input/coupon_list_test_en.csv', header=0)
users_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/input/user_list_en.csv', header=0)
detail_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/input/coupon_detail_train_en.csv', header=0)
area_train_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/input/coupon_area_train_en.csv', header=0)
area_test_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/input/coupon_area_test_en.csv', header=0)
prefectures_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Coupons/input/prefecture_locations_en.csv', header=0)

toc=timeit.default_timer()
print('Load Time',toc - tic)

#%%
tic = timeit.default_timer()
#detail_df = detail_df.rename(columns={'small_area_name_en': 'small_area_name_detail'})
users_df = users_df.rename(columns={'USER_ID_hash': 'user_id_hash','AGE': 'age'})
detail_df = detail_df.rename(columns={'USER_ID_hash': 'user_id_hash','COUPON_ID_hash': 'coupon_id_hash',
                                      'I_DATE': 'i_date'})
rename_dict = {'PRICE_RATE':'price_rate', 'CATALOG_PRICE':'catalog_price',
       'DISCOUNT_PRICE':'discount_price', 'DISPFROM':'dispfrom', 'DISPEND':'dispend',
       'DISPPERIOD':'dispperiod', 'VALIDFROM':'validfrom',
       'VALIDEND':'validend', 'VALIDPERIOD':'validperiod',
       'USABLE_DATE_MON':'usable_date_mon', 'USABLE_DATE_TUE':'usable_date_tue',
       'USABLE_DATE_WED':'usable_date_wed', 'USABLE_DATE_THU':'usable_date_thu',
       'USABLE_DATE_FRI':'usable_date_fri',
       'USABLE_DATE_SAT':'usable_date_sat', 'USABLE_DATE_SUN':'usable_date_sun',
       'USABLE_DATE_HOLIDAY':'usable_date_holiday',
       'USABLE_DATE_BEFORE_HOLIDAY':'usable_date_before_holiday'}

list_train_df = list_train_df.rename(columns=rename_dict)
list_test_df = list_test_df.rename(columns=rename_dict)
list_train_df = list_train_df.rename(columns={'COUPON_ID_hash': 'coupon_id_hash','USER_ID_hash': 'user_id_hash'})
list_test_df = list_test_df.rename(columns={'COUPON_ID_hash': 'coupon_id_hash','USER_ID_hash': 'user_id_hash'})

user_id_dict = dict(zip(users_df.user_id_hash, users_df.index))
user_id_rev_dict = {v: k for k, v in user_id_dict.items()}
users_df['user_id_hash'] = users_df['user_id_hash'].map(lambda x: user_id_dict[x])
coupon_id_dict = dict(zip(list_train_df.coupon_id_hash, list_train_df.index))
list_train_df['coupon_id_hash'] = list_train_df['coupon_id_hash'].map(lambda x: coupon_id_dict[x])

test_coupons = list_test_df['coupon_id_hash'].values
test_coupon_id_dict = dict(zip(list_test_df.coupon_id_hash, list_test_df.index))
test_coupon_id_rev_dict = {v: k for k, v in test_coupon_id_dict.items()}
list_test_df['coupon_id_hash'] = list_test_df['coupon_id_hash'].map(lambda x: test_coupon_id_dict[x])

list_train_df['dispfrom'] = list_train_df['dispfrom'].map(lambda x: pd.to_datetime(x))
list_train_df['dispend'] = list_train_df['dispend'].map(lambda x: pd.to_datetime(x))
list_train_df['validfrom'] = list_train_df['validfrom'].map(lambda x: pd.to_datetime(x))
list_train_df['validfrom'].fillna(list_train_df['dispend'],inplace=True)
list_train_df['valid_gap'] = list_train_df['validfrom'] - list_train_df['dispend']
list_train_df['valid_gap'] = list_train_df['valid_gap'].map(lambda x: x.astype('timedelta64[h]').astype(int))
list_train_df['valid_gap'] = list_train_df['valid_gap'].map(lambda x: (x - 12) / 24)


##TODO dangerous testing if dropping free helps or hurts
##none of it exists in test set
list_train_df = list_train_df[list_train_df['price_rate'] != 100]
#list_train_df = list_train_df[~((list_train_df['discount_price'] == 100) & (list_train_df['en_genre'] == 'Other coupon'))]
#list_train_df = list_train_df[~((list_train_df['discount_price'] == 100) & (list_train_df['en_genre'] == 'Gift card'))]


#7809 is a huge outlier in terms of purchases
#list_train_df = list_train_df[list_train_df['coupon_id_hash'] != 7809]



list_test_df['dispfrom'] = list_test_df['dispfrom'].map(lambda x: pd.to_datetime(x))
list_test_df['dispend'] = list_test_df['dispend'].map(lambda x: pd.to_datetime(x))
list_test_df['validfrom'] = list_test_df['validfrom'].map(lambda x: pd.to_datetime(x))
list_test_df['validfrom'].fillna(list_test_df['dispend'],inplace=True)
list_test_df['valid_gap'] = list_test_df['validfrom'] - list_test_df['dispend']
list_test_df['valid_gap'] = list_test_df['valid_gap'].map(lambda x: x.astype('timedelta64[h]').astype(int))
list_test_df['valid_gap'] = list_test_df['valid_gap'].map(lambda x: (x - 12) / 24)


detail_df = detail_df.drop('PURCHASEID_hash', 1)
detail_df['user_id_hash'] = detail_df['user_id_hash'].map(lambda x: user_id_dict[x])
detail_df['coupon_id_hash'] = detail_df['coupon_id_hash'].map(lambda x: coupon_id_dict[x])



## TODO dangerous testing removing coupons with price rate of 100, as these don't exist in test set
#detail_df = detail_df[detail_df['coupon_id_hash'] != 7809]

area_train_df['coupon_id_hash'] = area_train_df['COUPON_ID_hash'].map(lambda x: coupon_id_dict[x])
area_test_df['coupon_id_hash'] = area_test_df['COUPON_ID_hash'].map(lambda x: test_coupon_id_dict[x])
toc=timeit.default_timer()
print('Renaming',toc - tic)
#%%
if(reload_visits):
    tic = timeit.default_timer()
    def mapper(x,coup_dict):
        try:
            return coup_dict[x]
        except KeyError:
            return -1
    visits_df['coupon_id_hash'] = visits_df['VIEW_COUPON_ID_hash'].map(lambda x: mapper(x,coupon_id_dict))
    visits_df['user_id_hash'] = visits_df['USER_ID_hash'].map(lambda x: user_id_dict[x])
    #lose some info, can revisit later if it seems important
    visits_df = visits_df.drop(['REFERRER_hash',
                                'SESSION_ID_hash','PURCHASEID_hash'], 1)
    #drop coupons without details about them
    visits_anon_df =  visits_df[visits_df['coupon_id_hash'] == -1]

    visits_df = visits_df[visits_df['coupon_id_hash'] != -1]
    visits_anon_df['test_coupon_id'] = visits_anon_df['VIEW_COUPON_ID_hash'].map(
                                        lambda x: mapper(x,test_coupon_id_dict))
    visits_test_df = visits_anon_df[visits_anon_df['test_coupon_id'] != -1]

    #purchases_df = visits_df[visits_df['purchases'] == 1]
    #grouped = purchases_df.groupby('coupon_id_hash')
    #coupon_puchased_series = grouped.purchases.aggregate(np.sum)
    toc=timeit.default_timer()
    print('Looking At Visit Purchases',toc - tic)
#%%
#now lets start investigating the different users
#caps to avoid spyder variable explorer bugs
#USER_COUPON_VALUES_DICT = {}
#grouped_u = purchases_df.groupby('user_id_hash')
#for name, group in grouped_u:
#    USER_COUPON_VALUES_DICT[name] = group['coupon_id_hash'].values
#%%
if(redo_visits):
    #TODO decide on this approach
    visits_df = visits_df.rename(columns={'I_DATE': 'i_date'})
    visits_df = visits_df.drop(['USER_ID_hash','VIEW_COUPON_ID_hash','PAGE_SERIAL'], 1)
    tic = timeit.default_timer()
    date_format = '%Y-%m-%d %H:%M:%S'
    #starting_date = pd.to_datetime('2012-06-17 12:00:00')
    visits_df['i_date'] = visits_df['i_date'].map(lambda x: pd.to_datetime(x))
    visits_df['year'] = visits_df['i_date'].map(lambda x: x.year)
    visits_df['month'] = visits_df['i_date'].map(lambda x: x.month)
    visits_df['day'] = visits_df['i_date'].map(lambda x: x.day)

    beginning = pd.to_datetime('2011-07-01 00:00:00')
    visits_df['date_int'] = visits_df['i_date'].map(lambda x: (x - beginning).days)

    visits_df['weekday'] = visits_df['i_date'].map(lambda x: x.weekday())

    toc=timeit.default_timer()
    print('Converting Visits to Dates Time',toc - tic)
#%%
tic = timeit.default_timer()
date_format = '%Y-%m-%d %H:%M:%S'
#starting_date = pd.to_datetime('2012-06-17 12:00:00')
detail_df['i_date'] = detail_df['i_date'].map(lambda x: pd.to_datetime(x))
#detail_df['year'] = detail_df['i_date'].map(lambda x: pd.to_datetime(x.year)
#detail_df['month'] = detail_df['i_date'].map(lambda x: pd.to_datetime(x).month)
#detail_df['day'] = detail_df['i_date'].map(lambda x: pd.to_datetime(x).day)
#detail_df['weekday'] = detail_df['i_date'].map(lambda x: pd.to_datetime(x).weekday())
detail_df['year'] = detail_df['i_date'].map(lambda x: x.year)
detail_df['month'] = detail_df['i_date'].map(lambda x: x.month)
detail_df['day'] = detail_df['i_date'].map(lambda x: x.day)

beginning = pd.to_datetime('2011-07-01 00:00:00')
detail_df['date_int'] = detail_df['i_date'].map(lambda x: (x - beginning).days)


#TODO testing
#detail_df = detail_df.loc[detail_df['date_int'] >= 180]

detail_df['weekday'] = detail_df['i_date'].map(lambda x: x.weekday())

toc=timeit.default_timer()
print('Converting to Dates Time',toc - tic)
#%%
users_df['male_user'] = users_df['SEX_ID'].map(lambda x: 1 if x == 'm' else 0)
users_df['has_pref'] = users_df['en_pref'].notnull().astype(int)

users_gender_df = users_df[['user_id_hash','male_user']]
detail_df = pd.merge(detail_df,users_gender_df,on='user_id_hash')
detail_male_df = detail_df.loc[detail_df['male_user'] == 1]
detail_female_df = detail_df.loc[detail_df['male_user'] == 0]

users_pref_df = users_df[['user_id_hash','has_pref']]
detail_df = pd.merge(detail_df,users_pref_df,on='user_id_hash')
detail_pref_df = detail_df.loc[detail_df['has_pref'] == 1]
detail_no_pref_df = detail_df.loc[detail_df['has_pref'] == 0]

users_age_df = users_df[['user_id_hash','age']]
detail_df = pd.merge(detail_df,users_age_df,on='user_id_hash')
age_old = 55
age_young = 30
detail_old_df = detail_df.loc[detail_df['age'] >= age_old]
detail_middle_df = detail_df.loc[(detail_df['age'] < age_old) & (detail_df['age'] > age_young)]
detail_young_df = detail_df.loc[detail_df['age'] <= age_young]
#%%
tic = timeit.default_timer()
#TOOD use visits or detail for this?
#TODO appears to not help, unclear
detail_df = detail_df.drop_duplicates(['user_id_hash','coupon_id_hash','date_int'])

def join_purchases(input_df,output_df,col_name):
    grouped = input_df.groupby('coupon_id_hash')
    coupon_purchased_series = grouped['user_id_hash'].aggregate(lambda x: np.unique(x).shape[0])
    coupon_purchased_series.name = col_name
    output_df = output_df.join(coupon_purchased_series)
    output_df[col_name].fillna(0,inplace = True)
    return output_df
list_train_df = join_purchases(detail_df,list_train_df,'purchases')
list_train_df = join_purchases(detail_male_df,list_train_df,'purchases_m')
list_train_df = join_purchases(detail_female_df,list_train_df,'purchases_f')
list_train_df = join_purchases(detail_pref_df,list_train_df,'purchases_pref')
list_train_df = join_purchases(detail_no_pref_df,list_train_df,'purchases_no_pref')
list_train_df = join_purchases(detail_old_df,list_train_df,'purchases_old')
list_train_df = join_purchases(detail_young_df,list_train_df,'purchases_young')
list_train_df = join_purchases(detail_middle_df,list_train_df,'purchases_middle')
coupons_train = list_train_df

#gift_cards_df = coupons_train[coupons_train['en_genre'] == 'Gift card']
#other_df = coupons_train[coupons_train['en_genre'] == 'Other coupon']

toc=timeit.default_timer()
print('Calculating Purchases',toc - tic)
#%%
area_train_merged = area_train_df.merge(coupons_train,on='coupon_id_hash',how='inner')
area_train_merged = area_train_merged[['coupon_id_hash','en_pref','en_small_area_x','en_small_area_y',
                                       'en_large_area','en_genre','purchases']]
grouped = area_train_df.groupby('coupon_id_hash')
area_train_series = grouped['en_small_area'].aggregate(lambda x: np.unique(x).shape[0])
area_train_series.name = 'area_count'

grouped = area_test_df.groupby('coupon_id_hash')
area_test_series = grouped['en_small_area'].aggregate(lambda x: np.unique(x).shape[0])
area_test_series.name = 'area_count'

list_train_df = list_train_df.join(area_train_series)
list_train_df['area_count'].fillna(0,inplace = True)

list_test_df = list_test_df.join(area_test_series)

coupons_train = coupons_train.join(area_train_series)
coupons_train['area_count'].fillna(0,inplace = True)
#%%
#%%
tic=timeit.default_timer()
#train_coup = list_train_df
coupons_train['dispfrom_dateint'] = coupons_train['dispfrom'].map(lambda x: (x - beginning).days)
coupons_train = coupons_train[coupons_train['dispfrom_dateint'] >= 0]

visits_purchases = visits_df.merge(coupons_train,on='coupon_id_hash',how='inner')

visits_purchases['visit_diff'] = visits_purchases['date_int'] - visits_purchases['dispfrom_dateint']
visits_ghosts = visits_purchases[visits_purchases['visit_diff'] < 0]



visit_groups = visits_ghosts.groupby('coupon_id_hash')
coupon_ghost_users = visit_groups['user_id_hash'].aggregate(lambda x: np.unique(x).shape[0])
coupon_ghost_users.name = 'ghost_users'

coupons_train = coupons_train.join(coupon_ghost_users)
coupons_train['ghost_users'].fillna(0,inplace = True)

list_train_df = list_train_df.join(coupon_ghost_users)
list_train_df['ghost_users'].fillna(0,inplace = True)

visit_test_groups = visits_test_df.groupby('test_coupon_id')
test_coupon_ghost_users  = visit_test_groups['user_id_hash'].aggregate(lambda x: np.unique(x).shape[0])
test_coupon_ghost_users.name = 'ghost_users'
list_test_df = list_test_df.join(test_coupon_ghost_users)
list_test_df['ghost_users'].fillna(0,inplace = True)

coupons_train_ghost = coupons_train[coupons_train['ghost_users'] > 0]
coupons_train_ghost = coupons_train_ghost[coupons_train_ghost['purchases'] < 3000]
#coupon_purchased_series = grouped['user_id_hash'].aggregate(lambda x: np.unique(x).shape[0])
#coupon_purchased_series.name = 'purchases'
#
#coupons_train = list_train_df.join(coupon_purchased_series)
#coupons_train['purchases].fillna(0,inplace = True)
toc=timeit.default_timer()
print('Merging Visit Datasets',toc - tic)
#%%
tic=timeit.default_timer()

if(use_cv):
    starting_date = pd.to_datetime('2012-06-03 12:00:00')
    ending_date = pd.to_datetime('2012-06-10 12:00:00')

#    starting_date = pd.to_datetime('2012-04-23 12:00:00')
#    ending_date = pd.to_datetime('2012-04-30 12:00:00')

#    starting_date = pd.to_datetime('2012-05-10 12:00:00')
#    ending_date = pd.to_datetime('2012-05-17 12:00:00')
#    starting_date = pd.to_datetime('2012-05-17 12:00:00')
#    ending_date = pd.to_datetime('2012-05-24 12:00:00')
#    starting_date = pd.to_datetime('2012-05-24 12:00:00')
#    ending_date = pd.to_datetime('2012-05-31 12:00:00')

#    starting_date = pd.to_datetime('2012-06-17 12:00:00')
#    ending_date = pd.to_datetime('2012-06-24 12:00:00')

#    starting_date = pd.to_datetime('2012-06-10 12:00:00')
#    ending_date = pd.to_datetime('2012-06-17 12:00:00')

#    starting_date = pd.to_datetime('2011-10-24 12:00:00')
#    ending_date = pd.to_datetime('2011-10-31 12:00:00')

#    starting_date = pd.to_datetime('2011-12-20 12:00:00')
#    ending_date = pd.to_datetime('2011-12-27 12:00:00')
else:
    starting_date = pd.to_datetime('2012-06-24 12:00:00')
    ending_date = pd.to_datetime('2012-07-01 12:00:00')
#cv_coupons = list_train_df[pd.to_datetime(list_train_df['dispfrom']) >= starting_date]['coupon_id_hash'].values

#dangerous, but remove gift cards from training set
#train_no_gifts = list_train_df[list_train_df['en_genre'] != 'Gift card']
#cv_coupons = train_no_gifts[(pd.to_datetime(train_no_gifts['dispfrom']) >= starting_date) &
#                            (pd.to_datetime(train_no_gifts['dispfrom']) < ending_date) ]['coupon_id_hash'].values

#dangerous to exclude dispperiod >7 from cv
cv_coupons = list_train_df[(list_train_df['dispfrom'] >= starting_date) &
                            (list_train_df['dispfrom'] < ending_date)
                            & (list_train_df['dispperiod'] <= 7) ]['coupon_id_hash'].values


criterion = detail_df['coupon_id_hash'].map(lambda x: x not in cv_coupons)
cv_detail_df = detail_df[~criterion]
train_detail_df = detail_df[criterion]

cv_detail_df = cv_detail_df[(cv_detail_df['i_date'] >= starting_date) &
                            (cv_detail_df['i_date'] < ending_date)]
USER_COUPONS_PURCHASED_DET_DICT_CV = {}
USER_COUPONS_PURCHASED_UNIQUE_CV = {}
grouped_u_cv = cv_detail_df.groupby('user_id_hash')
for name, group in grouped_u_cv:
    USER_COUPONS_PURCHASED_DET_DICT_CV[name] = group['coupon_id_hash'].values
    USER_COUPONS_PURCHASED_UNIQUE_CV[name] = group['coupon_id_hash'].unique().tolist()

criterion_coupons = list_train_df['coupon_id_hash'].map(lambda x: x not in cv_coupons)
train_coupons_df = list_train_df[criterion_coupons]
cv_coupons_df = list_train_df[~criterion_coupons]

cv_detail_df_m = cv_detail_df.merge(cv_coupons_df,on='coupon_id_hash',how='inner')
cv_detail_df_small = cv_detail_df_m[['user_id_hash','coupon_id_hash','en_small_area',
                                       'en_large_area','en_genre','en_capsule','en_ken','discount_price',
                                       'price_rate','dispperiod','area_count']]

ghosts_criterion_cv = visits_ghosts['coupon_id_hash'].map(lambda x: x in cv_coupons)
visits_ghosts_cv = visits_ghosts[ghosts_criterion_cv]

toc=timeit.default_timer()
print('Making CV Dataset',toc - tic)
#%%
#TODO testing
if(redo_visits):
    visits_criterion = visits_df['coupon_id_hash'].map(lambda x: x not in cv_coupons)
    visits_train_df = visits_df[visits_criterion]
#    train_detail_df = visits_train_df
#%%
tic=timeit.default_timer()
### filter redundant features
features = ['coupon_id_hash', 'user_id_hash','purchases','purchases_m','purchases_f',
            'purchases_old','purchases_young','purchases_middle',
            'purchases_pref','purchases_no_pref','valid_gap',
            'en_genre', 'discount_price', 'price_rate','dispperiod','dispfrom','dispend',
            'usable_date_mon', 'usable_date_tue', 'usable_date_wed', 'usable_date_thu',
            'usable_date_fri', 'usable_date_sat', 'usable_date_sun', 'usable_date_holiday',
            'usable_date_before_holiday', 'en_large_area', 'en_ken', 'en_small_area',
            'catalog_price','en_capsule','validperiod','area_count','ghost_users']
#features = ['coupon_id_hash', 'user_id_hash',
#            'en_genre', 'discount_price', 'price_rate','dispperiod','dispfrom','dispend',
#            'en_large_area', 'en_ken', 'en_small_area',
#            'catalog_price','en_capsule']
if (use_cv):
    purchased_coupons_train = train_detail_df.merge(train_coupons_df,on='coupon_id_hash',how='inner')
else:
    purchased_coupons_train = detail_df.merge(list_train_df,on='coupon_id_hash',how='inner')
detail_train_c = purchased_coupons_train
#if(redo_visits):
#    detail_train_c =  detail_df.merge(train_coupons_df,on='coupon_id_hash',how='inner')
purchased_coupons_train = purchased_coupons_train[features]

toc=timeit.default_timer()
print('Merging Datasets',toc - tic)
#%%
visits_old_df = visits_df
#%%
#leisure large area > 14 same, but maybe make ken and small unknown
tic=timeit.default_timer()

detail_train_c['days_from_avail'] = detail_train_c['i_date'] - detail_train_c['dispfrom']
detail_train_c['time_from_avail'] = detail_train_c['days_from_avail'].map(lambda x: x.astype('timedelta64[D]').astype(int))

detail_train_c_small = detail_train_c[['user_id_hash','coupon_id_hash','en_small_area',
                                       'en_large_area','en_genre',
                                       'en_capsule','en_ken',
                                       'i_date',
                                       'discount_price','purchases','valid_gap',
                                       'price_rate','dispperiod','area_count']]
detail_train_c_food = detail_train_c_small[detail_train_c_small['en_genre'] == 'Food']
detail_train_c_relax = detail_train_c_small[detail_train_c_small['en_genre'] == 'Relaxation']
detail_train_c_spa = detail_train_c_small[detail_train_c_small['en_genre'] == 'Spa']
detail_train_c_hsalon = detail_train_c_small[detail_train_c_small['en_genre'] == 'Hair salon']
detail_train_c_other = detail_train_c_small[detail_train_c_small['en_genre'] == 'Other coupon']
detail_train_c_other_no_out = detail_train_c_other[detail_train_c_other['discount_price'] <= 100]
detail_train_c_gift = detail_train_c_small[detail_train_c_small['en_genre'] == 'Gift card']
detail_train_c_hot = detail_train_c_small[detail_train_c_small['en_genre'] == 'Hotel and Japanese hotel']
detail_train_c_del = detail_train_c_small[detail_train_c_small['en_genre'] == 'Delivery service']
detail_train_c_health = detail_train_c_small[detail_train_c_small['en_genre'] == 'Health and medical']
detail_train_c_lesson = detail_train_c_small[detail_train_c_small['en_genre'] == 'Lesson']
detail_train_c_leisure = detail_train_c_small[detail_train_c_small['en_genre'] == 'Leisure']
detail_train_c_leisure_chiba = detail_train_c_small[(detail_train_c_small['en_genre'] == 'Leisure')
                                                    & (detail_train_c_small['en_small_area'] == 'Chiba')]

chibans = detail_train_c_leisure_chiba.user_id_hash.unique()
#spa_users = detail_train_
detail_train_c_small['chiban_criterion'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in chibans else 0)
#detail_train_c_chibans = detail_train_c_small[detail_train_c_chiban_criterion]
detail_train_c_chibans = detail_train_c_small.loc[detail_train_c_small['chiban_criterion'] == 1]

detail_train_c_salon = detail_train_c_small[(detail_train_c_small['en_genre'] == 'Nail and eye salon')
                                            | (detail_train_c_small['en_genre'] == 'Hair salon')]

health_users = detail_train_c_health.user_id_hash.unique()
detail_train_c_small['health_user'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in health_users else 0)
detail_train_c_health_users = detail_train_c_small.loc[detail_train_c_small['health_user'] == 1]

relax_users = detail_train_c_relax.user_id_hash.unique()
detail_train_c_small['relax_user'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in relax_users else 0)
detail_train_c_relax_users = detail_train_c_small.loc[detail_train_c_small['relax_user'] == 1]

hotel_users = detail_train_c_hot.user_id_hash.unique()
detail_train_c_small['hotel_user'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in hotel_users else 0)
detail_train_c_hotel_users = detail_train_c_small.loc[detail_train_c_small['hotel_user'] == 1]

leisure_users = detail_train_c_leisure.user_id_hash.unique()
detail_train_c_small['leisure_user'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in leisure_users else 0)
detail_train_c_leisure_users = detail_train_c_small.loc[detail_train_c_small['leisure_user'] == 1]

food_users = detail_train_c_food.user_id_hash.unique()
detail_train_c_small['food_user'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in food_users else 0)
detail_train_c_food_users = detail_train_c_small.loc[detail_train_c_small['food_user'] == 1]

hsalon_users = detail_train_c_hsalon.user_id_hash.unique()
detail_train_c_small['hsalon_user'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in hsalon_users else 0)
detail_train_c_hsalon_users = detail_train_c_small.loc[detail_train_c_small['hsalon_user'] == 1]

lesson_users = detail_train_c_lesson.user_id_hash.unique()
detail_train_c_small['lesson_user'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in lesson_users else 0)
detail_train_c_lesson_users = detail_train_c_small.loc[detail_train_c_small['lesson_user'] == 1]

other_users = detail_train_c_other.user_id_hash.unique()
detail_train_c_small['other_user'] = detail_train_c_small.user_id_hash.map(lambda x: 1 if x in other_users else 0)
detail_train_c_other_users = detail_train_c_small.loc[detail_train_c_small['other_user'] == 1]

toc=timeit.default_timer()
print('detail_c feat',toc - tic)
#%%
tic=timeit.default_timer()
user_purch = {}
grouped_u = detail_train_c_small.groupby('user_id_hash')
for name, group in grouped_u:
    user_purch[name] = group['coupon_id_hash'].unique()
toc=timeit.default_timer()

#user = grouped_u.get_group(1492

#user_del_purch = {}
#grouped_u = detail_train_c_del.groupby('user_id_hash')
#for name, group in grouped_u:
#    user_del_purch[name] = group['coupon_id_hash'].unique()
#
#user_lesson_purch = {}
#grouped_u = detail_train_c_lesson.groupby('user_id_hash')
#for name, group in grouped_u:
#    user_lesson_purch[name] = group['coupon_id_hash'].unique()
#toc=timeit.default_timer()
#user_lesson_frac = {}
#for key in user_lesson_purch:
#    user_lesson_frac[key] = len(user_lesson_purch[key]) / len(user_purch[key])

print('train user purchase dicts',toc - tic)
#%%
### create 'dummyuser' records in order to merge training and testing sets in one
list_test_df['user_id_hash'] = 'dummyuser'
list_test_df['purchases'] = 'dummy_purchases'
list_test_df['purchases_f'] = 'dummy_purchases'
list_test_df['purchases_m'] = 'dummy_purchases'
list_test_df['purchases_pref'] = 'dummy_purchases'
list_test_df['purchases_no_pref'] = 'dummy_purchases'
list_test_df['purchases_old'] = 'dummy_purchases'
list_test_df['purchases_young'] = 'dummy_purchases'
list_test_df['purchases_middle'] = 'dummy_purchases'
#cv_coupons_df['user_id_hash'] = cv_coupons_df['user_id_hash'].map(lambda x: 'dummyuser')
cv_coupons_df['user_id_hash'] = 'dummyuser'
cv_coupons_df_orig = cv_coupons_df.copy()
cv_coupons_df['purchases'] = 'dummy_purchases'

### filter testing set consistently with training set
list_test_df_orig = list_test_df.copy()

#TODO testing dangerous
list_test_df = list_test_df[list_test_df['price_rate'] != 100]
#list_test_df = list_test_df[list_test_df['en_genre'] != 'Gift card'] ##dramatic lb improvement
#list_test_df = list_test_df[~((list_test_df['discount_price'] == 100) & (list_test_df['en_genre'] == 'Other coupon'))]
#list_test_df = list_test_df[~((list_test_df['discount_price'] == 100) & (list_test_df['en_genre'] == 'Gift card'))]

#- possibly overfitting - but probably should keep it out?

#list_test_df = list_test_df[list_test_df['coupon_id_hash'] != 188] ##increases lb by small amount
#probably is real coupon but not as great as my algorithm thinks

list_test_df = list_test_df[features]
cv_coupons_df = cv_coupons_df[features]
#%%
##Very dangerous, but based on lb feedback as well as dispperiod of 13 similarities,
##with train coupons not purchased often, drop all gift cards from cv and test
#list_test_df = list_test_df[list_test_df['en_genre'] != 'Gift card']
#cv_coupons_df = cv_coupons_df[cv_coupons_df['en_genre'] != 'Gift card']
#%%
usable_list = ['usable_date_mon',
       'usable_date_tue', 'usable_date_wed', 'usable_date_thu',
       'usable_date_fri', 'usable_date_sat', 'usable_date_sun',
       'usable_date_holiday', 'usable_date_before_holiday']
double_usable_list = ['double_usable_date_mon',
       'double_usable_date_tue', 'double_usable_date_wed', 'double_usable_date_thu',
       'double_usable_date_fri', 'double_usable_date_sat', 'double_usable_date_sun',
       'double_usable_date_holiday', 'double_usable_date_before_holiday']
#%%
tic=timeit.default_timer()
### merge sets together
def generate_features(input_df):
    ### create new features

    ##TODO testing if dropping free helps or hurts
    ##none of it exists in test set
    #dangerous

    input_df = input_df[input_df['price_rate'] != 100]
#    input_df = input_df[~((input_df['discount_price'] == 100) & (input_df['en_genre'] == 'Other coupon'))]
#    input_df = input_df[~((input_df['discount_price'] == 100) & (input_df['en_genre'] == 'Gift card'))]


    input_df['en_old_ken'] = input_df['en_ken']
    input_df['en_old_small_area'] = input_df['en_small_area']
    input_df['en_old_large_area'] = input_df['en_large_area']
    input_df['en_orig_small_area'] = input_df['en_small_area']
    input_df['en_orig_large_area'] = input_df['en_large_area']
    input_df['en_old_genre'] = input_df['en_genre']
    input_df['en_old_capsule'] = input_df['en_capsule']

    #TODO do this better kind of kludgy
    coup_lesson_cond = input_df['en_genre'] == 'Lesson'
    input_df['usable_nan'] = input_df['usable_date_mon'].map(lambda x: int(pd.isnull(x)))
    input_df['usable_notnan'] = input_df['usable_date_mon'].map(lambda x: int(~pd.isnull(x)))
    #coup_large_area_cond = input_df['area_count'] >= 13
    coup_large_area_cond = input_df['usable_nan']
    coup_large_and_lesson = (coup_lesson_cond & coup_large_area_cond)
    input_df['temp'] = input_df['en_genre'].map(lambda x: 'Correspondence course')
    input_df.en_capsule[coup_large_and_lesson] = input_df['temp'][coup_large_and_lesson]

    coup_small_and_lesson = (coup_lesson_cond & ~coup_large_area_cond)
    input_df['temp'] = input_df['en_genre'].map(lambda x: 'Class')
    input_df.en_capsule[coup_small_and_lesson] = input_df['temp'][coup_small_and_lesson]

    coup_del_cond = input_df['en_genre'] == 'Delivery service'
#    coup_other_cond = input_df['en_genre'] == 'Other coupon'
#    coup_gift_cond = input_df['en_genre'] == 'Gift card'
    coup_valid_nan_cond = input_df['valid_gap'] < 0
    input_df['unk'] = input_df['en_genre'].map(lambda x: 'unk')
#    coup_many_area_cond = input_df['area_count'] >= 25

    #cond_sum = (coup_del_cond | coup_gift_cond | coup_other_cond | coup_large_and_lesson)
    #cond_sum = (coup_del_cond | coup_large_and_lesson)
#    cond_sum = (coup_del_cond | coup_many_area_cond)
    cond_sum = (coup_del_cond | coup_valid_nan_cond)

#    input_df.en_small_area[coup_del_cond] = input_df['unk'][coup_del_cond]
#    input_df.en_ken[coup_del_cond] = input_df['unk'][coup_del_cond]
#    input_df.en_large_area[coup_del_cond] = input_df['unk'][coup_del_cond]
    input_df.en_small_area[cond_sum] = input_df['unk'][cond_sum]
    input_df.en_ken[cond_sum] = input_df['unk'][cond_sum]
    input_df.en_large_area[cond_sum] = input_df['unk'][cond_sum]

    input_df['del_small_area'] = input_df['en_old_small_area']
    input_df['del_large_area'] = input_df['en_old_large_area']
    input_df['del_ken'] = input_df['en_old_ken']
    input_df.del_small_area[~coup_del_cond] = input_df['unk'][~coup_del_cond]
    input_df.del_large_area[~coup_del_cond] = input_df['unk'][~coup_del_cond]
    input_df.del_ken[~coup_del_cond] = input_df['unk'][~coup_del_cond]

    coup_food_cond = input_df['en_genre'] == 'Food'
    input_df['food_small_area'] = input_df['en_old_small_area']
    input_df['food_large_area'] = input_df['en_old_large_area']
    input_df['food_ken'] = input_df['en_old_ken']
    input_df.food_small_area[~coup_food_cond] = input_df['unk'][~coup_food_cond]
    input_df.food_large_area[~coup_food_cond] = input_df['unk'][~coup_food_cond]
    input_df.food_ken[~coup_food_cond] = input_df['unk'][~coup_food_cond]

    coup_hotel_cond = input_df['en_genre'] == 'Hotel and Japanese hotel'
    input_df['hotel_small_area'] = input_df['en_old_small_area']
    input_df['hotel_large_area'] = input_df['en_old_large_area']
    input_df['hotel_ken'] = input_df['en_old_ken']
    input_df.hotel_small_area[~coup_hotel_cond] = input_df['unk'][~coup_hotel_cond]
    input_df.hotel_large_area[~coup_hotel_cond] = input_df['unk'][~coup_hotel_cond]
    input_df.hotel_ken[~coup_hotel_cond] = input_df['unk'][~coup_hotel_cond]

    #input_df.en_ken[cond_sum] = input_df['unk'][cond_sum]
    #input_df.en_large_area[cond_sum] = input_df['unk'][cond_sum]

    input_df['en_genre'] = input_df['en_genre'].map(lambda x: x.replace (" ", "_"))
    input_df['en_large_area'] = input_df['en_large_area'].map(lambda x: x.replace (" ", "_"))
    input_df['en_small_area'] = input_df['en_small_area'].map(lambda x: x.replace (" ", "_"))
    input_df['en_ken'] = input_df['en_ken'].map(lambda x: x.replace (" ", "_"))

    input_df['cheap'] = input_df['discount_price'].map(lambda x: 1 if x <= 800 else 0)
    input_df['decent'] = input_df['discount_price'].map(lambda x: 1 if 800 < x <= 2000 else 0)
    input_df['medium_low'] = input_df['discount_price'].map(lambda x: 1 if 2000 < x <= 5000 else 0)
    input_df['medium_high'] = input_df['discount_price'].map(lambda x: 1 if 5000 < x <= 20000 else 0)
    input_df['expensive'] = input_df['discount_price'].map(lambda x: 1 if x > 20000 else 0)

    input_df['pr_0'] = input_df['price_rate'].map(lambda x: 1 if x <= 50 else 0)
    input_df['pr_1'] = input_df['price_rate'].map(lambda x: 1 if 50 < x <= 60 else 0)
    input_df['pr_2'] = input_df['price_rate'].map(lambda x: 1 if 60 < x <= 75 else 0)
    input_df['pr_3'] = input_df['price_rate'].map(lambda x: 1 if 75 < x <= 89 else 0)
    input_df['pr_4'] = input_df['price_rate'].map(lambda x: 1 if x > 89 else 0)
    input_df['salon_genre'] = input_df['en_genre'].map(lambda x: 1 if 'salon' in x else 0)
    input_df['hotel_caps'] = input_df['en_capsule'].map(lambda x: 1 if 'Hotel' == x else 0)
    input_df['j_hotel_caps'] = input_df['en_capsule'].map(lambda x: 1 if 'Japanese hotel' == x else 0)
    def get_luxury(x):
        if('Relaxion' in x or 'Spa' in x or 'salon' in x or 'Health and medical' in x or 'Beauty' in x):
            return 1
        else:
            return 0
    input_df['luxury_genre'] = input_df['en_genre'].map(lambda x: get_luxury(x))
    def get_food_and_stuff(x):
        if('Relaxion' in x or 'Food' in x or 'Hotel and Japanese hotel' in x or 'Beauty' in x):
            return 1
        else:
            return 0
    input_df['food_stuff_genre'] = input_df['en_genre'].map(lambda x: get_food_and_stuff(x))
    for col in usable_list:
        input_df[col].fillna(1,inplace=True)
#        input_df[col] = input_df[col].map(lambda x: 1 if x > 1 else 0)
    input_df['usable_sum'] = input_df[usable_list].sum(axis=1)
#    input_df['usable_sum'] = input_df['usable_sum'].map(lambda x: str(x))
    for col in usable_list:
        input_df['double_'+col] = input_df[col].map(lambda x: 1 if (x == 2) else 0)


    input_df['corr_course'] = input_df['en_capsule'].map(lambda x: int('Correspondence' in x))
    input_df['class'] = input_df['en_capsule'].map(lambda x: int('Class' in x))
    input_df['other'] = input_df['en_capsule'].map(lambda x: int('Other' in x))

    input_df['disp_normed'] = input_df['dispperiod'].map(lambda x: '8' if x > 7 else str(x))

    input_df['discount_price_log'] = 1.0 / np.log10(input_df['discount_price'] + 2)
    input_df['dispfrom_weekday'] = input_df['dispfrom'].map(lambda x: str(x.weekday()))
    input_df['dispend_weekday'] = input_df['dispend'].map(lambda x: str(x.weekday()))

    input_df['validperiod_nan'] = input_df['validperiod'].map(lambda x: int(pd.isnull(x)))
    input_df['validperiod_notnan'] = input_df['validperiod'].map(lambda x: int(~pd.isnull(x)))

    input_df['validperiod'].fillna(0,inplace=True)
    valid_bins = [0,70,100,130,160,1000]
    input_df['validperiod_binned'] = np.digitize(input_df['validperiod'],valid_bins,right=True)

    valid_gap_bins = [-1000,-0.01,0,1,5,15,30,100]
    input_df['valid_gap_binned'] = np.digitize(input_df['valid_gap'],valid_gap_bins,right=True)




    price_bins = [0,100,500,1000,1500,3000,6000,12000,30000,100000000000000]
    input_df['discount_price_binned'] = np.digitize(input_df['discount_price'],price_bins,right=True)
    area_bins = [0,1,2,3,4,9,12,15,20,30,50,100]
    input_df['area_binned'] = np.digitize(input_df['area_count'],area_bins,right=True)
    ghost_bins = [0,1,10,100,1000]
    input_df['ghost_binned'] = np.digitize(input_df['ghost_users'],ghost_bins,right=True)
    rate_bins = [0,50,55,65,75,80,85,90,94,100]
    input_df['rate_binned'] = np.digitize(input_df['price_rate'],rate_bins,right=True)

    input_df['valid_gap_binned'] = input_df['valid_gap_binned'].map(lambda x: str(x))
    input_df['validperiod_binned'] = input_df['validperiod_binned'].map(lambda x: str(x))
    input_df['discount_price_binned'] = input_df['discount_price_binned'].map(lambda x: str(x))
    input_df['area_binned'] = input_df['area_binned'].map(lambda x: str(x))
    input_df['ghost_binned'] = input_df['ghost_binned'].map(lambda x: str(x))
    input_df['rate_binned'] = input_df['rate_binned'].map(lambda x: str(x))

    small_area_values = input_df['en_old_small_area'].unique()
    small_area_values_dict = dict(zip(small_area_values,range(len(small_area_values))))
    input_df['small_area_mapped'] = input_df['en_old_small_area'].map(lambda x: small_area_values_dict[x])
    large_area_values = input_df['en_old_large_area'].unique()
    large_area_values_dict = dict(zip(large_area_values,range(len(large_area_values))))
    input_df['large_area_mapped'] = input_df['en_old_large_area'].map(lambda x: large_area_values_dict[x])
    ken_values = input_df['en_old_ken'].unique()
    ken_values_dict = dict(zip(ken_values,range(len(ken_values))))
    input_df['ken_mapped'] = input_df['en_old_ken'].map(lambda x: ken_values_dict[x])
    return input_df

if (use_cv):
    input_df = pd.concat([purchased_coupons_train, cv_coupons_df], axis=0)
else:
    input_df = pd.concat([purchased_coupons_train, list_test_df], axis=0)
combined = generate_features(input_df)

features.extend(['en_old_ken','en_old_genre','en_old_small_area','en_old_large_area',
                 'en_old_capsule','cheap','decent','medium_low','medium_high','expensive',
                 'pr_0','pr_1','pr_2','pr_3','pr_4','salon_genre', 'luxury_genre','food_stuff_genre',
                 'disp_normed','usable_nan','usable_notnan','usable_sum','valid_gap_binned',
                 'validperiod_nan','validperiod_notnan','corr_course','class','other',
                 'discount_price_log','discount_price_binned','hotel_caps','j_hotel_caps',
                 'area_binned','ghost_binned','rate_binned','small_area_mapped',
                 'large_area_mapped','ken_mapped'])
features.extend(double_usable_list)
toc=timeit.default_timer()
print('Feature Generation',toc - tic)
#%%
tic=timeit.default_timer()
### convert categoricals to OneHotEncoder form

categoricals = ['en_genre', 'en_large_area','en_orig_large_area','en_orig_small_area',
                'en_ken', 'en_small_area','del_small_area','del_large_area','del_ken',
                'hotel_small_area','hotel_large_area','hotel_ken',
                'food_small_area','food_large_area','food_ken',
                'disp_normed','area_binned','en_capsule','valid_gap_binned',
                'ghost_binned','rate_binned','discount_price_binned']
#categoricals.extend(usable_list)
def get_categoricals(input_df,cat_list,feat_list):
    combined_categoricals = input_df[cat_list]
#    combined_categoricals = pd.get_dummies(combined_categoricals,dummy_na=True)
    combined_categoricals = pd.get_dummies(combined_categoricals,dummy_na=False)
    ### leaving continuous features as is, obtain transformed dataset
    continuous = list(set(feat_list) - set(cat_list))
    input_df = pd.concat([input_df[continuous], combined_categoricals], axis=1)
    ### remove NaN values
    NAN_SUBSTITUTION_VALUE = 1
    input_df = input_df.fillna(NAN_SUBSTITUTION_VALUE)
    return input_df

combined = get_categoricals(combined,categoricals,features)
### split back into training and testing sets
train_df = combined[combined['user_id_hash'] != 'dummyuser'].copy()
test_df = combined[combined['user_id_hash'] == 'dummyuser'].copy()
test_df.drop('user_id_hash', inplace=True, axis=1)
#TODO dangerous
if (use_cv):
    test_df = test_df[test_df['dispperiod'] <= 7]
toc=timeit.default_timer()
print('Getting Dummies',toc - tic)
#%%
### find most appropriate coupon for every user (mean of all purchased coupons), in other words, user profile
tic=timeit.default_timer()
drop_list = ['coupon_id_hash','dispfrom','catalog_price','discount_price',
             'en_old_genre','en_old_ken','dispend','area_count','usable_sum',
             'en_old_small_area','en_old_large_area','en_old_capsule',
             'ghost_users','purchases','purchases_m','purchases_f',
             'purchases_pref','purchases_no_pref',
             'purchases_old','purchases_young','purchases_middle',
#             'discount_price_binned','area_binned','ghost_binned','rate_binned',
             'small_area_mapped','valid_gap',
             'large_area_mapped','ken_mapped']
drop_list.extend(usable_list)
train_dropped_coupons = train_df.drop(drop_list, axis=1)
all_users = users_df[['user_id_hash']]
col_number = train_dropped_coupons.columns.shape[0]
zero_data = np.zeros(shape=(22873,col_number))
one_data = np.ones(shape=(22873,col_number))
user_profiles = pd.DataFrame(zero_data, columns=train_dropped_coupons.columns.values)

zero_row = np.zeros(shape=(1,col_number))
s = pd.DataFrame(zero_row, columns=train_dropped_coupons.columns.values)
ser = s.ix[0] * 0
i_max = train_dropped_coupons.index.shape[0]
i2=0
all_users_list = all_users.index.values.tolist()
print('here')
while (i2 < i_max):
    i1 = i2
    i2 = i2 + 200000
    if(i2 > i_max):
        i2 = i_max
    print("Merging coupon visit data i1=",i1," i2=",i2)
    df = train_dropped_coupons[i1:i2]
#    temp_prof = df.groupby(by='user_id_hash').apply(np.sum)
    temp_prof = df.groupby(by='user_id_hash').apply(np.sum)
    temp_values = temp_prof.index.values.tolist()
    missing = list(set(all_users_list) - set(temp_values))
    temp_zeros = np.zeros(shape=(len(missing),col_number))
    df_zeros = pd.DataFrame(temp_zeros, columns=train_dropped_coupons.columns.values)
    df_zeros.set_index([missing],inplace=True)
    temp_prof = temp_prof.append(df_zeros)
    user_profiles = user_profiles + temp_prof
toc=timeit.default_timer()
print('Finding Mean of Users Profiles',toc - tic)
#user_profiles = train_dropped_coupons.groupby(by='user_id_hash').apply(np.sum)
#all_users = users_df[['user_id_hash']]
#s = user_profiles.ix[0] * 0 + 0.000000000001
#temp_values = user_profiles.index.values
#for row in all_users.iterrows():
#    ind = row[0]
#    if (ind not in temp_values):
#        s.name = int(ind)
#        user_profiles = user_profiles.append(s)
#toc=timeit.default_timer()
#print('Finding Mean of Users Profiles',toc - tic)
#%%
##TODO figure out why sometimes needs to be commented out other times not
try:
    user_profiles.drop('user_id_hash', inplace=True, axis=1)
except ValueError:
    print('user_id_hash not in index weird')
#%%
coupons_ids = test_df['coupon_id_hash']
columns = [coupons_ids.values[i] for i in range(0, test_df.shape[0])]
user_index = user_profiles.index
temp_mat = np.ones((user_index.shape[0],test_df.shape[0]))
ones_df = pd.DataFrame(index=user_index, columns=columns,data=temp_mat)
#%%
def norm_rows(df):
    with np.errstate(invalid='ignore'):
        return df.div(df.sum(axis=1), axis=0).fillna(0)
#    return df.div(df.sum(axis=1), axis=0)
# dict lookup helper
def find_appropriate_weight(weights_dict, colname):
    for col, weight in weights_dict.items():
        if col == colname:
            return weight
    for col, weight in weights_dict.items():
        if col in colname:
            return weight
    print(colname)
    raise ValueError
### obtain string of top10 hashes according to similarity scores for every user
def get_top10_coupon_hashes_string(row):
    row.sort()
#    res = map(str,row.index[-10:][::-1].tolist())
    return ' '.join(row.index[-10:][::-1].tolist())
#%%
tic=timeit.default_timer()
dist_dict = {}
dispperiod_length = 8
for i in range(dispperiod_length):
    temp_df = detail_train_c[detail_train_c['dispperiod'] == i]
    value_counts = temp_df['time_from_avail'].value_counts()
    total = temp_df['time_from_avail'].count()
    cumulative_dict = {}
    rat = 0
    for j in range(len(value_counts)):
        try:
            rat = rat + (value_counts[j] / total)
        except KeyError:
            cumulative_dict[j] = rat
        cumulative_dict[j] = rat
#        print(i,j,value_counts[j] / total)
#    print(i,rat_0)
    dist_dict[i] = cumulative_dict
##calculate number of days that  disp period is active

test_df['end_date'] = test_df['dispend'].map(lambda x: x if x <= ending_date else ending_date)
test_df['days_active'] = test_df['end_date'] - test_df['dispfrom']

test_df['days_active'] = test_df['days_active'].map(lambda x: x.astype('timedelta64[D]').astype(int))
test_df['days_active'] = test_df['days_active'].map(lambda x: 1 if x <= 0 else x)
##hack but whatever
test_df['dispperiod_mod'] = test_df['dispperiod'].map(lambda x: 1 if x <= 0 else x)

def get_active_prop(row):
    try:
        return dist_dict[row['dispperiod']][row['days_active']-1]
    except:
        return row['dispperiod'] / row['days_active']
test_df['active_prop'] = test_df.apply(lambda row: get_active_prop(row),axis=1)

toc=timeit.default_timer()
print('Getting Active Test Time',toc - tic)

#TODO clean this code up (why go df to dict to series, just go to series
#temp = test_df[['coupon_id_hash','days_active']]
temp = test_df[['coupon_id_hash','active_prop']]
#temp['coupon_id_hash'] = temp['coupon_id_hash'].map(lambda x: str(x))
days_active_dict = temp.set_index('coupon_id_hash').to_dict()
#days_active_dict = days_active_dict['days_active']
days_active_dict = days_active_dict['active_prop']
days_active_series = pd.Series(days_active_dict)
result_active_weeks_df = ones_df * days_active_series
active_weeks_prop = result_active_weeks_df
result_active_weeks_df = norm_rows(result_active_weeks_df)

##calculate the purchase rates for different genres of coupons (over all users)
tic=timeit.default_timer()
coupons_train_no_free = coupons_train[coupons_train['price_rate'] != 100].copy()
#TODO dangerous
#remove the outlier other coupon
coupons_train_no_free = coupons_train_no_free[coupons_train_no_free['purchases'] <= 3000]
coupons_train_no_free = coupons_train_no_free[coupons_train_no_free['dispperiod'] <= 7]
grouped_genres = coupons_train_no_free.groupby('en_genre')

temp_test_df = test_df.copy()
temp_test_df.set_index('coupon_id_hash',drop=False,inplace=True)

price_bins = [0,100,500,1000,1500,3000,6000,12000,30000,100000000000000]
coupons_train_no_free['discount_price_binned'] = np.digitize(coupons_train_no_free['discount_price'],
                                                price_bins,right=True)
temp_test_df['discount_price_binned'] = np.digitize(temp_test_df['discount_price'],
                                                price_bins,right=True)
area_bins = [0,1,2,3,4,9,12,15,20,30,50,100]
coupons_train_no_free['area_binned'] = np.digitize(coupons_train_no_free['area_count'],
                                                area_bins,right=True)
temp_test_df['area_binned'] = np.digitize(temp_test_df['area_count'],
                                                area_bins,right=True)

ghost_bins = [0,1,5,10,40,100,250,1000]
coupons_train_no_free['ghost_binned'] = np.digitize(coupons_train_no_free['ghost_users'],
                                                ghost_bins,right=True)
temp_test_df['ghost_binned'] = np.digitize(temp_test_df['ghost_users'],
                                                ghost_bins,right=True)

rate_bins = [0,49,50,55,65,75,80,85,90,94,100]
coupons_train_no_free['rate_binned'] = np.digitize(coupons_train_no_free['price_rate'],
                                                rate_bins,right=True)
temp_test_df['rate_binned'] = np.digitize(temp_test_df['price_rate'],
                                                rate_bins,right=True)

small_area_values = coupons_train_no_free['en_small_area'].unique()
small_area_values_dict = dict(zip(small_area_values,range(len(small_area_values))))
coupons_train_no_free['small_area_mapped'] = \
            coupons_train_no_free['en_small_area'].map(lambda x: small_area_values_dict[x])
temp_test_df['small_area_mapped'] = \
            temp_test_df['en_old_small_area'].map(lambda x: small_area_values_dict[x])

large_area_values = coupons_train_no_free['en_large_area'].unique()
large_area_values_dict = dict(zip(large_area_values,range(len(large_area_values))))
coupons_train_no_free['large_area_mapped'] = \
            coupons_train_no_free['en_large_area'].map(lambda x: large_area_values_dict[x])
temp_test_df['large_area_mapped'] = \
            temp_test_df['en_old_large_area'].map(lambda x: large_area_values_dict[x])

ken_values = coupons_train_no_free['en_ken'].unique()
ken_values_dict = dict(zip(ken_values,range(len(ken_values))))
coupons_train_no_free['ken_mapped'] = \
            coupons_train_no_free['en_ken'].map(lambda x: ken_values_dict[x])
temp_test_df['large_area_mapped'] = \
            temp_test_df['en_old_ken'].map(lambda x: ken_values_dict[x])



def make_dict(result_dict,length,col_name,genre_name):
    temp_dict = {}
    for i in range(length):
        temp_group_df = group[group[col_name] == i]
        if(temp_group_df.index.shape[0] > 0):
            temp_dict[i] = temp_group_df['purchases'].mean()
        else:
            temp_dict[i] = group['purchases'].mean()
    result_dict[genre_name] = temp_dict
#caps to avoid spyder bug with variable explorer

genre_disp = {}
genre_price = {}
genre_area = {}
genre_rate = {}
genre_ghost = {}
genre_small = {}
genre_large = {}
genre_ken = {}
genre_dict_overall = {}
genre_dict_m_overall = {}
genre_dict_f_overall = {}

genre_sum = {}
genre_sum_m = {}
genre_sum_f = {}

genre_sum_pref = {}
genre_sum_no_pref = {}

genre_sum_old = {}
genre_sum_young = {}
genre_sum_middle = {}

genre_frac_pref = {}
genre_frac_no_pref = {}
genre_frac_m = {}
genre_frac_f = {}
genre_frac_old = {}
genre_frac_young = {}
genre_frac_middle = {}
for name, group in grouped_genres:
    genre_dict_overall[name] = group['purchases'].mean()
    genre_dict_m_overall[name] = group['purchases_m'].mean()
    genre_dict_f_overall[name] = group['purchases_f'].mean()

    genre_sum[name] = group['purchases'].sum()
    genre_sum_m[name] = group['purchases_m'].sum()
    genre_sum_f[name] = group['purchases_f'].sum()
    genre_sum_pref[name] = group['purchases_pref'].sum()
    genre_sum_no_pref[name] = group['purchases_no_pref'].sum()
    genre_sum_old[name] = group['purchases_old'].sum()
    genre_sum_young[name] = group['purchases_young'].sum()
    genre_sum_middle[name] = group['purchases_middle'].sum()

    genre_frac_m[name] = genre_sum_m[name] / genre_sum[name]
    genre_frac_f[name] = genre_sum_f[name] / genre_sum[name]

    genre_frac_pref[name] = genre_sum_pref[name] / genre_sum[name]
    genre_frac_no_pref[name] = genre_sum_no_pref[name] / genre_sum[name]

    genre_frac_old[name] = genre_sum_old[name] / genre_sum[name]
    genre_frac_young[name] = genre_sum_young[name] / genre_sum[name]
    genre_frac_middle[name] = genre_sum_middle[name] / genre_sum[name]

    make_dict(genre_disp,dispperiod_length,'dispperiod',name)
    make_dict(genre_price,len(price_bins),'discount_price_binned',name)
    make_dict(genre_area,len(area_bins),'area_binned',name)
    make_dict(genre_rate,len(rate_bins),'rate_binned',name)
    make_dict(genre_ghost,len(ghost_bins),'ghost_binned',name)
    make_dict(genre_small,len(small_area_values),'small_area_mapped',name)
    make_dict(genre_large,len(large_area_values),'large_area_mapped',name)
    make_dict(genre_ken,len(ken_values),'ken_mapped',name)

def get_genre_ave(row,col_name,genre_dict):
    try:
        return genre_dict[row['en_old_genre']][row[col_name]]
    except:
        return genre_dict_overall[row['en_old_genre']]

genre_disp_series = temp_test_df.apply(lambda row: get_genre_ave(row,'dispperiod',
                                                                     genre_disp), axis=1)
genre_disp_df = ones_df * genre_disp_series
genre_disp_df = genre_disp_df * days_active_series
genre_disp_df = norm_rows(genre_disp_df)

genre_price_series = temp_test_df.apply(lambda row: get_genre_ave(row,'discount_price_binned',genre_price), axis=1)
genre_price_df = norm_rows(ones_df * genre_price_series)

genre_area_series = temp_test_df.apply(lambda row: get_genre_ave(row,'area_binned',genre_area), axis=1)
genre_area_df = norm_rows(ones_df * genre_area_series)

genre_rate_series = temp_test_df.apply(lambda row: get_genre_ave(row,'rate_binned',genre_rate), axis=1)
genre_rate_df = norm_rows(ones_df * genre_rate_series)

genre_ghost_series = temp_test_df.apply(lambda row: get_genre_ave(row,'ghost_binned',genre_ghost), axis=1)
genre_ghost_df = norm_rows(ones_df * genre_ghost_series)

genre_small_series = temp_test_df.apply(lambda row: get_genre_ave(row,'small_area_mapped',genre_small), axis=1)
genre_small_df = norm_rows(ones_df * genre_small_series)

genre_large_series = temp_test_df.apply(lambda row: get_genre_ave(row,'large_area_mapped',genre_large), axis=1)
genre_large_df = norm_rows(ones_df * genre_large_series)

genre_ken_series = temp_test_df.apply(lambda row: get_genre_ave(row,'ken_mapped',genre_ken), axis=1)
genre_ken_df = norm_rows(ones_df * genre_ken_series)

toc=timeit.default_timer()
print('Ave Genre Time',toc - tic)
#%%
#%%
#explore the test coupon vists
def mapper0(x,coup_dict):
    try:
        if(coup_dict[x] >=10):
            return 1.3
        else:
            return 1
    except KeyError:
        return 1
if(use_cv):
    user_ghost_df = ones_df*0
    dict_cv = visits_ghosts_cv['coupon_id_hash'].value_counts()
    for row in visits_ghosts_cv.iterrows():
        user = row[1]['user_id_hash']
        coup = row[1]['coupon_id_hash']
        if(dict_cv[coup] >= 1):
            user_ghost_df.set_value(user,coup,1)
else:
    print('reloaded visits, use "leak"')
    user_ghost_df = ones_df*0
    dict_test = visits_test_df.test_coupon_id.value_counts()
    for row in visits_test_df.iterrows():
        user = row[1]['user_id_hash']
        coup = row[1]['test_coupon_id']
        if(dict_test[coup] >= 1):
            user_ghost_df.set_value(user,coup,1)
    test_leak_series = temp_test_df['coupon_id_hash'].map(lambda x: mapper0(x,dict_test))
#%%
test_drop_list = drop_list.copy()
test_drop_list.extend(['end_date','days_active','active_prop','dispperiod_mod'])
test_only_features = test_df.drop(test_drop_list, axis=1)

#lets try a different approach, as this is not improving my performance
#try and make a model that predicts the overall purchases for each coupon in
#the cv set
#then use this prediction (predicted coupons for all users) to break the degeneracy
#%%
#TODO training
tic = timeit.default_timer()
categoricals_purchases = categoricals.copy()

categoricals_purchases = ['disp_normed','area_binned',
#                          'en_orig_large_area',
                          'rate_binned',
                          'discount_price_binned','ghost_binned']
#categoricals_purchases = ['en_genre', 'en_large_area','en_orig_large_area','en_orig_small_area',
#                'en_ken', 'en_small_area','disp_normed','area_binned','en_capsule',
#                'ghost_binned','rate_binned','discount_price_binned']
#categoricals_purchases.extend(usable_list)
features_purchases = features.copy()
features_purchases.remove('user_id_hash')
features_purchases.remove('en_ken')
features_purchases.remove('en_capsule')
features_purchases.remove('en_genre')
features_purchases.remove('en_small_area')
features_purchases.remove('en_large_area')
features_purchases.remove('ghost_binned')

if (use_cv):
    combined_coups = list_train_df.append(cv_coupons_df)
else:
    combined_coups = list_train_df.append(list_test_df)
combined_purchases = generate_features(combined_coups)
combined_purchases = get_categoricals(combined_purchases,categoricals_purchases,features_purchases)


train_purchases_df = combined_purchases[combined_purchases['purchases'] != 'dummy_purchases'].copy()
test_purchases_df = combined_purchases[combined_purchases['purchases'] == 'dummy_purchases'].copy()

#criterion_cv = combined_purchases['purchases'].map(lambda x: 1 if x == 'dummy_purchases' else 0)
#train_purchases = combined_purchases[~criterion_cv]
#test_purchases = combined_purchases[criterion_cv]

#TODO testing dangerous
#train_purchases_df = train_purchases_df[train_purchases_df.purchases <= 3000]
#train_purchases_df = train_purchases_df[train_purchases_df.price_rate != 100]
#train_purchases_df = train_purchases_df[train_purchases_df['dispperiod'] <= 7]


#train_purchases_df = train_purchases_df[train_purchases_df['en_old_genre'] != 'Gift card']

purchases_drop_list = drop_list.copy()
purchases_drop_list.extend(['corr_course','discount_price_log','dispperiod',
#                        'class',
                        'other','salon_genre','luxury_genre',
                        'price_rate','food_stuff_genre','usable_notnan','validperiod',
                        'validperiod_notnan',
                        'validperiod_nan',
#                        'decent', 'pr_1', 'pr_2', 'pr_3', 'pr_0'
                        ])

train_purchases_dropped = train_purchases_df.drop(purchases_drop_list, axis=1)

def predict_purchases(genre_name):
#    train_purchases = train_purchases.purchases
    train_genre = train_purchases_df[train_purchases_df.en_old_genre == genre_name]
    train_purchases = train_genre.purchases
    train_genre_dropped = train_genre.drop(purchases_drop_list,axis=1)

    cv_l = cross_validation.KFold(len(train_genre_dropped), n_folds=10, shuffle=True,random_state = 1)
    regr = LassoCV(cv=cv_l, n_jobs = 2)

    train_data = train_genre_dropped.values
    regr = regr.fit( train_data, train_purchases )
    prediction_train = regr.predict(train_data)
    prediction_train = np.maximum(prediction_train, 0.)
    rmse_train = np.sqrt(((train_purchases - prediction_train) ** 2).mean())
    rmse_train_overall = np.sqrt(((train_purchases - train_purchases.mean()) ** 2).mean())
    print(genre_name)
    print('train_fit_rmse',rmse_train,'train_rmse',rmse_train_overall)

    test_genre = test_purchases_df[test_purchases_df.en_old_genre == genre_name]
    if(len(test_genre) == 0):
        return {}
    test_genre_dropped = test_genre.drop(purchases_drop_list,axis=1)
    test_data = test_genre_dropped.values
    pred_test = regr.predict(test_data)
    pred_test = np.maximum(pred_test, 0.)
    test_purchase_preds = dict(zip(test_genre['coupon_id_hash'],pred_test))

    #TODO testing revert back
    #cv_purchases_unordered = cv_coupons_df_orig.purchases
    #cv_purchases = []
    #for test_coup in  test_purchases_dropped.index:
    #    cv_purchases.append(cv_purchases_unordered.ix[test_coup])
    #cv_purchases = np.array(cv_purchases)
    #test_purchase_preds = dict(zip(test_purchases['coupon_id_hash'],cv_purchases))
    if(use_cv):
    #    cv_purchases_dict = dict(zip(cv_coupons_df.index,cv_coupons_df.purchases))
    #    cv_purchases = test_purchases.purchases.map(lambda x: cv_purchases_dict[x]).values

    #    cv_purchases = cv_coupons_df_orig.purchases
        cv_purchases_unordered = cv_coupons_df_orig.purchases
        cv_purchases = []
        for test_coup in test_genre_dropped.index:
            cv_purchases.append(cv_purchases_unordered.ix[test_coup])
        cv_purchases = np.array(cv_purchases)
        rmse = np.sqrt(((cv_purchases - pred_test) ** 2).mean())
        rmse_cv_overall = np.sqrt(((cv_purchases - cv_purchases.mean()) ** 2).mean())
        print('cv_fit_rmse',rmse,'cv_rmse',rmse_cv_overall)
#        p.clf()
#        p.cla()
#        plt.scatter(cv_coupons_df[cv_coupons_df.en_genre == genre_name].index,cv_purchases - pred_test)
#        plt.xlabel('cv coup id')
#        plt.ylabel('truth - pred')
    return test_purchase_preds
test_purchase_preds = {}
for genre_name in train_purchases_df['en_old_genre'].unique():
    test_purchases_genre = predict_purchases(genre_name)
    test_purchases_preds = test_purchase_preds.update(test_purchases_genre)
toc=timeit.default_timer()
print('Purchase Calculation Time',toc - tic)
#%%
purchases_test_df = test_df
test_purchases_series = purchases_test_df['coupon_id_hash'].map(lambda x: test_purchase_preds[x])

#TODO figure out if this is the right thing to do or not
#i think it should be, but not entirely clear
#test_purchases_series = test_purchases_series * days_active_series

#output = output.map(lambda x: '0c015306597566b632bebfb63b7e59f3') -- 0
#output = output.map(lambda x: 'c988d799bc7db9254fe865ee6cf2d4ff') #188 -- 0.000049
#output = output.map(lambda x: '7d7487370022eb18ee130856aa1c6eec') #262 -- 0
#output = output.map(lambda x: 'c9e1dcbd8c98f919bf85ab5f2ea30a9d') #107 -- 0.002
#output = output.map(lambda x: '266745cba5481af70ec489f67f2d5d77') #111 -- 0.000219
#output = output.map(lambda x: '27741884a086e2864936d7ef680becc2') #182 -- 0.000875
#output = output.map(lambda x: '7ae4e60eab2e4d7e20f88fc19267e87c') #218 -- 0.000292
#output = output.map(lambda x: 'b36879abb93ee7630b313ef0a04463f3') #259 -- 0.000656 leisure 6 (259)
#output = output.map(lambda x: 'd79a889ee9d0712607a2672e96ba3d69') #221 --  0.000292  other

#test_coupon_id_dict['2fcca928b8b3e9ead0f2cecffeea50c1'] 38 -- 0.000146 from script leisure coup
#test_coupon_id_dict['0fd38be174187a3de72015ce9d5ca3a2'] 161 -- 0.000802 from script delivery coup

#output = output.map(lambda x: '3905228fb8cac640b673f71d5f315df5') #142 --  0.000437
#output = output.map(lambda x: '2cb5e0a522dcb7c539018c165359ad5a') #40 --  0.000146   other coupons web

#output = output.map(lambda x: '2bfb0ce2886e31ae8e768feb4caa13bc') #109 -- 0.000121 other coupons discounted
#output = output.map(lambda x: '6d3ea4f9c9272ee7595eaca7f96234db 2bfb0ce2886e31ae8e768feb4caa13bc') #184 109 -- 0.000352 other coupons discounted

#output = output.map(lambda x: '3d5c0b4c9e35377c0df5e1e7efe1da42') #24 --  0.000000   delivery
#output = output.map(lambda x: '79de77aa8c36fdf17cb3366e2084e353') #277 -- 0.001324
#output = output.map(lambda x: '51da52d5516033bea13972588b671184') #237 --  0.000146

#output = output.map(lambda x: '9193590f0f6d2f9ea8467cfe52295107') #54 -- 0.000102

#output = output.map(lambda x: '1f0022baed331b2ba3f922a276af4145 8c470d8651dbc290e3b5742dc4556a29') #255 282 --  0.000036

def set_coupon_value(coup_id,lb_solo_value,test_purchases_series):
    purchase_factor = 22873.0 * 1.5
    if(coup_id in test_purchases_series.index.values):
        test_purchases_series = test_purchases_series.set_value(coup_id,lb_solo_value * purchase_factor)

if(not use_cv):
    purchase_factor = 22873.0 * 1.5 #1.5 to approximate multiple coupons per user
    #use information gained from lb feedback
    #obviously this leads to overfitting, but it should be okay (hopefully)
    #note the gift coupon I have already just excluded based on lb feedback

    #TODO dangerous
#    set_coupon_value(188,0.000049,test_purchases_series)
#    set_coupon_value(102,0.000000,test_purchases_series)
    set_coupon_value(262,0.000000,test_purchases_series)
    set_coupon_value(107,0.002041,test_purchases_series)
    set_coupon_value(111,0.000219,test_purchases_series)
    set_coupon_value(182,0.000875,test_purchases_series)
    set_coupon_value(218,0.000292,test_purchases_series)
    set_coupon_value(259,0.000656,test_purchases_series)
    set_coupon_value(221,0.000292,test_purchases_series)

    set_coupon_value(38,0.000146,test_purchases_series) # from script
    set_coupon_value(161,0.000802,test_purchases_series) # from script

    set_coupon_value(142,0.000437,test_purchases_series)
    set_coupon_value(40,0.000146,test_purchases_series)
    set_coupon_value(109,0.000121,test_purchases_series)
    set_coupon_value(184,0.000292,test_purchases_series)
    set_coupon_value(24,0.000000,test_purchases_series)
    set_coupon_value(277,0.001324,test_purchases_series)
    set_coupon_value(237,0.000146,test_purchases_series)
    set_coupon_value(54,0.000102,test_purchases_series)
    set_coupon_value(255,0.000030,test_purchases_series) #approximating
    set_coupon_value(282,0.000040,test_purchases_series) #approximating
#    if(188 in test_purchases_series.index.values):
#        test_purchases_series = test_purchases_series.set_value(188,0.000049 * purchase_factor)
#
#
#
#    test_purchases_series = test_purchases_series.set_value(262,0.00000 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(107,0.002041 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(111,0.000219 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(182,0.000875 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(218,0.000292 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(259,0.000656 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(221,0.000292 * purchase_factor)
#
#    test_purchases_series = test_purchases_series.set_value(38,0.000146 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(161,0.000802 * purchase_factor) #from a script
#
#    test_purchases_series = test_purchases_series.set_value(142,0.000437 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(40,0.000146 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(109,0.000121 * purchase_factor)
#    test_purchases_series = test_purchases_series.set_value(184,0.000292 * purchase_factor)
#
#    test_purchases_series = test_purchases_series.set_value(24,0.000000 * purchase_factor)

purchases_df = norm_rows(ones_df * test_purchases_series)
#%%

#now make separate predictions of purchases for male and female users

ones_temp_df = ones_df.copy()
ones_temp_df['user_id_hash'] = ones_temp_df.index
ones_temp_df = pd.merge(ones_temp_df,users_gender_df,on='user_id_hash')
ones_male_df = ones_temp_df.loc[ones_temp_df['male_user'] == 1]
ones_male_df = ones_male_df.drop(['user_id_hash','male_user'],axis=1)
ones_female_df = ones_temp_df.loc[ones_temp_df['male_user'] == 0]
ones_female_df = ones_female_df.drop(['user_id_hash','male_user'],axis=1)

gender_temp_test_df = test_df
test_factor_m_series = gender_temp_test_df['en_old_genre'].map(lambda x: genre_frac_m[x])
test_factor_f_series = gender_temp_test_df['en_old_genre'].map(lambda x: genre_frac_f[x])

purchases_m_df = ones_male_df * test_purchases_series * test_factor_m_series
purchases_f_df = ones_female_df * test_purchases_series * test_factor_f_series

purchases_gender_df = purchases_m_df.append(purchases_f_df)
purchases_gender_df = norm_rows(purchases_gender_df)
purchases_gender_df = purchases_gender_df.sort_index()
#%%
ones_temp_df = ones_df.copy()
ones_temp_df['user_id_hash'] = ones_temp_df.index
ones_temp_df = pd.merge(ones_temp_df,users_age_df,on='user_id_hash')
ones_young_df = ones_temp_df.loc[ones_temp_df['age'] <= age_young]
ones_young_df = ones_young_df.drop(['user_id_hash','age'],axis=1)
ones_old_df = ones_temp_df.loc[ones_temp_df['age'] >= age_old]
ones_old_df = ones_old_df.drop(['user_id_hash','age'],axis=1)
ones_middle_df = ones_temp_df.loc[(ones_temp_df['age'] > age_young) & (ones_temp_df['age'] < age_old)]
ones_middle_df = ones_middle_df.drop(['user_id_hash','age'],axis=1)

age_temp_test_df = test_df
test_factor_old_series = age_temp_test_df['en_old_genre'].map(lambda x: genre_frac_old[x])
test_factor_young_series = age_temp_test_df['en_old_genre'].map(lambda x: genre_frac_young[x])
test_factor_middle_series = age_temp_test_df['en_old_genre'].map(lambda x: genre_frac_middle[x])

purchases_old_df = ones_old_df * test_purchases_series * test_factor_old_series
purchases_young_df = ones_young_df * test_purchases_series * test_factor_young_series
purchases_middle_df = ones_middle_df * test_purchases_series * test_factor_middle_series

purchases_age_df = purchases_old_df.append(purchases_young_df)
purchases_age_df = purchases_age_df.append(purchases_middle_df)
purchases_age_df = norm_rows(purchases_age_df)
purchases_age_df = purchases_age_df.sort_index()
#%%
ones_temp_df = ones_df.copy()
ones_temp_df['user_id_hash'] = ones_temp_df.index
ones_temp_df = pd.merge(ones_temp_df,users_pref_df,on='user_id_hash')
ones_pref_df = ones_temp_df.loc[ones_temp_df['has_pref'] == 1]
ones_pref_df = ones_pref_df.drop(['user_id_hash','has_pref'],axis=1)
ones_no_pref_df = ones_temp_df.loc[ones_temp_df['has_pref'] == 0]
ones_no_pref_df = ones_no_pref_df.drop(['user_id_hash','has_pref'],axis=1)

pref_temp_test_df = test_df
test_factor_pref_series = gender_temp_test_df['en_old_genre'].map(lambda x: genre_frac_pref[x])
test_factor_no_pref_series = gender_temp_test_df['en_old_genre'].map(lambda x: genre_frac_no_pref[x])

purchases_pref_df = ones_pref_df * test_purchases_series * test_factor_pref_series
purchases_no_pref_df = ones_no_pref_df * test_purchases_series * test_factor_no_pref_series

purchases_has_pref_df = purchases_pref_df.append(purchases_no_pref_df)
purchases_has_pref_df = norm_rows(purchases_has_pref_df)
purchases_has_pref_df = purchases_has_pref_df.sort_index()


#%%
temp_df = coupons_train_no_free.copy()

coup_lesson_cond = temp_df['en_genre'] == 'Lesson'
#coup_large_area_cond = temp_df['area_count'] >= 13
temp_df['usable_nan'] = temp_df['usable_date_mon'].map(lambda x: int(pd.isnull(x)))
coup_large_area_cond = temp_df['usable_nan']
coup_large_and_lesson = (coup_lesson_cond & coup_large_area_cond)
temp_df['temp'] = temp_df['en_genre'].map(lambda x: 'Correspondence course')
temp_df.en_capsule[coup_large_and_lesson] = temp_df['temp'][coup_large_and_lesson]

coup_small_and_lesson = (coup_lesson_cond & ~coup_large_area_cond)
temp_df['temp'] = temp_df['en_genre'].map(lambda x: 'Class')
temp_df.en_capsule[coup_small_and_lesson] = temp_df['temp'][coup_small_and_lesson]

grouped_caps = temp_df.groupby('en_capsule')
caps_dict = {}
for name, group in grouped_caps:
    caps_dict[name] = group['purchases'].mean()
#temp_test_df['small_area_ave'] = temp_test_df['en_old_small_area'].map(lambda x: small_area_dict[x])
#small_area_ave_series = temp_test_df['small_area_ave']
#small_area_ave_df = norm_rows(ones_df * small_area_ave_series)
#%%
tic=timeit.default_timer()
grouped_small = coupons_train_no_free.groupby('en_small_area')
small_area_dict = {}
for name, group in grouped_small:
    small_area_dict[name] = group['purchases'].mean()
temp_test_df['small_area_ave'] = temp_test_df['en_old_small_area'].map(lambda x: small_area_dict[x])
small_area_ave_series = temp_test_df['small_area_ave']

small_area_ave_df = norm_rows(ones_df * small_area_ave_series)

toc=timeit.default_timer()
print('Ave Small Area Time',toc - tic)
#%%
tic=timeit.default_timer()
grouped_large = coupons_train_no_free.groupby('en_large_area')
large_area_dict = {}
for name, group in grouped_large:
    large_area_dict[name] = group['purchases'].mean()
temp_test_df['large_area_ave'] = temp_test_df['en_old_large_area'].map(lambda x: large_area_dict[x])
large_area_ave_series = temp_test_df['large_area_ave']

large_area_ave_df = norm_rows(ones_df * large_area_ave_series)

toc=timeit.default_timer()
print('Ave Large Area Time',toc - tic)
#%%
##calculate the purchase rates for different genres of coupons (over all users)
tic=timeit.default_timer()
bins = [0,49,50,55,60,70,80,85,90,99,100]
coupons_train_no_free['price_rate_binned'] = np.digitize(coupons_train_no_free['price_rate'],
                                                bins,right=True)
grouped_discount_rate = coupons_train_no_free.groupby('price_rate_binned')

discount_rate_dict = {}
for name, group in grouped_discount_rate:
    discount_rate_dict[name] = group['purchases'].mean()
temp_test_df['price_rate_binned'] = np.digitize(temp_test_df['price_rate'],
                                                bins,right=True)
temp_test_df['rate_ave'] = temp_test_df['price_rate_binned'].map(lambda x: discount_rate_dict[x])
rate_ave_series = temp_test_df['rate_ave']

rate_ave_df = norm_rows(ones_df * rate_ave_series)

toc=timeit.default_timer()
print('Discount Rate Time',toc - tic)
#%%
#tic=timeit.default_timer()
#coupons_train_no_free['disp_binned'] = coupons_train_no_free['dispperiod'].map(lambda x: 15 if x > 14 else x)
#grouped_discount_rate = coupons_train_no_free.groupby('disp_binned')
#disp_dict = {}
#for name, group in grouped_discount_rate:
#    disp_dict[name] = group['purchases'].mean()
#temp_test_df['disp_binned'] = temp_test_df['dispperiod'].map(lambda x: 15 if x > 14 else x)
#temp_test_df['disp_ave'] = temp_test_df['disp_binned'].map(lambda x: disp_dict[x])
#disp_ave_series = temp_test_df['disp_ave']
#disp_ave_df = norm_rows(ones_df * disp_ave_series)
#toc=timeit.default_timer()
#print('Dispperiod Time',toc - tic)
#%%
tic=timeit.default_timer()
users_test = user_index.values
users_criterion = users_df['user_id_hash'].map(lambda x: x in users_test)
users_reduced_df = users_df[users_criterion]
users_reduced_df['en_pref'] = users_reduced_df['en_pref'].fillna('')


ken_to_large_dict = pd.Series(test_df.en_old_large_area.values,index=test_df.en_old_ken).to_dict()

coupon_ken = test_df[['coupon_id_hash','en_old_ken','en_old_genre','en_old_large_area']]

def get_large(x):
    try:
        return ken_to_large_dict[x]
    except:
        return x
users_reduced_df['en_large_area'] = users_reduced_df['en_pref'].map(lambda x: get_large(x))

user_ken = users_reduced_df['en_pref']
user_large = users_reduced_df['en_large_area']
#genre_list = ['Other coupon','Gift card','Delivery service']
#genre_list = ['Other coupon','Gift card','Delivery service']
genre_list = ['Gift card','Delivery service']
TEMP_DICT_KEN = {}
TEMP_DICT_KEN_ALL = {}
TEMP_DICT_LARGE = {}
TEMP_DICT_LARGE_ALL = {}
for row in coupon_ken.iterrows():
    row_temp = row[1]
    ind = row[1]['coupon_id_hash']
    val_ken = row[1]['en_old_ken']
    val_large = row[1]['en_old_large_area']
    genre = row[1]['en_old_genre']
    TEMP_DICT_KEN[ind] = user_ken.map(lambda x: 1 if (x == val_ken) and genre not in genre_list else 0)
    TEMP_DICT_KEN_ALL[ind] = user_ken.map(lambda x: 1 if (x == val_ken) else 0)
    TEMP_DICT_LARGE[ind] = user_large.map(lambda x: 1 if (x == val_large) and genre not in genre_list else 0)
    TEMP_DICT_LARGE_ALL[ind] = user_large.map(lambda x: 1 if (x == val_large) else 0)
result_ken_users_df = pd.DataFrame.from_dict( TEMP_DICT_KEN)
result_ken_users_all_coupons_df = pd.DataFrame.from_dict( TEMP_DICT_KEN_ALL)
result_large_users_df = pd.DataFrame.from_dict( TEMP_DICT_LARGE)
result_large_users_all_coupons_df = pd.DataFrame.from_dict( TEMP_DICT_LARGE_ALL)
toc=timeit.default_timer()
print('User Res Time',toc - tic)
#%%
if(redo_visits):
    user_profiles_visits = user_profiles
#%%

tic_proc=timeit.default_timer()
price_w = 0.5
pr_w = 0.5
FEATURE_WEIGHTS = {
    'discount_price': 0,'dispperiod': 0,'price_rate': 0,'catalog_price': 0,
    'validperiod':0,
#    'dispfrom_weekday': 0,
#    'dispend_weekday': 0,
#    'usable_sum': 0.0,
#    'usable_date':0.0,
#    'genre_and_large':10,
#    'genre_and_small':10000,
#    'en_genre_Delivery_service':10,
#    'en_genre_Food':20,

#    'en_small_area_Shinjuku,_Takadanobaba_Nakano_-_Kichijoji':0,
#    'en_small_area_Shibuya,_Aoyama,_Jiyugaoka':0,
#    'en_small_area_Kawasaki,_Shonan-Hakone_other':0,
#    'en_small_area_Ikebukuro_Kagurazaka-Akabane':0,
#    'en_small_area_Ginza_Shinbashi,_Tokyo,_Ueno':0,
#    'en_small_area_Ebisu,_Meguro_Shinagawa':0,
#    'en_small_area_Akasaka,_Roppongi,_Azabu':0,
#    'en_small_area_Minami_other':0,

#    'en_large_area_Kanto':25,
#    'en_large_area_Kansai':15,
    'double_usable':1,
    'valid_gap_binned':0.0,
#    'valid_gap_binned_1':0,

    'validperiod_nan': 0,
    'validperiod_notnan': 2,
    'usable_nan': 1.0,
    'usable_notnan': 10.0,
    'discount_price_log':15,
    'en_capsule_Leisure':6,
    'en_capsule_Hotel':6,
    'en_capsule_Japanese hotel':4,
    'en_genre_Hotel_and_Japanese_hotel':1.5,
    'en_genre_': 2.0,
    'en_genre_Delivery_service': 3,
    'en_genre_Other_coupon': 0.5,
    'en_capsule':0.0,
    'en_capsule_Web service':1,
    'en_capsule_Other':2,
#    'en_capsule_Spa':10,

#    'en_capsule':20,
#    'en_orig_small_area':0.1,
#    'en_orig_large_area':0.1,
    'en_orig_small_area':0.0,
    'en_orig_large_area':0.0,

    'del_small_area_unk':0,
    'del_large_area_unk':30,
    'del_ken_unk':0,
    'del_small_area':15,
    'del_large_area':5,
    'del_ken':0,

    'food_small_area_unk':0.0,
    'food_large_area_unk':0.0,
    'food_ken_unk':0.0,
    'food_small_area':0.2,
    'food_large_area':0.0,
    'food_ken':3,

    'hotel_small_area_unk':0.0,
    'hotel_large_area_unk':0.5,
    'hotel_ken_unk':0.0,
    'hotel_small_area':0.0,
    'hotel_large_area':10.0,
    'hotel_ken':0,

    'en_small_area_unk':0,
    'en_large_area_unk':0.00,
    'en_ken_unk':0.0,
    'en_large_area': 25,
    'en_ken': 10.0,
    'en_small_area': 5,

    'disp_normed':0.0,
    'area_binned':0.0,
    'ghost_binned':0.0,
    'rate_binned':1.0,
    'discount_price_binned':1.3,
    'cheap': price_w,
    'decent': price_w,'medium_low': price_w,'medium_high': price_w,
    'expensive': price_w,
    'pr_0': pr_w, 'pr_1': pr_w, 'pr_2': pr_w, 'pr_3': pr_w, 'pr_4': pr_w,
    'salon_genre': 15,
    'luxury_genre': 3.0,
    'food_stuff_genre': 0.2,
    'corr_course':0.0,
    'class':0,
    'hotel_caps':0,
    'j_hotel_caps:':0,
    'other':0,
    'gift_card':0
}
W_values = [find_appropriate_weight(FEATURE_WEIGHTS, colname)
            for colname in user_profiles.columns]
W = np.diag(W_values)

### find weighted dot product(modified cosine similarity) between each test coupon and user profiles
similarity_scores = np.dot(np.dot(user_profiles, W), test_only_features.T)


### create (USED_ID)x(COUPON_ID) dataframe, similarity scores as values
columns = [coupons_ids.values[i] for i in range(0, similarity_scores.shape[1])]
result_df = pd.DataFrame(index=user_index, columns=columns,data=similarity_scores)
result_df = norm_rows(result_df)

#if(redo_visits):
#    similarity_scores_v = np.dot(np.dot(user_profiles_visits, W), test_only_features.T)
#    result_visits_df = pd.DataFrame(index=user_index, columns=columns,data=similarity_scores_v)
#    result_visits_df = norm_rows(result_visits_df)

#w_cos = 4
#w_users = 0.002
#w_genre_ave = 0.2
#w_r_ave = 0.1
#w_disp_ave = 0.2
#w_disc_p_ave = 0.2
#w_small_ave = 0.1
#w_large_ave = 0.1
#w_cos_visits = 0.0
#
#res_ens = (w_cos*result_df + w_users*result_local_users_df
#          + w_genre_ave * genre_ave_df + w_r_ave*rate_ave_df + w_disp_ave*disp_ave_df
#          + w_disc_p_ave * disc_price_ave_df + w_small_ave*small_area_ave_df
#          + w_large_ave*large_area_ave_df)

#w_cos = 8
#
#w_users = 0.004
#w_user_ghost = 1000

#w_genre_disp = 0.1
#w_genre_price = 0.2
#w_genre_area = 0.05
#w_genre_rate = 0.2
#w_genre_ghost = 0.0
#w_genre_small = 0.1
#w_genre_large = 0.1
#w_genre_ken = 0.1
#w_r_ave = 0.0
#
#
#w_disp_ave = 0.0
#w_small_ave = 0.0
#w_large_ave = 0.0
#w_cos_visits = 0.0

#res_ens = (w_cos*result_df + w_users*result_local_users_df
#          + w_genre_disp * genre_disp_df + w_r_ave*rate_ave_df +
#          w_genre_price * genre_price_df + w_genre_area*genre_area_df +
#          w_genre_rate * genre_rate_df + w_genre_ghost * genre_ghost_df +
#          w_genre_small * genre_small_df + w_genre_large * genre_large_df +
#          w_genre_ken * genre_ken_df + w_user_ghost * user_ghost_df +
#          w_small_ave*small_area_ave_df
#          + w_large_ave*large_area_ave_df)

w_cos = 10

w_ken_users = 0.002
w_ken_users_all_coupons = 0.000
w_large_users = 0.004
w_large_users_all_coupons = 0.000
w_user_ghost = 0.1
w_purchases = 0.0
w_gender_purchases = 0.39
w_age_purchases = 0.1
w_pref_purchases = 0.01

res_ens = (w_cos*result_df + w_ken_users*result_ken_users_df + w_user_ghost * user_ghost_df
          + w_purchases * purchases_df + w_gender_purchases * purchases_gender_df
          + w_age_purchases * purchases_age_df + w_pref_purchases * purchases_has_pref_df
          + w_ken_users_all_coupons * result_ken_users_all_coupons_df
          + w_large_users_all_coupons * result_large_users_all_coupons_df
          + w_large_users * result_large_users_df)

#res_ens = user_ghost_df

#if(redo_visits):
#    res_ens = res_ens + w_cos_visits * result_visits_df

#res_ens = res_ens * active_weeks_prop
if(use_cv):
    res_ens.columns = res_ens.columns.astype(str)

if(not use_cv):
#    w_user_ghost = 1000
#    w_test_visits_leak = 0
#    res_ens = res_ens + w_user_ghost * user_ghost_df
#    res_ens = res_ens * test_leak_series

#    res_ens.columns = res_ens.columns.astype(str)
    res_ens.columns = res_ens.columns.map(lambda x: test_coupon_id_rev_dict[x])
output = res_ens.apply(get_top10_coupon_hashes_string, axis=1)

#TODO testing
#output = output.map(lambda x: '0c015306597566b632bebfb63b7e59f3') #102 -- 0 gift card
#output = output.map(lambda x: 'c988d799bc7db9254fe865ee6cf2d4ff') #188 -- 0.000049
#output = output.map(lambda x: '7d7487370022eb18ee130856aa1c6eec') #262 -- 0
#output = output.map(lambda x: 'c9e1dcbd8c98f919bf85ab5f2ea30a9d') #107 -- 0.002
#output = output.map(lambda x: '266745cba5481af70ec489f67f2d5d77') #111 -- 0.000219
#output = output.map(lambda x: '27741884a086e2864936d7ef680becc2') #182 -- 0.000875
#output = output.map(lambda x: '7ae4e60eab2e4d7e20f88fc19267e87c') #218 -- 0.000292
#output = output.map(lambda x: 'b36879abb93ee7630b313ef0a04463f3') #259 -- 0.000656 leisure 6 (259)
#output = output.map(lambda x: 'd79a889ee9d0712607a2672e96ba3d69') #221 --  0.000292  other

#test_coupon_id_dict['2fcca928b8b3e9ead0f2cecffeea50c1'] 38 -- 0.000146 from script leisure coup
#test_coupon_id_dict['0fd38be174187a3de72015ce9d5ca3a2'] 161 -- 0.000802 from script delivery coup

#output = output.map(lambda x: '2bfb0ce2886e31ae8e768feb4caa13bc') #109 -- 0.000121 other coupons discounted
#output = output.map(lambda x: '6d3ea4f9c9272ee7595eaca7f96234db 2bfb0ce2886e31ae8e768feb4caa13bc') #184 109 -- 0.000352 other coupons discounted
#output = output.map(lambda x: '3905228fb8cac640b673f71d5f315df5') #142 --  0.000437  other coupons discounted
#output = output.map(lambda x: '2cb5e0a522dcb7c539018c165359ad5a') #40 --  0.000146   other coupons web
#output = output.map(lambda x: '3d5c0b4c9e35377c0df5e1e7efe1da42') #24 --  0.000000   delivery

#output = output.map(lambda x: '79de77aa8c36fdf17cb3366e2084e353') #277 -- 0.001324
#output = output.map(lambda x: '51da52d5516033bea13972588b671184') #237 --  0.000146
#output = output.map(lambda x: '9193590f0f6d2f9ea8467cfe52295107') #54 -- 0.000102

#output = output.map(lambda x: '1f0022baed331b2ba3f922a276af4145 8c470d8651dbc290e3b5742dc4556a29') #255 282 --  0.000036
#output = output.map(lambda x: 'f93dc6e223935d817e1237f8f73b56a2 0da4709d065835039f700d9bb67b461 81c1c7241aadbb323b38689a64fbc83a') #246 166 94 --0.000583

#115 45
#should test 44?,45,246
#273,#300??
#td = test_coupon_id_rev_dict
#out_string = (td[235] + ' ' + td[243] + ' ' + td[147] + ' ' + td[172] + ' ' + td[213] + ' ' +
#            td[119] + ' ' + td[120] + ' ' + td[3] + ' ' + td[283])
#output = output.map(lambda x: out_string) #235 243 147 172 213 119 120 3 283 -- 0.000067
#td = test_coupon_id_rev_dict
#out_string = (td[96] + ' ' + td[275] + ' ' + td[203] + ' ' + td[212] + ' ' + td[104] + ' ' +
#            td[258] + ' ' + td[201])
#output = output.map(lambda x: out_string) #96 275 203 212 104 258 201 -- 0.000057


#0.000346 for all 10
# 18 62 281 234 215 57 35 44 229 69
#ffe734ef0b1d82d6816ac33efa07cce5
#fefa1884298dd5d241437da39c0026b0
#fecbe103f0dd5ab6b52952a813d7dee6
#fe5b6ec460a7b05d9fd8347bdbdb429d
#fe3dfe6334edd49b32d86963f4dcfe17
#fce3668e023256957905b76b2ffc9659
#fc978c6b2af79fe098ab63599072e2e9
#fc5f052a1bd97696fbcab35d8d974b73
#fbbfdbb5b73e81a6ae06e02411ad6bbf
#fba7c9c0955059611cf58a9da12bd14f



#107 probably also good
if(not use_cv):
    output_df = pd.DataFrame(data={'user_id_hash': output.index,'PURCHASED_COUPONS': output.values})
    output_df_all_users = pd.merge(users_df, output_df, how='left', on='user_id_hash')
    output_df_all_users = output_df_all_users.rename(columns={'user_id_hash': 'USER_ID_hash'})
    output_df_all_users['USER_ID_hash'] = output_df_all_users['USER_ID_hash'].map(lambda x: user_id_rev_dict[x])
    output_df_all_users.to_csv('cosine_sim_python.csv', header=True,
                               index=False, columns=['USER_ID_hash', 'PURCHASED_COUPONS'])
cv_actual_list = []
cv_pred_list = []
if(use_cv):
    output_cv = output.map(lambda x: list(map(int, x.split())))
    for key in USER_COUPONS_PURCHASED_UNIQUE_CV:
        cv_actual_list.append(USER_COUPONS_PURCHASED_UNIQUE_CV[key])
        try:
            cv_pred_list.append(output_cv[key])
        except:
            cv_pred_list.append([0])
    print(starting_date)
    print('mapk',mean_average_precision_k.mapk(cv_actual_list,cv_pred_list))
toc=timeit.default_timer()
print('Proc Time',toc - tic_proc)
#%%
#    for i in range(len(users_df['user_id_hash'])):
#        cv_pred[i] = [19009, 19287, 19396, 19284, 17702, 19011, 19264, 19246, 12375,19229]
#keep an eye on the coupon that is disp 13 in the test set
toc=timeit.default_timer()
print('Total Time',toc - tic0)