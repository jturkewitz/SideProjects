# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:18:38 2015

@author: Jared
"""
import csv as csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import random
import sys
#from sklearn import preprocessing
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm, cross_validation
#from sklearn.metrics import mean_squared_error, roc_auc_score
#import sklearn.metrics as skm
from scipy.stats import linregress
#from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV,LogisticRegression
from sklearn.grid_search import RandomizedSearchCV
sys.path.append('/Users/Jared/DataAnalysis/xgboost-master/wrapper/')
# insert path to wrapper above
import xgboost as xgb
#import scipy.sparse
#import pickle
import timeit
#from scipy.optimize import minimize
#from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.metrics import log_loss,roc_auc_score
#%%
tic0=timeit.default_timer()
train_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook/train.csv', header=0)
test_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook/test.csv', header=0)
bids_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook/bids2.csv', header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic0)
#%%
df = pd.concat([train_df,test_df], ignore_index=True)
unique_bidders = train_df['bidder_id'].unique()
bids_df['time'] = bids_df['time'] - 9e15
bid_id_dict = dict(zip(df.bidder_id, df.index))
inv_bid_id_dict = {v: k for k, v in bid_id_dict.items()}
train_df['bidder_id'] = train_df['bidder_id'].map(lambda x: bid_id_dict[x])
#%%
tic=timeit.default_timer()
print('making dicts')
AUCTIONS = {}
for i in bids_df['auction_id'].unique():
    key = 'auc'+str(i)
    AUCTIONS[key] = bids_df[bids_df['auction_id'] == i]

auc1_df = AUCTIONS['auc1']
auc2_df = AUCTIONS['auc2']
auc6504_df = AUCTIONS['auc6504']

tic=timeit.default_timer()
BIDDERS = {}
for i in bids_df['bidder_id'].unique():
    key = 'bidder'+str(i)
    BIDDERS[key] = bids_df[bids_df['bidder_id'] == i]

bidder1_df = BIDDERS['bidder1']
bidder2_df = BIDDERS['bidder2']
bidder392_df = BIDDERS['bidder392']
bidder871_df = BIDDERS['bidder871']
bidder988_df = BIDDERS['bidder988']
bidder1211_df = BIDDERS['bidder1211']
bidder1534_df = BIDDERS['bidder1534']
bidder980_df = BIDDERS['bidder980']
bidder1102_df = BIDDERS['bidder1102']

toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
print('calculating bots, humans, and unknowns per auction id')
AUC_BOTS = {}
AUC_UNKNOWNS = {}
AUC_HUMANS = {}
for key in AUCTIONS:
    bots = 0
    unknowns = 0
    humans = 0
    auc = AUCTIONS[key]
    for bidder in auc.bidder_id.unique():
        subset =  train_df[train_df.bidder_id == bidder]
        if (not (subset.empty)):
            if(subset.outcome.iloc[0] == 1):
                bots = bots + 1
            else:
                humans = humans + 1
        else:
            unknowns = unknowns + 1
    AUC_BOTS[key] = bots
    AUC_UNKNOWNS[key] = unknowns
    AUC_HUMANS[key] = humans
toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
AUC_TIME0 = {}
AUC_TIME1 = {}
AUC_TIME2 = {}
AUC_TIME3 = {}
ending_times = []
starting_times = []
for key in AUCTIONS:
    temp_auc = AUCTIONS[key]
    starting_time = temp_auc['time'].iloc[0]
    ending_time = temp_auc['time'].iloc[len(temp_auc) - 1]
    starting_times.append(starting_time)
    ending_times.append(ending_time)
    if (starting_time <= 6.6e+14 and ending_time <= 6.6e14):
        AUC_TIME0[key] = 1
    else:
        AUC_TIME0[key] = 0
    if (starting_time <= 6.6e+14 and ending_time > 6.8e14):
        AUC_TIME1[key] = 1
    else:
        AUC_TIME1[key] = 0
    if (starting_time >= 6.6e+14 and starting_time <= 7.2e14):
        AUC_TIME2[key] = 1
    else:
        AUC_TIME2[key] = 0
    if (ending_time >= 7.4e+14):
        AUC_TIME3[key] = 1
    else:
        AUC_TIME3[key] = 0
toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
print('calculating overbids')
auc1_bidders = auc1_df['bidder_id']
BIDDERS_OVERBIDS = {}
for key in BIDDERS:
    BIDDERS_OVERBIDS[key] = 0
for auc_key in AUCTIONS:
    prev_bidder_id = 0
    auc_bidders = AUCTIONS[auc_key]['bidder_id']
    for index in auc_bidders.index.values:
        current_bidder_id = auc_bidders.ix[index]
        if (current_bidder_id == prev_bidder_id):
            current = 'bidder'+str(current_bidder_id)
            BIDDERS_OVERBIDS[current] = BIDDERS_OVERBIDS[current] + 1
        prev_bidder_id = current_bidder_id
toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
print('calculating won and started auctions')
auc1_bidders = auc1_df['bidder_id']
BIDDERS_WONAUCS = {}
BIDDERS_STARTEDAUCS = {}
for key in BIDDERS:
    BIDDERS_WONAUCS[key] = 0
    BIDDERS_STARTEDAUCS[key] = 0
for auc_key in AUCTIONS:
    prev_bidder_id = 0
    auc_ending_row = AUCTIONS[auc_key].tail(1)
    auc_ending_time = auc_ending_row['time'].iloc[0]
    auc_ending_bidder = auc_ending_row['bidder_id'].iloc[0]
    auc_starting_row = AUCTIONS[auc_key].head(1)
    auc_starting_time = auc_starting_row['time'].iloc[0]
    auc_starting_bidder = auc_starting_row['bidder_id'].iloc[0]
    bidder = 'bidder'+str(auc_ending_bidder)
    if(abs(auc_ending_time - 6.4555e14) >= 0.001e14 and
        abs(auc_ending_time - 7.0918e14) >= 0.001e14 and
        abs(auc_ending_time - 7.728e14) >= 0.001e14):
            BIDDERS_WONAUCS[bidder] = BIDDERS_WONAUCS[bidder] + 1
    if(abs(auc_starting_time - 6.31917e14) >= 0.001e14 and
        abs(auc_ending_time - 7.5924e14) >= 0.001e14):
            BIDDERS_STARTEDAUCS[bidder] = BIDDERS_STARTEDAUCS[bidder] + 1

toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
print('calculating auction switches')
BIDDERS_AUC_SWITCHES = {}
for key in BIDDERS:
    prev_auc = 0
    auc_switches = 0
    bidder_df = BIDDERS[key]
    auc_id = bidder_df['auction_id']
    for index in auc_id.index.values:
        current_auc = auc_id.ix[index]
        if(current_auc != prev_auc):
            auc_switches = auc_switches + 1
        prev_auc = current_auc
    BIDDERS_AUC_SWITCHES[key] = auc_switches

toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
print('calculating url switches')
BIDDERS_URL_SWITCHES = {}
for key in BIDDERS:
    prev_url = -100
    url_switches = 0
    bidder_df = BIDDERS[key]
    url_id = bidder_df['url_id']
    for index in url_id.index.values:
        current_url = url_id.ix[index]
        if(current_url != prev_url):
            url_switches = url_switches + 1
        prev_url = current_url
    BIDDERS_URL_SWITCHES[key] = url_switches

toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
print('calculating device switches')
BIDDERS_DEV_SWITCHES = {}
for key in BIDDERS:
    prev_dev = -100
    dev_switches = 0
    bidder_df = BIDDERS[key]
    dev_id = bidder_df['device_id']
    for index in dev_id.index.values:
        current_dev = dev_id.ix[index]
        if(current_dev != prev_dev):
            dev_switches = dev_switches + 1
        prev_dev = current_dev
    BIDDERS_DEV_SWITCHES[key] = dev_switches

toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
print('Iterating over bidders and auctions')
bidder_ids = []
num_bids = []
num_aucs = []
num_urls = []
num_countries = []
num_devices = []
num_ips = []
primary_device = []
primary_country = []
primary_url = []
primary_ip = []
secondary_device = []
secondary_country = []
secondary_url = []
secondary_ip = []
merc_id = []
outcomes = []
won_auctions = []
started_auctions = []
aucs_involved_all_bids = []
mean_frac_bids = []
mean_frac_bots_auctions = []
has_ip_79691 = []
has_auc_time0 = []
has_auc_time1 = []
has_auc_time2 = []
has_auc_time3 = []
num_self_overbids = []
time_diffs = []
auc_switches = []
url_switches = []
dev_switches = []
for key in BIDDERS:
    bidder_df = BIDDERS[key]
    if (bidder_df.empty):
        print(key)
        continue
    bidder_id = int(key.lstrip('bidder'))
    auctions = bidder_df['auction_id'].unique()
    outcomes.append(df['outcome'][bidder_id])
    bidder_ids.append(bidder_id)
    num_bids.append(len(bidder_df.index))
    num_aucs.append(len(auctions))
    num_urls.append(len(bidder_df['url_id'].unique()))
    num_countries.append(len(bidder_df['country_id'].unique()))
    num_devices.append(len(bidder_df['device_id'].unique()))
    num_ips.append(len(bidder_df['ip_id'].unique()))

    has_ip_79691.append(int(79691 in bidder_df['ip_id'].values))
    time_diffs.append(bidder_df['time'].max() - bidder_df['time'].min())

    devs = bidder_df['device_id'].value_counts()
    countries = bidder_df['country_id'].value_counts()
    ips = bidder_df['ip_id'].value_counts()
    urls = bidder_df['url_id'].value_counts()
    primary_device.append(devs.index[0])
    primary_country.append(countries.index[0])
    primary_ip.append(ips.index[0])
    primary_url.append(urls.index[0])
    try:
        secondary_device.append(devs.index[1])
    except:
        secondary_device.append(-2)
    try:
        secondary_country.append(countries.index[1])
    except:
        secondary_country.append(-2)
    try:
        secondary_ip.append(ips.index[1])
    except:
        secondary_ip.append(-2)
    try:
        secondary_url.append(urls.index[1])
    except:
        secondary_url.append(-2)

    merc_id.append(bidder_df['merchandise_id'].iloc[0])

    num_self_overbids.append(BIDDERS_OVERBIDS[key])

    won_auctions.append(BIDDERS_WONAUCS[key])
    started_auctions.append(BIDDERS_STARTEDAUCS[key])

    auc_switches.append(BIDDERS_AUC_SWITCHES[key])
    url_switches.append(BIDDERS_URL_SWITCHES[key])
    dev_switches.append(BIDDERS_DEV_SWITCHES[key])
    all_auc_bids = 0
    frac_bids = []
    #this may lead to leakage, but alternatives like using all auctions in fit are unwieldy
    status = 0
    subset =  train_df[train_df.bidder_id == bidder_id]
    if (not (subset.empty)):
        if(subset.outcome.iloc[0] == 1):
            status = 1
        else:
            status = 0
    else:
        status = 2

    bots_frac = []
    time0 = []
    time1 = []
    time2 = []
    time3 = []
    for auc in auctions:
        auc_key = 'auc'+str(auc)
        time0.append(AUC_TIME0[auc_key])
        time1.append(AUC_TIME1[auc_key])
        time2.append(AUC_TIME2[auc_key])
        time3.append(AUC_TIME3[auc_key])
        bots =  AUC_BOTS[auc_key]
        humans =  AUC_HUMANS[auc_key]
        unknowns =  AUC_UNKNOWNS[auc_key]
        if(status == 2):
            unknowns = unknowns - 1
        elif(status == 0):
            humans = humans -1
        else:
            bots = bots - 1
        total = bots + humans + unknowns
        if(total != 0):
            bots_frac.append(bots / total)
        else:
            bots_frac.append(0)
    if(all(time == 0 for time in time0)):
        has_auc_time0.append(0)
    else:
        has_auc_time0.append(1)
    if(all(time == 0 for time in time1)):
        has_auc_time1.append(0)
    else:
        has_auc_time1.append(1)
    if(all(time == 0 for time in time2)):
        has_auc_time2.append(0)
    else:
        has_auc_time2.append(1)
    if(all(time == 0 for time in time3)):
        has_auc_time3.append(0)
    else:
        has_auc_time3.append(1)
    mean_frac_bots_auctions.append(np.mean(bots_frac))
    aucs_involved_all_bids.append(1)
    mean_frac_bids.append(1)
toc=timeit.default_timer()
print('Time',toc - tic)
#%%
tic=timeit.default_timer()
df_dict = {'bidder_id': bidder_ids, 'outcome': outcomes, 'num_bids': num_bids,
           'num_aucs': num_aucs, 'num_urls': num_urls,
           'num_self_overbids':num_self_overbids,
           'primary_country': primary_country, 'primary_device': primary_device,
           'primary_ip': primary_ip, 'primary_url': primary_url,
           'secondary_country': secondary_country, 'secondary_device': secondary_device,
           'secondary_ip': secondary_ip, 'secondary_url': secondary_url,
           'has_ip_79691':has_ip_79691, 'time_diffs':time_diffs,
           'has_auc_time0':has_auc_time0,'has_auc_time1':has_auc_time1,
           'has_auc_time2':has_auc_time2, 'has_auc_time3':has_auc_time3,
           'num_countries': num_countries, 'num_devices': num_devices, 'num_ips': num_ips,
           'merc_id': merc_id, 'won_auctions': won_auctions, 'started_auctions': started_auctions,
           'aucs_involved_all_bids': aucs_involved_all_bids, 'auc_switches':auc_switches,
           'url_switches':url_switches, 'dev_switches':dev_switches,
           'mean_frac_bids': mean_frac_bids,
           'mean_frac_bots_auctions': mean_frac_bots_auctions}
bidders_df = pd.DataFrame(data=df_dict)
bidders_df['ratio_won'] = bidders_df['won_auctions'] / bidders_df['num_aucs']

bidders_df['always_lost'] = bidders_df['won_auctions'].map(lambda x: 0 if x >=1 else 1)
bidders_df['one_bid'] = bidders_df['num_bids'].map(lambda x: 1 if x ==1 else 0)
bins = [0.0,0.005,0.01,0.015,0.0199,0.025,0.035,0.047,0.08,0.13,0.249,1.0]
bidders_df['ratio_won_binned'] = np.digitize(bidders_df['ratio_won'], bins, right=True)
bins = [0.0,0.0001,0.0002,0.005,0.1,1.01]
bidders_df['mean_frac_bids_binned'] = np.digitize(bidders_df['mean_frac_bids'], bins, right=True)

bins = [0,1,2,4,6,8,10,15,20,30,40,50,60,70,80,90,100,125,150,175,200,
        250,300,400,500,750,1000,2500,5000,15000,100000000]
bidders_df['num_bids_binned'] = np.digitize(bidders_df['num_bids'],bins)

bins = [0,1,2,4,6,8,11,15,20,25,30,40,50,60,80,100,125,150,175,200,
        250,300,350,400,500,750,1000,2500,5000,15000,100000000]
bidders_df['num_self_overbids_binned'] = np.digitize(bidders_df['num_self_overbids'],bins)

bins = [0,1,2,4,6,8,11,15,20,30,45,60,75,100,140,280,580,1000,3000,100000000]
bidders_df['num_urls_binned'] = np.digitize(bidders_df['num_urls'],bins)

bins = [0,1,2,4,6,8,10,14,18,24,40,75,100,130,170,210,280,400,900,10000000]
bidders_df['num_devices_binned'] = np.digitize(bidders_df['num_devices'],bins)

bins = [0,1,2,4,6,8,10,12,15,21,30,50,75,100,150,250,350,500,1000,5000,15000,10000000]
bidders_df['num_ips_binned'] = np.digitize(bidders_df['num_ips'],bins)

bins = [0,1,2,4,8,12,15,18,24,32,45,55,70,90,110,150,250,500,1000000]
bidders_df['num_aucs_binned'] = np.digitize(bidders_df['num_aucs'],bins)

bins = [0,1,2,4,6,9,12,15,18,21,24,27,32,45,60,75,90,110,1000000]
bidders_df['num_countries_binned'] = np.digitize(bidders_df['num_countries'],bins)
#bidders_df['num_countries_binned'] = bidders_df['num_countries']

bins = [0,4e10,2.3e11,1e12,2e12,3.5e12,5e12,6.5e12,8e12,1e13,1.2e13,1.3e13,
        1.33e13,1.4e13,6.5e13,7.1e13,7.6e13,7.7e13,7.725e13,1e14]
bidders_df['time_diffs_binned'] = np.digitize(bidders_df['time_diffs'],bins)

bidders_df['aucs_inv_all_binned'] = np.digitize(np.log(bidders_df['aucs_involved_all_bids']),bins)
bidders_df['bids_per_auc'] = bidders_df['num_bids'] / bidders_df['num_aucs']
bidders_df['self_overbids_per_auc'] = bidders_df['num_self_overbids'] / bidders_df['num_aucs']
bidders_df['self_overbids_per_bid'] = bidders_df['num_self_overbids'] / bidders_df['num_bids']
bidders_df['bids_per_ip'] = bidders_df['num_bids'] / bidders_df['num_ips']
bidders_df['bids_per_device'] = bidders_df['num_bids'] / bidders_df['num_devices']
bidders_df['bids_per_country'] = bidders_df['num_bids'] / bidders_df['num_countries']
bidders_df['bids_per_url'] = bidders_df['num_bids'] / bidders_df['num_urls']
bins = [0.0,1.01,2.001,3.001,4.0001,5.0001,7.51,10.001,15.01,20.01,30.01,50.01,10000]
bidders_df['bids_per_auc_binned'] = np.digitize(bidders_df['bids_per_auc'],bins)

bins = [0.0,1.01,2.001,3.001,4.0001,5.0001,7.51,10.001,15.01,20.01,30.01,50.01,10000]
bidders_df['self_overbids_per_auc_binned'] = np.digitize(bidders_df['self_overbids_per_auc'],bins)

bins = [0.0,0.005,0.0075,0.013,0.019,0.025,0.036,0.06,0.099,0.15,0.33,0.45,0.6,0.75,1]
bidders_df['self_overbids_per_bid_binned'] = np.digitize(bidders_df['self_overbids_per_bid'],bins)

bins = [0.0,1.001,1.5,2.001,3.001,4.0001,5.0001,10.001,15.01,20.01,30.01,50.01,100,1000,10000]
bidders_df['bids_per_ip_binned'] = np.digitize(bidders_df['bids_per_ip'],bins)
bins = [0.0,1,1.15,1.26,1.33,1.49,1.51,1.6,1.75,1.99,2.001,2.19,2.45,2.99,3.001,3.46,
        3.99,4.5,5.5,6.5,8,10,13,18,25,50,150,300,1000,10000000]
bidders_df['bids_per_device_binned'] = np.digitize(bidders_df['bids_per_device'],bins,right=True)
bins = [0.0,1,1.39,1.6,1.99,2.001,2.5,2.99,3.0001,3.5,3.99,4.01,4.99,5.01,5.99,
        6.99,7.99,9.99,12,15,18,22,30,40,55,90,200,800,100000000]
bidders_df['bids_per_country_binned'] = np.digitize(bidders_df['bids_per_country'],bins,right=True)
bins = [0.0,1,1.3,1.49,1.55,1.7,1.99,2.001,2.49,2.99,3.01,3.99,4.5,5.99,7.99,
        11,15,25,40,70,150,300,900,1000000000]
bidders_df['bids_per_url_binned'] = np.digitize(bidders_df['bids_per_url'],bins,right=True)

bidders_df['devices_per_auc'] = bidders_df['num_devices'] / bidders_df['num_aucs']
bins = [0.0,0.161,0.499,0.5001,0.699,0.999,1.001,1.3,1.99,2.99,4.99,10000]
bidders_df['devices_per_auc_binned'] = np.digitize(bidders_df['devices_per_auc'],bins)

bidders_df['urls_per_device'] = bidders_df['num_urls'] / bidders_df['num_devices']

bidders_df['urls_per_auc'] = bidders_df['num_urls'] / bidders_df['num_aucs']
bidders_df['ips_per_device'] = bidders_df['num_ips'] / bidders_df['num_devices']
bidders_df['ips_per_auc'] = bidders_df['num_ips'] / bidders_df['num_aucs']
bidders_df['countries_per_device'] = bidders_df['num_countries'] / bidders_df['num_devices']
bidders_df['countries_per_auc'] = bidders_df['num_countries'] / bidders_df['num_aucs']

bins = [0.0,0.1,0.25,0.5,0.99,1.4,1.99,2,5,5,10,1000]
bidders_df['urls_per_device_binned'] = np.digitize(bidders_df['urls_per_device'],bins)
bins = [0.0,0.7,0.99,1.01,1.49,2,6,12,10000]
bidders_df['ips_per_device_binned'] = np.digitize(bidders_df['ips_per_device'],bins)
bins = [0.0,0.1,0.16,0.24,0.37,0.5,0.66,0.99,1.01,2.01,10000]
bidders_df['countries_per_device_binned'] = np.digitize(bidders_df['countries_per_device'],bins)

bins = [0.0,0.07,0.21,0.499,0.501,0.65,0.999,1.0001,2.001,10000]
bidders_df['urls_per_auc_binned'] = np.digitize(bidders_df['urls_per_auc'],bins)
bins = [0.0,0.25,0.66,0.999,1.0001,1.4,1.99,2.3,5,10000]
bidders_df['ips_per_auc_binned'] = np.digitize(bidders_df['ips_per_auc'],bins)
bins = [0.0,0.1,0.19,0.3,0.42,0.599,0.99,1.001,1.99,10000]
bidders_df['countries_per_auc_binned'] = np.digitize(bidders_df['countries_per_auc'],bins)

bidders_df['urls_per_ip'] = bidders_df['num_urls'] / bidders_df['num_ips']
bins = [0.0,0.05,0.105,0.19,0.3,0.499,0.5005,0.7,0.999,1.001,2.01,10000]
bidders_df['urls_per_ip'] = np.digitize(bidders_df['urls_per_ip'],bins)

bins = [0,1,2,3,4,6,8,11,18,50,1000000]
bidders_df['num_aucs_won_binned'] = np.digitize(bidders_df['won_auctions'],bins)

bins = [0,1,2,3,4,6,8,11,18,32,90,1000000]
bidders_df['num_aucs_started_binned'] = np.digitize(bidders_df['started_auctions'],bins)

bins = [0,1,2,3,4,6,8,11,15,20,30,45,60,90,130,190,280,400,550,700,1100,1800,3000,5000,
        10000,25000,10000000000]
bidders_df['auc_switches_binned'] = np.digitize(bidders_df['auc_switches'],bins,right=True)

bidders_df['auc_switches_per_bid'] = bidders_df['auc_switches'] / bidders_df['num_bids']
bidders_df['auc_switches_per_auc'] = bidders_df['auc_switches'] / bidders_df['num_aucs']

bins = [0,0.07,0.25,0.35,0.499,0.501,0.66,0.749,0.8,0.85,0.899,0.92,0.935,0.965,0.99,1,100]
bidders_df['auc_switches_per_bid_binned'] = np.digitize(bidders_df['auc_switches_per_bid'],bins,right=True)
bins = [0,1,1.1,1.15,1.24,1.35,1.49,1.6,1.76,1.99,2.2,2.5,3.0,4.0,6.0,8.5,12,17,25,70,100000000]
bidders_df['auc_switches_per_auc_binned'] = np.digitize(bidders_df['auc_switches_per_auc'],bins,right=True)

bins = [0,1,2,3,5,8,11,15,25,45,70,100,200,350,700,1500,3000,10000,1000000000]
bidders_df['url_switches_binned'] = np.digitize(bidders_df['url_switches'],bins,right=True)

bidders_df['url_switches_per_url'] = bidders_df['url_switches'] / bidders_df['num_urls']
bidders_df['url_switches_per_bid'] = bidders_df['url_switches'] / bidders_df['num_bids']
bins = [0,1,1.2,1.35,1.5,1.75,1.99,2.2,2.5,3,3.8,5,8,14,40,100000000]
bidders_df['url_switches_per_url_binned'] = np.digitize(bidders_df['url_switches_per_url'],bins,right=True)
bins = [0,0.005,0.04,0.143,0.329,0.4,0.499,0.501,0.64,0.72,0.8,0.86,0.9,0.94,0.99,1,2]
bidders_df['url_switches_per_bid_binned'] = np.digitize(bidders_df['url_switches_per_bid'],bins,right=True)

bidders_df['dev_switches_per_dev'] = bidders_df['dev_switches'] / bidders_df['num_devices']
bins = [0,1,1.2,1.35,1.49,1.7,2,2.5,3,4.5,7.5,15,40,70,100000]
bidders_df['dev_switches_per_dev_binned'] = np.digitize(bidders_df['dev_switches_per_dev'],bins,right=True)
bins = [0,1,2,3,5,8,13,20,30,45,75,120,170,280,500,800,1600,3200,10000,40000,1000000000]
bidders_df['dev_switches_binned'] = np.digitize(bidders_df['dev_switches'],bins,right=True)

bidders_df['num_countries_per_bid'] = bidders_df['num_countries'] / bidders_df['num_bids']
bins = [0.0,0.1,0.3,0.5,1.0]
bidders_df['num_con_per_bid_binned'] = np.digitize(bidders_df['num_countries_per_bid'],bins)

bidders_df['num_urls_per_bid'] = bidders_df['num_urls'] / bidders_df['num_bids']
bins = [0.0,0.1,0.3,0.5,1.0]
bidders_df['num_urls_per_bid_binned'] = np.digitize(bidders_df['num_urls_per_bid'],bins)
bins = [0.0,0.02,0.05,0.1,0.15,0.2,0.3,1.0]
bidders_df['mean_frac_bots_auctions_binned'] = np.digitize(bidders_df['mean_frac_bots_auctions'],bins)

def combine(row,ident):
    prim_ident = row['primary_'+ident]
    sec_ident = row['secondary_'+ident]
    if(sec_ident == -2):
        sec_ident = 0
    if(prim_ident >= sec_ident):
        return int(str(int(prim_ident)) + str(int(sec_ident)))
    else:
        return int(str(int(sec_ident)) + str(int(prim_ident)))
bidders_df['combined_countries'] = bidders_df.apply(lambda row: combine(row,'country'), axis=1)
bidders_df['combined_devices'] = bidders_df.apply(lambda row: combine(row,'device'), axis=1)
bidders_df['combined_ips'] = bidders_df.apply(lambda row: combine(row,'ip'), axis=1)
bidders_df['combined_urls'] = bidders_df.apply(lambda row: combine(row,'url'), axis=1)

bidders_df['devices_shared'] = bidders_df.groupby(['primary_device'])['primary_device'].transform('count')
bidders_df['countries_shared'] = bidders_df.groupby(['primary_country'])['primary_country'].transform('count')
bidders_df['ips_shared'] = bidders_df.groupby(['primary_ip'])['primary_ip'].transform('count')
bidders_df['urls_shared'] = bidders_df.groupby(['primary_url'])['primary_url'].transform('count')

dev_shared_lim = 10
countries_shared_lim = 20
ips_shared_lim = 10
urls_shared_lim = 4

bidders_df['devices_shared'] = bidders_df['devices_shared'].map(lambda x: 1 if x >= dev_shared_lim else 0)
bidders_df['countries_shared'] = bidders_df['countries_shared'].map(lambda x: 1 if x >= countries_shared_lim else 0)
bidders_df['ips_shared'] = bidders_df['ips_shared'].map(lambda x: 1 if x >= ips_shared_lim else 0)
bidders_df['urls_shared'] = bidders_df['urls_shared'].map(lambda x: 1 if x >= urls_shared_lim else 0)

bidders_df['devices_shared_sec'] = bidders_df.groupby(['secondary_device'])['secondary_device'].transform('count')
bidders_df['countries_shared_sec'] = bidders_df.groupby(['secondary_country'])['secondary_country'].transform('count')
bidders_df['ips_shared_sec'] = bidders_df.groupby(['secondary_ip'])['secondary_ip'].transform('count')
bidders_df['urls_shared_sec'] = bidders_df.groupby(['secondary_url'])['secondary_url'].transform('count')

bidders_df['devices_shared_sec'] = bidders_df['devices_shared_sec'].map(lambda x: 1 if x >= dev_shared_lim else 0)
bidders_df['countries_shared_sec'] = bidders_df['countries_shared_sec'].map(lambda x: 1 if x >= countries_shared_lim else 0)
bidders_df['ips_shared_sec'] = bidders_df['ips_shared_sec'].map(lambda x: 1 if x >= ips_shared_lim else 0)
bidders_df['urls_shared_sec'] = bidders_df['urls_shared_sec'].map(lambda x: 1 if x >= urls_shared_lim else 0)

bidders_df['devices_shared_com'] = bidders_df.groupby(['combined_devices'])['combined_devices'].transform('count')
bidders_df['countries_shared_com'] = bidders_df.groupby(['combined_countries'])['combined_countries'].transform('count')
bidders_df['ips_shared_com'] = bidders_df.groupby(['combined_ips'])['combined_ips'].transform('count')
bidders_df['urls_shared_com'] = bidders_df.groupby(['combined_urls'])['combined_urls'].transform('count')

bidders_df['devices_shared_com'] = bidders_df['devices_shared_com'].map(lambda x: 1 if x >= dev_shared_lim else 0)
bidders_df['countries_shared_com'] = bidders_df['countries_shared_com'].map(lambda x: 1 if x >= countries_shared_lim else 0)
bidders_df['ips_shared_com'] = bidders_df['ips_shared_com'].map(lambda x: 1 if x >= ips_shared_lim else 0)
bidders_df['urls_shared_com'] = bidders_df['urls_shared_com'].map(lambda x: 1 if x >= urls_shared_lim else 0)

bidders_df['primary_device'] = (bidders_df['primary_device']+1) * bidders_df['devices_shared']
bidders_df['primary_country'] = (bidders_df['primary_country']+1) * bidders_df['countries_shared']
bidders_df['primary_ip'] = (bidders_df['primary_ip']+1) * bidders_df['ips_shared']
bidders_df['primary_url'] = (bidders_df['primary_url']+1) * bidders_df['urls_shared']

bidders_df['secondary_device'] = (bidders_df['secondary_device']+1) * bidders_df['devices_shared_sec']
bidders_df['secondary_country'] = (bidders_df['secondary_country']+1) * bidders_df['countries_shared_sec']
bidders_df['secondary_ip'] = (bidders_df['secondary_ip']+1) * bidders_df['ips_shared_sec']
bidders_df['secondary_url'] = (bidders_df['secondary_url']+1) * bidders_df['urls_shared_sec']

bidders_df['combined_devices'] = (bidders_df['combined_devices']+1) * bidders_df['devices_shared_com']
bidders_df['combined_countries'] = (bidders_df['combined_countries']+1) * bidders_df['countries_shared_com']
bidders_df['combined_ips'] = (bidders_df['combined_ips']+1) * bidders_df['ips_shared_com']
bidders_df['combined_urls'] = (bidders_df['combined_urls']+1) * bidders_df['urls_shared_com']

bidders_train = bidders_df.dropna()
toc=timeit.default_timer()
print('Time',toc - tic)
#%%
#visualize / explore dataset
num_bots = len(bidders_train[bidders_train['outcome'] == 1])
num_humans = len(bidders_train[bidders_train['outcome'] == 0])
w_bots = np.ones(num_bots) / num_bots
w_humans = np.ones(num_humans) / num_humans
def make_comparison_hist(var, bins = 10, x_low = 0, x_high = 10):
    p.clf()
    p.cla()
    bidders_train[bidders_train['outcome'] == 1][var].hist(
          bins=bins, alpha = 0.5, weights = w_bots,
          label = 'bot', range = (x_low,x_high))
    bidders_train[bidders_train['outcome'] == 0][var].hist(
        bins=bins, alpha = 0.5, weights = w_humans,
        label = 'human', range = (x_low,x_high))
    plt.xlabel(var)
    plt.ylabel('norm bidders')
    plt.title(var)
    p.legend()
    pic_name = 'Plots/'
    pic_name = pic_name + var + '.png'
    p.savefig(pic_name, bbox_inches='tight')
print('making comparison hists')
p.figure()
for feature in bidders_train.columns.values:
    if(feature == 'outcome'):
        continue
    make_comparison_hist(feature, bins = 15, x_low = 0, x_high = 15)
#%%
tic=timeit.default_timer()
random.seed(13)
np.random.seed(1)
rows = random.sample(range(len(bidders_train.index)),
                     int(len(bidders_train.index)/4)+1)
df_cv = bidders_train.ix[bidders_train.index[rows]]
df_train = bidders_train.drop(bidders_train.index[rows])
#uncomment next line to use all of training data
#df_train = bidders_train
outcome_train = df_train['outcome'].values
outcome_cv = df_cv['outcome'].values
df_test = bidders_df[bidders_df.bidder_id >= 2013]
drop_list = ['bidder_id', 'num_bids','num_aucs', 'num_urls','num_countries','num_self_overbids',
             'num_devices', 'mean_frac_bots_auctions',
             'aucs_involved_all_bids','mean_frac_bids', 'num_urls_per_bid','mean_frac_bids_binned',
             'always_lost','one_bid',
             'num_ips', 'ratio_won',
             'num_countries_per_bid',
             'mean_frac_bids',
             'aucs_inv_all_binned',
             'bids_per_auc','devices_per_auc','urls_per_auc','ips_per_auc','countries_per_auc',
             'self_overbids_per_auc','self_overbids_per_bid',
             'time_diffs','num_con_per_bid_binned',
             'devices_shared_sec','countries_shared_sec','ips_shared_sec','urls_shared_sec',
             'devices_shared','countries_shared','ips_shared','urls_shared',
             'devices_shared_com','countries_shared_com','ips_shared_com','urls_shared_com',
             'ips_per_device','urls_per_device','countries_per_device','urls_per_ip',
             'bids_per_country','bids_per_device','bids_per_ip', 'bids_per_url',
             'auc_switches','auc_switches_per_auc','auc_switches_per_bid',
             'url_switches','url_switches_per_url','url_switches_per_bid',
             'dev_switches','dev_switches_per_dev',
             'won_auctions', 'started_auctions', 'outcome']
df_train_df = df_train.drop(drop_list, axis=1)
df_cv_df = df_cv.drop(drop_list, axis=1)
df_test_df = df_test.drop(drop_list, axis=1)

train_data = df_train_df.values
cv_data = df_cv_df.values
test_data = df_test_df.values
test_ids = df_test['bidder_id'].map(lambda x: inv_bid_id_dict[x]).values

dtrain = xgb.DMatrix( train_data, label=outcome_train)
dcv = xgb.DMatrix( cv_data, label=outcome_cv)
dtest = xgb.DMatrix( test_data, )

param = {'objective':'binary:logistic','eval_metric':'auc','max_depth':6,
         'gamma': 2.213, 'learning_rate':0.273,'max_delta_step': 1.444,
         'subsample': 0.847}
num_round = 75
plst = param.items()
# specify validations set to watch performance
watchlist  = [(dcv,'eval'), (dtrain,'train')]
bst_search = xgb.XGBClassifier()
clf = RandomizedSearchCV(bst_search, {'max_depth': sp_randint(1,13), 'learning_rate':sp_rand(0,1),
                                'gamma':sp_rand(0,3), 'subsample':sp_rand(0,1),'max_delta_step':sp_rand(0,3),
                               'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100, 125, 150,200,]},
                   verbose=1, n_jobs=2, cv = 4, scoring='roc_auc', n_iter = 1000)
clf.fit(train_data, outcome_train)
print('best clf score',clf.best_score_)
print('best params:', clf.best_params_)
bst = xgb.train(plst, dtrain, num_round, watchlist)
# this is prediction
preds = bst.predict(dcv)
pred_test = bst.predict(dtest)
labels = dcv.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

print('{0:<25} {1:>5}'.format('Feature','Importance'))
print("--------------------------------------")
for i in range(len(df_train_df.columns.values)):
    key = 'f' + str(i)
    try:
        score = bst.get_fscore()[key]
    except:
        score = 0
    print ('{0:25} {1:5.3f}'.format(df_train_df.columns.values[i],score))
p.figure()
p.hist(preds[outcome_cv == 1], label = 'comp', alpha = 0.5,
       bins = 20, range = (0,1.0))
p.hist(preds[outcome_cv == 0], label = 'human', alpha = 0.5,
       bins = 20, range = (0,1.0))
plt.xlabel('Pred. Prob of Comp')
plt.ylabel('Bidders')
p.legend()
toc=timeit.default_timer()
print('Time',toc - tic)
#%%
print ('Predicting...')
unknown_bidders = list(set(test_df['bidder_id'].values) - set(test_ids))
unknown_bidders_vals = np.zeros(len(unknown_bidders))
predictions_file = open('/Users/Jared/DataAnalysis/Kaggle/Facebook/fb_forest.csv','wt')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['bidder_id','prediction'])
open_file_object.writerows(zip(test_ids, pred_test))
open_file_object.writerows(zip(unknown_bidders, unknown_bidders_vals))
predictions_file.close()
print ('Done.')
toc=timeit.default_timer()
print('Total Time',toc - tic0)