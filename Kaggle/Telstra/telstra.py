# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:46:15 2015

@author: Jared Turkewitz
"""
import pandas as pd
import numpy as np
import random
from sklearn import cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import operator
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
import timeit

tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'
#%%
tic=timeit.default_timer()

train_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/train.csv', header=0)
test_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/test.csv', header=0)
severity_type = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/severity_type.csv', header=0)
log_feature = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/log_feature.csv', header=0)
event_type = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/event_type.csv', header=0)
resource_type = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/resource_type.csv', header=0)

toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
#controls whether to run on local cv or on test dataset for Kaggle leader board
is_sub_run = False
#is_sub_run = True
#%%
def norm_rows(df):
    with np.errstate(invalid='ignore'):
        return df.div(df.sum(axis=1), axis=0).fillna(0)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def print_value_counts(input_df,col_name,is_normalized=False):
    for value in input_df[col_name].unique():
        print(value)
        print(input_df.loc[input_df[col_name] == value].fault_severity.value_counts(normalize=is_normalized))

def print_value_counts_spec(input_df,col_name,col_value,is_normalized=False):
    print(col_value)
    print(input_df.loc[input_df[col_name] == col_value].fault_severity.value_counts(normalize=is_normalized))

def convert_strings_to_ints(input_df,col_name,output_col_name):
    labels, levels = pd.factorize(input_df[col_name])
    input_df[output_col_name] = labels
    output_dict = dict(zip(input_df[col_name],input_df[output_col_name]))
    return (output_dict,input_df)

#assumes row normalized, doesn't do eps thing
def get_log_loss_row(row):
    ans = row['fault_severity']
    if (ans == 0):
        return -1.0 * np.log(row['predict_0'])
    elif (ans == 1):
        return -1.0 * np.log(row['predict_1'])
    elif (ans == 2):
        return -1.0 * np.log(row['predict_2'])
    else:
        print('not_exceptable_value')
        raise ValueError('Not of correct class')
        return -1000
def get_log_loss_row_two_classes(row):
    ans = row['fault_severity']
    if (ans == 0):
        return -1.0 * np.log(row['predict_low'])
    elif (ans == 1):
        return -1.0 * np.log(row['predict_high'])
    else:
        print('not_exceptable_value')
        raise ValueError('Not of correct class')
        return -1000

#%%
test_orig['fault_severity'] = 'dummy'
#%%
severity_type['severity_type'] = severity_type['severity_type'].map(lambda x: x.replace(' ','_'))
severity_type['severity_type'] = severity_type['severity_type'].map(lambda x: x[-1])
#%%
event_type['event_type'] = event_type['event_type'].map(lambda x: x.replace(' ','_'))
event_type['event_type'] = event_type['event_type'].map(lambda x: x[11:]).astype(int)
event_type_dummies = pd.get_dummies(event_type,columns=['event_type'])
#%%
resource_type['resource_type'] = resource_type['resource_type'].map(lambda x: x.replace(' ','_'))
resource_type['resource_type'] = resource_type['resource_type'].map(lambda x: x[14:]).astype(int)
resource_type_dummies = pd.get_dummies(resource_type,columns=['resource_type'])
#%%
resource_type['resource_shifted_up'] = resource_type['resource_type'].shift(1)
resource_type['resource_shifted_up'].fillna(8,inplace=True)
resource_type['is_next_resource_type_repeat'] = 0
resource_type['is_next_resource_type_different'] = 0
repeat_cond = resource_type['resource_shifted_up'] == resource_type['resource_type']
resource_type['is_next_resource_type_repeat'][repeat_cond] = 1
resource_type['is_next_resource_type_different'][~repeat_cond] = 1
resource_type['switches_resource_type'] = resource_type['is_next_resource_type_different'].cumsum()
#%%
log_feature['log_feature'] = log_feature['log_feature'].map(lambda x: x.replace(' ','_'))
log_feature['log_feature'] = log_feature['log_feature'].map(lambda x: x[8:]).astype(int)

log_feature.reset_index('count_of_log_feature_seen',inplace=True)
log_feature.rename(columns={'index':'count_of_log_feature_seen'},inplace=True)

log_feature_value_counts = log_feature.log_feature.value_counts().to_dict()
log_feature['num_ids_with_log_feature'] = log_feature['log_feature'].map(lambda x: log_feature_value_counts[x])

bins = [-10,0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,
        200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,
        370,380,390,400]
log_feature['binned_log_feature'] = np.digitize(log_feature['log_feature'], bins, right=True)
bins_offset = list(map(lambda x:x+5, bins))
log_feature['binned_offset_log_feature'] = np.digitize(log_feature['log_feature'], bins_offset, right=True)
#%%
log_feature['position_of_log_feature'] = 1
log_feature['position_of_log_feature'] = log_feature.groupby(['id'])['position_of_log_feature'].cumsum()
log_feature['log_feature'] = log_feature['log_feature'].astype(int)
#%%
event_type.reset_index('count_of_event_type_seen',inplace=True)
event_type.rename(columns={'index':'count_of_event_type_seen'},inplace=True)

temp_combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)
event_type['event_type'] = event_type['event_type'].astype(int)
event_type_combined = pd.merge(temp_combined,event_type,left_on = ['id'],
                               right_on = ['id'],how='left')
event_type_combined.sort('count_of_event_type_seen',inplace=True)

most_common_event_type_dict = event_type_combined.groupby(['location'])['event_type'].agg(lambda x: x.value_counts().index[0]).to_dict()
event_type_no_dups = event_type_combined.drop_duplicates('location')
event_type_no_dups_ids = event_type_combined.drop_duplicates('id')
#%%
event_type_combined['location_number'] = event_type_combined['location'].map(lambda x: x[9:]).astype(int)
event_type_combined.sort(['location_number','count_of_event_type_seen'],inplace=True)
#%%
log_combined = pd.merge(temp_combined,log_feature,left_on = ['id'],
                               right_on = ['id'],how='left')
log_combined = pd.merge(log_combined,event_type_no_dups_ids[['id','event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
log_combined.sort('count_of_log_feature_seen',inplace=True)
log_feature_unique_features_per_location_dict = log_combined.groupby('location').log_feature.nunique().to_dict()
most_common_feature_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: x.value_counts().index[0]).to_dict()

log_combined['location_number'] = log_combined['location'].map(lambda x: x[9:]).astype(int)
most_common_location_dict = log_combined.groupby(['log_feature'])['location_number'].agg(lambda x: x.value_counts().index[0]).to_dict()
most_common_event_dict = log_combined.groupby(['log_feature'])['event_type'].agg(lambda x: x.value_counts().index[0]).to_dict()

mean_volume_dict = log_combined.groupby(['location'])['volume'].agg(lambda x: x.mean()).to_dict()
max_volume_dict = log_combined.groupby(['location'])['volume'].agg(lambda x: x.max()).to_dict()
std_log_feature_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: x.std()).to_dict()

unique_locations_per_feature_dict = log_combined.groupby('log_feature').location.nunique().to_dict()
median_ids_with_feature_number_dict = log_combined.groupby(['location'])['num_ids_with_log_feature'].agg(lambda x: x.median()).to_dict()
median_feature_number_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: x.median()).to_dict()

log_combined['location_number'] = log_combined['location'].map(lambda x: x[9:]).astype(int)
log_combined.sort(['location_number','count_of_log_feature_seen'],inplace=True)
#%%
log_feature['locations_per_feature'] = log_feature['log_feature'].map(unique_locations_per_feature_dict)
log_feature['most_common_location_of_log_feature'] = log_feature['log_feature'].map(most_common_location_dict)
log_feature['most_common_event_of_log_feature'] = log_feature['log_feature'].map(most_common_event_dict)
#%%
has_event_type_dict = {}
event_grouped = event_type_combined.groupby(['location'])
for i in event_type_combined.event_type.unique():
    has_event_type_dict[i] = event_grouped['event_type'].agg(lambda x: 1 if i in x.values else 0).to_dict()

event_type_number_dict = event_type_combined.groupby(['location'])['event_type'].agg(lambda x: x.nunique()).to_dict()
#%%
event_type_no_dups = event_type_combined.drop_duplicates('id')
#%%
resource_type.reset_index('count_of_resource_type_seen',inplace=True)
resource_type.rename(columns={'index':'count_of_resource_type_seen'},inplace=True)
resource_type_combined = pd.merge(temp_combined,resource_type,left_on = ['id'],
                               right_on = ['id'],how='left')
resource_type_combined.sort('count_of_resource_type_seen',inplace=True)
most_common_resource_type_dict = resource_type_combined.groupby(['location'])['resource_type'].agg(lambda x: x.value_counts().index[0]).to_dict()
std_resource_type_dict = resource_type_combined.groupby(['location'])['resource_type'].agg(lambda x: x.std()).to_dict()
resource_type_number_dict = resource_type_combined.groupby(['location'])['resource_type'].agg(lambda x: x.nunique()).to_dict()

has_resource_type_dict = {}
resource_grouped = resource_type_combined.groupby(['location'])
for i in resource_type_combined.resource_type.unique():
    has_resource_type_dict[i] = resource_grouped['resource_type'].agg(lambda x: 1 if i in x.values else 0).to_dict()

#%%
severity_type.reset_index('count_of_severity_type_seen',inplace=True)
severity_type.rename(columns={'index':'count_of_severity_type_seen'},inplace=True)
severity_type_combined = pd.merge(temp_combined,severity_type,left_on = ['id'],
                               right_on = ['id'],how='left')
#%%
severity_type_combined.sort('count_of_severity_type_seen',inplace=True)
#%%
log_feature_dummies = pd.get_dummies(log_feature,columns=['log_feature',
                                                          'most_common_location_of_log_feature',
                                                          'binned_log_feature','binned_offset_log_feature'])
#%%
log_feature_dummies_cols = [col for col in list(log_feature_dummies) if col.startswith('log_feature_')]
log_feature_dummies_binned_cols = [col for col in list(log_feature_dummies) if col.startswith('binned_log_feature_')]
log_feature_dummies_binned_offset_cols = [col for col in list(log_feature_dummies) if col.startswith('binned_offset_log_feature_')]
log_feature_dummies_by_vol = log_feature_dummies[(['id','volume'] + log_feature_dummies_cols + log_feature_dummies_binned_cols +
                                                    log_feature_dummies_binned_offset_cols
                                                   )]
log_feature_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('log_feature_') else x,inplace=True)
log_feature_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('binned_log_feature_') else x,inplace=True)
log_feature_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('binned_offset_log_feature_') else x,inplace=True)

log_feature_dummies_by_vol_cols = [col for col in list(log_feature_dummies_by_vol) if col.startswith('vol_log_feature_')]
log_feature_dummies_by_vol_binned_cols = [col for col in list(log_feature_dummies_by_vol) if col.startswith('vol_binned_log_feature_')]
log_feature_dummies_by_vol_binned_offset_cols = [col for col in list(log_feature_dummies_by_vol) if col.startswith('vol_binned_offset_log_feature_')]
log_feature_dummies_by_vol[log_feature_dummies_by_vol_cols] = \
    log_feature_dummies_by_vol[log_feature_dummies_by_vol_cols].multiply(log_feature_dummies_by_vol['volume'],axis='index')
log_feature_dummies_by_vol[log_feature_dummies_by_vol_binned_cols] = \
    log_feature_dummies_by_vol[log_feature_dummies_by_vol_binned_cols].multiply(log_feature_dummies_by_vol['volume'],axis='index')
log_feature_dummies_by_vol[log_feature_dummies_by_vol_binned_offset_cols] = \
    log_feature_dummies_by_vol[log_feature_dummies_by_vol_binned_offset_cols].multiply(log_feature_dummies_by_vol['volume'],axis='index')

log_feature_dummies_by_vol_collapse = log_feature_dummies_by_vol.groupby('id').sum()
log_feature_dummies_by_vol_collapse.reset_index('id',inplace=True)

log_feature_dummies_collapse = log_feature_dummies.groupby('id').sum()
log_feature_dummies_collapse.reset_index('id',inplace=True)
#%%
log_feature_dummies_loc = pd.merge(temp_combined[['id','location']],
                                   log_feature_dummies[['id'] + log_feature_dummies_cols +
                                   log_feature_dummies_binned_cols + log_feature_dummies_binned_offset_cols],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_dummies_loc_collapse = log_feature_dummies_loc.groupby('location').max()
log_feature_dummies_loc_collapse.reset_index('location',inplace=True)
log_feature_dummies_loc_collapse.rename(columns = lambda x : ('loc_' + x) if x.startswith('log_feature_') else x,inplace=True)
log_feature_dummies_loc_collapse.drop('id',axis=1,inplace=True)

log_feature_dummies_loc_collapse_sum = log_feature_dummies_loc.groupby('location').sum()
log_feature_dummies_loc_collapse_sum.reset_index('location',inplace=True)
log_feature_dummies_loc_collapse_sum.rename(columns = lambda x : ('summed_loc_' + x) if x.startswith('log_feature_') else x,inplace=True)
log_feature_dummies_loc_collapse_sum.drop('id',axis=1,inplace=True)
#%%
log_feature_count = log_feature.groupby('id').count().reset_index('id')
log_feature_count['count_log_features'] = log_feature_count['log_feature']
#%%
unique_volumes_dict = log_feature.groupby('id')['volume'].nunique().to_dict()
#%%
log_feature_sum = log_feature.groupby('id').sum().reset_index('id')
log_feature_sum['sum_log_features_volume'] = log_feature_sum.volume
log_feature_sum['sum_num_ids_with_log_feature'] = log_feature_sum['num_ids_with_log_feature']
log_feature_sum['sum_locations_per_feature'] = log_feature_sum['locations_per_feature']

log_feature_mean = log_feature.groupby('id').mean().reset_index('id')
log_feature_mean['mean_log_features_volume'] = log_feature_mean.volume
log_feature_mean['mean_num_ids_with_log_feature'] = log_feature_mean['num_ids_with_log_feature']

log_feature_median = log_feature.groupby('id').median().reset_index('id')
log_feature_median['median_log_features_volume'] = log_feature_median.volume
log_feature_median['median_log_feature'] = log_feature_median['log_feature']
log_feature_median['median_log_feature_most_common_event'] = log_feature_median['most_common_event_of_log_feature']
log_feature_median['median_log_feature_most_common_location'] = log_feature_median['most_common_location_of_log_feature']

log_feature_max = log_feature.groupby('id').max().reset_index('id')
log_feature_max['max_log_features_volume'] = log_feature_max.volume
log_feature_max['max_num_ids_with_log_feature'] = log_feature_max['num_ids_with_log_feature']
log_feature_max['max_locations_per_feature'] = log_feature_max['locations_per_feature']
log_feature_max['max_log_feature'] = log_feature_max['log_feature']

log_feature_min = log_feature.groupby('id').min().reset_index('id')
log_feature_min['min_log_features_volume'] = log_feature_min.volume
log_feature_min['min_num_ids_with_log_feature'] = log_feature_min['num_ids_with_log_feature']
log_feature_min['min_locations_per_feature'] = log_feature_min['locations_per_feature']
log_feature_min['min_log_feature'] = log_feature_min['log_feature']

min_log_feature_dict = pd.Series(log_feature_min['min_log_feature'].values,index=log_feature_min['id']).to_dict()

log_feature_std = log_feature.groupby('id').std().reset_index('id')
log_feature_std['std_log_features_volume'] = log_feature_std.volume
log_feature_std['std_log_feature'] = log_feature_std['log_feature']
log_feature_std['std_num_ids_with_log_feature'] = log_feature_std['num_ids_with_log_feature']
#%%
event_type_median = event_type.groupby('id').median().reset_index('id')
event_type_median['median_event_type'] = event_type_median['event_type']

event_type_min = event_type.groupby('id').min().reset_index('id')
event_type_min['min_event_type'] = event_type_min['event_type']

event_type_max = event_type.groupby('id').max().reset_index('id')
event_type_max['max_event_type'] = event_type_max['event_type']
#%%
resource_type_median = resource_type.groupby('id').median().reset_index('id')
resource_type_median['median_resource_type'] = resource_type_median['resource_type']

resource_type_min = resource_type.groupby('id').min().reset_index('id')
resource_type_min['min_resource_type'] = resource_type_min['resource_type']

resource_type_max = resource_type.groupby('id').max().reset_index('id')
resource_type_max['max_resource_type'] = resource_type_max['resource_type']
resource_type_max['max_is_next_resource_type_repeat'] = resource_type_max['is_next_resource_type_repeat']
#%%
log_feature_volume_of_min_feature = log_feature.copy()
log_feature_volume_of_min_feature.sort(['id','log_feature'],ascending=True,inplace=True)
log_feature_volume_of_min_feature.drop_duplicates('id',inplace=True)
log_feature_volume_of_min_feature.rename(columns = {'volume':'volume_of_min_log_feature'},inplace=True)
#%%
log_feature_volume_of_max_feature = log_feature.copy()
log_feature_volume_of_max_feature.sort(['id','log_feature'],ascending=False,inplace=True)
log_feature_volume_of_max_feature.drop_duplicates('id',inplace=True)
log_feature_volume_of_max_feature.rename(columns = {'volume':'volume_of_max_log_feature'},inplace=True)
#%%
log_feature_volume_of_rarest_feature = log_feature.copy()
log_feature_volume_of_rarest_feature.sort(['id','num_ids_with_log_feature'],ascending=True,inplace=True)
log_feature_volume_of_rarest_feature.drop_duplicates('id',inplace=True)
log_feature_volume_of_rarest_feature.rename(columns = {'volume':'volume_of_rarest_log_feature',
                                                       'log_feature':'rarest_log_feature',
                                                      },inplace=True)

log_feature_volume_of_least_rare_feature = log_feature.copy()
log_feature_volume_of_least_rare_feature.sort(['id','num_ids_with_log_feature'],ascending=False,inplace=True)
log_feature_volume_of_least_rare_feature.drop_duplicates('id',inplace=True)
log_feature_volume_of_least_rare_feature.rename(columns = {'volume':'volume_of_least_rare_log_feature','log_feature':'least_rare_log_feature'},inplace=True)
#%%
LOG_FEATURE_DICT = {}
for i in range(1,21):
    temp_df = log_feature.loc[log_feature['position_of_log_feature'] == i]
    temp_df = temp_df[['id','log_feature','volume']]
    log_feat_name = 'position_log_feature_' + str(i)
    volume_name = 'position_volume_'+ str(i)
    temp_df.rename(columns = {'log_feature':log_feat_name,'volume':volume_name},inplace=True)
    LOG_FEATURE_DICT[i] = temp_df
#%%
resource_type['position_of_resource_type'] = 1
resource_type['position_of_resource_type'] = resource_type.groupby(['id'])['position_of_resource_type'].cumsum()

RESOURCE_TYPE_DICT = {}
for i in range(1,6):
    temp_df = resource_type.loc[resource_type['position_of_resource_type'] == i]
    temp_df = temp_df[['id','resource_type']]
    resource_type_feat_name = 'position_resource_type_' + str(i)
    temp_df.rename(columns = {'resource_type':resource_type_feat_name},inplace=True)
    RESOURCE_TYPE_DICT[i] = temp_df
#%%
event_type['position_of_event_type'] = 1
event_type['position_of_event_type'] = event_type.groupby(['id'])['position_of_event_type'].cumsum()

EVENT_TYPE_DICT = {}
for i in range(1,11):
    temp_df = event_type.loc[event_type['position_of_event_type'] == i]
    temp_df = temp_df[['id','event_type']]
    event_type_feat_name = 'position_event_type_' + str(i)
    temp_df.rename(columns = {'event_type':event_type_feat_name},inplace=True)
    EVENT_TYPE_DICT[i] = temp_df
#%%
event_type_count = event_type.groupby('id').count().reset_index('id')
event_type_count['count_event_types'] = event_type_count.event_type
#%%
resource_type_count = resource_type.groupby('id').count().reset_index('id')
resource_type_count['count_resource_types'] = resource_type_count.resource_type
#%%
log_feature.sort('count_of_log_feature_seen',inplace=True)
log_feature_no_dups = log_feature.drop_duplicates('id')

resource_type.sort('count_of_resource_type_seen',inplace=True)
resource_type_no_dups = resource_type.drop_duplicates('id')
event_type_no_dups = event_type.drop_duplicates('id')
log_feature_no_dups.reset_index(drop=True,inplace=True)
log_feature_no_dups.reset_index('order_of_log_feature',inplace=True)
log_feature_no_dups.rename(columns={'index':'order_of_log_feature','count_of_log_feature_seen':'count_of_log_feature_seen_no_dups'},inplace=True)

temp_combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)
log_feature_no_dups_combined = pd.merge(temp_combined,log_feature_no_dups,left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , severity_type[['id','severity_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , log_feature_sum[['id','sum_log_features_volume']],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , resource_type_no_dups[['id','switches_resource_type','resource_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , event_type_no_dups[['id','event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')

log_feature_no_dups_combined.sort('order_of_log_feature',inplace=True)

(location_dict_ordered,log_feature_no_dups_combined) = convert_strings_to_ints(log_feature_no_dups_combined,'location','location_hash')

log_feature_no_dups_combined.sort('order_of_log_feature',ascending=False,inplace=True)
(location_dict_ordered,log_feature_no_dups_combined) = convert_strings_to_ints(log_feature_no_dups_combined,'location','location_hash_rev')
log_feature_no_dups_combined.sort('order_of_log_feature',inplace=True)

has_event_type_names = []
for key in has_event_type_dict:
    col_name = 'location_by_has_event_type_' + str(key)
    log_feature_no_dups_combined[col_name] = (
            log_feature_no_dups_combined['location'].map(lambda x: has_event_type_dict[key][x]))
    has_event_type_names.append(col_name)
has_resource_type_names = []
for key in has_resource_type_dict:
    col_name = 'location_by_has_resource_type_' + str(key)
    log_feature_no_dups_combined[col_name] = (
            log_feature_no_dups_combined['location'].map(lambda x: has_resource_type_dict[key][x]))
    has_resource_type_names.append(col_name)

log_feature_no_dups_combined['location_by_unique_features'] = (
            log_feature_no_dups_combined['location'].map(lambda x: log_feature_unique_features_per_location_dict[x]))
log_feature_no_dups_combined['location_by_unique_resource_types'] = (
            log_feature_no_dups_combined['location'].map(lambda x: resource_type_number_dict[x]))
log_feature_no_dups_combined['location_by_unique_event_types'] = (
            log_feature_no_dups_combined['location'].map(lambda x: event_type_number_dict[x]))

log_feature_no_dups_combined['location_by_most_common_event'] = (
            log_feature_no_dups_combined['location'].map(lambda x: most_common_event_type_dict[x]))

log_feature_no_dups_combined['count_of_volumes'] = (
            log_feature_no_dups_combined['id'].map(lambda x: unique_volumes_dict[x]))

log_feature_no_dups_combined['location_by_most_common_resource'] = (
            log_feature_no_dups_combined['location'].map(lambda x: most_common_resource_type_dict[x]))
log_feature_no_dups_combined['location_by_std_resource'] = (
            log_feature_no_dups_combined['location'].map(lambda x: std_resource_type_dict[x]))
log_feature_no_dups_combined['location_by_std_log_feature'] = (
            log_feature_no_dups_combined['location'].map(lambda x: std_log_feature_dict[x]))
log_feature_no_dups_combined['location_by_most_common_feature'] = (
            log_feature_no_dups_combined['location'].map(lambda x: most_common_feature_dict[x]))

log_feature_no_dups_combined['location_by_median_feature_number'] = (
            log_feature_no_dups_combined['location'].map(lambda x: median_feature_number_dict[x]))

log_feature_no_dups_combined['location_by_mean_volume'] = (
            log_feature_no_dups_combined['location'].map(lambda x: mean_volume_dict[x]))
log_feature_no_dups_combined['location_by_max_volume'] = (
            log_feature_no_dups_combined['location'].map(lambda x: max_volume_dict[x]))

log_feature_no_dups_combined['location_hash_next'] = log_feature_no_dups_combined['location_hash'].shift(-1)
log_feature_no_dups_combined['location_hash_next'].fillna(-1,inplace=True)

log_feature_no_dups_combined['order_by_location'] = 1
log_feature_no_dups_combined['order_by_location'] = log_feature_no_dups_combined.groupby(['location'])['order_by_location'].cumsum()

log_feature_no_dups_combined['location_max'] = log_feature_no_dups_combined.groupby(['location'])['order_by_location'].transform(max)
log_feature_no_dups_combined['location_next_max'] = log_feature_no_dups_combined['location_max'] - 1

log_feature_no_dups_combined.sort('order_of_log_feature',ascending = False , inplace=True)
log_feature_no_dups_combined['reverse_order_by_location'] = 1
log_feature_no_dups_combined['reverse_order_by_location'] = log_feature_no_dups_combined.groupby(['location'])['reverse_order_by_location'].cumsum()

log_feature_no_dups_combined.sort('order_of_log_feature',inplace=True)

log_feature_no_dups_combined.sort('order_of_log_feature',inplace=True)
#%%
tic=timeit.default_timer()
combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)

combined = pd.merge(combined,severity_type[['id','severity_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_dummies_collapse,left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_dummies_by_vol_collapse[['id'] + log_feature_dummies_by_vol_cols +
                                    log_feature_dummies_by_vol_binned_cols + log_feature_dummies_by_vol_binned_offset_cols
                                    ],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_dummies_loc_collapse,left_on = ['location'],
                               right_on = ['location'],how='left')
combined = pd.merge(combined,log_feature_dummies_loc_collapse_sum,left_on = ['location'],
                               right_on = ['location'],how='left')

combined = pd.merge(combined,log_feature_volume_of_max_feature[['id','volume_of_max_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_volume_of_min_feature[['id','volume_of_min_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_volume_of_rarest_feature[['id','volume_of_rarest_log_feature','rarest_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_volume_of_least_rare_feature[['id','volume_of_least_rare_log_feature','least_rare_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_count[['id','count_log_features']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,event_type_count[['id','count_event_types']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,resource_type_count[['id','count_resource_types']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_sum[['id','sum_log_features_volume',
                                    'sum_num_ids_with_log_feature','sum_locations_per_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_mean[['id','mean_log_features_volume','mean_num_ids_with_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_max[['id','max_log_features_volume','max_log_feature',
                                              'max_num_ids_with_log_feature','max_locations_per_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_min[['id','min_log_features_volume','min_num_ids_with_log_feature',
                                                'min_locations_per_feature','min_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_std[['id','std_log_features_volume','std_num_ids_with_log_feature','std_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_median[['id','median_log_feature','median_log_feature_most_common_event',
                                                 'median_log_feature_most_common_location']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,resource_type_max[['id','max_is_next_resource_type_repeat']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,event_type_max[['id','max_event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,event_type_min[['id','min_event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,event_type_median[['id','median_event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,resource_type_max[['id','max_resource_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,resource_type_min[['id','min_resource_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,resource_type_median[['id','median_resource_type']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined['range_log_features_volume'] = combined['max_log_features_volume'] - combined['min_locations_per_feature']

combined = pd.merge(combined,log_feature_no_dups_combined[
                    ['id',
                      'location_by_unique_features',
                      'location_by_unique_event_types',
                      'location_by_unique_resource_types',
                      'location_by_std_resource',
                      'location_by_std_log_feature',
                      'location_by_most_common_event',
                      'location_by_most_common_resource',
                      'location_by_most_common_feature',
                      'location_by_median_feature_number',
                      'location_by_mean_volume',
                      'location_by_max_volume',
                      'count_of_volumes',
                      'order_of_log_feature',
                      'location_max',
                      'reverse_order_by_location','order_by_location',
                      'switches_resource_type',
                      ]
                      + has_event_type_names + has_resource_type_names],
                      left_on = ['id'],
                      right_on = ['id'],how='left')
combined.fillna(0,inplace=True)

for key in LOG_FEATURE_DICT:
    temp_df = LOG_FEATURE_DICT[key]
    combined = pd.merge(combined,temp_df,left_on = ['id'],
                               right_on = ['id'],how='left')
    combined.fillna(-1,inplace=True)
for key in RESOURCE_TYPE_DICT:
    temp_df = RESOURCE_TYPE_DICT[key]
    combined = pd.merge(combined,temp_df,left_on = ['id'],
                               right_on = ['id'],how='left')
    combined.fillna(-1,inplace=True)
for key in EVENT_TYPE_DICT:
    temp_df = EVENT_TYPE_DICT[key]
    combined = pd.merge(combined,temp_df,left_on = ['id'],
                               right_on = ['id'],how='left')
    combined.fillna(-1,inplace=True)
toc=timeit.default_timer()
print('Merging Time',toc - tic)
#%%
tic=timeit.default_timer()
combined['location_number'] = combined['location'].map(lambda x: x[9:]).astype(int)

combined.sort(['median_log_feature','min_log_feature'],inplace=True)

combined_temp = combined.copy()
log_feature_cols_all = [col for col in list(combined) if col.startswith('log_feature_')]

combined_temp[log_feature_cols_all] = combined_temp[log_feature_cols_all].astype(str)
combined_temp['combo_log_features'] = combined_temp[log_feature_cols_all].sum(axis=1)
(combo_log_feature_dict,combined_temp) = convert_strings_to_ints(combined_temp,'combo_log_features','combo_log_features_hash')

event_type_cols_all = [col for col in list(combined) if col.startswith('event_type_')]
resource_type_cols_all = [col for col in list(combined) if col.startswith('resource_type_')]

combined.sort(['median_event_type','min_event_type'],inplace=True)
combined_temp2 = combined.copy()
combined_temp2[event_type_cols_all] = combined_temp2[event_type_cols_all].astype(str)
combined_temp2['combo_event_types'] = combined_temp2[event_type_cols_all].sum(axis=1)
(combo_event_dict,combined_temp2) = convert_strings_to_ints(combined_temp2,'combo_event_types','combo_event_types_hash')

combined.sort(['median_resource_type','min_resource_type'],inplace=True)
combined_temp3 = combined.copy()
combined_temp3[resource_type_cols_all] = combined_temp3[resource_type_cols_all].astype(str)
combined_temp3['combo_resource_types'] = combined_temp3[resource_type_cols_all].sum(axis=1)
(combo_resource_dict,combined_temp3) = convert_strings_to_ints(combined_temp3,'combo_resource_types','combo_resource_types_hash')

combined['combo_log_features_hash'] = combined_temp['combo_log_features_hash']
combined['combo_event_types_hash'] = combined_temp2['combo_event_types_hash']
combined['combo_resource_types_hash'] = combined_temp3['combo_resource_types_hash']
combined['log_features_range'] = combined['max_log_feature'] - combined['min_log_feature']

toc=timeit.default_timer()
print('Combination Hash Time',toc - tic)
#%%
combined['order_by_location_fraction'] = combined['order_by_location'] / combined['location_max']
combined['reverse_order_by_location_fraction'] = combined['reverse_order_by_location'] / combined['location_max']

event_combo_value_counts = combined['combo_event_types_hash'].value_counts().to_dict()
combined['num_ids_with_event_type_combo'] = combined['combo_event_types_hash'].map(lambda x: event_combo_value_counts[x])
resource_combo_value_counts = combined['combo_resource_types_hash'].value_counts().to_dict()
combined['num_ids_with_resource_type_combo'] = combined['combo_resource_types_hash'].map(lambda x: resource_combo_value_counts[x])
log_feature_combo_value_counts = combined['combo_log_features_hash'].value_counts().to_dict()
combined['num_ids_with_log_feature_combo'] = combined['combo_log_features_hash'].map(lambda x: log_feature_combo_value_counts[x])

count_median_log_features_dict = combined.groupby(['location'])['count_log_features'].agg(lambda x: x.median()).to_dict()
combined['location_by_median_count_log_features'] = combined['location'].map(count_median_log_features_dict)

combined['log_203_82_vol_diff'] = combined['vol_log_feature_203'] - (combined['vol_log_feature_82'])
combined['log_201_80_vol_diff'] = combined['vol_log_feature_201'] - (combined['vol_log_feature_80'])
combined['log_312_232_vol_diff'] = combined['vol_log_feature_312'] - (combined['vol_log_feature_232'])
combined['log_171_55_vol_diff'] = combined['vol_log_feature_171'] - (combined['vol_log_feature_55'])
combined['log_170_54_vol_diff'] = combined['vol_log_feature_170'] - (combined['vol_log_feature_54'])
combined['log_193_71_vol_diff'] = combined['vol_log_feature_193'] - (combined['vol_log_feature_71'])

#%%
combo_feature_nunique_location_dict = combined.groupby(['combo_log_features_hash'])['location_number'].agg(lambda x: x.nunique()).to_dict()

combined['combo_log_features_by_nunique_loc'] = combined['combo_log_features_hash'].map(lambda x: combo_feature_nunique_location_dict[x])
combined['combo_log_features_by_nunique_loc_per_ids'] = combined['combo_log_features_by_nunique_loc'] / combined['num_ids_with_log_feature_combo']
#%%
if (is_sub_run):
    train = combined.loc[combined['fault_severity'] != 'dummy' ]
    test = combined.loc[combined['fault_severity'] == 'dummy' ]
else:
    train = combined.loc[(combined['fault_severity'] != 'dummy') &
                         (combined['id'] > 5000)]
    test = combined.loc[(combined['fault_severity'] != 'dummy') &
                         (combined['id'] <= 5000)]
#    train = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         ((combined['id'] > 10000) | (combined['id'] <= 5000))]
#    test = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] > 5000) & (combined['id'] <= 10000)]
#    train = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         ((combined['id'] > 15000) | (combined['id'] <= 10000))]
#    test = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] > 10000) & (combined['id'] <= 15000)]
#    train = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] < 15000)]
#    test = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] >= 15000)]
#%%
def fit_xgb_model(train, test, params, xgb_features, num_rounds = 10, num_boost_rounds = 10,
                  do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
                  random_seed = 5, reweight_probs = True, calculate_log_loss = True, use_two_classes = False):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    X_train, X_watch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
    train_data = X_train[xgb_features].values
    train_data_full = train[xgb_features].values
    watch_data = X_watch[xgb_features].values
    train_fault_severity = X_train['fault_severity'].astype(int).values
    train_fault_severity_full = train['fault_severity'].astype(int).values
    watch_fault_severity = X_watch['fault_severity'].astype(int).values
    test_data = test[xgb_features].values
    dtrain = xgb.DMatrix(train_data, train_fault_severity)
    dtrain_full = xgb.DMatrix(train_data_full, train_fault_severity_full)
    dwatch = xgb.DMatrix(watch_data, watch_fault_severity)
    dtest = xgb.DMatrix(test_data)
    watchlist = [(dtrain, 'train'),(dwatch, 'watch')]
    if(do_grid_search):
        print('Random search cv')
        #usually random search for xgb is not needed, just tune parameters manually as much faster
        gbm_search = xgb.XGBClassifier()
        clf = RandomizedSearchCV(gbm_search,
                                 {'max_depth': sp_randint(1,20), 'learning_rate':sp_rand(0.005,0.2),
                                  'objective':['multi:softprob'],
                                  'subsample':sp_rand(0.1,0.99),
                                  'colsample_bytree':sp_rand(0.1,0.99),'seed':[random_seed],
                                  'gamma':sp_rand(0,2.5),
                                  'min_child_weight':sp_randint(1,20),
                                  'max_delta_step':sp_rand(0,5),
                                  'reg_alpha':sp_rand(0,0.3),
                                  'n_estimators': [300,500,600,750,850,1000]},
                                  verbose=10, n_jobs=1, cv = 4, scoring='log_loss', n_iter = 30,
                                  refit=False)
        clf.fit(train_data_full, train_fault_severity_full)
        print('best clf score',clf.best_score_)
        print('best params:', clf.best_params_)
        toc=timeit.default_timer()
        print('Grid search time',toc - tic)
    if (use_early_stopping):
        xgb_classifier = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=watchlist,
                            early_stopping_rounds=100, verbose_eval=50)
        y_pred = xgb_classifier.predict(dtest,ntree_limit=xgb_classifier.best_iteration)
    else:
        xgb_classifier = xgb.train(params, dtrain_full, num_boost_round=num_boost_rounds, evals=[(dtrain_full,'train')],
                            verbose_eval=50)
        y_pred = xgb_classifier.predict(dtest)


    if(print_feature_imp):
        create_feature_map(xgb_features)
        imp_dict = xgb_classifier.get_fscore(fmap='xgb.fmap')
        imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1),reverse=True)
        print('{0:<20} {1:>5}'.format('Feature','Imp'))
        print("--------------------------------------")
        num_to_print = 30
        num_printed = 0
        for i in imp_dict:
            num_printed = num_printed + 1
            if (num_printed > num_to_print):
                continue
            print ('{0:20} {1:5.0f}'.format(i[0], i[1]))
    columns = ['predict_0','predict_1','predict_2']

    if (use_two_classes):
        columns = ['predict_low','predict_high']
    result_xgb_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)

    result_xgb_df = norm_rows(result_xgb_df)

    if(is_sub_run):
        print('creating xgb output')
    else:
        if(calculate_log_loss):
            result_xgb_df.reset_index('id',inplace=True)
            result_xgb_df = pd.merge(result_xgb_df,test[['id','fault_severity']],left_on = ['id'],
                                   right_on = ['id'],how='left')
            if(not use_two_classes):
                result_xgb_df['log_loss'] = result_xgb_df.apply(lambda row: get_log_loss_row(row),axis=1)
            else:
                result_xgb_df['log_loss'] = result_xgb_df.apply(lambda row: get_log_loss_row_two_classes(row),axis=1)
            print('log_loss',round(result_xgb_df['log_loss'].mean(),5))
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_xgb_df
#%%
vol_log_feature_cols = [col for col in list(train) if col.startswith('vol_log_feature_')]
vol_binned_log_feature_cols = [col for col in list(train) if col.startswith('vol_binned_log_feature_')]
vol_binned_offset_log_feature_cols = [col for col in list(train) if col.startswith('vol_binned_offset_log_feature_')]

log_by_loc_cols = [col for col in list(train) if col.startswith('most_common_location_of_log_feature_')]

loc_by_logs_cols = [col for col in list(train) if col.startswith('loc_log_feature_')]
loc_by_logs_summed_cols = [col for col in list(train) if col.startswith('summed_loc_log_feature_')]

#%%
count_features = ['count_log_features','count_event_types',
                  'count_resource_types','count_of_volumes']
location_by_features = ['location_by_unique_features','location_by_unique_resource_types',
                        'location_by_unique_event_types','location_by_most_common_event',
                        'location_by_most_common_resource','location_by_most_common_feature',
                        'location_max','location_number','location_by_mean_volume',
                        'location_by_max_volume','location_by_std_resource',
                        'location_by_median_feature_number','location_by_median_count_log_features',
                        'location_by_std_log_feature']
unique_combinations_hashed_features = ['combo_log_features_hash','combo_resource_types_hash',
                                       'combo_event_types_hash','num_ids_with_log_feature_combo',
                                       'num_ids_with_resource_type_combo','num_ids_with_event_type_combo']
vol_diff_features = ['log_203_82_vol_diff','log_201_80_vol_diff','log_312_232_vol_diff',
                     'log_170_54_vol_diff','log_171_55_vol_diff','log_193_71_vol_diff']

aggregate_volume_features = ['sum_log_features_volume','mean_log_features_volume','max_log_features_volume']

log_feature_stat_features = ['std_log_feature','median_log_feature','min_log_feature',
                               'max_log_feature','log_features_range']

log_feature_volume_features = (['volume_of_min_log_feature','volume_of_max_log_feature'] +
                                vol_log_feature_cols + vol_binned_log_feature_cols +
                                vol_binned_offset_log_feature_cols)
#%%
xgb_features = count_features
xgb_features = (xgb_features + location_by_features)
xgb_features.remove('location_by_std_log_feature')
xgb_features = (xgb_features + has_event_type_names)
xgb_features = (xgb_features + has_resource_type_names)
xgb_features = (xgb_features + loc_by_logs_cols)

xgb_features = (xgb_features + aggregate_volume_features)
xgb_features = (xgb_features + ['min_num_ids_with_log_feature'])

xgb_features = (xgb_features + vol_diff_features)

xgb_features = (xgb_features + log_feature_stat_features)
xgb_features = (xgb_features + log_feature_volume_features)

xgb_features = (xgb_features + ['position_event_type_1','position_event_type_2'])

#the so called "magic features"
xgb_features = (xgb_features + ['order_by_location'])
xgb_features = (xgb_features + ['reverse_order_by_location'])
xgb_features = (xgb_features + ['order_by_location_fraction'])
xgb_features = (xgb_features + ['reverse_order_by_location_fraction'])

xgb_features = (xgb_features + ['severity_type'])

xgb_features = (xgb_features + log_by_loc_cols)

xgb_features = (xgb_features + ['median_log_feature_most_common_location'])
xgb_features = (xgb_features + ['median_log_feature_most_common_event'])

xgb_features = (xgb_features + unique_combinations_hashed_features)

xgb_features = (xgb_features + ['median_event_type'])
xgb_features = (xgb_features + ['max_event_type'])
xgb_features = (xgb_features + ['min_event_type'])

xgb_features = (xgb_features + ['min_resource_type'])

params = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.5,
              'lambda': 0.95,
              'gamma': 1.0,
              'seed': 6,
              'colsample_bytree': 0.4,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
              'max_depth': 8,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#this is used to find early stopping rounds, then scale up when using all of dataset
#as number of rounds should be proportional to size of dataset

#result_xgb_df = fit_xgb_model(train,test,params,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
result_xgb_df = fit_xgb_model(train,test,params,xgb_features,
                              num_rounds = num_rounds, num_boost_rounds = 2300,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%
train_low = train.copy()
train_low['fault_severity'] = train_low['fault_severity'].map(lambda x: 0 if x == 0 else 1)
test_low = test.copy()
if(not is_sub_run):
    test_low['fault_severity'] = test_low['fault_severity'].map(lambda x: 0 if x == 0 else 1)
train_high = train.copy()
train_high['fault_severity'] = train_high['fault_severity'].map(lambda x: 0 if x < 2 else 1)
test_high = test.copy()
if(not is_sub_run):
    test_high['fault_severity'] = test_high['fault_severity'].map(lambda x: 0 if x < 2 else 1)
#%%
params_low = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.5,
              'lambda': 0.95,
              'gamma': 1.0,
              'seed': 6,
              'colsample_bytree': 0.4,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
              'max_depth': 8,
              'num_class':2}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_low = fit_xgb_model(train_low,test_low,params_low,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6,use_two_classes=True)
result_xgb_df_low = fit_xgb_model(train_low,test_low,params_low,xgb_features,
                              num_rounds = num_rounds, num_boost_rounds = 1600,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6,
                              use_two_classes=True)
#%%
params_high = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.5,
              'gamma': 0.5,
              'seed': 6,
              'colsample_bytree': 0.8,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
              'max_depth': 6,
              'num_class':2}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_high = fit_xgb_model(train_high,test_high,params_high,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6,use_two_classes=True)
result_xgb_df_high = fit_xgb_model(train_high,test_high,params_high,xgb_features,
                              num_rounds = num_rounds, num_boost_rounds = 1600,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6,
                              use_two_classes=True)
#%%
result_xgb_df_low_high = result_xgb_df.copy()
result_xgb_df_low_high['predict_0'] = result_xgb_df_low['predict_low']
result_xgb_df_low_high['predict_2'] = result_xgb_df_high['predict_high']
result_xgb_df_low_high['predict_1'] = result_xgb_df_low['predict_high'] * result_xgb_df_high['predict_low']
result_xgb_df_low_high[['predict_0','predict_1','predict_2']] = norm_rows(result_xgb_df_low_high[['predict_0','predict_1','predict_2']])
#%%
if (not is_sub_run):
    #useful for inspecting how fit performs
    ans_xgb_df = pd.merge(result_xgb_df,combined[['id','order_by_location','location_number','median_log_feature','combo_log_features_hash','num_ids_with_log_feature_combo',
                                                  'severity_type','location_max','median_event_type','combo_resource_types_hash'] + vol_log_feature_cols],
                          left_on = ['id'],
                          right_on = ['id'],how='left')
    ans_bad_xgb_df = ans_xgb_df.loc[ans_xgb_df.log_loss >= 2.5]
#%%
xgb_features_4 = count_features
xgb_features_4 = (xgb_features_4 + location_by_features)

xgb_features_4 = (xgb_features_4 + has_event_type_names)

xgb_features_4 = (xgb_features_4 + loc_by_logs_cols)

xgb_features_4 = (xgb_features_4 + aggregate_volume_features)

xgb_features_4 = (xgb_features_4 + ['min_num_ids_with_log_feature'])

xgb_features_4 = (xgb_features_4 + ['std_log_feature'])
xgb_features_4 = (xgb_features_4 + ['median_log_feature'])
xgb_features_4 = (xgb_features_4 + ['min_log_feature'])

xgb_features_4 = (xgb_features_4 + log_feature_volume_features)
xgb_features_4.remove('volume_of_max_log_feature')

xgb_features_4 = (xgb_features_4 + ['position_event_type_1'])
xgb_features_4 = (xgb_features_4 + ['position_resource_type_1'])

xgb_features_4 = (xgb_features_4 + ['switches_resource_type'])

xgb_features_4 = (xgb_features_4 + ['reverse_order_by_location'])
xgb_features_4 = (xgb_features_4 + ['reverse_order_by_location_fraction'])

xgb_features_4 = (xgb_features_4 + ['severity_type'])

xgb_features_4 = (xgb_features_4 + ['combo_log_features_by_nunique_loc_per_ids'])

xgb_features_4 = (xgb_features_4 + ['median_log_feature_most_common_location'])

xgb_features_4 = (xgb_features_4 + ['combo_resource_types_hash'])
xgb_features_4 = (xgb_features_4 + ['num_ids_with_resource_type_combo'])
xgb_features_4 = (xgb_features_4 + ['num_ids_with_event_type_combo'])

params_4 = {'learning_rate': 0.015,
              'subsample': 0.95,
              'reg_alpha': 0.5,
              'gamma': 0.7,
              'seed': 6,
              'colsample_bytree': 0.4,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
              'max_depth': 7,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_4 = fit_xgb_model(train,test,params_4,xgb_features_4,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
result_xgb_df_4 = fit_xgb_model(train,test,params_4,xgb_features_4,
                              num_rounds = num_rounds, num_boost_rounds = 1200,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%
xgb_features_5 = count_features
xgb_features_5 = (xgb_features_5 + location_by_features)
xgb_features_5.remove('location_by_std_log_feature')
xgb_features_5 = (xgb_features_5 + has_event_type_names)
xgb_features_5 = (xgb_features_5 + loc_by_logs_cols)

xgb_features_5 = (xgb_features_5 + aggregate_volume_features)
xgb_features_5 = (xgb_features_5 + ['min_num_ids_with_log_feature'])

xgb_features_5 = (xgb_features_5 + vol_diff_features)

xgb_features_5 = (xgb_features_5 + log_feature_stat_features)

xgb_features_5 = (xgb_features_5 + log_feature_volume_features)

xgb_features_5 = (xgb_features_5 + ['position_event_type_1','position_event_type_2'])

xgb_features_5 = (xgb_features_5 + ['order_by_location'])
xgb_features_5 = (xgb_features_5 + ['reverse_order_by_location'])
xgb_features_5 = (xgb_features_5 + ['order_by_location_fraction'])
xgb_features_5 = (xgb_features_5 + ['reverse_order_by_location_fraction'])

xgb_features_5 = (xgb_features_5 + ['severity_type'])

xgb_features_5 = (xgb_features_5 + log_by_loc_cols)

xgb_features_5 = (xgb_features_5 + ['median_log_feature_most_common_location'])
xgb_features_5 = (xgb_features_5 + ['median_log_feature_most_common_event'])

xgb_features_5 = (xgb_features_5 + unique_combinations_hashed_features)

xgb_features_5 = (xgb_features_5 + ['median_event_type'])
xgb_features_5 = (xgb_features_5 + ['max_event_type'])
xgb_features_5 = (xgb_features_5 + ['min_event_type'])

xgb_features_5 = (xgb_features_5 + ['min_resource_type'])

params_5 = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.4,
              'lambda': 0.9,
              'gamma': 2.0,
              'seed': 8,
              'colsample_bytree': 0.7,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
              'max_depth': 10,
              'min_child_weight': 4,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_5 = fit_xgb_model(train,test,params_5,xgb_features_5,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
result_xgb_df_5 = fit_xgb_model(train,test,params_5,xgb_features_5,
                              num_rounds = num_rounds, num_boost_rounds = 3400,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%
xgb_features_6 = count_features
xgb_features_6 = (xgb_features_6 + location_by_features)
xgb_features_6.remove('location_by_std_resource')
xgb_features_6.remove('location_by_std_log_feature')
xgb_features_6 = (xgb_features_6 + loc_by_logs_cols)
xgb_features_6 = (xgb_features_6 + loc_by_logs_summed_cols)

xgb_features_6 = (xgb_features_6 + ['mean_log_features_volume'])
xgb_features_6 = (xgb_features_6 + ['min_num_ids_with_log_feature'])

xgb_features_6 = (xgb_features_6 + log_feature_stat_features)

xgb_features_6 = (xgb_features_6 + ['rarest_log_feature'])
xgb_features_6 = (xgb_features_6 + ['volume_of_rarest_log_feature'])
xgb_features_6 = (xgb_features_6 + ['least_rare_log_feature'])
xgb_features_6 = (xgb_features_6 + ['volume_of_least_rare_log_feature'])

xgb_features_6 = (xgb_features_6 + log_feature_volume_features)

xgb_features_6 = (xgb_features_6 + ['order_by_location'])
xgb_features_6 = (xgb_features_6 + ['reverse_order_by_location'])

xgb_features_6 = (xgb_features_6 + ['severity_type'])

xgb_features_6 = (xgb_features_6 + log_by_loc_cols)

xgb_features_6 = (xgb_features_6 + ['median_log_feature_most_common_location'])
xgb_features_6 = (xgb_features_6 + ['median_log_feature_most_common_event'])

xgb_features_6 = (xgb_features_6 + unique_combinations_hashed_features)
xgb_features_6.remove('combo_log_features_hash')

xgb_features_6 = (xgb_features_6 + ['median_event_type'])
xgb_features_6 = (xgb_features_6 + ['max_event_type'])
xgb_features_6 = (xgb_features_6 + ['min_event_type'])

xgb_features_6 = (xgb_features_6 + ['min_resource_type'])
#%%
params_6 = {'learning_rate': 0.005,
              'subsample': 0.95,
              'reg_alpha': 0.05,
              'lambda': 0.98,
              'gamma': 2.0,
              'seed': 8,
              'colsample_bytree': 0.6,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
              'max_depth': 11,
              'min_child_weight': 2,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_6 = fit_xgb_model(train,test,params_6,xgb_features_6,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
result_xgb_df_6 = fit_xgb_model(train,test,params_6,xgb_features_6,
                              num_rounds = num_rounds, num_boost_rounds = 6000,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%
tic = timeit.default_timer()
train_data = train[xgb_features_6].values
train_fault_severity = train['fault_severity'].astype(int).values
test_data = test[xgb_features_6].values

extc = ExtraTreesClassifier(
                            n_estimators=2000,
                            max_features = None,
                            criterion= 'entropy',
                            min_samples_split= 1,
                            max_depth= None,
                            min_samples_leaf= 4,
                            random_state = 1,
                            n_jobs = 2
                            )
extc = extc.fit(train_data, train_fault_severity)

y_pred = extc.predict_proba(test_data)
columns = ['predict_0','predict_1','predict_2']
result_extc_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
result_extc_df.replace([0],[0.00001],inplace=True)
result_extc_df = norm_rows(result_extc_df)

if(is_sub_run):
    print('is a sub run')
else:
    result_extc_df.reset_index('id',inplace=True)
    result_extc_df = pd.merge(result_extc_df,test[['id','fault_severity']],left_on = ['id'],
                           right_on = ['id'],how='left')
    result_extc_df['log_loss'] = result_extc_df.apply(lambda row: get_log_loss_row(row),axis=1)
    print('log_loss',round(result_extc_df['log_loss'].mean(),5))

toc=timeit.default_timer()
print('extc Time',toc - tic)
#%%
tic = timeit.default_timer()
extc_features_2 = count_features
extc_features_2 = (extc_features_2 + location_by_features)
extc_features_2 = (extc_features_2 + has_event_type_names)
extc_features_2 = (extc_features_2 + loc_by_logs_cols)
extc_features_2 = (extc_features_2 + aggregate_volume_features)
extc_features_2 = (extc_features_2 + ['min_num_ids_with_log_feature'])

extc_features_2 = (extc_features_2 + ['std_log_feature'])
extc_features_2 = (extc_features_2 + ['median_log_feature'])
extc_features_2 = (extc_features_2 + ['min_log_feature'])
extc_features_2 = (extc_features_2 + log_feature_volume_features)
extc_features_2.remove('volume_of_max_log_feature')

extc_features_2 = (extc_features_2 + ['position_event_type_1'])
extc_features_2 = (extc_features_2 + ['position_resource_type_1'])
extc_features_2 = (extc_features_2 + ['switches_resource_type'])
extc_features_2 = (extc_features_2 + ['reverse_order_by_location'])
extc_features_2 = (extc_features_2 + ['reverse_order_by_location_fraction'])

extc_features_2 = (extc_features_2 + ['severity_type'])

extc_features_2 = (extc_features_2 + ['combo_log_features_by_nunique_loc_per_ids'])

extc_features_2 = (extc_features_2 + ['median_log_feature_most_common_location'])

extc_features_2 = (extc_features_2 + ['median_event_type'])
extc_features_2 = (extc_features_2 + ['max_event_type'])
extc_features_2 = (extc_features_2 + ['min_event_type'])
train_data = train[extc_features_2].values
train_fault_severity = train['fault_severity'].astype(int).values
test_data = test[extc_features_2].values

extc_2 = ExtraTreesClassifier(n_estimators=2000,
                            max_features = None,
                            criterion= 'entropy',
                            min_samples_split= 1,
                            max_depth= None,
                            min_samples_leaf= 5,
                            random_state = 1,
                            n_jobs = 2
                            )
extc_2 = extc_2.fit(train_data, train_fault_severity)
y_pred = extc_2.predict_proba(test_data)
columns = ['predict_0','predict_1','predict_2']
result_extc_df_2 = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
result_extc_df_2.replace([0],[0.00001],inplace=True)
result_extc_df_2 = norm_rows(result_extc_df_2)

if(is_sub_run):
    print('is a sub run')
else:
    result_extc_df_2.reset_index('id',inplace=True)
    result_extc_df_2 = pd.merge(result_extc_df_2,test[['id','fault_severity']],left_on = ['id'],
                           right_on = ['id'],how='left')
    result_extc_df_2['log_loss'] = result_extc_df_2.apply(lambda row: get_log_loss_row(row),axis=1)
    print('log_loss',round(result_extc_df_2['log_loss'].mean(),5))
toc=timeit.default_timer()
print('extc Time',toc - tic)
#%%
result_xgb_ens = (2.0 * result_xgb_df + 1.0 * result_xgb_df_4 + 1.0 * result_xgb_df_5 + 5.0 * result_xgb_df_6 +
                 + 3.0 * result_xgb_df_low_high
                 + 1.0 * result_extc_df
                 + 2.0 * result_extc_df_2
                 )
result_xgb_ens = result_xgb_ens[['predict_0','predict_1','predict_2']]
result_xgb_ens = norm_rows(result_xgb_ens)
if(is_sub_run):
    print('is a submission run')
else:
    result_xgb_ens['id'] = result_xgb_df['id']
    result_xgb_ens = pd.merge(result_xgb_ens,test[['id','fault_severity']],left_on = ['id'],
                           right_on = ['id'],how='left')
    result_xgb_ens['log_loss'] = result_xgb_ens.apply(lambda row: get_log_loss_row(row),axis=1)
    print('log_loss',round(result_xgb_ens['log_loss'].mean(),5))
#%%
if(is_sub_run):
    submission = result_xgb_ens.copy()
    submission.reset_index('id',inplace=True)
    submission = submission[['id','predict_0','predict_1','predict_2']]
    submission.to_csv('telstra_2.csv', index=False)
    print('xgb submission created')
#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)