# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:52:25 2016

@author: Jared Turkewitz
"""

#%%
import pandas as pd
import numpy as np
import random
from sklearn import cross_validation
import xgboost as xgb
import operator
from sklearn.neighbors import KNeighborsClassifier
import timeit

import warnings
warnings.filterwarnings("ignore")

tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'
#%%
tic=timeit.default_timer()
train_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/train.csv', header=0)
test_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/test.csv', header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
train_orig.sort_values('time',inplace=True)
test_orig.sort_values('time',inplace=True)
train_orig_copy = train_orig.copy()
train_orig_copy.drop_duplicates('place_id',inplace=True)
train_orig_copy['counting'] = 1
TEMP_VALUES = train_orig_copy['counting'].cumsum().values
PLACE_ID_DICT = pd.Series(TEMP_VALUES,index=train_orig_copy['place_id']).to_dict()
train_orig_copy['place_id_hash'] = train_orig_copy['place_id'].map(lambda x: PLACE_ID_DICT[x])
REVERSE_PLACE_ID_DICT = dict(zip(train_orig_copy.place_id_hash,train_orig_copy.place_id))
train_orig['place_id'] = train_orig['place_id'].map(lambda x: PLACE_ID_DICT[x])
del train_orig_copy
#%%
def norm_rows(df):
    with np.errstate(invalid='ignore'):
        return df.div(df.sum(axis=1), axis=0).fillna(0)

def get_datetimes(input_df):
    input_df['time_dt'] = pd.to_datetime(input_df['time'] * 60, unit='s')
    input_df['hour'] = input_df['time_dt'].map(lambda x: x.hour)
    input_df['minute'] = input_df['time_dt'].map(lambda x: x.minute)
    input_df['weekday'] = input_df['time_dt'].map(lambda x: x.weekday())
    input_df['month'] = input_df['time_dt'].map(lambda x: x.month)
    input_df['year'] = input_df['time_dt'].map(lambda x: x.year - 1970)
    input_df['day'] = input_df['time_dt'].map(lambda x: x.day)
    input_df['day_of_year'] = input_df['time_dt'].map(lambda x: x.dayofyear)
    beginning = pd.to_datetime('1970-01-01 00:00:00')
    input_df['date_int'] = input_df['time_dt'].map(lambda x: (x - beginning).days)
    input_df.drop(['time_dt'],axis=1,inplace=True)
    return input_df
#tic=timeit.default_timer()
#train_orig_3 = get_datetimes(train_orig.copy())
#test_orig_3 = get_datetimes(test_orig.copy())
#
#
#test_orig_3.to_csv('test_orig_3.csv', index=False,float_format='%.4f')
#train_orig_3.to_csv('train_orig_3.csv', index=False)
#toc=timeit.default_timer()
#print('Combining time',toc - tic)
#%%
def get_top6_places_and_probs(row):
    row.sort()
    inds = row.index[-6:][::-1].tolist()
    probs = row[-6:][::-1].tolist()
    return inds + probs
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
def apply_dict(common_dict,x,def_val=0):
    try:
        return common_dict[x]
    except KeyError:
        return def_val
def fit_xgb_model(train, test, params, xgb_features, num_rounds = 10, num_boost_rounds = 10,
                  use_early_stopping = True, print_feature_imp = False,
                  random_seed = 5, reweight_probs = True, calculate_log_loss = True,
                  is_sub_run = True):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    if(use_early_stopping):
        X_train, X_watch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
        watch_data = X_watch[xgb_features].values
        watch_place_id = X_watch['labels'].astype(int).values
        dwatch = xgb.DMatrix(watch_data, watch_place_id)
        train_data = X_train[xgb_features].values
        train_place_id = X_train['labels'].astype(int).values
        dtrain = xgb.DMatrix(train_data, train_place_id)
        watchlist = [(dtrain, 'train'),(dwatch, 'watch')]
    else:
        train_data_full = train[xgb_features].values
        train_place_id_full = train['labels'].astype(int).values

        weights = (train['date_int']) + 1000
        weights = weights / weights.mean()
        dtrain_full = xgb.DMatrix(train_data_full, train_place_id_full, weight = weights)

    test_data = test[xgb_features].values
    dtest = xgb.DMatrix(test_data)

    if (use_early_stopping):
        xgb_classifier = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=watchlist,
                            early_stopping_rounds=10, verbose_eval=50)
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
    result_xgb_df = pd.DataFrame(index=test.row_id,data=y_pred)
    result_xgb_df['pred'] = result_xgb_df.apply(get_top6_places_and_probs,axis=1)
    result_xgb_df['pred_0'] = result_xgb_df['pred'].map(lambda x: x[0])
    result_xgb_df['pred_1'] = result_xgb_df['pred'].map(lambda x: x[1])
    result_xgb_df['pred_2'] = result_xgb_df['pred'].map(lambda x: x[2])
    result_xgb_df['pred_3'] = result_xgb_df['pred'].map(lambda x: x[3])
    result_xgb_df['pred_4'] = result_xgb_df['pred'].map(lambda x: x[4])
    result_xgb_df['pred_5'] = result_xgb_df['pred'].map(lambda x: x[5])

    result_xgb_df['prob_0'] = result_xgb_df['pred'].map(lambda x: x[6])
    result_xgb_df['prob_1'] = result_xgb_df['pred'].map(lambda x: x[7])
    result_xgb_df['prob_2'] = result_xgb_df['pred'].map(lambda x: x[8])
    result_xgb_df['prob_3'] = result_xgb_df['pred'].map(lambda x: x[9])
    result_xgb_df['prob_4'] = result_xgb_df['pred'].map(lambda x: x[10])
    result_xgb_df['prob_5'] = result_xgb_df['pred'].map(lambda x: x[11])
    result_xgb_df.drop('pred',axis=1,inplace=True)

    if(is_sub_run):
        print('creating xgb output')
        result_xgb_df.reset_index('row_id',inplace=True)
    else:
        print('cv')
        result_xgb_df.reset_index('row_id',inplace=True)
        result_xgb_df = pd.merge(result_xgb_df,test[['row_id','place_id','labels']],left_on = ['row_id'],
                               right_on = ['row_id'],how='left')
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_xgb_df
#%%
def fit_knn_model(train, test, xgb_features, num_neighbors = 25,
                  random_seed = 5,
                  compute_probs = False,
                  pred_ratio_dict = {},
                  pred_ratio_dict_0 = {},
                  pred_ratio_dict_1 = {},
                  pred_ratio_dict_2 = {},
                  pred_ratio_dict_3 = {},
                  low_max_dict = {},
                  high_min_dict = {},
                  high_mean_dict = {},
                  low_mean_dict = {},
                  is_sub_run = True):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    xgb_features_1 = xgb_features + ['hour_and_minute']
    xgb_features_2 = xgb_features + ['hour_shift_and_minute']
    xgb_features_3 = xgb_features + ['hour_shift_1_and_minute']
    xgb_features_4 = xgb_features + ['hour_shift_2_and_minute']

    train_data_full_1 = train[xgb_features_1].values
    train_data_full_2 = train[xgb_features_2].values
    train_data_full_3 = train[xgb_features_3].values
    train_data_full_4 = train[xgb_features_4].values
    train_place_id_full = train['labels'].astype(int).values

    if(compute_probs):
        if(is_sub_run):
            time_min = 786242
            time_max = 1006589
        else:
            time_min = 600001
            time_max = 786239
        time_diff = time_max - time_min
        test_0_cond = (test.time >= (time_min + 0.0 * time_diff)) & (test.time < (time_min + 0.25 * time_diff))
        test_1_cond = (test.time >= (time_min + 0.25 * time_diff)) & (test.time < (time_min + 0.5 * time_diff))
        test_2_cond = (test.time >= (time_min + 0.5 * time_diff)) & (test.time < (time_min + 0.75 * time_diff))
        test_3_cond = (test.time >= (time_min + 0.75 * time_diff)) & (test.time <= time_max)
        test_0 = test[test_0_cond]
        test_1 = test[test_1_cond]
        test_2 = test[test_2_cond]
        test_3 = test[test_3_cond]

        test_data_1_0 = test_0[xgb_features_1].values
        test_data_2_0 = test_0[xgb_features_2].values
        test_data_3_0 = test_0[xgb_features_3].values
        test_data_4_0 = test_0[xgb_features_4].values
        test_data_1_1 = test_1[xgb_features_1].values
        test_data_2_1 = test_1[xgb_features_2].values
        test_data_3_1 = test_1[xgb_features_3].values
        test_data_4_1 = test_1[xgb_features_4].values
        test_data_1_2 = test_2[xgb_features_1].values
        test_data_2_2 = test_2[xgb_features_2].values
        test_data_3_2 = test_2[xgb_features_3].values
        test_data_4_2 = test_2[xgb_features_4].values
        test_data_1_3 = test_3[xgb_features_1].values
        test_data_2_3 = test_3[xgb_features_2].values
        test_data_3_3 = test_3[xgb_features_3].values
        test_data_4_3 = test_3[xgb_features_4].values

    test_data_1 = test[xgb_features_1].values
    test_data_2 = test[xgb_features_2].values
    test_data_3 = test[xgb_features_3].values
    test_data_4 = test[xgb_features_4].values

    def weight_func(distances):
        return (distances ** -2)
    knn_classifier_1 = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weight_func,
                               metric='manhattan',n_jobs = 2)
    knn_classifier_2 = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weight_func,
                               metric='manhattan',n_jobs = 2)
    knn_classifier_3 = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weight_func,
                               metric='manhattan',n_jobs = 2)
    knn_classifier_4 = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weight_func,
                               metric='manhattan',n_jobs = 2)
    knn_classifier_1.fit(train_data_full_1,train_place_id_full)
    knn_classifier_2.fit(train_data_full_2,train_place_id_full)
    knn_classifier_3.fit(train_data_full_3,train_place_id_full)
    knn_classifier_4.fit(train_data_full_4,train_place_id_full)
    y_pred_1 = knn_classifier_1.predict_proba(test_data_1)
    y_pred_2 = knn_classifier_2.predict_proba(test_data_2)
    y_pred_3 = knn_classifier_3.predict_proba(test_data_3)
    y_pred_4 = knn_classifier_4.predict_proba(test_data_4)
    y_pred =  (y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4) / 4.0

    if(compute_probs):
        y_pred_1_0 = knn_classifier_1.predict_proba(test_data_1_0)
        y_pred_2_0 = knn_classifier_2.predict_proba(test_data_2_0)
        y_pred_3_0 = knn_classifier_3.predict_proba(test_data_3_0)
        y_pred_4_0 = knn_classifier_4.predict_proba(test_data_4_0)
        y_pred_0 =  (y_pred_1_0 + y_pred_2_0 + y_pred_3_0 + y_pred_4_0) / 4.0
        y_pred_1_1 = knn_classifier_1.predict_proba(test_data_1_1)
        y_pred_2_1 = knn_classifier_2.predict_proba(test_data_2_1)
        y_pred_3_1 = knn_classifier_3.predict_proba(test_data_3_1)
        y_pred_4_1 = knn_classifier_4.predict_proba(test_data_4_1)
        y_pred_1 =  (y_pred_1_1 + y_pred_2_1 + y_pred_3_1 + y_pred_4_1) / 4.0
        y_pred_1_2 = knn_classifier_1.predict_proba(test_data_1_2)
        y_pred_2_2 = knn_classifier_2.predict_proba(test_data_2_2)
        y_pred_3_2 = knn_classifier_3.predict_proba(test_data_3_2)
        y_pred_4_2 = knn_classifier_4.predict_proba(test_data_4_2)
        y_pred_2 =  (y_pred_1_2 + y_pred_2_2 + y_pred_3_2 + y_pred_4_2) / 4.0
        y_pred_1_3 = knn_classifier_1.predict_proba(test_data_1_3)
        y_pred_2_3 = knn_classifier_2.predict_proba(test_data_2_3)
        y_pred_3_3 = knn_classifier_3.predict_proba(test_data_3_3)
        y_pred_4_3 = knn_classifier_4.predict_proba(test_data_4_3)
        y_pred_3 =  (y_pred_1_3 + y_pred_2_3 + y_pred_3_3 + y_pred_4_3) / 4.0
        result_knn_df_0 = pd.DataFrame(index=test_0.row_id,data=y_pred_0)
        result_knn_df_1 = pd.DataFrame(index=test_1.row_id,data=y_pred_1)
        result_knn_df_2 = pd.DataFrame(index=test_2.row_id,data=y_pred_2)
        result_knn_df_3 = pd.DataFrame(index=test_3.row_id,data=y_pred_3)

        result_knn_df_0 = result_knn_df_0 * pd.Series(pred_ratio_dict_0)
        result_knn_df_1 = result_knn_df_1 * pd.Series(pred_ratio_dict_1)
        result_knn_df_2 = result_knn_df_2 * pd.Series(pred_ratio_dict_2)
        result_knn_df_3 = result_knn_df_3 * pd.Series(pred_ratio_dict_3)
        result_knn_df = pd.concat([result_knn_df_0,result_knn_df_1,result_knn_df_2,result_knn_df_3], axis=0)

        result_knn_df = result_knn_df * pd.Series(low_max_dict)
        result_knn_df = norm_rows(result_knn_df)
        result_knn_df['pred'] = result_knn_df.apply(get_top6_places_and_probs,axis=1)
        result_knn_df['pred_0'] = result_knn_df['pred'].map(lambda x: x[0])
        result_knn_df['pred_1'] = result_knn_df['pred'].map(lambda x: x[1])
        result_knn_df['pred_2'] = result_knn_df['pred'].map(lambda x: x[2])
        result_knn_df['pred_3'] = result_knn_df['pred'].map(lambda x: x[3])
        result_knn_df['pred_4'] = result_knn_df['pred'].map(lambda x: x[4])
        result_knn_df['pred_5'] = result_knn_df['pred'].map(lambda x: x[5])

        result_knn_df['prob_0'] = result_knn_df['pred'].map(lambda x: x[6])
        result_knn_df['prob_1'] = result_knn_df['pred'].map(lambda x: x[7])
        result_knn_df['prob_2'] = result_knn_df['pred'].map(lambda x: x[8])
        result_knn_df['prob_3'] = result_knn_df['pred'].map(lambda x: x[9])
        result_knn_df['prob_4'] = result_knn_df['pred'].map(lambda x: x[10])
        result_knn_df['prob_5'] = result_knn_df['pred'].map(lambda x: x[11])
        result_knn_df.drop('pred',axis=1,inplace=True)
    else:
        result_knn_df = pd.DataFrame(index=test.row_id,data=y_pred)
        result_knn_df = result_knn_df * pd.Series(high_min_dict)
        result_knn_df = result_knn_df * pd.Series(high_mean_dict)
        result_knn_df = result_knn_df * pd.Series(low_mean_dict)
        result_knn_df = norm_rows(result_knn_df)

    if(is_sub_run):
        print('creating xgb output')
        result_knn_df.reset_index('row_id',inplace=True)
    else:
        result_knn_df.reset_index('row_id',inplace=True)
        result_knn_df = pd.merge(result_knn_df,test[['row_id','place_id','labels']],left_on = ['row_id'],
                               right_on = ['row_id'],how='left')
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_knn_df
#%%
train_orig_3 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/train_orig_3.csv', header=0)
test_orig_3 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/test_orig_3.csv', header=0)

#is_sub_run = False
is_sub_run = True
if(is_sub_run):
    TRAIN = train_orig_3.copy()
    TEST = test_orig_3.copy()
else:
    time_cond = train_orig_3['time'] <= 600000
    TRAIN = train_orig_3[time_cond]
    TEST = train_orig_3.loc[~time_cond & (train_orig_3['time'] < 786240)]
    train_place_set = set(TRAIN.place_id.unique())
    TEST['unique_place'] = TEST['place_id'].map(lambda x: x in train_place_set)
    TEST = TEST[TEST['unique_place']]

del train_orig_3
del test_orig_3

TRAIN['hour_shift'] = TRAIN['hour'].map(lambda x: x if x >= 12 else x + 24)
TRAIN['hour_shift_1'] = TRAIN['hour'].map(lambda x: x if x >= 6 else x + 24)
TRAIN['hour_shift_2'] = TRAIN['hour'].map(lambda x: x if x >= 18 else x + 24)
TRAIN['hour_and_minute'] = TRAIN['hour'] + TRAIN['minute'] / 60
TRAIN['hour_shift_and_minute'] = TRAIN['hour_shift'] + TRAIN['minute'] / 60
TRAIN['hour_shift_1_and_minute'] = TRAIN['hour_shift_1'] + TRAIN['minute'] / 60
TRAIN['hour_shift_2_and_minute'] = TRAIN['hour_shift_2'] + TRAIN['minute'] / 60

TEST['hour_shift'] = TEST['hour'].map(lambda x: x if x >= 12 else x + 24)
TEST['hour_shift_1'] = TEST['hour'].map(lambda x: x if x >= 6 else x + 24)
TEST['hour_shift_2'] = TEST['hour'].map(lambda x: x if x >= 18 else x + 24)
TEST['hour_and_minute'] = TEST['hour'] + TEST['minute'] / 60
TEST['hour_shift_and_minute'] = TEST['hour_shift'] + TEST['minute'] / 60
TEST['hour_shift_1_and_minute'] = TEST['hour_shift_1'] + TEST['minute'] / 60
TEST['hour_shift_2_and_minute'] = TEST['hour_shift_2'] + TEST['minute'] / 60

TRAIN.drop(['minute','day','day_of_year'],axis=1,inplace=True)
TEST.drop(['minute','day','day_of_year'],axis=1,inplace=True)

PLACE_DICT = TRAIN['place_id'].value_counts().to_dict()
TRAIN['place_freq_all'] = TRAIN['place_id'].map(lambda x: PLACE_DICT[x])
del PLACE_DICT
#%%
def get_probs(df,prob_name):
    temp_probs = pd.read_csv(prob_name, header=0)
    for index,row in temp_probs.iterrows():
        col_name = 'pred_cv_' + str(row['place'])
        col_value = row['pred_place_prob']
        df[col_name] = col_value
    return df
#%%
tic_xgb=timeit.default_timer()
def run_xgb(train,test,output_name = 'res.csv',is_sub_run=True,is_xgb_run = True):

    time_max_dict = train.groupby('place_id')['time'].max().to_dict()
    time_min_dict = train.groupby('place_id')['time'].min().to_dict()
    time_mean_dict = train.groupby('place_id')['time'].mean().to_dict()
    train['max_time'] = train['place_id'].map(time_max_dict)
    train['min_time'] = train['place_id'].map(time_min_dict)
    train['mean_time'] = train['place_id'].map(time_mean_dict)
    if(is_xgb_run or compute_knn_probs):
        temp_probs = pd.read_csv(prob_name, header=0)
        temp_probs_0 = pd.read_csv(prob_name_0, header=0)
        temp_probs_1 = pd.read_csv(prob_name_1, header=0)
        temp_probs_2 = pd.read_csv(prob_name_2, header=0)
        temp_probs_3 = pd.read_csv(prob_name_3, header=0)
        temp_dict = pd.Series(temp_probs.place_prob_ratio.values,index=temp_probs.place).to_dict()
        temp_dict_0 = pd.Series(temp_probs_0.place_prob_ratio.values,index=temp_probs_0.place).to_dict()
        temp_dict_1 = pd.Series(temp_probs_1.place_prob_ratio.values,index=temp_probs_1.place).to_dict()
        temp_dict_2 = pd.Series(temp_probs_2.place_prob_ratio.values,index=temp_probs_2.place).to_dict()
        temp_dict_3 = pd.Series(temp_probs_3.place_prob_ratio.values,index=temp_probs_3.place).to_dict()
        temp_max_dict = pd.Series(temp_probs['pred_place_prob_max'].values,index=temp_probs.place).to_dict()
        temp_prob_dict = pd.Series(temp_probs['pred_place_prob'].values,index=temp_probs.place).to_dict()
        train['pred_ratio'] = train['place_id'].map(lambda x: apply_dict(temp_dict,x,1))
        train['pred_ratio_0'] = train['place_id'].map(lambda x: apply_dict(temp_dict_0,x,1))
        train['pred_ratio_1'] = train['place_id'].map(lambda x: apply_dict(temp_dict_1,x,1))
        train['pred_ratio_2'] = train['place_id'].map(lambda x: apply_dict(temp_dict_2,x,1))
        train['pred_ratio_3'] = train['place_id'].map(lambda x: apply_dict(temp_dict_3,x,1))

        train['pred_ratio'] = train['place_id'].map(lambda x: apply_dict(temp_dict,x,1))
        train['pred_max'] = train['place_id'].map(lambda x: apply_dict(temp_max_dict,x,1))
        train['pred_prob'] = train['place_id'].map(lambda x: apply_dict(temp_prob_dict,x,1))
    time_cutoff_percent = 0.8
    time_loc_percent = 0.0
    time_high_mean_time_percent = 0.75
    time_high_min_time_percent = 0.9
    time_low_mean_time_percent = 0.25
    if(is_sub_run):
        time_test = 786240
    else:
        time_test = 600001
    time_cutoff = time_cutoff_percent * time_test
    time_high_cutoff = time_loc_percent * time_test
    time_high_mean_time = time_high_mean_time_percent * time_test
    time_high_min_time = time_high_min_time_percent * time_test
    time_low_mean_time = time_low_mean_time_percent * time_test

    place_value_counts_dict = train['place_id'].value_counts().to_dict()
    train['place_freq'] = train['place_id'].map(lambda x: place_value_counts_dict[x])

    if(not is_xgb_run):
        train_low_freq_cond = train['place_freq'] <= 3
    else:
        train_low_freq_cond = train['place_freq'] <= 10
        train_high_freq_cond = train['place_freq'] >= 8
    train = train[(~train_low_freq_cond)]
    if(is_xgb_run):
        train = train[train_high_freq_cond]

    time_high_cond = train['time'] >= time_high_cutoff
    train_high = train[time_high_cond]
    high_place_value_counts_dict = train_high['place_id'].value_counts().to_dict()
    train['high_time_place_freq'] = train['place_id'].map(lambda x: apply_dict(high_place_value_counts_dict,x,0))

    train_low = train[~time_high_cond]
    low_place_value_counts_dict = train_low['place_id'].value_counts().to_dict()
    train['low_time_place_freq'] = train['place_id'].map(lambda x: apply_dict(low_place_value_counts_dict,x,0))

    label_mapping = dict(zip(list(set(train.place_id.values)),range(len(list(set(train.place_id.values))))))
    train['labels'] = train['place_id'].map(lambda x: label_mapping[x])
    label_id_rev_dict = {v: k for k, v in label_mapping.items()}
    if(not is_sub_run):
        test['labels'] =  test['place_id'].map(lambda x: apply_dict(label_mapping,x,-1))

    low_max_dict = {}
    high_min_dict = {}
    high_mean_dict = {}
    low_mean_dict = {}
    temp_dict = {}
    temp_dict_0 = {}
    temp_dict_1 = {}
    temp_dict_2 = {}
    temp_dict_3 = {}
    train_no_dups = train.drop_duplicates('place_id')
    train_no_dups['high_mean_time'] = train_no_dups['mean_time'].map(lambda x: 1.3 if x >= time_high_mean_time else 1.0)
    high_mean_dict = pd.Series(train_no_dups.high_mean_time.values,index=train_no_dups['labels'])

    train_no_dups['low_mean_time'] = train_no_dups['mean_time'].map(lambda x: 0.5 if x <= time_low_mean_time else 1.0)
    low_mean_dict = pd.Series(train_no_dups.low_mean_time.values,index=train_no_dups['labels'])

    train_no_dups['high_min_time'] = train_no_dups['min_time'].map(lambda x: 1.3 if x >= time_high_min_time else 1.0)
    high_min_dict = pd.Series(train_no_dups.high_min_time.values,index=train_no_dups['labels'])

    if(compute_knn_probs):

        train_no_dups['low_max_time'] = train_no_dups['max_time'].map(lambda x: 0.5 if x <= time_cutoff else 1.0)
        train_no_dups_low_ratio_cond = train_no_dups['pred_ratio'] <= 0.35
        train_no_dups['low_max_time'][~train_no_dups_low_ratio_cond] = 1.0
        low_max_dict = pd.Series(train_no_dups.low_max_time.values,index=train_no_dups['labels'])
        temp_probs = pd.read_csv(prob_name, header=0)
        temp_probs['place'] = temp_probs['place'].map(lambda x: apply_dict(label_mapping,x,-1))
        place_weight = 0.9
        temp_probs['place_prob_ratio'] = temp_probs['place_prob_ratio'] ** place_weight
        temp_dict = pd.Series(temp_probs.place_prob_ratio.values,index=temp_probs.place).to_dict()

        temp_probs_0 = pd.read_csv(prob_name_0, header=0)
        temp_probs_1 = pd.read_csv(prob_name_1, header=0)
        temp_probs_2 = pd.read_csv(prob_name_2, header=0)
        temp_probs_3 = pd.read_csv(prob_name_3, header=0)

        temp_probs_0['place'] = temp_probs_0['place'].map(lambda x: apply_dict(label_mapping,x,-1))
        temp_probs_1['place'] = temp_probs_1['place'].map(lambda x: apply_dict(label_mapping,x,-1))
        temp_probs_2['place'] = temp_probs_2['place'].map(lambda x: apply_dict(label_mapping,x,-1))
        temp_probs_3['place'] = temp_probs_3['place'].map(lambda x: apply_dict(label_mapping,x,-1))

        temp_probs_0['place_prob_ratio'] = temp_probs_0['place_prob_ratio'] ** place_weight
        temp_probs_1['place_prob_ratio'] = temp_probs_1['place_prob_ratio'] ** place_weight
        temp_probs_2['place_prob_ratio'] = temp_probs_2['place_prob_ratio'] ** place_weight
        temp_probs_3['place_prob_ratio'] = temp_probs_3['place_prob_ratio'] ** place_weight

        temp_dict_0 = pd.Series(temp_probs_0.place_prob_ratio.values,index=temp_probs_0.place).to_dict()
        temp_dict_1 = pd.Series(temp_probs_1.place_prob_ratio.values,index=temp_probs_1.place).to_dict()
        temp_dict_2 = pd.Series(temp_probs_2.place_prob_ratio.values,index=temp_probs_2.place).to_dict()
        temp_dict_3 = pd.Series(temp_probs_3.place_prob_ratio.values,index=temp_probs_3.place).to_dict()

    xgb_features = []
    xgb_features = (xgb_features + ['x','y','accuracy','weekday','year','month'])
    xgb_features = (xgb_features + ['hour_shift_and_minute'])
    xgb_features = (xgb_features + ['hour_and_minute'])

    knn_features = []
    knn_features = (knn_features + ['x','y','weekday','year','month'])
    knn_features = (knn_features + ['accuracy'])
    knn_features = (knn_features + ['acc_binned'])

    if(not is_xgb_run):
        weights_dict = {}
        if(not compute_knn_probs):
            weights_dict['x'] = 500
            weights_dict['y'] = 1100
            weights_dict['hour_shift'] = 3.5
            weights_dict['weekday'] = 3
            weights_dict['month'] = 0.3
            weights_dict['year'] = 5
            weights_dict['accuracy'] = 2.0
            weights_dict['acc_binned'] = 2.0
            number_neighbors = 45
        else:
            weights_dict['x'] = 500
            weights_dict['y'] = 1100
            weights_dict['hour_shift'] = 3.5
            weights_dict['weekday'] = 3.0
            weights_dict['month'] = 0.2
            weights_dict['year'] = 3
            weights_dict['accuracy'] = 2.0
            weights_dict['acc_binned'] = 2.0
            number_neighbors = 50
            if(is_sub_run):
                number_neighbors = number_neighbors * 1.1
        train['x'] = train['x'] * weights_dict['x']
        train['y'] = train['y'] * weights_dict['y']
        train['hour_and_minute'] = train['hour_and_minute'] * weights_dict['hour_shift']
        train['hour_shift_and_minute'] = train['hour_shift_and_minute'] * weights_dict['hour_shift']
        train['hour_shift_1_and_minute'] = train['hour_shift_1_and_minute'] * weights_dict['hour_shift']
        train['hour_shift_2_and_minute'] = train['hour_shift_2_and_minute'] * weights_dict['hour_shift']
        train['weekday'] = train['weekday'] * weights_dict['weekday']
        train['month'] = train['month'] * weights_dict['month']
        train['year'] = train['year'] * weights_dict['year']
        acc_bins = [0,10,30,60,70,100,120,150,180,200,500,10000]
        train['acc_binned'] = np.digitize(train['accuracy'],acc_bins,right=True)
        train['acc_binned'] = train['acc_binned'] * weights_dict['acc_binned']
        train['accuracy'] = np.log10(train['accuracy']) * weights_dict['accuracy']

        test['x'] = test['x'] * weights_dict['x']
        test['y'] = test['y'] * weights_dict['y']
        test['hour_and_minute'] = test['hour_and_minute'] * weights_dict['hour_shift']
        test['hour_shift_and_minute'] = test['hour_shift_and_minute'] * weights_dict['hour_shift']
        test['hour_shift_1_and_minute'] = test['hour_shift_1_and_minute'] * weights_dict['hour_shift']
        test['hour_shift_2_and_minute'] = test['hour_shift_2_and_minute'] * weights_dict['hour_shift']
        test['weekday'] = test['weekday'] * weights_dict['weekday']
        test['month'] = test['month'] * weights_dict['month']
        test['year'] = test['year'] * weights_dict['year']
        test['acc_binned'] = np.digitize(test['accuracy'],acc_bins,right=True)
        test['acc_binned'] = test['acc_binned'] * weights_dict['acc_binned']
        test['accuracy'] = np.log10(test['accuracy']) * weights_dict['accuracy']

    number_classes = train['place_id'].nunique()
    params = {'learning_rate': 0.05,
              'subsample': 0.9,
              'reg_alpha': 0.8,
#              'lambda': 0.95,
              'gamma': 2.0,
              'seed': 6,
#              'colsample_bytree': 0.8,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
              'max_depth': 6,
#              'min_child_weight': 2,
              'num_class':number_classes}
    num_rounds = 10000
    if(number_classes > 10):
#        result_xgb_df = fit_xgb_model(train,test,params,xgb_features,
#                                                       num_rounds = num_rounds, num_boost_rounds = num_rounds,
#                                                       use_early_stopping = True, print_feature_imp = False,
#                                                       random_seed = 6, is_sub_run = is_sub_run)
        if(is_xgb_run):
            result_xgb_df = fit_xgb_model(train,test,params,xgb_features,
                                  num_rounds = num_rounds, num_boost_rounds = 20,
                                  use_early_stopping = False, print_feature_imp = False,random_seed = 6,
                                  is_sub_run = is_sub_run)
        else:
            result_xgb_df = fit_knn_model(train,test,knn_features,
                                          num_neighbors = number_neighbors,
                                          compute_probs=compute_knn_probs,
                                          pred_ratio_dict = temp_dict,
                                          pred_ratio_dict_0 = temp_dict_0,
                                          pred_ratio_dict_1 = temp_dict_1,
                                          pred_ratio_dict_2 = temp_dict_2,
                                          pred_ratio_dict_3 = temp_dict_3,
                                          low_max_dict = low_max_dict,
                                          high_min_dict = high_min_dict,
                                          high_mean_dict = high_mean_dict,
                                          low_mean_dict = low_mean_dict,
                                          is_sub_run = is_sub_run)
            if(not compute_knn_probs):
                if (not is_sub_run):
                    result_xgb_df.drop(['row_id','labels','place_id'],axis=1,inplace=True)
                else:
                    result_xgb_df.drop(['row_id'],axis=1,inplace=True)
                mean_probs = result_xgb_df.mean()
                max_probs = result_xgb_df.max()
                std_probs = result_xgb_df.std()

                mean_probs.name = 'pred_place_prob'
                max_probs.name = 'pred_place_prob_max'
                std_probs.name = 'pred_place_prob_std'
                mean_df = pd.concat([mean_probs,max_probs,std_probs],axis=1)
                mean_df.fillna(0,inplace=True)

                mean_df.reset_index(inplace=True)
                mean_df.rename(columns = {'index':'place'},inplace=True)

                place_value_counts_norm_dict = train['place_id'].value_counts(1).to_dict()
                mean_df['place'] = mean_df['place'].map(lambda x: label_id_rev_dict[x])
                mean_df['train_place_prob'] = mean_df['place'].map(lambda x: place_value_counts_norm_dict[x])
                mean_df['place_prob_ratio'] = mean_df['pred_place_prob'] / mean_df['train_place_prob']
                mean_df['place_prob_diff'] = mean_df['pred_place_prob'] - mean_df['train_place_prob']
                if (not is_sub_run):
                    test_place_value_counts_norm_dict = test['place_id'].value_counts(1).to_dict()
                    mean_df['test_place_prob'] = mean_df['place'].map(lambda x: apply_dict(test_place_value_counts_norm_dict,x,0))
                    mean_df['test_place_prob_ratio'] = mean_df['test_place_prob'] / mean_df['train_place_prob']
                mean_df.to_csv(output_name, index=False)
                return mean_df
        result_xgb_df['pred_0'] = result_xgb_df['pred_0'].map(lambda x: label_id_rev_dict[x])
        result_xgb_df['pred_1'] = result_xgb_df['pred_1'].map(lambda x: label_id_rev_dict[x])
        result_xgb_df['pred_2'] = result_xgb_df['pred_2'].map(lambda x: label_id_rev_dict[x])
        result_xgb_df['pred_3'] = result_xgb_df['pred_3'].map(lambda x: label_id_rev_dict[x])
        result_xgb_df['pred_4'] = result_xgb_df['pred_4'].map(lambda x: label_id_rev_dict[x])
        result_xgb_df['pred_5'] = result_xgb_df['pred_5'].map(lambda x: label_id_rev_dict[x])

        if (is_sub_run):
            result_xgb_df = result_xgb_df[['row_id','pred_0','pred_1','pred_2','pred_3','pred_4','pred_5',
                               'prob_0','prob_1','prob_2','prob_3','prob_4','prob_5']]
        else:
            result_xgb_df = result_xgb_df[['row_id','place_id','pred_0','pred_1','pred_2','pred_3','pred_4','pred_5',
                               'prob_0','prob_1','prob_2','prob_3','prob_4','prob_5']]

        result_xgb_df.to_csv(output_name, index=False)

        if(not is_sub_run):
            result_xgb_df['pred_0_res'] = result_xgb_df['pred_0'] - result_xgb_df['place_id']
            result_xgb_df['pred_1_res'] = result_xgb_df['pred_1'] - result_xgb_df['place_id']
            result_xgb_df['pred_2_res'] = result_xgb_df['pred_2'] - result_xgb_df['place_id']
            result_xgb_df['pred_0_res'] = (result_xgb_df['pred_0_res'] == 0).astype(int)
            result_xgb_df['pred_1_res'] = (result_xgb_df['pred_1_res'] == 0).astype(int) / 2
            result_xgb_df['pred_2_res'] = (result_xgb_df['pred_2_res'] == 0).astype(int) / 3
            result_xgb_df['apk'] = result_xgb_df['pred_0_res'] + result_xgb_df['pred_1_res'] + result_xgb_df['pred_2_res']
            print(output_name)
            print('apk',result_xgb_df['apk'].mean())
    else:
        print(output_name,'less than 10 classes in train')
        return 7
    return result_xgb_df

if (is_sub_run):
    base_str = '/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/Test_Results/Run23/'
else:
    base_str = '/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/Train_Results/Run23/'
is_xgb_run = False
#is_xgb_run = False
#compute_knn_probs = False
compute_knn_probs = True
if(is_xgb_run):
    x_range = 40
    y_range = 50
else:
    x_range = 20
    y_range = 25
for i in range(x_range):
    x_width = 0.5
    y_width = 0.4
    if (is_xgb_run):
        extra_edge_x = 0.02
        extra_edge_y = 0.02
    else:
        extra_edge_x = 0.04
        extra_edge_y = 0.04
    if (i != (x_range - 1)):
        train_x = TRAIN.loc[(TRAIN.x < ((i+1)*x_width + extra_edge_x)) & (TRAIN.x >= (i*x_width - extra_edge_x))]
        test_x = TEST.loc[(TEST.x < (i+1)*x_width) & (TEST.x >= i*x_width)]
    else:
        train_x = TRAIN.loc[(TRAIN.x <= ((i+1)*x_width + extra_edge_x)) & (TRAIN.x >= (i*x_width - extra_edge_x))]
        test_x = TEST.loc[(TEST.x <= (i+1)*x_width) & (TEST.x >= i*x_width)]
    for j in range(y_range):
        tic=timeit.default_timer()
        if (j != y_range - 1):
            cond1_train = train_x.y < (j+1) * y_width + extra_edge_y
            cond1_test = test_x.y < (j+1) * y_width
        else:
            cond1_train = train_x.y <= (j+1) * y_width + extra_edge_y
            cond1_test = test_x.y <= (j+1) * y_width
        cond2_train = train_x.y >= j*y_width - extra_edge_y
        cond2_test = test_x.y >= j*y_width
        train_temp = train_x[cond1_train & cond2_train].copy()
        test_temp = test_x[cond1_test & cond2_test].copy()
        if(is_xgb_run):
            name = 'res_x_'+str(i)+'_y_'+str(j)+'.csv'
            prob_name = base_str+'prob_x_'+str(i)+'_y_'+str(j)+'.csv'
            result_xgb = run_xgb(train_temp.copy(),test_temp.copy(), base_str + name, is_sub_run = is_sub_run,is_xgb_run=is_xgb_run)
        else:
            if(compute_knn_probs):
                name = 'res_x_'+str(i)+'_y_'+str(j)+'.csv'
                prob_name = base_str+'prob_x_'+str(i)+'_y_'+str(j)+'.csv'
                prob_name_0 = base_str+'prob_x_'+str(i)+'_y_'+str(j)+'_0.csv'
                prob_name_1 = base_str+'prob_x_'+str(i)+'_y_'+str(j)+'_1.csv'
                prob_name_2 = base_str+'prob_x_'+str(i)+'_y_'+str(j)+'_2.csv'
                prob_name_3 = base_str+'prob_x_'+str(i)+'_y_'+str(j)+'_3.csv'
                result_xgb = run_xgb(train_temp.copy(),test_temp.copy(), base_str + name, is_sub_run = is_sub_run,is_xgb_run=is_xgb_run)
            else:
                name = 'prob_x_'+str(i)+'_y_'+str(j)+'.csv'
                result_xgb = run_xgb(train_temp.copy(),test_temp.copy(), base_str + name, is_sub_run = is_sub_run,is_xgb_run=is_xgb_run)
                name_0 = 'prob_x_'+str(i)+'_y_'+str(j)+'_0.csv'
                name_1 = 'prob_x_'+str(i)+'_y_'+str(j)+'_1.csv'
                name_2 = 'prob_x_'+str(i)+'_y_'+str(j)+'_2.csv'
                name_3 = 'prob_x_'+str(i)+'_y_'+str(j)+'_3.csv'

                if(is_sub_run):
                    time_min = 786242
                    time_max = 1006589
                else:
                    time_min = 600001
                    time_max = 786239
                time_diff = time_max - time_min

                test_0_cond = (test_temp.time >= (time_min + 0.0 * time_diff)) & (test_temp.time < (time_min + 0.25 * time_diff))
                test_1_cond = (test_temp.time >= (time_min + 0.25 * time_diff)) & (test_temp.time < (time_min + 0.5 * time_diff))
                test_2_cond = (test_temp.time >= (time_min + 0.5 * time_diff)) & (test_temp.time < (time_min + 0.75 * time_diff))
                test_3_cond = (test_temp.time >= (time_min + 0.75 * time_diff)) & (test_temp.time <= time_max)
                test_0 = test_temp[test_0_cond]
                test_1 = test_temp[test_1_cond]
                test_2 = test_temp[test_2_cond]
                test_3 = test_temp[test_3_cond]
                result_xgb_0 = run_xgb(train_temp.copy(),test_0.copy(), base_str + name_0, is_sub_run = is_sub_run,is_xgb_run=is_xgb_run)
                result_xgb_1 = run_xgb(train_temp.copy(),test_1.copy(), base_str + name_1, is_sub_run = is_sub_run,is_xgb_run=is_xgb_run)
                result_xgb_2 = run_xgb(train_temp.copy(),test_2.copy(), base_str + name_2, is_sub_run = is_sub_run,is_xgb_run=is_xgb_run)
                result_xgb_3 = run_xgb(train_temp.copy(),test_3.copy(), base_str + name_3, is_sub_run = is_sub_run,is_xgb_run=is_xgb_run)
toc=timeit.default_timer()
print('XGB time',toc - tic_xgb)
tic=timeit.default_timer()

#is_xgb_run = True
#is_xgb_run = False
#is_sub_run = True
#is_sub_run = False
if(is_xgb_run):
    if(is_sub_run):
        base_str = '/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/Test_Results/Run13/'
    else:
        base_str = '/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/Train_Results/Run13/'
if (not is_xgb_run):
    if(is_sub_run):
        base_str = '/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/Test_Results/Run23/'
    else:
        base_str = '/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/Train_Results/Run23/'

results = pd.DataFrame()
RES_LIST = []

if(is_xgb_run):
    x_range = 40
    y_range = 50
else:
    x_range = 20
    y_range = 25
for i in range(x_range):
    for j in range(y_range):
        if(is_xgb_run or compute_knn_probs):
            name = 'res_x_'+str(i)+'_y_'+str(j)+'.csv'
        else:
            name = 'prob_x_'+str(i)+'_y_'+str(j)+'.csv'
        try:
            temp_res = pd.read_csv(base_str+name, header=0)
        except OSError:
            continue
        RES_LIST.append(temp_res)

test =  pd.concat(RES_LIST)
if(is_xgb_run):
    test_xgb =  pd.concat(RES_LIST)
else:
    test_knn =  pd.concat(RES_LIST)
del RES_LIST
if(not is_sub_run and (is_xgb_run or compute_knn_probs)):
    results = test.copy()
    results['pred_0_res'] = results['pred_0'] - results['place_id']
    results['pred_1_res'] = results['pred_1'] - results['place_id']
    results['pred_2_res'] = results['pred_2'] - results['place_id']
    results['pred_0_res'] = (results['pred_0_res'] == 0).astype(int)
    results['pred_1_res'] = (results['pred_1_res'] == 0).astype(int) / 2
    results['pred_2_res'] = (results['pred_2_res'] == 0).astype(int) / 3
    results['apk'] = results['pred_0_res'] + results['pred_1_res'] + results['pred_2_res']
    print('apk',results['apk'].mean())
toc=timeit.default_timer()
print('Stitching time',toc - tic)
#%%
tic=timeit.default_timer()
test_knn.columns = [str(col) + '_x' for col in test_knn.columns]
test_knn.rename(columns = {'row_id_x':'row_id'},inplace=True)
test_combined_3 = pd.merge(test_xgb,test_knn,how='left',on='row_id')

#test_combined_3 = test_combined_3.loc[test_combined_2.row_id <= 100000]

def get_place_diffs(df,rank):
    pred_name = 'pred_'+str(rank)
    prob_name_new = 'prob_'+str(rank)+'_new'
    prob_name = 'prob_'+str(rank)

    df[prob_name] = df[prob_name]
    df[prob_name_new] = df[prob_name]
    for i in range(6):
        col_name = 'place_diff_'+str(rank)+'_'+str(i)
        pred_name_x = 'pred_'+str(i)+'_x'
        prob_name_x = 'prob_'+str(i)+'_x'
        df[prob_name_x] = df[prob_name_x]
        df[col_name] = df[pred_name] - df[pred_name_x]
        df[col_name] = (df[col_name] == 0).astype(int)
        df[prob_name_new] = df[prob_name_new] + (df[prob_name_x] * df[col_name] * 1.0)

    df[prob_name_new] = df[prob_name_new] / 2
    return df

test_combined_3 = get_place_diffs(test_combined_3,0)
test_combined_3 = get_place_diffs(test_combined_3,1)
test_combined_3 = get_place_diffs(test_combined_3,2)
test_combined_3 = get_place_diffs(test_combined_3,3)
test_combined_3 = get_place_diffs(test_combined_3,4)
test_combined_3 = get_place_diffs(test_combined_3,5)

test_combined_3['no_first_knn'] = (test_combined_3['place_diff_0_0'] +
        test_combined_3['place_diff_1_0'] +
        test_combined_3['place_diff_2_0'] +
        test_combined_3['place_diff_3_0'] +
        test_combined_3['place_diff_4_0'] +
        test_combined_3['place_diff_5_0'])
test_combined_3['no_second_knn'] = (test_combined_3['place_diff_0_1'] +
        test_combined_3['place_diff_1_1'] +
        test_combined_3['place_diff_2_1'] +
        test_combined_3['place_diff_3_1'] +
        test_combined_3['place_diff_4_1'] +
        test_combined_3['place_diff_5_1'])
no_1knn_cond = test_combined_3['no_first_knn'] == 0
no_2knn_cond = test_combined_3['no_second_knn'] == 0

test_combined_3['prob_6_new'] = 0
test_combined_3['prob_6_new'][no_1knn_cond] = test_combined_3['prob_0_x'][no_1knn_cond] / 3.0

test_combined_3['prob_7_new'] = 0
test_combined_3['prob_7_new'][no_2knn_cond] = test_combined_3['prob_1_x'][no_2knn_cond] / 3.0

test_combined_3['pred_6'] = test_combined_3['pred_0_x']
test_combined_3['pred_7'] = test_combined_3['pred_1_x']

def get_top_3_combined(row):
    prob_0 = row['prob_0_new']
    prob_1 = row['prob_1_new']
    prob_2 = row['prob_2_new']
    prob_3 = row['prob_3_new']
    prob_4 = row['prob_4_new']
    prob_5 = row['prob_5_new']
    prob_6 = row['prob_6_new']
    prob_7 = row['prob_7_new']
    inds = np.argsort([prob_0,prob_1,prob_2,prob_3,prob_4,prob_5,prob_6,prob_7])[::-1][:3]
    col_name_0 = 'pred_'+str(inds[0])
    col_name_1 = 'pred_'+str(inds[1])
    col_name_2 = 'pred_'+str(inds[2])
    val_0 = row[col_name_0]
    val_1 = row[col_name_1]
    val_2 = row[col_name_2]
    return [val_0,val_1,val_2]
test_combined_3['ans_list'] = test_combined_3.apply(lambda row: get_top_3_combined(row),axis=1)
test_combined_3['pred_0_new'] = test_combined_3['ans_list'].map(lambda x: x[0])
test_combined_3['pred_1_new'] = test_combined_3['ans_list'].map(lambda x: x[1])
test_combined_3['pred_2_new'] = test_combined_3['ans_list'].map(lambda x: x[2])
test_combined_3.drop('ans_list',axis=1,inplace=True)
if(not is_sub_run):
    results = test_combined_3.copy()
    results['pred_0_res'] = results['pred_0_new'] - results['place_id']
    results['pred_1_res'] = results['pred_1_new'] - results['place_id']
    results['pred_2_res'] = results['pred_2_new'] - results['place_id']
    results['pred_0_res'] = (results['pred_0_res'] == 0).astype(int)
    results['pred_1_res'] = (results['pred_1_res'] == 0).astype(int) / 2
    results['pred_2_res'] = (results['pred_2_res'] == 0).astype(int) / 3
    results['apk'] = results['pred_0_res'] + results['pred_1_res'] + results['pred_2_res']
    print('apk',results['apk'].mean())
toc=timeit.default_timer()
print('Ens time',toc - tic)

test = test_combined_3.copy()
test['pred_0'] = test['pred_0_new']
test['pred_1'] = test['pred_1_new']
test['pred_2'] = test['pred_2_new']
#%%
if(is_sub_run):
    tic=timeit.default_timer()
    test_all = test.copy()

    test_all = test_all[['row_id','pred_0','pred_1','pred_2']].copy()
    test_all['pred_0'] = test_all['pred_0'].map(REVERSE_PLACE_ID_DICT)
    test_all['pred_1'] = test_all['pred_1'].map(REVERSE_PLACE_ID_DICT)
    test_all['pred_2'] = test_all['pred_2'].map(REVERSE_PLACE_ID_DICT)

    test_all['pred_0'] = test_all['pred_0'].astype(str)
    test_all['pred_1'] = test_all['pred_1'].astype(str)
    test_all['pred_2'] = test_all['pred_2'].astype(str)

    test_all['pred_0'] = test_all['pred_0'].map(lambda x: x.split('.')[0])
    test_all['pred_1'] = test_all['pred_1'].map(lambda x: x.split('.')[0])
    test_all['pred_2'] = test_all['pred_2'].map(lambda x: x.split('.')[0])

    test_all['pred_0'] = test_all['pred_0'].astype(str)
    test_all['pred_1'] = test_all['pred_1'].astype(str)
    test_all['pred_2'] = test_all['pred_2'].astype(str)

    test_all['place_id'] = test_all['pred_0'] + ' ' + test_all['pred_1'] + ' ' + test_all['pred_2']

    test_sub = test_all[['row_id','place_id']].copy()
    test_sub.to_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook_Checkin/basic_fb_checkin.csv', index=False)
    print('submission created')
    toc=timeit.default_timer()
    print('Making Sub Time',toc - tic)

    del test_all
    del test_sub
toc=timeit.default_timer()
print('Total Time',toc - tic0)