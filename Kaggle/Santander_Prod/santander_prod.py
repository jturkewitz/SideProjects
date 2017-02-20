#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:13:13 2016

@author: Jared Turkewitz
"""
#%%
import pandas as pd
import numpy as np
import random
from itertools import compress

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import cross_validation
    import xgboost as xgb

import operator
import timeit

warnings.filterwarnings("ignore")

tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'
#%%
train_dtype_dict = {'ncodpers':np.int32,
'ind_empleado':np.int8,
'pais_residencia':np.int8,
'sexo':np.int8,
'age':np.int16,
'ind_nuevo':np.int8,
'antiguedad':np.int16,
'indrel':np.int8,
'indrel_1mes':np.int8,
'tiprel_1mes':np.int8,
'indresi':np.int8,
'indext':np.int8,
'conyuemp':np.int8,
'canal_entrada':np.int16,
'indfall':np.int8,
'tipodom':np.int8,
'cod_prov':np.int8,
'ind_actividad_cliente':np.int8,
#'renta':np.float64,
'renta':np.int64,
'segmento':np.int8,
'ind_ahor_fin_ult1':np.int8,
'ind_aval_fin_ult1':np.int8,
'ind_cco_fin_ult1':np.int8,
'ind_cder_fin_ult1':np.int8,
'ind_cno_fin_ult1':np.int8,
'ind_ctju_fin_ult1':np.int8,
'ind_ctma_fin_ult1':np.int8,
'ind_ctop_fin_ult1':np.int8,
'ind_ctpp_fin_ult1':np.int8,
'ind_deco_fin_ult1':np.int8,
'ind_deme_fin_ult1':np.int8,
'ind_dela_fin_ult1':np.int8,
'ind_ecue_fin_ult1':np.int8,
'ind_fond_fin_ult1':np.int8,
'ind_hip_fin_ult1':np.int8,
'ind_plan_fin_ult1':np.int8,
'ind_pres_fin_ult1':np.int8,
'ind_reca_fin_ult1':np.int8,
'ind_tjcr_fin_ult1':np.int8,
'ind_valo_fin_ult1':np.int8,
'ind_viv_fin_ult1':np.int8,
'ind_nomina_ult1':np.int8,
'ind_nom_pens_ult1':np.int8,
'ind_recibo_ult1':np.int8,
'fecha_dato_month':np.int8,
'fecha_dato_year':np.int8,
'month_int':np.int8,
'fecha_alta_month':np.int8,
'fecha_alta_year':np.int8,
'fecha_alta_day':np.int8,
'fecha_alta_month_int':np.int16,
'fecha_alta_day_int':np.int32,
'ult_fec_cli_1t_month':np.int8,
'ult_fec_cli_1t_year':np.int8,
'ult_fec_cli_1t_day':np.int8,
'ult_fec_cli_1t_month_int':np.int8}
#%%
test_dtype_dict = {'ncodpers':np.int32,'ind_empleado':np.int8,'pais_residencia':np.int8,
'sexo':np.int8,'age':np.int16,'ind_nuevo':np.int8,'antiguedad':np.int16,'indrel':np.int8,
'indrel_1mes':np.int8,'tiprel_1mes':np.int8,'indresi':np.int8,'indext':np.int8,'conyuemp':np.int8,
'canal_entrada':np.int16,'indfall':np.int8,'tipodom':np.int8,'cod_prov':np.int8,
'ind_actividad_cliente':np.int8,'renta':np.int64,'segmento':np.int8,'fecha_dato_month':np.int8,
'fecha_dato_year':np.int8,'month_int':np.int8,'fecha_alta_month':np.int8,'fecha_alta_year':np.int8,
'fecha_alta_day':np.int8,'fecha_alta_month_int':np.int16,'fecha_alta_day_int':np.int32,'ult_fec_cli_1t_month':np.int8,
'ult_fec_cli_1t_year':np.int8,'ult_fec_cli_1t_day':np.int8,'ult_fec_cli_1t_month_int':np.int8}
#%%
tic=timeit.default_timer()

train_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/train_hash.csv',
                          dtype = train_dtype_dict,header=0)
test_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/test_hash.csv',
                        dtype = test_dtype_dict, header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)

train_orig.rename(columns={'ncodpers':'id'},inplace=True)
test_orig.rename(columns={'ncodpers':'id'},inplace=True)

test_orig['renta'] = test_orig['renta'].map(lambda x: -1 if x == -2 else x)

target_cols_all = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
               'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
               'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

very_low_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cder_fin_ult1'] # << 0.005
low_cols = ['ind_viv_fin_ult1','ind_pres_fin_ult1','ind_deme_fin_ult1','ind_deco_fin_ult1'] # < 0.005
medium_low_cols = ['ind_hip_fin_ult1','ind_plan_fin_ult1'] # < 0.01 #'ind_ctju_fin_ult1'

target_cols = [col for col in target_cols if col not in very_low_cols]
target_cols = [col for col in target_cols if col not in low_cols]
#%%
for col in target_cols:
    test_orig[col] = 0
    test_orig[col] = test_orig[col].astype(np.int8)

for col in target_cols:
    train_orig[col].replace(-1,0,inplace=True)
combined_orig = pd.concat([train_orig,test_orig],axis=0)
#%%
train_small = train_orig[['id','month_int'] + target_cols].copy()
test_small = test_orig[['id','month_int'] + target_cols].copy()
combined_small = pd.concat([train_small,test_small],axis=0)
combined_small.fillna(0,inplace=True)

combined_small.sort_values(by = ['id','month_int'],inplace=True)
tic=timeit.default_timer()
DIFF_CONDS = {}
for shift_val in range(1,18):
    name = 'id_shift_' + str(shift_val)
    combined_small[name] = combined_small['id'].shift(shift_val).fillna(0).astype(np.int32)
    DIFF_CONDS[shift_val] = ((combined_small['id'] - combined_small[name]) != 0)
    combined_small.drop(name,axis = 1,inplace=True)

for col in target_cols:
    for shift_val in range(1,18):
        name = col + '_s_' + str(shift_val)
        combined_small[name] = combined_small[col].shift(shift_val).fillna(0).astype(np.int8)
        combined_small[name][DIFF_CONDS[shift_val]] = 0
toc=timeit.default_timer()
print('Shift Time',toc - tic)

for col in target_cols:
    combined_small[col] = (combined_small[col] - combined_small[col + '_s_1']).astype(np.int8)
    combined_small[col] = (combined_small[col] > 0).astype(np.int8)

MIN_MONTH_DICT = combined_small.groupby('id')['month_int'].min().to_dict()
combined_small['min_month_int'] = combined_small['id'].map(lambda x: MIN_MONTH_DICT[x])

combined_small = combined_small[combined_small['min_month_int'] != combined_small['month_int']]

combined_small['sum_inds'] = combined_small[target_cols].sum(axis=1)
combined_small = combined_small[(combined_small['sum_inds'] != 0) | (combined_small['month_int'] == 18)].copy()
#combined_small = combined_small[(combined_small['sum_inds'] != 0) | (combined_small['month_int'] >= 17)].copy()

combined_small.to_csv('combined_shifted_small.csv', index=False)
#combined_small.to_csv('combined_shifted_small_larger.csv', index=False)
#%%
cols_to_combine = ['age', 'antiguedad', 'canal_entrada', 'cod_prov', 'conyuemp',
       'fecha_alta_day', 'fecha_alta_month', 'fecha_alta_month_int','fecha_alta_day_int',
       'fecha_alta_year', 'fecha_dato_month', 'fecha_dato_year',
       'ind_actividad_cliente',
       'ind_empleado',
       'ind_nuevo',
       'indext',
       'indfall', 'indrel', 'indrel_1mes', 'indresi',
       'pais_residencia', 'renta', 'segmento', 'sexo',
       'tiprel_1mes', 'ult_fec_cli_1t_day', 'ult_fec_cli_1t_month',
       'ult_fec_cli_1t_month_int', 'ult_fec_cli_1t_year']


cols_to_combine += very_low_cols
cols_to_combine += low_cols

combined_orig.sort_values(by = ['id','month_int'],inplace=True)
combined_orig.fillna(0,inplace=True)
#%%
tic=timeit.default_timer()
DIFF_CONDS = {}
for shift_val in [1]:
    name = 'id_shift_' + str(shift_val)
    combined_orig[name] = combined_orig['id'].shift(shift_val).fillna(0).astype(np.int32)
    DIFF_CONDS[shift_val] = ((combined_orig['id'] - combined_orig[name]) != 0)
    combined_orig.drop(name,axis = 1,inplace=True)
shifted_feature_names = []
for col in cols_to_combine + target_cols:
    for shift_val in [1]:
        name = col + '_s_' + str(shift_val)
        combined_orig[name] = combined_orig[col].shift(shift_val).fillna(0).astype(train_dtype_dict[col])
        combined_orig[name][DIFF_CONDS[shift_val]] = 0
        if col in cols_to_combine:
            shifted_feature_names.append(name)

toc=timeit.default_timer()
print('Shift Time',toc - tic)

cols_to_combine = [x for x in cols_to_combine if x not in very_low_cols]
cols_to_combine = [x for x in cols_to_combine if x not in low_cols]
#%%
diff_feautres_s1 = []
for col in cols_to_combine:
    name = col + '_s1_diff'
    diff_feautres_s1.append(name)
    combined_orig[name] = (combined_orig[col] - combined_orig[col + '_s_1']).astype(train_dtype_dict[col])

MIN_MONTH_DICT = combined_orig.groupby('id')['month_int'].min().to_dict()
combined_orig['min_month_int'] = combined_orig['id'].map(lambda x: MIN_MONTH_DICT[x]).astype(np.int8)

MIN_ANTIGUEDAD_DICT = combined_orig.groupby('id')['antiguedad'].min().to_dict()
combined_orig['min_antiguedad'] = combined_orig['id'].map(lambda x: MIN_ANTIGUEDAD_DICT[x]).astype(np.int16)

MAX_ANTIGUEDAD_DICT = combined_orig.groupby('id')['antiguedad'].max().to_dict()
combined_orig['max_antiguedad'] = combined_orig['id'].map(lambda x: MAX_ANTIGUEDAD_DICT[x]).astype(np.int16)

MIN_AGE_DICT = combined_orig.groupby('id')['age'].min().to_dict()
combined_orig['min_age'] = combined_orig['id'].map(lambda x: MIN_AGE_DICT[x]).astype(np.int16)

MAX_AGE_DICT = combined_orig.groupby('id')['age'].max().to_dict()
combined_orig['max_age'] = combined_orig['id'].map(lambda x: MAX_AGE_DICT[x]).astype(np.int16)

STD_RENTA_DICT = combined_orig.groupby('id')['renta'].std().to_dict()
combined_orig['std_renta'] = combined_orig['id'].map(lambda x: STD_RENTA_DICT[x])
MIN_RENTA_DICT = combined_orig.groupby('id')['renta'].min().to_dict()
combined_orig['min_renta'] = combined_orig['id'].map(lambda x: MIN_RENTA_DICT[x])
MAX_RENTA_DICT = combined_orig.groupby('id')['renta'].max().to_dict()
combined_orig['max_renta'] = combined_orig['id'].map(lambda x: MAX_RENTA_DICT[x])
#%%
RENTA_VAL_COUNTS = combined_orig.groupby('renta')['id'].nunique().to_dict()
combined_orig['renta_freq'] = combined_orig['renta'].map(lambda x: RENTA_VAL_COUNTS[x])
#%%
combined_orig.sort_values(by = ['id','month_int'],inplace=True)
combined_nd = combined_orig.drop_duplicates('id')

#%%
for col in target_cols:
    combined_orig[col] = (combined_orig[col] - combined_orig[col + '_s_1']).astype(np.int8)
    combined_orig[col] = (combined_orig[col] > 0).astype(np.int8)

#%%
def apply_dict(dict_to_apply,x):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
target_col_sum_all_others = []

def get_truth_values(input_list,truth):
    return [ int(x == truth) for x in input_list ]
#%%
def apk(actual, predicted, k=7):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

#%%
def get_top7_preds_string(row):
    row.sort(inplace=True)
    return row.index[-7:][::-1].tolist()
#%%
combined_dtype_dict = {}
combined_dtype_dict['id'] = np.int32
combined_dtype_dict['month_int'] = np.int8
combined_dtype_dict['sum_inds'] = np.int8
combined_dtype_dict['min_month_int'] = np.int8
shift_cols = []
for col in target_cols:
    combined_dtype_dict[col] = np.int8
    for shift_val in range(1,17):
        combined_dtype_dict[col + '_s_' + str(shift_val)] = np.int8
        shift_cols += [col + '_s_' + str(shift_val)]

del train_orig
#%%
tic=timeit.default_timer()

combined = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/combined_shifted_small_condensed_dec14.csv',
                          dtype = combined_dtype_dict,header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)

#%%
LEN_ID_DICT = combined_orig.groupby('id')['month_int'].count().to_dict()
combined_orig['id_counts'] = combined_orig['id'].map(lambda x: LEN_ID_DICT[x])

MAX_MONTH_DICT = combined_orig.groupby('id')['month_int'].max().to_dict()
combined_orig['max_month_int'] = combined_orig['id'].map(lambda x: MAX_MONTH_DICT[x]).astype(np.int8)

combined_orig['min_max_month_int'] = (combined_orig['max_month_int'] - combined_orig['min_month_int']) + 1
combined_orig['skipped_months'] = combined_orig['min_max_month_int'] -  combined_orig['id_counts']

combined['months_active'] = combined['month_int'] - combined['min_month_int']
combined['months_active_ratio'] = (combined['month_int'] - combined['min_month_int']) / (combined['month_int'] - 1)
combined['months_active_this_year'] = (combined['month_int'] % 12) - (combined['min_month_int'] % 12)
combined['months_active_this_year_ratio'] = (combined['month_int'] % 12) - (combined['min_month_int'] % 12) / ((combined['month_int'] % 12) + 1)

combined = combined[combined['min_month_int'] != combined['month_int']]

UNIQ_SET = set(test_orig['id'].unique())
combined['is_test_id'] = combined['id'].map(lambda x: 1 if x in UNIQ_SET else 0)
combined = combined[(combined['is_test_id'] == 1)]

combined = pd.merge(combined,combined_orig[(['id','month_int','min_antiguedad','max_antiguedad','min_age','max_age',
                                             'std_renta','min_renta','max_renta','max_month_int',
                                             'skipped_months','id_counts','min_max_month_int','renta_freq'] +
                                            cols_to_combine + diff_feautres_s1 +
                                            shifted_feature_names +
                                            target_col_sum_all_others)],on=['id','month_int'])
combined['min_antiguedad_plus_months_active'] = combined['months_active'] + combined['min_antiguedad']
combined['antiguedad_no_change'] = (combined['min_antiguedad'] == combined['max_antiguedad']).astype(np.int8)
combined['antiguedad_minus_month_int'] = combined['antiguedad'] - combined['month_int']
combined['min_month_int_binned'] = (combined['min_month_int'] > 1).astype(np.int8)
combined['month_int_minus_fecha_alta_month_int'] = combined['month_int'] - combined['fecha_alta_month_int']
combined['fecha_dato_month_minus_fecha_alta_month'] = combined['fecha_alta_month'] - combined['fecha_dato_month']
combined['fecha_dato_year_minus_fecha_alta_year'] = combined['fecha_alta_year'] - combined['fecha_dato_year']

combined['fecha_dato_month_minus_fecha_alta_month'] = combined['fecha_dato_month_minus_fecha_alta_month'].map(lambda x: -1 if x == 11 else x)

combined['fecha_dato_month_s1_diff'] = combined['fecha_dato_month_s1_diff'].map(lambda x: 1 if x == -11 else x)

first_col_names = []
for col in target_cols:
    name = col + '_first'
    first_col_names.append(name)
    FIRST_DICT = dict(zip(combined_nd['id'],combined_nd[col]))
    combined[name] = combined['id'].map(lambda x: apply_dict(FIRST_DICT,x))

mean_shifted_names = []
mean_shifted_small_names = []
shift_small_cols = []

for col in target_cols:
    shift_names_temp = []
    shift_small_temp = []
    for shift_val in range(1,6):
        shift_small_cols += [col + '_s_' + str(shift_val)]
        shift_small_temp += [col + '_s_' + str(shift_val)]
    for shift_val in range(1,18):
        shift_names_temp.append(col + '_s_' + str(shift_val))
    name = 'mean_shifted_' + col
    mean_shifted_names.append(name)
    combined[name] = combined[shift_names_temp].sum(axis = 1) / (combined['month_int'] - combined['min_month_int'])

    name = 'mean_shifted_small_' + col
    mean_shifted_small_names.append(name)
    combined[name] = combined[shift_small_temp].sum(axis = 1) / 5
    close_cond = (combined['month_int'] - combined['min_month_int']) < 5
    combined[name][close_cond] = combined[name][close_cond] * 5 / (combined['month_int'] - combined['min_month_int'])

combined['nomina_diff'] = combined['ind_nom_pens_ult1'] - combined['ind_nomina_ult1']
combined['nomina_diff_pens_1_mi6'] = (combined['nomina_diff'] == 1) & (combined['month_int'] == 6)

del combined_orig
#%%
shift_1_cols = []
shift_1to4_cols = []
shift_1to12_cols = []
for col in target_cols:
    shift_1_cols += col + '_s_1'
    for shift_val in range(1,5):
        shift_1to4_cols += [col + '_s_' + str(shift_val)]
    for shift_val in range(1,13):
        shift_1to12_cols += [col + '_s_' + str(shift_val)]

#%%
tic=timeit.default_timer()
is_sub_run = True
if (is_sub_run):
    train = combined[(combined['month_int'] != 18)].copy()
    test = combined[(combined['month_int'] == 18) ].copy()
else:
    train = combined[(combined['month_int'] < 17)].copy()
    test = combined[(combined['month_int'] == 17) ].copy()

train['truth_list'] = train[target_cols].apply(lambda x: list(compress(target_cols, x.values)), axis=1)

random.seed(5)
np.random.seed(5)

train['target'] = train['truth_list'].map(lambda x: np.random.choice(x))

def convert_strings_to_ints(input_df,col_name,output_col_name,do_sort=True):
    labels, levels = pd.factorize(input_df[col_name],sort = do_sort)
    input_df[output_col_name] = labels
    input_df[output_col_name] = input_df[output_col_name].astype(int)
    output_dict = dict(zip(input_df[col_name],input_df[output_col_name]))
    return (output_dict,input_df)

(target_dict,train) = convert_strings_to_ints(train,'target','target_hash')

np.random.seed(5)
train['dela_choice'] = (train['target'] == 'ind_dela_fin_ult1') & (train['month_int'] <= 13) & (np.random.uniform(size=len(train)) >= 0.8)
train = train[~train['dela_choice']]

toc=timeit.default_timer()
print('duplicating Time',toc - tic)

train_ma1 = train[(train['months_active'] <= 1) & (train['min_month_int'] >= 2)].copy()
test_ma1 = test[test['months_active'] <= 1].copy()
train_samp_3g = train[(train['month_int'] >= 3)].copy()
train_samp_2g = train[(train['month_int'] >= 2)].copy()
train_samp_6 = train[(train['month_int'] == 6)].copy()
train_samp_12 = train[(train['month_int'] == 12)].copy()
if is_sub_run:
    train_samp = train[(train['month_int'] == 6)].copy()
    train_samp_15g = train[(train['month_int'] >= 15)].copy()
else:
    train_samp_15g = train[(train['month_int'] >= 15)].copy()
    train_samp = train[(train['month_int'] == 5)].copy()
test_samp = test[test['id'] < 50000].copy()
#%%
pred_cols = []
for col in target_cols:
    name = 'pred_' + col
    pred_cols.append('pred_' + col)
def get_results(df):
    df['added_products'] = df[pred_cols].apply(lambda row: get_top7_preds_string(row), axis=1)
    df['added_products'] = df['added_products'].map(lambda x: [x[5:] for x in x])
    if not is_sub_run:
        df['truth_list'] = df[target_cols].apply(lambda x: list(compress(target_cols, x.values)), axis=1)
        df['apk'] = df.apply(lambda x: apk(x['truth_list'],x['added_products']),axis=1)
        print(df['apk'].mean())

    if is_sub_run:
        df['added_products'] = df['added_products'].map(lambda x: ' '.join(x))
def probe_results(df,prod):
    df['added_products'] = 1
    df['added_products'] = df['added_products'].map(lambda x: [prod])
    df['truth_list'] = df[target_cols].apply(lambda x: list(compress(target_cols, x.values)), axis=1)
    df['apk'] = df.apply(lambda x: apk(x['truth_list'],x['added_products']),axis=1)
    return df['apk'].mean()
#%%
lb_probing_dict = {
'ind_tjcr_fin_ult1':0.0041178,
'ind_ahor_fin_ult1':0,
'ind_aval_fin_ult1':0,
'ind_cder_fin_ult1':0,
'ind_ecue_fin_ult1':0.0019961,
'ind_fond_fin_ult1':0.000104,
'ind_ctpp_fin_ult1':0.0001142,
'ind_ctju_fin_ult1':0.0000502,
'ind_hip_fin_ult1':0.0000161,
'ind_plan_fin_ult1':0.0000126,
'ind_viv_fin_ult1':0,
'ind_pres_fin_ult1':0.0000054,
'ind_deme_fin_ult1':0,
'ind_deco_fin_ult1':0,
'ind_reca_fin_ult1':0.0032092,
'ind_cco_fin_ult1':0.0096681,
'ind_cno_fin_ult1':0.0017839,
'ind_nom_pens_ult1':0.0021801,
'ind_nomina_ult1':0.0021478,
'ind_recibo_ult1':0.0086845,
'ind_valo_fin_ult1':0.000278,
'ind_dela_fin_ult1':0.0000933,
'ind_ctma_fin_ult1':0.0004488,
'ind_ctop_fin_ult1':0.0001949}

sum_probing_lb = 0
for item,val in lb_probing_dict.items():
    sum_probing_lb += val
print('lb',sum_probing_lb)
cv_probing_dict = {'ind_cco_fin_ult1': 0.0038806574244755294,
 'ind_cno_fin_ult1': 0.0016603628953902965,
 'ind_ctju_fin_ult1': 4.1333271780755445e-05,
 'ind_ctma_fin_ult1': 0.0004969297073139132,
 'ind_ctop_fin_ult1': 0.0002224123672012079,
 'ind_ctpp_fin_ult1': 0.00013384107052816048,
 'ind_dela_fin_ult1': 4.276472708052187e-05,
 'ind_ecue_fin_ult1': 0.002723683678439314,
 'ind_fond_fin_ult1': 5.690034816571529e-05,
 'ind_hip_fin_ult1': 3.22077442447445e-06,
 'ind_nom_pens_ult1': 0.00273849924079194,
 'ind_nomina_ult1': 0.002745298653465832,
 'ind_plan_fin_ult1': 1.9897228666753267e-05,
 'ind_reca_fin_ult1': 0.0002822114123489502,
 'ind_recibo_ult1': 0.010384957695127938,
 'ind_tjcr_fin_ult1': 0.0043231023644420856,
 'ind_valo_fin_ult1': 0.0001831725987963608}

cv_month6_dict = {'ind_cco_fin_ult1': 0.009808023919887356,
 'ind_cno_fin_ult1': 0.001905654606529451,
 'ind_ctju_fin_ult1': 1.2656025058929617e-05,
 'ind_ctma_fin_ult1': 0.00028752906930755727,
 'ind_ctop_fin_ult1': 0.0003079632764339541,
 'ind_ctpp_fin_ult1': 0.0002159434275679866,
 'ind_dela_fin_ult1': 0.0013423823912504683,
 'ind_ecue_fin_ult1': 0.0016051267448176218,
 'ind_fond_fin_ult1': 0.00033222065779690244,
 'ind_hip_fin_ult1': 4.007741268661046e-06,
 'ind_nom_pens_ult1': 0.008105841282898912,
 'ind_nomina_ult1': 0.003608733711959415,
 'ind_plan_fin_ult1': 2.6630386061497733e-05,
 'ind_reca_fin_ult1': 0.004102608723165276,
 'ind_recibo_ult1': 0.013143677524481456,
 'ind_tjcr_fin_ult1': 0.0067138103600111925,
 'ind_valo_fin_ult1': 0.0002197666018045382}

cv_month12_dict = {'ind_cco_fin_ult1': 0.009383501037805038,
 'ind_cno_fin_ult1': 0.002941598932480755,
 'ind_ctju_fin_ult1': 7.382870935354925e-05,
 'ind_ctma_fin_ult1': 0.0005398815743643329,
 'ind_ctop_fin_ult1': 0.00028124352399780264,
 'ind_ctpp_fin_ult1': 0.00015222602695917452,
 'ind_dela_fin_ult1': 0.001171738370059462,
 'ind_ecue_fin_ult1': 0.00172567298340718,
 'ind_fond_fin_ult1': 0.0001916988022571118,
 'ind_hip_fin_ult1': 2.74116495124564e-06,
 'ind_nom_pens_ult1': 0.0024797491870618502,
 'ind_nomina_ult1': 0.002447220696307065,
 'ind_plan_fin_ult1': 0.00015257324118633233,
 'ind_reca_fin_ult1': 0.00024140526003969937,
 'ind_recibo_ult1': 0.010134342666817247,
 'ind_tjcr_fin_ult1': 0.004416144657487792,
 'ind_valo_fin_ult1': 0.00024473120684721084}

cv_month5_dict = {'ind_cco_fin_ult1': 0.003423360555649619,
 'ind_cno_fin_ult1': 0.0021088987594619977,
 'ind_ctju_fin_ult1': 1.58238614336102e-05,
 'ind_ctma_fin_ult1': 0.0002846976402930368,
 'ind_ctop_fin_ult1': 0.0004308046275300377,
 'ind_ctpp_fin_ult1': 0.0002560828242005917,
 'ind_dela_fin_ult1': 0.0012721065937503133,
 'ind_ecue_fin_ult1': 0.00202933110955334,
 'ind_fond_fin_ult1': 0.0005576592500228126,
 'ind_hip_fin_ult1': 1.2395358122994656e-05,
 'ind_nom_pens_ult1': 0.0023884536447890906,
 'ind_nomina_ult1': 0.002436980153185493,
 'ind_plan_fin_ult1': 2.8614816092445108e-05,
 'ind_reca_fin_ult1': 0.0005070228934352602,
 'ind_recibo_ult1': 0.010516643801186878,
 'ind_tjcr_fin_ult1': 0.00619905046282158,
 'ind_valo_fin_ult1': 0.0002985435190474458}

sum_probing_cv = 0
for item,val in cv_probing_dict.items():
    sum_probing_cv += val
print('cv',sum_probing_cv)

def norm_rows(df):
    with np.errstate(invalid='ignore'):
        return df.div(df.sum(axis=1), axis=0).fillna(0)


res_xgb_1_cv = {'ind_cco_fin_ult1': 0.1575418301318839,
 'ind_cno_fin_ult1': 0.059691923422850734,
 'ind_ctju_fin_ult1': 0.000589820887330269,
 'ind_ctma_fin_ult1': 0.009044724284009337,
 'ind_ctop_fin_ult1': 0.0067197584234381415,
 'ind_ctpp_fin_ult1': 0.003634448867890694,
 'ind_dela_fin_ult1': 0.021045530399504058,
 'ind_ecue_fin_ult1': 0.05180295442502181,
 'ind_fond_fin_ult1': 0.006811399182983569,
 'ind_hip_fin_ult1': 7.291344553694143e-05,
 'ind_nom_pens_ult1': 0.08167439250734301,
 'ind_nomina_ult1': 0.08173363484939439,
 'ind_plan_fin_ult1': 0.0005478325047780407,
 'ind_reca_fin_ult1': 0.053179919230947614,
 'ind_recibo_ult1': 0.32730510776744537,
 'ind_tjcr_fin_ult1': 0.13426388077942283,
 'ind_valo_fin_ult1': 0.0043399288902195325}

res_xgb_2_cv = {'ind_cco_fin_ult1': 0.14791336394397508,
 'ind_cno_fin_ult1': 0.062388839876009454,
 'ind_ctju_fin_ult1': 0.0005713689621044011,
 'ind_ctma_fin_ult1': 0.005881650007232238,
 'ind_ctop_fin_ult1': 0.005467240540285032,
 'ind_ctpp_fin_ult1': 0.0030839151907816735,
 'ind_dela_fin_ult1': 0.023086006926253046,
 'ind_ecue_fin_ult1': 0.05149678730202095,
 'ind_fond_fin_ult1': 0.00710734949142769,
 'ind_hip_fin_ult1': 0.00014349581789407923,
 'ind_nom_pens_ult1': 0.10511176129770099,
 'ind_nomina_ult1': 0.10337854944013818,
 'ind_plan_fin_ult1': 0.0006719997986786143,
 'ind_reca_fin_ult1': 0.05600035815260505,
 'ind_recibo_ult1': 0.2967676189370012,
 'ind_tjcr_fin_ult1': 0.12683876942231523,
 'ind_valo_fin_ult1': 0.004090924893577121}

res_xgb_2b_cv = {'ind_cco_fin_ult1': 0.15274627596058377,
 'ind_cno_fin_ult1': 0.05949766414646308,
 'ind_ctju_fin_ult1': 0.0006536546099392526,
 'ind_ctma_fin_ult1': 0.006978504572990394,
 'ind_ctop_fin_ult1': 0.005601268165904341,
 'ind_ctpp_fin_ult1': 0.0031800915325912514,
 'ind_dela_fin_ult1': 0.021826865610150407,
 'ind_ecue_fin_ult1': 0.048307869713348564,
 'ind_fond_fin_ult1': 0.006916636365323528,
 'ind_hip_fin_ult1': 0.0001200352446390862,
 'ind_nom_pens_ult1': 0.10367278591315975,
 'ind_nomina_ult1': 0.10224244495419404,
 'ind_plan_fin_ult1': 0.0005912342698393012,
 'ind_reca_fin_ult1': 0.054503533733348576,
 'ind_recibo_ult1': 0.3010891001156668,
 'ind_tjcr_fin_ult1': 0.12797114275806803,
 'ind_valo_fin_ult1': 0.004100892333790926}

res_xgb_4_cv = {'ind_cco_fin_ult1': 0.12316667894028037,
 'ind_cno_fin_ult1': 0.06062860712146262,
 'ind_ctju_fin_ult1': 0.0012766024355967151,
 'ind_ctma_fin_ult1': 0.014980273219976557,
 'ind_ctop_fin_ult1': 0.006715529380793333,
 'ind_ctpp_fin_ult1': 0.004356995184287496,
 'ind_dela_fin_ult1': 0.003424327668748079,
 'ind_ecue_fin_ult1': 0.08294908078813817,
 'ind_fond_fin_ult1': 0.00283156378477682,
 'ind_hip_fin_ult1': 0.00014795258299037754,
 'ind_nom_pens_ult1': 0.10617079458633588,
 'ind_nomina_ult1': 0.10564770249894162,
 'ind_plan_fin_ult1': 0.0007310280349206093,
 'ind_reca_fin_ult1': 0.020754374471015788,
 'ind_recibo_ult1': 0.32190597450829594,
 'ind_tjcr_fin_ult1': 0.13992383695321964,
 'ind_valo_fin_ult1': 0.004388677840222683}
res_xgb_5_cv = {'ind_cco_fin_ult1': 0.13466939585566146,
 'ind_cno_fin_ult1': 0.06247751099336744,
 'ind_ctju_fin_ult1': 0.001376417207162024,
 'ind_ctma_fin_ult1': 0.013095743318009332,
 'ind_ctop_fin_ult1': 0.006680175530347622,
 'ind_ctpp_fin_ult1': 0.004510092184502863,
 'ind_dela_fin_ult1': 0.005616006386946235,
 'ind_ecue_fin_ult1': 0.07097528818565485,
 'ind_fond_fin_ult1': 0.0030345690466101016,
 'ind_hip_fin_ult1': 0.00013245542094087197,
 'ind_nom_pens_ult1': 0.10733950425787181,
 'ind_nomina_ult1': 0.1072469716074887,
 'ind_plan_fin_ult1': 0.0007987080450540371,
 'ind_reca_fin_ult1': 0.0206349503182134,
 'ind_recibo_ult1': 0.3183683709896341,
 'ind_tjcr_fin_ult1': 0.1381908598087636,
 'ind_valo_fin_ult1': 0.004852980843771987}
res_xgb_5b_cv = {'ind_cco_fin_ult1': 0.1431940414371315,
 'ind_cno_fin_ult1': 0.05365678024797945,
 'ind_ctju_fin_ult1': 0.0014384551513186336,
 'ind_ctma_fin_ult1': 0.0151407240988091,
 'ind_ctop_fin_ult1': 0.007326709050581161,
 'ind_ctpp_fin_ult1': 0.0050698283665101445,
 'ind_dela_fin_ult1': 0.00612599633614946,
 'ind_ecue_fin_ult1': 0.07774997578292998,
 'ind_fond_fin_ult1': 0.0023921200886889518,
 'ind_hip_fin_ult1': 8.564325094127323e-05,
 'ind_nom_pens_ult1': 0.08638591831683694,
 'ind_nomina_ult1': 0.08632608014993802,
 'ind_plan_fin_ult1': 0.0007088418262189719,
 'ind_reca_fin_ult1': 0.021991107113503168,
 'ind_recibo_ult1': 0.34208483494991604,
 'ind_tjcr_fin_ult1': 0.14469633356003,
 'ind_valo_fin_ult1': 0.005626610272510002}
res_xgb_6_cv = {'ind_cco_fin_ult1': 0.14749211753434416,
 'ind_cno_fin_ult1': 0.08605644337718346,
 'ind_ctju_fin_ult1': 0.0012738778936674173,
 'ind_ctma_fin_ult1': 0.011608996229137787,
 'ind_ctop_fin_ult1': 0.006183543233969116,
 'ind_ctpp_fin_ult1': 0.00351560555419552,
 'ind_dela_fin_ult1': 0.021678845546758602,
 'ind_ecue_fin_ult1': 0.05970668742277326,
 'ind_fond_fin_ult1': 0.004807264713425918,
 'ind_hip_fin_ult1': 8.743741394584486e-05,
 'ind_nom_pens_ult1': 0.10643820566247844,
 'ind_nomina_ult1': 0.10751347232223737,
 'ind_plan_fin_ult1': 0.004068462342959938,
 'ind_reca_fin_ult1': 0.006032197982055503,
 'ind_recibo_ult1': 0.298532791747581,
 'ind_tjcr_fin_ult1': 0.12841629385871337,
 'ind_valo_fin_ult1': 0.006587757164572376}
res_xgb_6b_cv = {'ind_cco_fin_ult1': 0.16306896725407086,
 'ind_cno_fin_ult1': 0.07655161593466307,
 'ind_ctju_fin_ult1': 0.0014102908747953378,
 'ind_ctma_fin_ult1': 0.013007464341427194,
 'ind_ctop_fin_ult1': 0.007011403449548706,
 'ind_ctpp_fin_ult1': 0.0037711570315326548,
 'ind_dela_fin_ult1': 0.02104068165283255,
 'ind_ecue_fin_ult1': 0.0674165924868359,
 'ind_fond_fin_ult1': 0.004243657013196282,
 'ind_hip_fin_ult1': 8.666631727620557e-05,
 'ind_nom_pens_ult1': 0.08014803800621556,
 'ind_nomina_ult1': 0.08461430348308324,
 'ind_plan_fin_ult1': 0.0033663064684911464,
 'ind_reca_fin_ult1': 0.006215036630505901,
 'ind_recibo_ult1': 0.3268279323463584,
 'ind_tjcr_fin_ult1': 0.13515370033522023,
 'ind_valo_fin_ult1': 0.006066186373944297}

def get_pred_dict(df):
    mean_pred_dict = {}
    for col in target_cols:
        name = 'pred_' + col
        mean_pred_dict[col] = df[name].mean()
    return mean_pred_dict
    
def update_probs(df,update_factor = 1.0,a1=0,a2=1,a2b=0,a4=0,a5=0,a5b=0,a6=0,a6b=0,print_odds=False):
    for col in target_cols:
        name = 'pred_' + col
        if is_sub_run:
            probed_apk = lb_probing_dict[col]
            mean_pred = df[name].mean() * sum_probing_lb
            mean_pred = ((a1 * res_xgb_1_cv[col] + a2 * res_xgb_2_cv[col] + a2b * res_xgb_2b_cv[col]
                          + a4 * res_xgb_4_cv[col] + a5 * res_xgb_5_cv[col] + a5b * res_xgb_5b_cv[col] + a6 * res_xgb_6_cv[col]
                          + a6b * res_xgb_6b_cv[col])
                         / (a1 + a2 + a2b + a4 + a5 + a5b + a6 + a6b)
                         * sum_probing_lb)
        else:
            probed_apk = cv_probing_dict[col]
            mean_pred = df[name].mean() * sum_probing_cv
        if print_odds:
            print(col,'adj_factor: ',(probed_apk / mean_pred) ** update_factor)
        df['odds_temp'] = (df[name] / (1 - df[name])
                     * ((probed_apk / mean_pred) ** update_factor))
        df[name] = df['odds_temp'] / (1 + df['odds_temp'])
    return df
def get_top_7_pred_cols(df):
    if is_sub_run:
        df['pred_0'] = df['added_products'].map(lambda x: x.split()[0])
        df['pred_1'] = df['added_products'].map(lambda x: x.split()[1])
        df['pred_2'] = df['added_products'].map(lambda x: x.split()[2])
        df['pred_3'] = df['added_products'].map(lambda x: x.split()[3])
        df['pred_4'] = df['added_products'].map(lambda x: x.split()[4])
        df['pred_5'] = df['added_products'].map(lambda x: x.split()[5])
        df['pred_6'] = df['added_products'].map(lambda x: x.split()[6])
    else:
        df['pred_0'] = df['added_products'].map(lambda x: x[0])
        df['pred_1'] = df['added_products'].map(lambda x: x[1])
        df['pred_2'] = df['added_products'].map(lambda x: x[2])
        df['pred_3'] = df['added_products'].map(lambda x: x[3])
        df['pred_4'] = df['added_products'].map(lambda x: x[4])
        df['pred_5'] = df['added_products'].map(lambda x: x[5])
        df['pred_6'] = df['added_products'].map(lambda x: x[6])
#%%
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
#%%
es_rounds_dict = {}
def fit_xgb_model(train, test, params, xgb_features, target_col, num_rounds = 10, num_rounds_es = 200000,
                  use_early_stopping = True, print_feature_imp = False,
                  random_seed = 123, use_weights = False, use_multi_output = False,
                  use_weights_late = False, weights_late_factor = 1.0,
                  ):
    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    if not use_multi_output:
        s1_name = target_col + '_s_1'
        train = train[train[s1_name] != 1]

    X_train, X_watch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
    train_data = X_train[xgb_features].values
    train_data_full = train[xgb_features].values
    watch_data = X_watch[xgb_features].values
    train_prods = X_train[target_col].astype(np.int8).values
    train_prods_full = train[target_col].astype(np.int8).values
    watch_prods = X_watch[target_col].astype(np.int8).values
    test_data = test[xgb_features].values

    if use_weights:
        weights_tr_full = (train['month_int'] == 6).astype(int) * 11 + 1
        weights_tr_full = weights_tr_full / weights_tr_full.mean()
        weights_tr = (X_train['month_int'] == 6).astype(int) * 11 + 1
        weights_tr = weights_tr / weights_tr.mean()
        dtrain_full = xgb.DMatrix(train_data_full, train_prods_full, weight = weights_tr_full)
        dtrain = xgb.DMatrix(train_data, train_prods,weight = weights_tr)
        dwatch = xgb.DMatrix(watch_data, watch_prods)
    elif use_weights_late:
        weights_tr_full = (train['month_int'] >= 12).astype(int) * weights_late_factor + 1
        weights_tr_full = weights_tr_full / weights_tr_full.mean()
        weights_tr = (X_train['month_int'] >= 12).astype(int) * weights_late_factor + 1
        weights_tr = weights_tr / weights_tr.mean()
        dtrain_full = xgb.DMatrix(train_data_full, train_prods_full, weight = weights_tr_full)
        dtrain = xgb.DMatrix(train_data, train_prods,weight = weights_tr)
        dwatch = xgb.DMatrix(watch_data, watch_prods)
    else:
        dtrain = xgb.DMatrix(train_data, train_prods)
        dtrain_full = xgb.DMatrix(train_data_full, train_prods_full)
        dwatch = xgb.DMatrix(watch_data, watch_prods)
    dtest = xgb.DMatrix(test_data)
    watchlist = [(dtrain, 'train'),(dwatch, 'watch')]

    if use_early_stopping:
        xgb_classifier = xgb.train(params, dtrain, num_boost_round=num_rounds_es, evals=watchlist,
                            early_stopping_rounds=20, verbose_eval=50)
        y_pred = xgb_classifier.predict(dtest,ntree_limit=xgb_classifier.best_iteration)
        es_rounds_dict[target_col] = xgb_classifier.best_iteration
    else:
        xgb_classifier = xgb.train(params, dtrain_full, num_boost_round=num_rounds, evals=[(dtrain_full,'train')],
                            verbose_eval=50)
        y_pred = xgb_classifier.predict(dtest)
    if(print_feature_imp):
        create_feature_map(xgb_features)
        imp_dict = xgb_classifier.get_fscore(fmap='xgb.fmap')
        imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1),reverse=True)
        print('{0:<20} {1:>5}'.format('Feature','Imp'))
        print("--------------------------------------")
        num_to_print = 40
        num_printed = 0
        for i in imp_dict:
            num_printed = num_printed + 1
            if (num_printed > num_to_print):
                continue
            print ('{0:20} {1:5.0f}'.format(i[0], i[1]))


    if use_multi_output:
        target_rev_dict = {v: k for k, v in target_dict.items()}
        columns = [target_rev_dict[i] for i in range(0, y_pred.shape[1])]
        columns = ['pred_' + x for x in columns]
    else:
        columns = ['pred_' + target_col]

    result_xgb_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
    result_xgb_df.reset_index('id',inplace=True)
    if use_multi_output:
        s1_names = []
        for col in target_cols:
            s1_names.append(col+'_s_1')
            result_xgb_df[col+'_s_1'] = test[col+'_s_1'].values
    else:
         result_xgb_df[s1_name] = test[s1_name].values
    if use_multi_output:
        for col in target_cols:
            s1_cond = result_xgb_df[col + '_s_1'] == 1
            result_xgb_df['pred_' + col][s1_cond] = 0
            result_xgb_df['pred_' + col] = result_xgb_df['pred_' + col] + 1e-7
            result_xgb_df.drop(col + '_s_1',axis=1,inplace=True)
    else:
        s1_cond = result_xgb_df[s1_name] == 1
        result_xgb_df['pred_' + target_col][s1_cond] = 0
        result_xgb_df['pred_' + target_col] = result_xgb_df['pred_' + target_col] + 1e-7
        result_xgb_df.drop(s1_name,axis=1,inplace=True)

    if(is_sub_run):
        print('creating xgb output',target_col)
        result_xgb_df.index = result_xgb_df['id']
    else:
        if use_multi_output:
            result_xgb_df = pd.merge(result_xgb_df,test[['id'] + target_cols],left_on = ['id'],
                                   right_on = ['id'],how='left')
        else:
            result_xgb_df = pd.merge(result_xgb_df,test[['id'] + [target_col]],left_on = ['id'],
                                    right_on = ['id'],how='left')
            result_xgb_df[target_col + '_loss'] = -1 * (result_xgb_df[target_col] * np.log(result_xgb_df['pred_' + target_col])
                                                  + (1 - result_xgb_df[target_col]) * np.log(1 - result_xgb_df['pred_' + target_col]))
            print(target_col,'logloss',round(result_xgb_df[target_col + '_loss'].mean(),5))
    if use_multi_output and not is_sub_run:
        get_results(result_xgb_df)
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    del train
    del test
    return result_xgb_df
#%%
low_target_s1_names =  ['ind_ahor_fin_ult1_s_1',
 'ind_aval_fin_ult1_s_1',
 'ind_cder_fin_ult1_s_1',
 'ind_viv_fin_ult1_s_1',
 'ind_pres_fin_ult1_s_1',
 'ind_deme_fin_ult1_s_1',
 'ind_deco_fin_ult1_s_1']

diff_feautres_s1.remove('fecha_dato_year_s1_diff')
#%%
num_classes = 17
#%%
tic=timeit.default_timer()
xgb_features_5 = []
xgb_features_5 += shift_cols
xgb_features_5 += mean_shifted_names
xgb_features_5 += mean_shifted_small_names
xgb_features_5 += cols_to_combine
xgb_features_5 += diff_feautres_s1
xgb_features_5 += ['min_month_int']
xgb_features_5 += ['max_age']
xgb_features_5 += ['skipped_months']
xgb_features_5 += ['fecha_dato_month_minus_fecha_alta_month']
xgb_features_5 += ['months_active']
xgb_features_5 += ['renta_freq']

xgb_features_5.remove('fecha_alta_day_int')
xgb_features_5.remove('fecha_alta_month_int')
xgb_features_5.remove('fecha_alta_month')

params_5 = {'learning_rate': 0.1,
           'subsample': 0.98,
           'gamma': 0.05,
           'seed': 5,
           'colsample_bytree': 0.3,
           'eval_metric': 'logloss',
           'objective': 'binary:logistic',
           'max_depth': 4,
           }

es_rounds_dict = {}

es_xgb_5 = {'ind_cco_fin_ult1': 713,
 'ind_cno_fin_ult1': 517,
 'ind_ctju_fin_ult1': 190,
 'ind_ctma_fin_ult1': 350,
 'ind_ctop_fin_ult1': 250,
 'ind_ctpp_fin_ult1': 150,
 'ind_dela_fin_ult1': 538,
 'ind_ecue_fin_ult1': 634,
 'ind_fond_fin_ult1': 281,
 'ind_hip_fin_ult1': 115,
 'ind_nom_pens_ult1': 775,
 'ind_nomina_ult1': 830,
 'ind_plan_fin_ult1': 160,
 'ind_reca_fin_ult1': 445,
 'ind_recibo_ult1': 866,
 'ind_tjcr_fin_ult1': 597,
 'ind_valo_fin_ult1': 361}

num_boost_rounds = 15000
num_rounds = 150
result_xgb_5 = pd.DataFrame(test[['id']])
for col in target_cols:
    rounds = int(es_xgb_5[col] / 0.8)
    result_xgb_df = fit_xgb_model(train_samp_3g.copy(),test.copy(),params_5,xgb_features_5,col,
                                               num_rounds = rounds, num_rounds_es = num_boost_rounds,
                                               use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 5, use_weights=False,use_multi_output=False)
    result_xgb_5['pred_' + col] = result_xgb_df['pred_' + col].values
    if not is_sub_run:
        result_xgb_5[col] = result_xgb_df[col].values

toc=timeit.default_timer()
print('Xgb Time',toc - tic)
tic=timeit.default_timer()

result_xgb_5[pred_cols] = norm_rows(result_xgb_5[pred_cols])
result_xgb_5u = result_xgb_5.copy()
result_xgb_5u = update_probs(result_xgb_5u,update_factor = 0.7,print_odds=True)
if not is_sub_run:
    get_results(result_xgb_5)
    get_results(result_xgb_5u)
toc=timeit.default_timer()
print('Sorting Time',toc - tic)

if is_sub_run:
    result_xgb_5.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_5.csv',index=False)
#%%
tic=timeit.default_timer()
xgb_features_1 = []
xgb_features_1 += shift_small_cols
xgb_features_1 += mean_shifted_small_names
xgb_features_1 += cols_to_combine
xgb_features_1 += diff_feautres_s1
xgb_features_1 += low_target_s1_names
xgb_features_1 += ['min_month_int_binned']
xgb_features_1 += ['skipped_months','id_counts','min_max_month_int']
xgb_features_1 += ['fecha_dato_month_minus_fecha_alta_month']
xgb_features_1 += ['months_active']
xgb_features_1 += ['renta_freq']

xgb_features_1.remove('renta')
xgb_features_1.remove('fecha_alta_day_int')
xgb_features_1.remove('fecha_alta_month_int')
xgb_features_1.remove('fecha_alta_month')

params_1 = {'learning_rate': 0.02,
           'subsample': 0.98,
           'gamma': 2.0,
           'seed': 5,
           'colsample_bytree': 0.5,
           'objective': 'multi:softprob',
           'eval_metric': 'mlogloss',
           'max_depth': 4,
           'num_class':num_classes
           }

num_rounds = 10
num_boost_rounds = 15000
result_xgb_1 = fit_xgb_model(train_samp_6.copy(),test.copy(),params_1,xgb_features_1,'target_hash',
                              num_rounds = 1576, num_rounds_es = num_boost_rounds,
                              use_early_stopping = False, print_feature_imp = True,
                              random_seed = 5,use_weights=False,use_multi_output=True)

result_xgb_1_samp = result_xgb_1.sample(frac= 0.01,random_state=111)
result_xgb_1[pred_cols] = norm_rows(result_xgb_1[pred_cols])
tic=timeit.default_timer()
result_xgb_1u = result_xgb_1.copy()
result_xgb_1u = update_probs(result_xgb_1u,update_factor = 1.0)
if not is_sub_run:
    get_results(result_xgb_1u)

toc=timeit.default_timer()
print('xgb1 res Time',toc - tic)

if is_sub_run:
    result_xgb_1.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_1.csv',index=False)
#%%
tic=timeit.default_timer()

xgb_features_2 = []
xgb_features_2 += shift_small_cols
xgb_features_2 += mean_shifted_small_names
xgb_features_2 += cols_to_combine
xgb_features_2 += diff_feautres_s1
xgb_features_2 += ['min_month_int_binned']
xgb_features_2 += ['months_active']
xgb_features_2 += ['skipped_months']
xgb_features_2 += ['renta_freq']
xgb_features_2 += ['fecha_dato_month_minus_fecha_alta_month']
xgb_features_2 += low_target_s1_names
xgb_features_2 += ['antiguedad_no_change']

xgb_features_2.remove('antiguedad')
xgb_features_2.remove('renta')
xgb_features_2.remove('fecha_alta_day_int')
xgb_features_2.remove('fecha_alta_month_int')
xgb_features_2.remove('fecha_alta_month')

params_2 = {'learning_rate': 0.02,
           'subsample': 0.98,
           'gamma': 0.2,
           'seed': 4,
           'colsample_bytree': 0.5,
           'eval_metric': 'logloss',
           'objective': 'binary:logistic',
           'max_depth': 4,
           }

num_rounds = 10
num_boost_rounds = 15000

result_xgb_2 = pd.DataFrame(test[['id']])

es_rounds_dict = {}

es_xgb_2 = {'ind_cco_fin_ult1': 922,
 'ind_cno_fin_ult1': 627,
 'ind_ctju_fin_ult1': 565,
 'ind_ctma_fin_ult1': 517,
 'ind_ctop_fin_ult1': 690,
 'ind_ctpp_fin_ult1': 521,
 'ind_dela_fin_ult1': 913,
 'ind_ecue_fin_ult1': 657,
 'ind_fond_fin_ult1': 634,
 'ind_hip_fin_ult1': 411,
 'ind_nom_pens_ult1': 831,
 'ind_nomina_ult1': 1117,
 'ind_plan_fin_ult1': 356,
 'ind_reca_fin_ult1': 599,
 'ind_recibo_ult1': 910,
 'ind_tjcr_fin_ult1': 614,
 'ind_valo_fin_ult1': 285}
for col in target_cols:
    result_xgb_df = fit_xgb_model(train_samp_6.copy(),test.copy(),params_2,xgb_features_2,col,
                                               num_rounds = int(es_xgb_2[col] / 0.8), num_rounds_es = num_boost_rounds,
                                               use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 5, use_weights=False,use_multi_output=False)
    result_xgb_2['pred_' + col] = result_xgb_df['pred_' + col].values
    if not is_sub_run:
        result_xgb_2[col] = result_xgb_df[col].values

result_xgb_2[pred_cols] = norm_rows(result_xgb_2[pred_cols])
toc=timeit.default_timer()
print('Xgb Time',toc - tic)

tic=timeit.default_timer()
result_xgb_2u = result_xgb_2.copy()

result_xgb_2u = update_probs(result_xgb_2u,update_factor =  1.0)
if not is_sub_run:
    get_results(result_xgb_2)
    get_results(result_xgb_2u)
toc=timeit.default_timer()
print('Sorting Time',toc - tic)
#%%
if is_sub_run:
    result_xgb_2.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_2.csv',index=False)
#%%
tic=timeit.default_timer()

xgb_features_2b = []
xgb_features_2b += shift_small_cols
xgb_features_2b += mean_shifted_small_names
xgb_features_2b += cols_to_combine
xgb_features_2b += diff_feautres_s1
xgb_features_2b += ['min_month_int_binned']
xgb_features_2b += ['skipped_months']
xgb_features_2b += ['fecha_dato_month_minus_fecha_alta_month']
xgb_features_2b += ['months_active']
xgb_features_2b += ['renta_freq']
xgb_features_2b += ['min_antiguedad_plus_months_active','antiguedad_no_change']

xgb_features_2b.remove('renta')
xgb_features_2b.remove('antiguedad_s1_diff')
xgb_features_2b.remove('fecha_alta_day_int')
xgb_features_2b.remove('fecha_alta_month_int')
xgb_features_2b.remove('fecha_alta_month')

params_2b = {'learning_rate': 0.01,
           'subsample': 0.98,
           'gamma': 3.0,
           'alpha': 0.1,
           'seed': 4,
           'colsample_bytree': 0.3,
           'eval_metric': 'logloss',
           'objective': 'binary:logistic',
           'max_depth': 6,
           }

num_rounds = 10
num_boost_rounds = 15000

result_xgb_2b = pd.DataFrame(test[['id']])

es_rounds_dict = {}

es_xgb_2b = {'ind_cco_fin_ult1': 1494,
 'ind_cno_fin_ult1': 986,
 'ind_ctju_fin_ult1': 1062,
 'ind_ctma_fin_ult1': 1018,
 'ind_ctop_fin_ult1': 1061,
 'ind_ctpp_fin_ult1': 1066,
 'ind_dela_fin_ult1': 1650,
 'ind_ecue_fin_ult1': 955,
 'ind_fond_fin_ult1': 936,
 'ind_hip_fin_ult1': 906,
 'ind_nom_pens_ult1': 1057,
 'ind_nomina_ult1': 1707,
 'ind_plan_fin_ult1': 703,
 'ind_reca_fin_ult1': 1232,
 'ind_recibo_ult1': 1020,
 'ind_tjcr_fin_ult1': 987,
 'ind_valo_fin_ult1': 743}
for col in target_cols:
    result_xgb_df = fit_xgb_model(train_samp_6.copy(),test.copy(),params_2b,xgb_features_2b,col,
                                               num_rounds = int(es_xgb_2b[col] / 0.8), num_rounds_es = num_boost_rounds,
                                               use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 5, use_weights=False,use_multi_output=False)
    result_xgb_2b['pred_' + col] = result_xgb_df['pred_' + col].values
    if not is_sub_run:
        result_xgb_2b[col] = result_xgb_df[col].values

result_xgb_2b[pred_cols] = norm_rows(result_xgb_2b[pred_cols])
toc=timeit.default_timer()
print('Xgb Time',toc - tic)

tic=timeit.default_timer()
result_xgb_2ub = result_xgb_2b.copy()

result_xgb_2ub = update_probs(result_xgb_2ub,update_factor =  1.0)
if not is_sub_run:
    get_results(result_xgb_2b)
    get_results(result_xgb_2ub)
toc=timeit.default_timer()
print('Sorting Time',toc - tic)
if is_sub_run:
    result_xgb_2b.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_2b.csv',index=False)
#%%
tic=timeit.default_timer()

xgb_features_2ma1 = []
xgb_features_2ma1 += first_col_names
xgb_features_2ma1 += cols_to_combine
xgb_features_2ma1 += diff_feautres_s1
xgb_features_2ma1 += ['month_int']
xgb_features_2ma1 += ['min_age']
xgb_features_2ma1 += ['max_age']
xgb_features_2ma1 += ['fecha_dato_month_minus_fecha_alta_month']
xgb_features_2ma1 += low_target_s1_names
xgb_features_2ma1 += ['month_int_minus_fecha_alta_month_int']

xgb_features_2ma1.remove('antiguedad')
xgb_features_2ma1.remove('renta')
xgb_features_2ma1.remove('fecha_alta_day_int')
xgb_features_2ma1.remove('fecha_alta_month_int')
xgb_features_2ma1.remove('fecha_alta_month')

params_2ma1 = {'learning_rate': 0.01,
           'subsample': 0.98,
           'gamma': 0.07,
           'seed': 4,
           'colsample_bytree': 0.3,
           'eval_metric': 'logloss',
           'objective': 'binary:logistic',
           'max_depth': 4,
           }

num_rounds = 10
num_boost_rounds = 15000

result_xgb_2ma1 = pd.DataFrame(test_ma1[['id']])

es_rounds_dict = {}

es_xgb_2ma1 = {'ind_cco_fin_ult1': 1023,
 'ind_cno_fin_ult1': 1115,
 'ind_ctju_fin_ult1': 1196,
 'ind_ctma_fin_ult1': 1318,
 'ind_ctop_fin_ult1': 870,
 'ind_ctpp_fin_ult1': 834,
 'ind_dela_fin_ult1': 1018,
 'ind_ecue_fin_ult1': 805,
 'ind_fond_fin_ult1': 875,
 'ind_hip_fin_ult1': 1319,
 'ind_nom_pens_ult1': 1123,
 'ind_nomina_ult1': 1337,
 'ind_plan_fin_ult1': 826,
 'ind_reca_fin_ult1': 1102,
 'ind_recibo_ult1': 1375,
 'ind_tjcr_fin_ult1': 912,
 'ind_valo_fin_ult1': 729}
for col in target_cols:
    rounds = int(es_xgb_2ma1[col] / 0.8)
    result_xgb_df = fit_xgb_model(train_ma1.copy(),test_ma1.copy(),params_2ma1,xgb_features_2ma1,col,
                                               num_rounds = rounds, num_rounds_es = num_boost_rounds,
                                               use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 5, use_weights=False,use_multi_output=False,
                                               use_weights_late = False, weights_late_factor = 1.0,)
    result_xgb_2ma1['pred_' + col] = result_xgb_df['pred_' + col].values
    if not is_sub_run:
        result_xgb_2ma1[col] = result_xgb_df[col].values

toc=timeit.default_timer()
print('Xgb Time',toc - tic)

tic=timeit.default_timer()
result_xgb_2ma1[pred_cols] = norm_rows(result_xgb_2ma1[pred_cols])

result_xgb_2ma1u = result_xgb_2ma1.copy()
if not is_sub_run:
    get_results(result_xgb_2ma1)
toc=timeit.default_timer()
print('Sorting Time',toc - tic)

if is_sub_run:
    result_xgb_2ma1.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_2ma1.csv',index=False)
#%%
tic=timeit.default_timer()

xgb_features_2ma1_mo = []
xgb_features_2ma1_mo += first_col_names
xgb_features_2ma1_mo += cols_to_combine
xgb_features_2ma1_mo += diff_feautres_s1
xgb_features_2ma1_mo += ['month_int']
xgb_features_2ma1_mo += ['min_age']
xgb_features_2ma1_mo += ['max_age']
xgb_features_2ma1_mo += ['fecha_dato_month_minus_fecha_alta_month']
xgb_features_2ma1_mo += low_target_s1_names

xgb_features_2ma1_mo += ['month_int_minus_fecha_alta_month_int']

xgb_features_2ma1_mo.remove('antiguedad')
xgb_features_2ma1_mo.remove('renta')
xgb_features_2ma1_mo.remove('fecha_alta_day_int')
xgb_features_2ma1_mo.remove('fecha_alta_month_int')
xgb_features_2ma1_mo.remove('fecha_alta_month')

params_2ma1_mo = {'learning_rate': 0.06,
           'subsample': 0.98,
           'gamma': 5.0,
           'alpha': 0.1,
           'seed': 4,
           'colsample_bytree': 0.4,
           'eval_metric': 'mlogloss',
           'objective': 'multi:softprob',
           'max_depth': 8,
           'num_class': num_classes,
           }

num_rounds = 10
num_boost_rounds = 15000

result_xgb_2ma1_mo = fit_xgb_model(train_ma1.copy(),test_ma1.copy(),params_2ma1_mo,xgb_features_2ma1_mo,'target_hash',
                                           num_rounds = 705, num_rounds_es = num_boost_rounds,
                                           use_early_stopping = False, print_feature_imp = True,
                                           random_seed = 5, use_weights=False,use_multi_output=True,
                                           use_weights_late = False, weights_late_factor = 1.0)

toc=timeit.default_timer()
print('Xgb Time',toc - tic)

tic=timeit.default_timer()
result_xgb_2ma1_mo[pred_cols] = norm_rows(result_xgb_2ma1_mo[pred_cols])

toc=timeit.default_timer()
print('Sorting Time',toc - tic)

if is_sub_run:
    result_xgb_2ma1_mo.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_2ma1_mo.csv',index=False)

#%%
tic=timeit.default_timer()
xgb_features_4 = []
xgb_features_4 += shift_cols
xgb_features_4 += mean_shifted_names
xgb_features_4 += mean_shifted_small_names
xgb_features_4 += cols_to_combine
xgb_features_4 += diff_feautres_s1
xgb_features_4 += low_target_s1_names
xgb_features_4 += ['min_month_int']
xgb_features_4 += ['months_active']
xgb_features_4 += ['antiguedad_no_change']
xgb_features_4 += ['skipped_months']

xgb_features_4.remove('fecha_alta_day_int')
xgb_features_4.remove('fecha_alta_month_int')
xgb_features_4.remove('fecha_alta_month')

params_4 = {'learning_rate': 0.03,
           'subsample': 0.98,
           'gamma': 0.5,
           'alpha': 0.05,
           'seed': 5,
           'colsample_bytree': 0.3,
           'eval_metric': 'logloss',
           'objective': 'binary:logistic',
           'max_depth': 3,
           }

es_rounds_dict = {}           
es_xgb_4 = {'ind_cco_fin_ult1': 1161,
 'ind_cno_fin_ult1': 1062,
 'ind_ctju_fin_ult1': 562,
 'ind_ctma_fin_ult1': 708,
 'ind_ctop_fin_ult1': 608,
 'ind_ctpp_fin_ult1': 598,
 'ind_dela_fin_ult1': 418,
 'ind_ecue_fin_ult1': 1085,
 'ind_fond_fin_ult1': 369,
 'ind_hip_fin_ult1': 279,
 'ind_nom_pens_ult1': 1650,
 'ind_nomina_ult1': 1168,
 'ind_plan_fin_ult1': 394,
 'ind_reca_fin_ult1': 657,
 'ind_recibo_ult1': 1318,
 'ind_tjcr_fin_ult1': 1270,
 'ind_valo_fin_ult1': 600}           
result_xgb_4 = pd.DataFrame(test[['id']])
for col in target_cols:
    rounds = int(es_xgb_4[col] / 0.8)
    result_xgb_df = fit_xgb_model(train_samp_15g.copy(),test.copy(),params_4,xgb_features_4,col,
                                               num_rounds = rounds, num_rounds_es = num_boost_rounds,
                                               use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 5, use_weights=False,use_multi_output=False,
                                               use_weights_late = False, weights_late_factor = 1.0,)
    result_xgb_4['pred_' + col] = result_xgb_df['pred_' + col].values
    if not is_sub_run:
        result_xgb_4[col] = result_xgb_df[col].values

toc=timeit.default_timer()
print('Xgb Time',toc - tic)
tic=timeit.default_timer()
result_xgb_4[pred_cols] = norm_rows(result_xgb_4[pred_cols])
result_xgb_4u = result_xgb_4.copy()
result_xgb_4u = update_probs(result_xgb_4u,update_factor = 1.0)
if not is_sub_run:
    get_results(result_xgb_4)
    get_results(result_xgb_4u)
toc=timeit.default_timer()
print('Sorting Time',toc - tic)
#%%
if is_sub_run:
    result_xgb_4.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_4.csv',index=False)

#%%
tic=timeit.default_timer()
xgb_features_5b = []
xgb_features_5b += shift_cols
xgb_features_5b += first_col_names
xgb_features_5b += mean_shifted_names
xgb_features_5b += mean_shifted_small_names
xgb_features_5b += cols_to_combine
xgb_features_5b += diff_feautres_s1
xgb_features_5b += ['min_month_int']
xgb_features_5b += ['max_age']
xgb_features_5b += ['skipped_months']
xgb_features_5b += ['months_active']
xgb_features_5b.remove('fecha_alta_day_int')
xgb_features_5b.remove('fecha_alta_month_int')
xgb_features_5b.remove('fecha_alta_month')

params_5b = {'learning_rate': 0.1,
           'subsample': 0.99,
           'gamma': 3.0,
           'alpha': 0.05,
           'seed': 5,
           'colsample_bytree': 0.5,
           'eval_metric': 'mlogloss',
           'objective': 'multi:softprob',
           'max_depth': 6,
           'num_class': num_classes,
           }

result_xgb_5b = fit_xgb_model(train_samp_3g.copy(),test.copy(),params_5b,xgb_features_5b,'target_hash',
                              num_rounds = 770, num_rounds_es = num_boost_rounds,
                              use_early_stopping = False, print_feature_imp = True,
                              random_seed = 5,use_weights=False,use_multi_output=True)
toc=timeit.default_timer()
print('Xgb Time',toc - tic)
result_xgb_5b[pred_cols] = norm_rows(result_xgb_5b[pred_cols])
result_xgb_5ub = result_xgb_5b.copy()
result_xgb_5ub = update_probs(result_xgb_5ub,update_factor = 0.7,print_odds=True)
if not is_sub_run:
    get_results(result_xgb_5ub)
toc=timeit.default_timer()
print('Sorting Time',toc - tic)
if is_sub_run:
    result_xgb_5b.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_5b.csv',index=False)
#%%
tic=timeit.default_timer()
xgb_features_6 = []
xgb_features_6 += shift_1to12_cols
xgb_features_6 += mean_shifted_names
xgb_features_6 += mean_shifted_small_names
xgb_features_6 += cols_to_combine
xgb_features_6 += diff_feautres_s1
xgb_features_6 += ['min_month_int']
xgb_features_6 += ['max_age']

xgb_features_6 += ['months_active']

xgb_features_6 += ['min_age','std_renta']
xgb_features_6 += ['skipped_months']

xgb_features_6.remove('fecha_alta_day_int')
xgb_features_6.remove('fecha_alta_month_int')
xgb_features_6.remove('fecha_alta_month')

params_6 = {'learning_rate': 0.02,
           'subsample': 0.98,
           'gamma': 0.3,
           'alpha': 0.05,
           'seed': 5,
           'colsample_bytree': 0.3,
           'eval_metric': 'logloss',
           'objective': 'binary:logistic',
           'max_depth': 4,
           }

es_rounds_dict = {}
es_xgb_6 ={'ind_cco_fin_ult1': 1161,
 'ind_cno_fin_ult1': 1061,
 'ind_ctju_fin_ult1': 569,
 'ind_ctma_fin_ult1': 552,
 'ind_ctop_fin_ult1': 534,
 'ind_ctpp_fin_ult1': 454,
 'ind_dela_fin_ult1': 835,
 'ind_ecue_fin_ult1': 940,
 'ind_fond_fin_ult1': 588,
 'ind_hip_fin_ult1': 672,
 'ind_nom_pens_ult1': 848,
 'ind_nomina_ult1': 1011,
 'ind_plan_fin_ult1': 433,
 'ind_reca_fin_ult1': 482,
 'ind_recibo_ult1': 1057,
 'ind_tjcr_fin_ult1': 699,
 'ind_valo_fin_ult1': 484}
result_xgb_6 = pd.DataFrame(test[['id']])
for col in target_cols:
    result_xgb_df = fit_xgb_model(train_samp_12.copy(),test.copy(),params_6,xgb_features_6,col,
                                               num_rounds = int(es_xgb_6[col] / 0.8), num_rounds_es = num_boost_rounds,
                                               use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 5, use_weights=False,use_multi_output=False)

    result_xgb_6['pred_' + col] = result_xgb_df['pred_' + col].values
    if not is_sub_run:
        result_xgb_6[col] = result_xgb_df[col].values

toc=timeit.default_timer()
print('Xgb Time',toc - tic)

tic=timeit.default_timer()
result_xgb_6[pred_cols] = norm_rows(result_xgb_6[pred_cols])
result_xgb_6u = result_xgb_6.copy()
result_xgb_6u = update_probs(result_xgb_6u,update_factor = 1.0)
if not is_sub_run:
    get_results(result_xgb_6)
    get_results(result_xgb_6u)
toc=timeit.default_timer()
print('Sorting Time',toc - tic)
if is_sub_run:
    result_xgb_6.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_6.csv',index=False)
#%%
tic=timeit.default_timer()
xgb_features_6b = []
xgb_features_6b += shift_1to12_cols
xgb_features_6b += mean_shifted_names
xgb_features_6b += mean_shifted_small_names
xgb_features_6b += cols_to_combine
xgb_features_6b += diff_feautres_s1
xgb_features_6b += ['min_month_int']
xgb_features_6b += ['max_age']

xgb_features_6b += ['months_active']

xgb_features_6b += ['min_age','std_renta']
xgb_features_6b += ['skipped_months']

xgb_features_6b += ['min_antiguedad_plus_months_active','antiguedad_no_change']

xgb_features_6b.remove('fecha_alta_day_int')
xgb_features_6b.remove('fecha_alta_month_int')
xgb_features_6b.remove('fecha_alta_month')

params_6b = {'learning_rate': 0.02,
           'subsample': 0.98,
           'gamma': 1.5,
           'alpha': 0.05,
           'seed': 5,
           'colsample_bytree': 0.3,
           'eval_metric': 'mlogloss',
           'objective': 'multi:softprob',
           'max_depth': 6,
           'num_class': num_classes,
           }

result_xgb_6b = fit_xgb_model(train_samp_12.copy(),test.copy(),params_6b,xgb_features_6b,'target_hash',
                                           num_rounds = 1260, num_rounds_es = num_boost_rounds,
                                           use_early_stopping = False, print_feature_imp = False,
                                           random_seed = 5, use_weights=False,use_multi_output=True)


toc=timeit.default_timer()
print('Xgb Time',toc - tic)

tic=timeit.default_timer()
result_xgb_6b[pred_cols] = norm_rows(result_xgb_6b[pred_cols])
result_xgb_6bu = result_xgb_6b.copy()
result_xgb_6bu = update_probs(result_xgb_6bu,update_factor = 1.0)
if not is_sub_run:
    get_results(result_xgb_6bu)
toc=timeit.default_timer()
print('Sorting Time',toc - tic)
if is_sub_run:
    result_xgb_6b.to_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_6b.csv',index=False)
#%%
steve_probs = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/raw_probs20161221b.csv',header=0)
steve_probs.rename(columns={'ncodpers':'id'},inplace=True)
steve_probs.rename(columns=lambda x: 'pred_' + x if x.startswith('ind') else x, inplace=True)

steve_probs.replace(-1,1e-8,inplace=True)
if is_sub_run:
    result_xgb_5b = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Santander_Prod/Predictions/result_xgb_5b.csv',header=0)
#%%
result_temp = pd.DataFrame(test[['id','months_active','min_month_int','renta','age',
                                 'antiguedad','canal_entrada','cod_prov','ind_actividad_cliente',
                                 'tiprel_1mes_s1_diff']].copy())

for col in pred_cols:
    result_temp[col + '_xgb_2'] = result_xgb_2[col].values
    result_temp[col + '_xgb_4'] = result_xgb_4[col].values
    result_temp[col + '_xgb_5'] = result_xgb_5[col].values
    result_temp[col + '_xgb_6'] = result_xgb_6[col].values
    if is_sub_run:
        result_temp[col + '_steve'] = steve_probs[col].values
        result_temp[col + '_std'] = result_temp[[col + '_xgb_2',col + '_xgb_4',col + '_xgb_5',col + '_xgb_6',col+'_steve']].std(axis=1)
    else:
        result_temp[col + '_std'] = result_temp[[col + '_xgb_2',col + '_xgb_4',col + '_xgb_5',col + '_xgb_6']].std(axis=1)
result_temp_small = result_temp.sample(frac = 0.02,random_state = 111)
result_xgb_2ma1_copy = pd.DataFrame(test[['id','months_active']].copy())
result_xgb_2ma1_mo_copy = pd.DataFrame(test[['id','months_active']].copy())
for col in pred_cols:
    PRED_DICT = dict(zip(result_xgb_2ma1['id'],result_xgb_2ma1[col]))
    result_xgb_2ma1_copy[col] = result_xgb_2ma1_copy['id'].map(lambda x: apply_dict(PRED_DICT,x))
    PRED_DICT_MO = dict(zip(result_xgb_2ma1_mo['id'],result_xgb_2ma1_mo[col]))
    result_xgb_2ma1_mo_copy[col] = result_xgb_2ma1_mo_copy['id'].map(lambda x: apply_dict(PRED_DICT_MO,x))
#%%
tic=timeit.default_timer()

result_ens = pd.DataFrame(test[['id','months_active','min_month_int']].copy())
if not is_sub_run:
    result_ens = pd.merge(result_ens,result_xgb_2[['id'] + target_cols])



result_ens.index = result_ens.id
result_xgb_1.index = result_xgb_1.id
result_xgb_2.index = result_xgb_2.id
result_xgb_2b.index = result_xgb_2b.id
result_xgb_4.index = result_xgb_4.id
result_xgb_5.index = result_xgb_5.id
result_xgb_5b.index = result_xgb_5b.id
result_xgb_6.index = result_xgb_6.id
result_xgb_2ma1_copy.index = result_xgb_2ma1_copy.id
result_xgb_2ma1_mo_copy.index = result_xgb_2ma1_mo_copy.id
steve_probs.index = steve_probs.id

a1 = 6.0 
a2 = 3.0 
a2b = 4.0 
a4 = 2.0 
a5 = 4.0 
a5b = 4.0 
a6 = 2.0 
a6b = 2.0
for col in pred_cols:
    result_ens[col] = (
                       a1 * result_xgb_1[col].values +
                       a2 * result_xgb_2[col].values +
                       a2b * result_xgb_2b[col].values +
                       a4 * result_xgb_4[col].values +
                       a5 * result_xgb_5[col].values +
                       a5b * result_xgb_5b[col].values +
                       a6 * result_xgb_6[col].values +
                       a6b * result_xgb_6b[col].values
                        ) / (a1 + a2 + a2b + a4 + a5 + a5b + a6 + a6b)

result_ens = update_probs(result_ens,update_factor = 1.0,a1=a1,a2=a2,a2b=a2b,a4=a4,a5=a5,a5b=a5b,a6=a6,a6b=a6b)
        
b1 = 1.0
b2 = 1.5
b3 = 1.5
ma8_cond = result_ens['months_active'] <= 12
for col in pred_cols:
    result_ens[col][ma8_cond] = (b1 * result_ens[col][ma8_cond] +
                                 b2 * result_xgb_5[col][ma8_cond] + 
                                 b3 * result_xgb_5b[col][ma8_cond]
                                ) / (b1 + b2 + b3)        

bb0 = 4
bb1 = 4
bb2 = 4
bb3 = 4
cond_reca_ma8_high = ((result_xgb_1['pred_ind_reca_fin_ult1'] >= 0.25) & (result_ens['months_active'] <= 12))
for col in pred_cols:
    result_ens[col][cond_reca_ma8_high] = ((bb0 * result_ens[col][cond_reca_ma8_high].values +
                                       bb1 * result_xgb_1[col][cond_reca_ma8_high].values +
                                       bb2 * result_xgb_2[col][cond_reca_ma8_high].values +
                                       bb3 * result_xgb_2b[col][cond_reca_ma8_high].values ) /
                                         (bb0 + bb1 + bb2 + bb3))
d1 = 0.8
d2 = 0.2
for col in pred_cols:
    if col == 'pred_ind_dela_fin_ult1':
        continue
    result_ens[col] = (d1 * result_ens[col].values + d2 * steve_probs[col].values) / (d1 + d2)           

e0 = 0.3
e1 = 0.7        
cond_nom_pens_high_steve = steve_probs['pred_ind_nom_pens_ult1'] >= 0.3
for col in pred_cols:
    result_ens[col][cond_nom_pens_high_steve] = ((e0 * result_ens[col][cond_nom_pens_high_steve].values +
                                       e1 * steve_probs[col][cond_nom_pens_high_steve].values ) /
                                         (e0 + e1))
e0nom = 0.3        
e1nom = 0.7        
cond_nomina_high_steve = steve_probs['pred_ind_nomina_ult1'] >= 0.3
for col in pred_cols:
    result_ens[col][cond_nomina_high_steve] = ((e0nom * result_ens[col][cond_nomina_high_steve].values +
                                       e1nom * steve_probs[col][cond_nomina_high_steve].values ) /
                                         (e0nom + e1nom))

dd1 = 0.5
dd2 = 0.5
cond_sister = steve_probs['sister'] == 1
for col in pred_cols:
    if col == 'pred_ind_dela_fin_ult1':
        continue
    result_ens[col][cond_sister] = (dd1 * result_ens[col][cond_sister].values + dd2 * steve_probs[col][cond_sister].values) / (dd1 + dd2)        
                
f1 = 1.0
f2 = 1.0
f3 = 1.0
ma_cond = result_ens['months_active'] <= 1
for col in pred_cols:
    result_ens[col] = (f1 * result_ens[col] + f2 * result_xgb_2ma1_copy[col] + f3 * result_xgb_2ma1_mo_copy[col])
    result_ens[col][ma_cond] = result_ens[col][ma_cond] / (f1 + f2 + f3)

cond_nom_greater = (result_ens['pred_ind_nomina_ult1'] > result_ens['pred_ind_nom_pens_ult1']) & (result_ens['pred_ind_nom_pens_ult1'] > 1e-4)        
result_ens['pred_ind_nom_pens_ult1'][cond_nom_greater] = result_ens['pred_ind_nomina_ult1'][cond_nom_greater] + 0.00001        
        
get_results(result_ens)
toc=timeit.default_timer()
print('Ensembling Time',toc - tic)

if is_sub_run:
    submission = result_ens[['id','added_products']].copy()
    submission.rename(columns={'id':'ncodpers'},inplace=True)
    submission.to_csv('submission.csv',index=False)
#%%
result_ens[['id'] + pred_cols].to_csv('result_ens_probs_sub_112.csv',index=False)
#%%
lb_probing_dict = {
'ind_tjcr_fin_ult1':0.0041178,
'ind_ahor_fin_ult1':0,
'ind_aval_fin_ult1':0,
'ind_cder_fin_ult1':0,
'ind_ecue_fin_ult1':0.0019961,
'ind_fond_fin_ult1':0.000104,
'ind_ctpp_fin_ult1':0.0001142,
'ind_ctju_fin_ult1':0.0000502,
'ind_hip_fin_ult1':0.0000161,
'ind_plan_fin_ult1':0.0000126,
'ind_viv_fin_ult1':0,
'ind_pres_fin_ult1':0.0000054,
'ind_deme_fin_ult1':0,
'ind_deco_fin_ult1':0,
'ind_reca_fin_ult1':0.0032092,
'ind_cco_fin_ult1':0.0096681,
'ind_cno_fin_ult1':0.0017839,
'ind_nom_pens_ult1':0.0021801,
'ind_nomina_ult1':0.0021478,
'ind_recibo_ult1':0.0086845,
'ind_valo_fin_ult1':0.000278,
'ind_dela_fin_ult1':0.0000933,
'ind_ctma_fin_ult1':0.0004488,
'ind_ctop_fin_ult1':0.0001949}

sum_probing_lb = 0
for item,val in lb_probing_dict.items():
    sum_probing_lb += val
print('lb',sum_probing_lb)
cv_probing_dict = {'ind_cco_fin_ult1': 0.003881731015950354,
 'ind_cno_fin_ult1': 0.0016603628953902967,
 'ind_ctju_fin_ult1': 4.1333271780755445e-05,
 'ind_ctma_fin_ult1': 0.0004969297073139132,
 'ind_ctop_fin_ult1': 0.0002224123672012079,
 'ind_ctpp_fin_ult1': 0.00013384107052816048,
 'ind_dela_fin_ult1': 4.276472708052187e-05,
 'ind_ecue_fin_ult1': 0.002724220474176726,
 'ind_fond_fin_ult1': 5.6900348165715294e-05,
 'ind_nom_pens_ult1': 0.002740342239490389,
 'ind_nomina_ult1': 0.002747141652164281,
 'ind_reca_fin_ult1': 0.00028221141234895016,
 'ind_recibo_ult1': 0.010396284085187339,
 'ind_tjcr_fin_ult1': 0.004323156044015828,
 'ind_valo_fin_ult1': 0.00018317259879636078}

cv_month6_dict = {'ind_cco_fin_ult1': 0.010039260044401548,
 'ind_cno_fin_ult1': 0.0019260360802181014,
 'ind_ctju_fin_ult1': 1.2656025058929616e-05,
 'ind_ctma_fin_ult1': 0.0003160051256901489,
 'ind_ctop_fin_ult1': 0.00030796327643395403,
 'ind_ctpp_fin_ult1': 0.00021910743383271903,
 'ind_dela_fin_ult1': 0.0013487104037799332,
 'ind_ecue_fin_ult1': 0.0016968829264948616,
 'ind_fond_fin_ult1': 0.0003338026609292687,
 'ind_nom_pens_ult1': 0.008183965870918908,
 'ind_nomina_ult1': 0.0036306971887804344,
 'ind_reca_fin_ult1': 0.0041450591405504346,
 'ind_recibo_ult1': 0.013364446061603173,
 'ind_tjcr_fin_ult1': 0.006769444136832738,
 'ind_valo_fin_ult1': 0.00022055760337072133}
#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)