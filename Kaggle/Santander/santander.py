# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:46:15 2015

@author: Jared Turkewitz
"""
#%%
base_features = ['var3', 'var15', 'imp_ent_var16_ult1',
       'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3',
       'imp_op_var40_comer_ult1', 'imp_op_var40_comer_ult3',
       'imp_op_var40_efect_ult1', 'imp_op_var40_efect_ult3',
       'imp_op_var40_ult1', 'imp_op_var41_comer_ult1',
       'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1',
       'imp_op_var41_efect_ult3', 'imp_op_var41_ult1',
       'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3',
       'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var1_0', 'ind_var1',
       'ind_var2_0', 'ind_var2', 'ind_var5_0', 'ind_var5', 'ind_var6_0',
       'ind_var6', 'ind_var8_0', 'ind_var8', 'ind_var12_0', 'ind_var12',
       'ind_var13_0', 'ind_var13_corto_0', 'ind_var13_corto',
       'ind_var13_largo_0', 'ind_var13_largo', 'ind_var13_medio_0',
       'ind_var13_medio', 'ind_var13', 'ind_var14_0', 'ind_var14',
       'ind_var17_0', 'ind_var17', 'ind_var18_0', 'ind_var18', 'ind_var19',
       'ind_var20_0', 'ind_var20', 'ind_var24_0', 'ind_var24',
       'ind_var25_cte', 'ind_var26_0', 'ind_var26_cte', 'ind_var26',
       'ind_var25_0', 'ind_var25', 'ind_var27_0', 'ind_var28_0',
       'ind_var28', 'ind_var27', 'ind_var29_0', 'ind_var29', 'ind_var30_0',
       'ind_var30', 'ind_var31_0', 'ind_var31', 'ind_var32_cte',
       'ind_var32_0', 'ind_var32', 'ind_var33_0', 'ind_var33',
       'ind_var34_0', 'ind_var34', 'ind_var37_cte', 'ind_var37_0',
       'ind_var37', 'ind_var39_0', 'ind_var40_0', 'ind_var40',
       'ind_var41_0', 'ind_var41', 'ind_var39', 'ind_var44_0', 'ind_var44',
       'ind_var46_0', 'ind_var46', 'num_var1_0', 'num_var1', 'num_var4',
       'num_var5_0', 'num_var5', 'num_var6_0', 'num_var6', 'num_var8_0',
       'num_var8', 'num_var12_0', 'num_var12', 'num_var13_0',
       'num_var13_corto_0', 'num_var13_corto', 'num_var13_largo_0',
       'num_var13_largo', 'num_var13_medio_0', 'num_var13_medio',
       'num_var13', 'num_var14_0', 'num_var14', 'num_var17_0', 'num_var17',
       'num_var18_0', 'num_var18', 'num_var20_0', 'num_var20',
       'num_var24_0', 'num_var24', 'num_var26_0', 'num_var26',
       'num_var25_0', 'num_var25', 'num_op_var40_hace2',
       'num_op_var40_hace3', 'num_op_var40_ult1', 'num_op_var40_ult3',
       'num_op_var41_hace2', 'num_op_var41_hace3', 'num_op_var41_ult1',
       'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_hace3',
       'num_op_var39_ult1', 'num_op_var39_ult3', 'num_var27_0',
       'num_var28_0', 'num_var28', 'num_var27', 'num_var29_0', 'num_var29',
       'num_var30_0', 'num_var30', 'num_var31_0', 'num_var31',
       'num_var32_0', 'num_var32', 'num_var33_0', 'num_var33',
       'num_var34_0', 'num_var34', 'num_var35', 'num_var37_med_ult2',
       'num_var37_0', 'num_var37', 'num_var39_0', 'num_var40_0',
       'num_var40', 'num_var41_0', 'num_var41', 'num_var39', 'num_var42_0',
       'num_var42', 'num_var44_0', 'num_var44', 'num_var46_0', 'num_var46',
       'saldo_var1', 'saldo_var5', 'saldo_var6', 'saldo_var8',
       'saldo_var12', 'saldo_var13_corto', 'saldo_var13_largo',
       'saldo_var13_medio', 'saldo_var13', 'saldo_var14', 'saldo_var17',
       'saldo_var18', 'saldo_var20', 'saldo_var24', 'saldo_var26',
       'saldo_var25', 'saldo_var28', 'saldo_var27', 'saldo_var29',
       'saldo_var30', 'saldo_var31', 'saldo_var32', 'saldo_var33',
       'saldo_var34', 'saldo_var37', 'saldo_var40', 'saldo_var41',
       'saldo_var42', 'saldo_var44', 'saldo_var46', 'var36',
       'delta_imp_amort_var18_1y3', 'delta_imp_amort_var34_1y3',
       'delta_imp_aport_var13_1y3', 'delta_imp_aport_var17_1y3',
       'delta_imp_aport_var33_1y3', 'delta_imp_compra_var44_1y3',
       'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var17_1y3',
       'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3',
       'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_in_1y3',
       'delta_imp_trasp_var33_out_1y3', 'delta_imp_venta_var44_1y3',
       'delta_num_aport_var13_1y3', 'delta_num_aport_var17_1y3',
       'delta_num_aport_var33_1y3', 'delta_num_compra_var44_1y3',
       'delta_num_reemb_var13_1y3', 'delta_num_reemb_var17_1y3',
       'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_in_1y3',
       'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_in_1y3',
       'delta_num_trasp_var33_out_1y3', 'delta_num_venta_var44_1y3',
       'imp_amort_var18_hace3', 'imp_amort_var18_ult1',
       'imp_amort_var34_hace3', 'imp_amort_var34_ult1',
       'imp_aport_var13_hace3', 'imp_aport_var13_ult1',
       'imp_aport_var17_hace3', 'imp_aport_var17_ult1',
       'imp_aport_var33_hace3', 'imp_aport_var33_ult1',
       'imp_var7_emit_ult1', 'imp_var7_recib_ult1',
       'imp_compra_var44_hace3', 'imp_compra_var44_ult1',
       'imp_reemb_var13_hace3', 'imp_reemb_var13_ult1',
       'imp_reemb_var17_hace3', 'imp_reemb_var17_ult1',
       'imp_reemb_var33_hace3', 'imp_reemb_var33_ult1',
       'imp_var43_emit_ult1', 'imp_trans_var37_ult1',
       'imp_trasp_var17_in_hace3', 'imp_trasp_var17_in_ult1',
       'imp_trasp_var17_out_hace3', 'imp_trasp_var17_out_ult1',
       'imp_trasp_var33_in_hace3', 'imp_trasp_var33_in_ult1',
       'imp_trasp_var33_out_hace3', 'imp_trasp_var33_out_ult1',
       'imp_venta_var44_hace3', 'imp_venta_var44_ult1',
       'ind_var7_emit_ult1', 'ind_var7_recib_ult1', 'ind_var10_ult1',
       'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var9_ult1',
       'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'var21',
       'num_var2_0_ult1', 'num_var2_ult1', 'num_aport_var13_hace3',
       'num_aport_var13_ult1', 'num_aport_var17_hace3',
       'num_aport_var17_ult1', 'num_aport_var33_hace3',
       'num_aport_var33_ult1', 'num_var7_emit_ult1', 'num_var7_recib_ult1',
       'num_compra_var44_hace3', 'num_compra_var44_ult1',
       'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3',
       'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3',
       'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var8_ult3',
       'num_meses_var12_ult3', 'num_meses_var13_corto_ult3',
       'num_meses_var13_largo_ult3', 'num_meses_var13_medio_ult3',
       'num_meses_var17_ult3', 'num_meses_var29_ult3',
       'num_meses_var33_ult3', 'num_meses_var39_vig_ult3',
       'num_meses_var44_ult3', 'num_op_var39_comer_ult1',
       'num_op_var39_comer_ult3', 'num_op_var40_comer_ult1',
       'num_op_var40_comer_ult3', 'num_op_var40_efect_ult1',
       'num_op_var40_efect_ult3', 'num_op_var41_comer_ult1',
       'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1',
       'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1',
       'num_op_var39_efect_ult3', 'num_reemb_var13_hace3',
       'num_reemb_var13_ult1', 'num_reemb_var17_hace3',
       'num_reemb_var17_ult1', 'num_reemb_var33_hace3',
       'num_reemb_var33_ult1', 'num_sal_var16_ult1', 'num_var43_emit_ult1',
       'num_var43_recib_ult1', 'num_trasp_var11_ult1',
       'num_trasp_var17_in_hace3', 'num_trasp_var17_in_ult1',
       'num_trasp_var17_out_hace3', 'num_trasp_var17_out_ult1',
       'num_trasp_var33_in_hace3', 'num_trasp_var33_in_ult1',
       'num_trasp_var33_out_hace3', 'num_trasp_var33_out_ult1',
       'num_venta_var44_hace3', 'num_venta_var44_ult1', 'num_var45_hace2',
       'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3',
       'saldo_var2_ult1', 'saldo_medio_var5_hace2',
       'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1',
       'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2',
       'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1',
       'saldo_medio_var8_ult3', 'saldo_medio_var12_hace2',
       'saldo_medio_var12_hace3', 'saldo_medio_var12_ult1',
       'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2',
       'saldo_medio_var13_corto_hace3', 'saldo_medio_var13_corto_ult1',
       'saldo_medio_var13_corto_ult3', 'saldo_medio_var13_largo_hace2',
       'saldo_medio_var13_largo_hace3', 'saldo_medio_var13_largo_ult1',
       'saldo_medio_var13_largo_ult3', 'saldo_medio_var13_medio_hace2',
       'saldo_medio_var13_medio_hace3', 'saldo_medio_var13_medio_ult1',
       'saldo_medio_var13_medio_ult3', 'saldo_medio_var17_hace2',
       'saldo_medio_var17_hace3', 'saldo_medio_var17_ult1',
       'saldo_medio_var17_ult3', 'saldo_medio_var29_hace2',
       'saldo_medio_var29_hace3', 'saldo_medio_var29_ult1',
       'saldo_medio_var29_ult3', 'saldo_medio_var33_hace2',
       'saldo_medio_var33_hace3', 'saldo_medio_var33_ult1',
       'saldo_medio_var33_ult3', 'saldo_medio_var44_hace2',
       'saldo_medio_var44_hace3', 'saldo_medio_var44_ult1',
       'saldo_medio_var44_ult3', 'var38']

#%%
has_equal_in_train_cols = ['ind_var6_0','ind_var6','ind_var13_medio_0',
                           'ind_var18_0','ind_var26_0','ind_var25_0','ind_var32_0',
                           'ind_var34_0','ind_var37_0','ind_var40','num_var6_0',
                           'num_var6','num_var13_medio_0','num_var18_0','num_var26_0',
                           'num_var25_0','num_var32_0','num_var34_0','num_var37_0',
                           'num_var40','saldo_var6','saldo_var13_medio','delta_imp_reemb_var13_1y3',
                           'delta_imp_reemb_var17_1y3','delta_imp_reemb_var33_1y3',
                           'delta_imp_trasp_var17_in_1y3','delta_imp_trasp_var17_out_1y3',
                           'delta_imp_trasp_var33_in_1y3','delta_imp_trasp_var33_out_1y3']
#%%
zero_std_cols = ['ind_var2_0','ind_var2','ind_var27_0','ind_var28_0','ind_var28',
            'ind_var27','ind_var41','ind_var46_0','ind_var46','num_var27_0',
            'num_var28_0','num_var28','num_var27','num_var41','num_var46_0',
            'num_var46','saldo_var28','saldo_var27','saldo_var41','saldo_var46',
            'imp_amort_var18_hace3','imp_amort_var34_hace3','imp_reemb_var13_hace3',
            'imp_reemb_var33_hace3','imp_trasp_var17_out_hace3','imp_trasp_var33_out_hace3',
            'num_var2_0_ult1','num_var2_ult1','num_reemb_var13_hace3',
            'num_reemb_var33_hace3','num_trasp_var17_out_hace3','num_trasp_var33_out_hace3',
            'saldo_var2_ult1','saldo_medio_var13_medio_hace3'
            ]
#%%
base_features = [x for x in base_features if x not in has_equal_in_train_cols]
base_features = [x for x in base_features if x not in zero_std_cols]
#%%
import pandas as pd
import numpy as np
import random
from sklearn import cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV,Lasso,ElasticNetCV
import sklearn.metrics as skm
import xgboost as xgb
from scipy.sparse import csr_matrix
import operator
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
import decimal
import timeit

tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'
#%%
tic=timeit.default_timer()

train_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Santander/train.csv', header=0)
test_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Santander/test.csv', header=0)

toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
train_orig = train_orig.rename(columns={'TARGET': 'target','ID':'id'})
test_orig = test_orig.rename(columns={'ID':'id'})
#%%
#controls whether to run on local cv or on test dataset for Kaggle leader board
#is_sub_run = False
##is_sub_run = True
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
        print(input_df.loc[input_df[col_name] == value].target.value_counts(normalize=is_normalized))

def print_value_counts_spec(input_df,col_name,col_value,is_normalized=False):
    print(col_value)
    print(input_df.loc[input_df[col_name] == col_value].target.value_counts(normalize=is_normalized))

def convert_strings_to_ints(input_df,col_name,output_col_name):
    labels, levels = pd.factorize(input_df[col_name])
    input_df[output_col_name] = labels
    output_dict = dict(zip(input_df[col_name],input_df[output_col_name]))
    return (output_dict,input_df)

#assumes row normalized, doesn't do eps thing
def get_log_loss_row(row):
    ans = row['target']
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
    ans = row['target']
    if (ans == 0):
        return -1.0 * np.log(row['predict_low'])
    elif (ans == 1):
        return -1.0 * np.log(row['predict_high'])
    else:
        print('not_exceptable_value')
        raise ValueError('Not of correct class')
        return -1000

#%%
test_orig['target'] = 'dummy'
#%%
tic=timeit.default_timer()
combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)

combined.fillna(-1,inplace=True)

toc=timeit.default_timer()
print('Merging Time',toc - tic)
#%%
tic=timeit.default_timer()
#maxes = combined.max()
mins = combined.min().to_dict()

max_1_columns = ['ind_var1_0', 'ind_var1', 'ind_var5_0', 'ind_var5', 'ind_var6_0',
       'ind_var6', 'ind_var8_0', 'ind_var8', 'ind_var12_0', 'ind_var12',
       'ind_var13_0', 'ind_var13_corto_0', 'ind_var13_corto',
       'ind_var13_largo_0', 'ind_var13_largo', 'ind_var13_medio_0',
       'ind_var13_medio', 'ind_var13', 'ind_var14_0', 'ind_var14',
       'ind_var17_0', 'ind_var17', 'ind_var18_0', 'ind_var18', 'ind_var19',
       'ind_var20_0', 'ind_var20', 'ind_var24_0', 'ind_var24',
       'ind_var25_cte', 'ind_var26_0', 'ind_var26_cte', 'ind_var26',
       'ind_var25_0', 'ind_var25', 'ind_var29_0', 'ind_var29',
       'ind_var30_0', 'ind_var30', 'ind_var31_0', 'ind_var31',
       'ind_var32_cte', 'ind_var32_0', 'ind_var32', 'ind_var33_0',
       'ind_var33', 'ind_var34_0', 'ind_var34', 'ind_var37_cte',
       'ind_var37_0', 'ind_var37', 'ind_var39_0', 'ind_var40_0',
       'ind_var40', 'ind_var41_0', 'ind_var39', 'ind_var44_0', 'ind_var44',
       'ind_var7_emit_ult1', 'ind_var7_recib_ult1', 'ind_var10_ult1',
       'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var9_ult1',
       'ind_var43_emit_ult1', 'ind_var43_recib_ult1']

max_huge_columns = ['delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_in_1y3',
       'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var17_out_1y3',
       'delta_num_trasp_var17_in_1y3', 'delta_num_reemb_var33_1y3',
       'delta_num_reemb_var17_1y3', 'delta_num_reemb_var13_1y3',
       'delta_num_compra_var44_1y3', 'delta_num_aport_var33_1y3',
       'delta_num_aport_var17_1y3', 'delta_num_aport_var13_1y3',
       'delta_imp_venta_var44_1y3', 'delta_imp_trasp_var33_out_1y3',
       'delta_num_trasp_var33_out_1y3', 'delta_num_venta_var44_1y3',
       'delta_imp_trasp_var17_in_1y3', 'delta_imp_reemb_var33_1y3',
       'delta_imp_reemb_var17_1y3', 'delta_imp_reemb_var13_1y3',
       'delta_imp_compra_var44_1y3', 'delta_imp_aport_var33_1y3',
       'delta_imp_aport_var17_1y3', 'delta_imp_aport_var13_1y3',
       'delta_imp_amort_var34_1y3', 'delta_imp_amort_var18_1y3']

combined['sum_bool_cols']= combined[max_1_columns].sum(axis=1)

def count_large_values(row):
    res = 0
    for col_name in max_huge_columns:
        if(row[col_name] >= 1e8):
            res += 1
    return res

combined['large_max_values'] = (combined[base_features] >= 1e8).astype(int).sum(axis=1)
combined['zero_values'] = (combined[base_features] == 0).astype(int).sum(axis=1)
combined['negative_values'] = (combined[base_features] < 0).astype(int).sum(axis=1)
#combined['unique_values'] = combined[base_features].apply(lambda x: len(x.unique()),axis=1)
combined['unique_values'] = combined[base_features].apply(lambda x: x.nunique(),axis=1)



toc=timeit.default_timer()
print('Maxes Time',toc - tic)
#%%
tic=timeit.default_timer()
freq_1_columns = []
freq_2_columns = []
freq_3_columns = []
freq_4_columns = []

cumulative_1_columns = []
cumulative_2_columns = []
cumulative_3_columns = []
cumulative_4_columns = []

for col in base_features:
    number_unique = combined[col].nunique()
    combined_value_counts = combined[col].value_counts().to_dict()
    most_freq_number = combined[col].value_counts().max()
    col_freq = col + '_freq'
#    col_freq_number = col + '_freq_number'
    col_cumulative = col + '_cumulative'
    combined[col_freq] = combined[col].map(lambda x: combined_value_counts[x])


    combined.sort(col,inplace=True)
    TEMP_VALUES = combined.drop_duplicates(col)[col_freq].cumsum().values
    TEMP_DICT = pd.Series(TEMP_VALUES,index=combined.drop_duplicates(col)[col]).to_dict()
    combined[col_cumulative] = combined[col].map(lambda x: TEMP_DICT[x])

    if (number_unique <= 200):
        freq_1_columns.append(col_freq)
        cumulative_1_columns.append(col_freq)
    elif (number_unique > 200 and number_unique <= 1000):
        freq_2_columns.append(col_freq)
        cumulative_2_columns.append(col_freq)
    elif (number_unique > 1000 and number_unique <= 15000):
        freq_3_columns.append(col_freq)
        cumulative_3_columns.append(col_freq)
    else:
        freq_4_columns.append(col_freq)
        cumulative_4_columns.append(col_freq)

    combined[col_freq] = combined[col_freq] / most_freq_number

combined['freq_1_mean'] = combined[freq_1_columns].mean(axis=1)
combined['freq_1_max'] = combined[freq_1_columns].max(axis=1)
combined['freq_1_min'] = combined[freq_1_columns].min(axis=1)
combined['freq_1_std'] = combined[freq_1_columns].std(axis=1)

combined['freq_2_mean'] = combined[freq_2_columns].mean(axis=1)
combined['freq_2_max'] = combined[freq_2_columns].max(axis=1)
combined['freq_2_min'] = combined[freq_2_columns].min(axis=1)
combined['freq_2_std'] = combined[freq_2_columns].std(axis=1)

combined['freq_3_mean'] = combined[freq_3_columns].mean(axis=1)
combined['freq_3_max'] = combined[freq_3_columns].max(axis=1)
combined['freq_3_min'] = combined[freq_3_columns].min(axis=1)
combined['freq_3_std'] = combined[freq_3_columns].std(axis=1)

combined['freq_4_mean'] = combined[freq_4_columns].mean(axis=1)
combined['freq_4_max'] = combined[freq_4_columns].max(axis=1)
combined['freq_4_min'] = combined[freq_4_columns].min(axis=1)
combined['freq_4_std'] = combined[freq_4_columns].std(axis=1)

combined['cumulative_1_mean'] = combined[cumulative_1_columns].mean(axis=1)
combined['cumulative_1_max'] = combined[cumulative_1_columns].max(axis=1)
combined['cumulative_1_min'] = combined[cumulative_1_columns].min(axis=1)
combined['cumulative_1_std'] = combined[cumulative_1_columns].std(axis=1)

combined['cumulative_2_mean'] = combined[cumulative_2_columns].mean(axis=1)
combined['cumulative_2_max'] = combined[cumulative_2_columns].max(axis=1)
combined['cumulative_2_min'] = combined[cumulative_2_columns].min(axis=1)
combined['cumulative_2_std'] = combined[cumulative_2_columns].std(axis=1)

combined['cumulative_3_mean'] = combined[cumulative_3_columns].mean(axis=1)
combined['cumulative_3_max'] = combined[cumulative_3_columns].max(axis=1)
combined['cumulative_3_min'] = combined[cumulative_3_columns].min(axis=1)
combined['cumulative_3_std'] = combined[cumulative_3_columns].std(axis=1)

combined['cumulative_4_mean'] = combined[cumulative_4_columns].mean(axis=1)
combined['cumulative_4_max'] = combined[cumulative_4_columns].max(axis=1)
combined['cumulative_4_min'] = combined[cumulative_4_columns].min(axis=1)
combined['cumulative_4_std'] = combined[cumulative_4_columns].std(axis=1)


toc=timeit.default_timer()
print('Freq Time',toc - tic)
#%%
tic=timeit.default_timer()
combined['temp_one'] = 1
combined['num_duplicate_rows'] = combined.groupby(base_features)['temp_one'].transform(pd.Series.count)
#combined.groupby(base_features).size().reset_index(name='num_identical_rows',inplace=True)
#correlation_matrix = train_orig.corr()
toc=timeit.default_timer()
print('Duplicate counting Time',toc - tic)
#%%
tic=timeit.default_timer()
combined['temp_one'] = 1
combined['num_duplicate_bool_rows'] = combined.groupby(max_1_columns)['temp_one'].transform(pd.Series.count)
#combined.groupby(base_features).size().reset_index(name='num_identical_rows',inplace=True)
#correlation_matrix = train_orig.corr()
toc=timeit.default_timer()
print('Duplicate counting Time2',toc - tic)
#%%
tic=timeit.default_timer()

combined['var15_last_digit'] = combined['var15'].map(lambda x: np.round(x) % 10)
combined['var38_mod_10'] = combined['var38'].map(lambda x: np.round(x) % 10)
combined['var38_mod_100'] = combined['var38'].map(lambda x: np.round(x) % 100)
combined['var38_decimal_count'] = combined['var38'].map(lambda x: decimal.Decimal(str(x)).as_tuple().exponent)
combined['saldo_var30_decimal_count'] = combined['saldo_var30'].map(lambda x: decimal.Decimal(str(x)).as_tuple().exponent)
combined['saldo_var30_last_digit'] = combined['saldo_var30'].map(lambda x: np.round(x) % 10)

combined['saldo_medio_var5_ult3_mod_3'] = combined['saldo_medio_var5_ult3'].map(lambda x: x % 3)

combined['saldo_var42_var30_diff'] = combined['saldo_var42'] - combined['saldo_var30']

combined['saldo_var30_var5_diff'] = combined['saldo_var30'] - combined['saldo_var5']
combined['saldo_var5_saldo_medio_var5_ult3_diff'] = combined['saldo_var5'] - combined['saldo_medio_var5_ult3']

#combined['var38_modified'] = combined['var38'].map(lambda x: -1 if np.isclose(x,117310.979016) else x)
combined['var38_modified'] = combined['var38'].map(lambda x: -1 if x == 117310.979016494 else x)

#combined['saldo_medio_var5_hace3_log1p'] = np.log1p(combined['saldo_medio_var5_hace3'].map(lambda x: x if x >=0 else 0))

#tic=timeit.default_timer()
combined['saldo_var30_str'] = combined['saldo_var30'].astype(str)
saldo_var30_zero_cond = combined['saldo_var30'] == 0.0
saldo_var30_neg_cond = combined['saldo_var30'] < 0
combined['saldo_var30_num_zeros'] = combined['saldo_var30_str'].map(lambda x: x.count('0'))
combined['saldo_var30_num_digits'] = combined['saldo_var30_str'].map(lambda x: len(x.rsplit('.', 1)[0]))
combined['saldo_var30_num_consec_zeros'] = combined['saldo_var30_str'].map(lambda x: x.rsplit('.', 1)[0].count('0') -
                                                x.rsplit('.', 1)[0].rstrip('0').count('0'))
combined['saldo_var30_decimal_count'] = combined['saldo_var30_str'].map(lambda x: decimal.Decimal(x).as_tuple().exponent)

combined['saldo_var30_num_digits'][saldo_var30_zero_cond] = -1
combined['saldo_var30_num_digits'][saldo_var30_neg_cond] = -2

combined['saldo_var30_decimal_count'][saldo_var30_zero_cond] = -3
combined['saldo_var30_decimal_count'][saldo_var30_neg_cond] = -4

combined['saldo_var30_num_zeros'][saldo_var30_zero_cond] = -1
combined['saldo_var30_num_zeros'][saldo_var30_neg_cond] = -2

combined['saldo_var30_num_consec_zeros'][saldo_var30_zero_cond] = -1
combined['saldo_var30_num_consec_zeros'][saldo_var30_neg_cond] = -2
#%%
combined['saldo_var8_str'] = combined['saldo_var8'].astype(str)
saldo_var8_zero_cond = combined['saldo_var8'] == 0.0
saldo_var8_neg_cond = combined['saldo_var8'] < 0
#saldo_var8_last_0_cond = combined['saldo_var8'] < 0

#combined['saldo_var8_decimal_count'] = combined['saldo_var8_str'].map(lambda x: decimal.Decimal(x).as_tuple().exponent)
combined['saldo_var8_last_dec_0'] = combined['saldo_var8_str'].map(lambda x: x.count('0') - x.rstrip('0').count('0'))
combined['saldo_var8_num_consec_zeros'] = combined['saldo_var8_str'].map(lambda x: x.rsplit('.', 1)[0].count('0') -
                                                x.rsplit('.', 1)[0].rstrip('0').count('0'))
combined['saldo_var8_num_digits'] = combined['saldo_var8_str'].map(lambda x: len(x.rsplit('.', 1)[0]))

combined['saldo_var8_num_consec_zeros'] = combined['saldo_var8_num_consec_zeros'] / combined['saldo_var8_num_digits']

combined['saldo_var8_last_dec_0'][saldo_var8_zero_cond] = -1

combined['saldo_var8_num_consec_zeros'][saldo_var8_zero_cond] = -1
combined['saldo_var8_num_consec_zeros'][saldo_var8_neg_cond] = -2

#%%
#imp_trans_var37_ult1
combined['imp_trans_var37_ult1_str'] = combined['imp_trans_var37_ult1'].astype(str)
imp_trans_var37_ult1_zero_cond = combined['imp_trans_var37_ult1'] == 0.0

combined['imp_trans_var37_ult1_last_dec_0'] = combined['imp_trans_var37_ult1_str'].map(lambda x: x.count('0') - x.rstrip('0').count('0'))
combined['imp_trans_var37_ult1_num_consec_zeros'] = combined['imp_trans_var37_ult1_str'].map(lambda x: x.rsplit('.', 1)[0].count('0') -
                                                x.rsplit('.', 1)[0].rstrip('0').count('0'))
combined['imp_trans_var37_ult1_num_digits'] = combined['imp_trans_var37_ult1_str'].map(lambda x: len(x.rsplit('.', 1)[0]))

combined['imp_trans_var37_ult1_num_consec_zeros'] = combined['imp_trans_var37_ult1_num_consec_zeros'] / combined['imp_trans_var37_ult1_num_digits']

combined['imp_trans_var37_ult1_last_dec_0'][imp_trans_var37_ult1_zero_cond] = -1

combined['imp_trans_var37_ult1_num_consec_zeros'][imp_trans_var37_ult1_zero_cond] = -1

toc=timeit.default_timer()
print('FE Time',toc - tic)
#%%
freq_col_feats = freq_1_columns
freq_col_feats = [x[:-5] for x in freq_col_feats]
float_cols = [x for x in ['id','target'] + base_features if x not in freq_col_feats]
combined_freq_vals = combined.loc[combined['freq_1_mean'] == 1][float_cols]
#%%
#for col in base_features:
#    max_count = combined[col].value_counts().max()
#    if(max_count <  151788):
#        continue
#    print(col)
#    print_value_counts(combined,col,0)

#%%

low_count_features = []
for col in base_features:
    max_val = combined[col].value_counts().max()
    if (max_val >= 151788):
        low_count_features.append(col)
#%%
#ind_var_1
combined['ind_var1_diff'] = combined['ind_var1'] - combined['ind_var1_0']
combined['ind_var5_diff'] = combined['ind_var5'] - combined['ind_var5_0']
combined['ind_var8_diff'] = combined['ind_var8'] - combined['ind_var8_0']
combined['ind_var12_diff'] = combined['ind_var12'] - combined['ind_var12_0']
combined['ind_var13_diff'] = combined['ind_var13'] - combined['ind_var13_0']
combined['ind_var14_diff'] = combined['ind_var14'] - combined['ind_var14_0']
combined['ind_var17_diff'] = combined['ind_var17'] - combined['ind_var17_0']
combined['ind_var20_diff'] = combined['ind_var20'] - combined['ind_var20_0']
combined['ind_var24_diff'] = combined['ind_var24'] - combined['ind_var24_0']
combined['ind_var29_diff'] = combined['ind_var29'] - combined['ind_var29_0']
combined['ind_var30_diff'] = combined['ind_var30'] - combined['ind_var30_0']
combined['ind_var31_diff'] = combined['ind_var31'] - combined['ind_var31_0']
combined['ind_var33_diff'] = combined['ind_var33'] - combined['ind_var33_0']
combined['ind_var39_diff'] = combined['ind_var39'] - combined['ind_var39_0']
combined['ind_var44_diff'] = combined['ind_var44'] - combined['ind_var44_0']

combined['num_var20_diff'] = combined['num_var20'] - combined['num_var20_0']
combined['num_var29_diff'] = combined['num_var29'] - combined['num_var29_0']

ind_diff_list = ['ind_var1_diff','ind_var5_diff','ind_var8_diff','ind_var12_diff',
                 'ind_var13_diff','ind_var14_diff','ind_var17_diff','ind_var20_diff',
                 'ind_var24_diff','ind_var29_diff','ind_var30_diff','ind_var31_diff',
                 'ind_var33_diff','ind_var39_diff','ind_var44_diff']
#%%
combined['saldo_var25_26_diff'] = combined['saldo_var25'] - combined['saldo_var26']
combined['saldo_var12_14_diff'] = combined['saldo_var12'] - combined['saldo_var14']
combined['saldo_var24_30_diff'] = combined['saldo_var24'] - combined['saldo_var30']
combined['imp_var43_37_diff'] = combined['imp_var43_emit_ult1'] - combined['imp_trans_var37_ult1']
combined['imp_var10_cte_diff'] = combined['ind_var10_ult1'] - combined['ind_var10cte_ult1']
combined['imp_op39_ult1_ult3_diff'] = combined['imp_op_var39_comer_ult1'] - combined['imp_op_var39_comer_ult3']
combined['num_var45_ult1_ult3_diff'] = combined['num_var45_ult1'] - combined['num_var45_ult3']

combined['saldo_var30_var38_diff'] = combined['saldo_var30'] - combined['var38']
combined['saldo_var30_var38_ratio'] = combined['saldo_var30'] / combined['var38']

#combined['saldo_medio_var5_hace2_hace3_diff'] = combined['saldo_medio_var5_hace2'] - combined['saldo_medio_var5_hace3']
combined['saldo_var5_hace2_diff'] = combined['saldo_var5'] - combined['saldo_medio_var5_hace2']

saldo_diff_list = ['saldo_var25_26_diff','saldo_var12_14_diff','saldo_var24_30_diff',
                   'imp_var43_37_diff','imp_var10_cte_diff','imp_op39_ult1_ult3_diff',
                   'num_var45_ult1_ult3_diff','saldo_var30_var38_diff','saldo_var30_var38_ratio']

#combined['saldo_medio_var5_hace2_hace3_diff'][combined['saldo_medio_var5_hace2'] == 0] = 1e11
combined['saldo_var24_30_diff'][combined['saldo_var24'] == 0] = 1e11
combined['saldo_var30_var38_diff'][combined['saldo_var30'] == 0] = 1e11
combined['imp_var43_37_diff'][combined['imp_var43_emit_ult1'] == 0] = 1e11
combined['imp_op39_ult1_ult3_diff'][combined['imp_op_var39_comer_ult1'] == 0] = 1e11
combined['num_var45_ult1_ult3_diff'][combined['num_var45_ult1'] == 0] = 1e11
#%%
#for col in ind_diff_list:
#    print(col)
#    print_value_counts(combined,col)
#%%
#saldo_var5
#saldo var14
#imp_aport_v13_hace3
#imp_var7_recib_ult1

#saldo_var20 - no nonzero are target 1
#saldo_medio_var13_largo_hace2 - no nonzero are target 1
#saldo_medio_var13_largo_hace3 - no nonzero are target 1
#saldo_medio_var13_largo_ult1 - no nonzero are target 1
#saldo_medio_var13_largo_ult3 - no nonzero are target 1
#num_meses_var13_largo_ult3 - no nonzero are target 1
#ind_var20_0 - no nonzero are target 1
#ind_var20 - no nonzero are target 1
#ind_var33_0 - no nonzero are target 1
#ind_var33 - no nonzero are target 1
#num_var20_0 - no nonzero are target 1
#num_var20 - no nonzero are target 1
#num_var33_0 - no nonzero are target 1
#num_var33 - no nonzero are target 1
#saldo_var33 - no nonzero are target 1
#delta_imp_venta_var44_1y3 - no nonzero are target 1
#delta_num_venta_var44_1y3 - no nonzero are target 1
#imp_venta_var44_ult1 - no nonzero are target 1
#num_aport_var17_hace3 - no nonzero are target 1
#num_aport_var33_hace3 - no nonzero are target 1
#num_compra_var44_hace3 - no nonzero are target 1
#num_meses_var33_ult3 - no nonzero are target 1
#num_venta_var44_ult1 - no nonzero are target 1
#saldo_medio_var17_hace2 - no nonzero are target 1
#saldo_medio_var33_hace2 - no nonzero are target 1
#saldo_medio_var33_hace3 - no nonzero are target 1
#saldo_medio_var33_ult1 - no nonzero are target 1
#saldo_medio_var33_ult3 - no nonzero are target 1
#saldo_medio_var44_hace2 - no nonzero are target 1
#saldo_medio_var44_hace3 - no nonzero are target 1



#saldo var26 investigate high values
#train_subset_float_cols_2 = train[float_cols]
#train_subset_float_cols_2_t1 = train.loc[train.target == 1][float_cols]
#train_subset_float_cols_2_t0 = train.loc[train.target == 0][float_cols]

#%%
#high target potentials for low variance
#imp_op_var40_efect_ult1
#imp_op_var40_efect_ult3
#num_op_var40_efect_ult1
#num_op_var40_efect_ult3
#num_sal_var16_ult1
#%%
#for (index,val) in target_std.iteritems():
#    if (val != 0):
#        continue
#    print(index,val)
#    print_value_counts(combined,index,0)
#%%
#for col in base_features:
#    number_unique = combined[col].nunique()
#    if (number_unique > 200):
#        continue
#    print(col)
#    print_value_counts(combined,col)
#%%

#investigate ind and corresponding name_0
#ind_var8_0
#ind_var32_cte
#ind_var39

#num_var13_largo_0
#ind_var13_largo_0
#ind_var_20_0
#num_var20_0
#ind_var33_0
#num_var33_0
#num_var1_0
#saldo_medio_var44_hace3

#num_meses_var13_largo_ult3

#var36 - 0


#num_var13_0
#num_var26
#num_op_var40_ult3
#num_op_var39_ult1
#num_var30
#num_var40_0

#delta_imp_aport_var13_1y3 -1
#delta_num_aport_var13_1y3
#num_aport_var13_hace3
#deltas,cte

#num_var22_ult1
#num_var22_ult3

#num_meses_var12_ult3
#num_meses_var13_corto_ult3

#%%
#%%
is_sub_run = False
#is_sub_run = True
if (is_sub_run):
    train = combined.loc[combined['target'] != 'dummy' ]
    test = combined.loc[combined['target'] == 'dummy' ]
else:
#    train = combined.loc[(combined['target'] != 'dummy') & (combined['id'] > 75000)]
#    test = combined.loc[(combined['target'] != 'dummy') & (combined['id'] <= 75000)]
#    train = combined.loc[(combined['target'] != 'dummy') & (combined['id'] <= 110000)]
#    test = combined.loc[(combined['target'] != 'dummy') & (combined['id'] > 110000)]
#    train = combined.loc[(combined['target'] != 'dummy') & (combined['id'] >= 40000)]
#    test = combined.loc[(combined['target'] != 'dummy') & (combined['id'] < 40000)]
#    test.drop_duplicates(['var38','num_duplicate_rows'],inplace=True)
#    random.seed(5)
#    np.random.seed(5)

#    train, test = cross_validation.train_test_split( combined.loc[(combined['target'] != 'dummy')], test_size=0.3,random_state=7)
#    train, test = cross_validation.train_test_split( combined.loc[(combined['target'] != 'dummy')], test_size=0.3,random_state=42)
    train, test = cross_validation.train_test_split( combined.loc[(combined['target'] != 'dummy')], test_size=0.3,random_state=47)


test.set_index('id',drop=False,inplace=True)
train['target'] = train['target'].astype(int)

#train_all = train.copy()
#test_all = test.copy()

#train = train_all.loc[train_all['var15'] > 22]
#train_var15_small = train.loc[train_all['var15'] <= 22]
#test = test_all.loc[test_all['var15'] > 22]
#test_var15_small = test_all.loc[test_all['var15'] <= 22]
#%%
#train_delta_imp_aport_var13_1y3 = train.loc[train.delta_imp_aport_var13_1y3 != 0][['id','target'] + base_features]
#train_delta_imp_aport_var13_1y3_corr = train_delta_imp_aport_var13_1y3.corr()

#imp_sal_var16_ult1
#num_sal_var16_ult1
#var38
#imp_reemb_var13_ult1
#%%
#train_var15_23 = train.loc[train.var15 == 23][['id','target'] + base_features]
#train_var15_23_corr = train_var15_23.corr()
#%%
tic=timeit.default_timer()
min_cols = []
max_cols = []
#for col in base_features:
for col in train.columns.values:
    if(col == 'id' or col == 'target'):
        continue
    lim_min = train[col].min()
    lim_max = train[col].max()
    min_col = col + '_min'
    max_col = col + '_max'

    min_cols.append(min_col)
    max_cols.append(max_col)

#    test[min_col] = test[col].map(lambda x: 1 if x < lim_min else 0)
#    test[max_col] = test[col].map(lambda x: 1 if x > lim_max else 0)
    test_max = test[col].max()
    test_min = test[col].min()
    if(test_min < lim_min):
        test[col] = test[col].map(lambda x: lim_min if x < lim_min else x)
    if(test_max > lim_max):
        test[col] = test[col].map(lambda x: lim_max if x > lim_max else x)


#test['below_min_values'] = test[min_cols].sum(axis=1)
#test['above_max_values'] = test[max_cols].sum(axis=1)

toc=timeit.default_timer()
print('Limit Setting Time',toc - tic)
#%%


train_t1 = train.loc[train.target == 1]
train_t0 = train.loc[train.target == 0]
def get_distributin_diff(col_name):
    print(round(train_t1[col_name].mean() - train_t0[col_name].mean(),5))
    print(round((train_t1[col_name].median() - train_t0[col_name].median()) / train_t0[col_name].std(),5))
#%%
#for col in freq_columns:
#    print(col)
#    print_value_counts(combined,col,0)
#for col in base_features:
#    print(col)
#    print(combined[col].nunique())
#%%
def fit_xgb_model(train, test, params, xgb_features, num_rounds = 10, num_boost_rounds = 10,
                  do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
                  random_seed = 5, reweight_probs = True, calculate_auc = True):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    X_train, X_watch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
    train_data = X_train[xgb_features].values
    train_data_full = train[xgb_features].values
    watch_data = X_watch[xgb_features].values
    train_target = X_train['target'].astype(int).values
    train_target_full = train['target'].astype(int).values
    watch_target = X_watch['target'].astype(int).values
    test_data = test[xgb_features].values
    dtrain = xgb.DMatrix(train_data, train_target)
    dtrain_full = xgb.DMatrix(train_data_full, train_target_full)
    dwatch = xgb.DMatrix(watch_data, watch_target)
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
        clf.fit(train_data_full, train_target_full)
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
        num_to_print = 50
        num_printed = 0
        for i in imp_dict:
            num_printed = num_printed + 1
            if (num_printed > num_to_print):
                continue
            print ('{0:20} {1:5.0f}'.format(i[0], i[1]))


    columns = ['pred']
    result_xgb_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)

#    result_xgb_df = norm_rows(result_xgb_df)
    result_xgb_df['id'] = result_xgb_df.index
    if(is_sub_run):
        print('creating xgb output')

    else:
        if(calculate_auc):
#            result_xgb_df.reset_index('id',inplace=True)
            result_xgb_df = pd.merge(result_xgb_df,test[['id','target']],left_on = ['id'],
                                   right_on = ['id'],how='left')
            result_xgb_df['target'] = result_xgb_df['target'].astype(int)
            result_xgb_df.set_index('id',drop=False,inplace=True)
            print('auc',round(skm.roc_auc_score(result_xgb_df.target.values,result_xgb_df.pred.values),5))
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_xgb_df
#%%
#%%
xgb_features = []
xgb_features = xgb_features + base_features
#xgb_features = [x for x in xgb_features if x not in low_count_features]


xgb_features = xgb_features + ['sum_bool_cols']
xgb_features = xgb_features + ['zero_values']
xgb_features = xgb_features + ['large_max_values']
xgb_features = xgb_features + ['unique_values']
xgb_features = xgb_features + ['negative_values']
#xgb_features = xgb_features + freq_1_columns



xgb_features = xgb_features + [
#                                'var15_freq',
#                               'saldo_medio_var5_ult3_freq',
                                'saldo_medio_var5_hace3_freq',
                                'saldo_medio_var5_hace2_freq',
#                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
#                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
                               'num_var22_ult3_freq','num_var22_ult1_freq',
                               ]
#xgb_features = xgb_features + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
xgb_features = xgb_features + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
xgb_features = xgb_features + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
xgb_features = xgb_features + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

#xgb_features = xgb_features + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
#xgb_features = xgb_features + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
xgb_features = xgb_features + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
xgb_features = xgb_features + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']

#xgb_features = xgb_features + ['saldo_var30_num_digits']
xgb_features = xgb_features + ['saldo_var30_decimal_count']
xgb_features = xgb_features + ['saldo_var30_num_zeros']
xgb_features = xgb_features + ['saldo_var30_num_consec_zeros']

xgb_features = xgb_features + ['imp_trans_var37_ult1_num_consec_zeros']
xgb_features = xgb_features + ['saldo_var8_num_consec_zeros']

xgb_features = xgb_features + ['num_duplicate_rows']
xgb_features = xgb_features + ['num_duplicate_bool_rows']

xgb_features = xgb_features + ['var38_modified']
xgb_features = xgb_features + ['saldo_var42_var30_diff']
xgb_features = xgb_features + ['saldo_var30_var5_diff']
xgb_features = xgb_features + ['saldo_var5_saldo_medio_var5_ult3_diff']

xgb_features = xgb_features + ind_diff_list
xgb_features = xgb_features + saldo_diff_list



#xgb_features = xgb_features + ['mean_1_min','mean_1_std']
#xgb_features = xgb_features + ['mean_2_min','mean_2_std']

#xgb_features.remove('saldo_var42')
xgb_features.remove('var38')
#xgb_features.remove('var38_modified')


params = {'learning_rate': 0.02,
              'subsample': 0.77,
              'reg_alpha': 0.25,
#              'lambda': 0.99,
#              'gamma': 0.05,
              'seed': 6,
              'colsample_bytree': 0.5,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 1.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 4,
#              'min_child_weight': 2,
              }
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df = fit_xgb_model(train,test,params,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
result_xgb_df = fit_xgb_model(train,test,params,xgb_features,
                              num_rounds = num_rounds, num_boost_rounds = 700,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
xgb_features_3 = []
xgb_features_3 = xgb_features_3 + base_features
xgb_features_3 = [x for x in xgb_features_3 if x not in low_count_features]

xgb_features_3 = xgb_features_3 + ['sum_bool_cols']
xgb_features_3 = xgb_features_3 + ['zero_values']
#xgb_features_3 = xgb_features_3 + ['large_max_values']
xgb_features_3 = xgb_features_3 + ['unique_values']
xgb_features_3 = xgb_features_3 + ['negative_values']
#xgb_features_3 = xgb_features_3 + freq_1_columns



#xgb_features_3 = xgb_features_3 + [
#                                'var15_freq',
##                               'saldo_medio_var5_ult3_freq',
#                                'saldo_medio_var5_hace3_freq',
#                                'saldo_medio_var5_hace2_freq',
##                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
##                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
#                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
#                               'num_var22_ult3_freq','num_var22_ult1_freq',
#                               ]
xgb_features_3 = xgb_features_3 + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
xgb_features_3 = xgb_features_3 + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
xgb_features_3 = xgb_features_3 + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
xgb_features_3 = xgb_features_3 + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

#xgb_features_3 = xgb_features_3 + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
#xgb_features_3 = xgb_features_3 + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
#xgb_features_3 = xgb_features_3 + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
#xgb_features_3 = xgb_features_3 + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']

#xgb_features_3 = xgb_features_3 + ['saldo_var30_num_digits']
xgb_features_3 = xgb_features_3 + ['saldo_var30_decimal_count']
#xgb_features_3 = xgb_features_3 + ['saldo_var30_num_zeros']
xgb_features_3 = xgb_features_3 + ['saldo_var30_num_consec_zeros']

#xgb_features_3 = xgb_features_3 + ['num_duplicate_rows']
xgb_features_3 = xgb_features_3 + ['num_duplicate_bool_rows']

xgb_features_3 = xgb_features_3 + ['var38_modified']
xgb_features_3 = xgb_features_3 + ['saldo_var42_var30_diff']
xgb_features_3 = xgb_features_3 + ['saldo_var30_var5_diff']
xgb_features_3 = xgb_features_3 + ['saldo_var5_saldo_medio_var5_ult3_diff']

xgb_features_3 = xgb_features_3 + ['saldo_var8_num_consec_zeros']
xgb_features_3 = xgb_features_3 + ind_diff_list
#xgb_features_3 = xgb_features_3 + saldo_diff_list

#xgb_features_3 = xgb_features_3 + ['var15_last_digit']
#xgb_features_3 = xgb_features_3 + ['var38_mod_10']
#xgb_features_3 = xgb_features_3 + ['var38_mod_100']
#xgb_features_3 = xgb_features_3 + ['var38_decimal_count']
#xgb_features_3 = xgb_features_3 + ['saldo_var30_decimal_count']
#xgb_features_3 = xgb_features_3 + ['saldo_var30_last_digit']
#xgb_features_3 = xgb_features_3 + ['saldo_medio_var5_ult3_mod_3']

#xgb_features_3.remove('saldo_var42')
xgb_features_3.remove('var38')


params_3 = {'learning_rate': 0.015,
              'subsample': 0.85,
              'reg_alpha': 0.4,
#              'lambda': 0.99,
              'gamma': 0.5,
              'seed': 3,
              'colsample_bytree': 0.6,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 1.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 4,
#              'min_child_weight': 5,
              }
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_3 = fit_xgb_model(train,test,params_3,xgb_features_3,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
result_xgb_3 = fit_xgb_model(train,test,params_3,xgb_features_3,
                              num_rounds = num_rounds, num_boost_rounds = 1050,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
xgb_features_4 = []
xgb_features_4 = xgb_features_4 + base_features

#xgb_features_4 = xgb_features_4 + ['sum_bool_cols']
xgb_features_4 = xgb_features_4 + ['zero_values']
#xgb_features_4 = xgb_features_4 + ['large_max_values']
#xgb_features_4 = xgb_features_4 + ['unique_values']
#xgb_features_4 = xgb_features_4 + ['negative_values']
#xgb_features_4 = xgb_features_4 + freq_1_columns

#xgb_features_4 = xgb_features_4 + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
#xgb_features_4 = xgb_features_4 + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
#xgb_features_4 = xgb_features_4 + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
#xgb_features_4 = xgb_features_4 + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

#xgb_features_4 = xgb_features_4 + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
#xgb_features_4 = xgb_features_4 + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
#xgb_features_4 = xgb_features_4 + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
#xgb_features_4 = xgb_features_4 + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']



#xgb_features_4 = xgb_features_4 + ['num_duplicate_rows']
#xgb_features_4 = xgb_features_4 + ['num_duplicate_bool_rows']

xgb_features_4 = xgb_features_4 + ['var38_modified']
#xgb_features_4 = xgb_features_4 + ['id']
#xgb_features_4 = xgb_features_4 + ['saldo_var42_var30_diff']
#xgb_features_4 = xgb_features_4 + ['saldo_var30_var5_diff']
#xgb_features_4 = xgb_features_4 + ['saldo_var5_saldo_medio_var5_ult3_diff']

#xgb_features_4 = xgb_features_4 + ['mean_1_min','mean_1_std']
#xgb_features_4 = xgb_features_4 + ['mean_2_min','mean_2_std']

#xgb_features_4 = xgb_features_4 + ['var15_last_digit']
#xgb_features_4 = xgb_features_4 + ['var38_mod_10']
#xgb_features_4 = xgb_features_4 + ['var38_mod_100']
#xgb_features_4 = xgb_features_4 + ['var38_decimal_count']
#xgb_features_4 = xgb_features_4 + ['saldo_var30_decimal_count']
#xgb_features_4 = xgb_features_4 + ['saldo_var30_last_digit']
#xgb_features_4 = xgb_features_4 + ['saldo_medio_var5_ult3_mod_3']

#xgb_features_4.remove('saldo_var42')
xgb_features_4.remove('var38')


params_4 = {'learning_rate': 0.02,
              'subsample': 0.68,
#              'reg_alpha': 0.3,
#              'lambda': 0.99,
#              'gamma': 0.5,
              'seed': 10,
              'colsample_bytree': 0.7,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 1.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 5,
#              'min_child_weight': 2,
              }
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_4 = fit_xgb_model(train,test,params_4,xgb_features_4,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
result_xgb_4 = fit_xgb_model(train,test,params_4,xgb_features_4,
                              num_rounds = num_rounds, num_boost_rounds = 550,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
xgb_features_6 = []
xgb_features_6 = xgb_features_6 + base_features

xgb_features_6 = xgb_features_6 + ['sum_bool_cols']
xgb_features_6 = xgb_features_6 + ['zero_values']
#xgb_features_6 = xgb_features_6 + ['large_max_values']
#xgb_features_6 = xgb_features_6 + ['unique_values']
#xgb_features_6 = xgb_features_6 + ['negative_values']
#xgb_features_6 = xgb_features_6 + freq_1_columns



#xgb_features_6 = xgb_features_6 + [
#                                'var15_freq',
##                               'saldo_medio_var5_ult3_freq',
#                                'saldo_medio_var5_hace3_freq',
#                                'saldo_medio_var5_hace2_freq',
##                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
##                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
#                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
#                               'num_var22_ult3_freq','num_var22_ult1_freq',
#                               ]
#xgb_features_6 = xgb_features_6 + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
#xgb_features_6 = xgb_features_6 + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
#xgb_features_6 = xgb_features_6 + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
#xgb_features_6 = xgb_features_6 + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

xgb_features_6 = xgb_features_6 + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
xgb_features_6 = xgb_features_6 + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
xgb_features_6 = xgb_features_6 + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
xgb_features_6 = xgb_features_6 + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']

#xgb_features_6 = xgb_features_6 + ['saldo_var30_num_digits']
xgb_features_6 = xgb_features_6 + ['saldo_var30_decimal_count']
xgb_features_6 = xgb_features_6 + ['saldo_var30_num_zeros']
xgb_features_6 = xgb_features_6 + ['saldo_var30_num_consec_zeros']

xgb_features_6 = xgb_features_6 + ['saldo_var8_num_consec_zeros']

#xgb_features_6 = xgb_features_6 + ['num_duplicate_rows']
#xgb_features_6 = xgb_features_6 + ['num_duplicate_bool_rows']

xgb_features_6 = xgb_features_6 + ['var38_modified']
xgb_features_6 = xgb_features_6 + ind_diff_list
#xgb_features_6 = xgb_features_6 + ['saldo_var42_var30_diff']
#xgb_features_6 = xgb_features_6 + ['saldo_var30_var5_diff']
#xgb_features_6 = xgb_features_6 + ['saldo_var5_saldo_medio_var5_ult3_diff']

#xgb_features_6 = xgb_features_6 + ['mean_1_min','mean_1_std']
#xgb_features_6 = xgb_features_6 + ['mean_2_min','mean_2_std']

#xgb_features_6 = xgb_features_6 + ['var15_last_digit']
#xgb_features_6 = xgb_features_6 + ['var38_mod_10']
#xgb_features_6 = xgb_features_6 + ['var38_mod_100']
#xgb_features_6 = xgb_features_6 + ['var38_decimal_count']
#xgb_features_6 = xgb_features_6 + ['saldo_var30_decimal_count']
#xgb_features_6 = xgb_features_6 + ['saldo_var30_last_digit']
#xgb_features_6 = xgb_features_6 + ['saldo_medio_var5_ult3_mod_3']

#xgb_features_6.remove('saldo_var42')
xgb_features_6.remove('var38')


params_6 = {'learning_rate': 0.01,
              'subsample': 0.5,
              'reg_alpha': 0.3,
#              'lambda': 0.99,
              'gamma': 0.5,
              'seed': 6,
              'colsample_bytree': 0.65,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 1.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
#              'objective': 'rank:pairwise',
#              'eval_metric':'auc',
              'eval_metric':'logloss',
              'max_depth': 8,
              'min_child_weight': 3,
              }
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_6 = fit_xgb_model(train,test,params_6,xgb_features_6,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
result_xgb_6 = fit_xgb_model(train,test,params_6,xgb_features_6,
                              num_rounds = num_rounds, num_boost_rounds = 900,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
xgb_features_7_high = []
xgb_features_7_high = xgb_features_7_high + base_features
#xgb_features_7_high = [x for x in xgb_features_7_high if x not in low_count_features]

xgb_features_7_high = xgb_features_7_high + ['sum_bool_cols']
xgb_features_7_high = xgb_features_7_high + ['zero_values']
#xgb_features_7_high = xgb_features_7_high + ['large_max_values']
#xgb_features_7_high = xgb_features_7_high + ['unique_values']
#xgb_features_7_high = xgb_features_7_high + ['negative_values']
#xgb_features_7_high = xgb_features_7_high + freq_1_columns

#xgb_features_7_high = xgb_features_7_high + [
#                                'var15_freq',
##                               'saldo_medio_var5_ult3_freq',
#                                'saldo_medio_var5_hace3_freq',
#                                'saldo_medio_var5_hace2_freq',
##                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
##                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
#                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
#                               'num_var22_ult3_freq','num_var22_ult1_freq',
#                               ]
xgb_features_7_high = xgb_features_7_high + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
#xgb_features_7_high = xgb_features_7_high + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
#xgb_features_7_high = xgb_features_7_high + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
#xgb_features_7_high = xgb_features_7_high + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

#xgb_features_7_high = xgb_features_7_high + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
#xgb_features_7_high = xgb_features_7_high + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
#xgb_features_7_high = xgb_features_7_high + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
#xgb_features_7_high = xgb_features_7_high + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']

#xgb_features_7_high = xgb_features_7_high + ['saldo_var30_num_digits']
#xgb_features_7_high = xgb_features_7_high + ['saldo_var30_decimal_count']
#xgb_features_7_high = xgb_features_7_high + ['saldo_var30_num_zeros']
#xgb_features_7_high = xgb_features_7_high + ['saldo_var30_num_consec_zeros']

#xgb_features_7_high = xgb_features_7_high + ['num_duplicate_rows']
#xgb_features_7_high = xgb_features_7_high + ['num_duplicate_bool_rows']

xgb_features_7_high = xgb_features_7_high + ['var38_modified']
xgb_features_7_high = xgb_features_7_high + ['saldo_var42_var30_diff']
xgb_features_7_high = xgb_features_7_high + ['saldo_var30_var5_diff']
xgb_features_7_high = xgb_features_7_high + ['saldo_var5_saldo_medio_var5_ult3_diff']

xgb_features_7_high = xgb_features_7_high + ind_diff_list
xgb_features_7_high = xgb_features_7_high + saldo_diff_list

#xgb_features_7_high = xgb_features_7_high + ['mean_1_min','mean_1_std']
#xgb_features_7_high = xgb_features_7_high + ['mean_2_min','mean_2_std']

#xgb_features_7_high = xgb_features_7_high + ['saldo_var30_decimal_count']
#xgb_features_7_high = xgb_features_7_high + ['saldo_var30_last_digit']
#xgb_features_7_high = xgb_features_7_high + ['saldo_medio_var5_ult3_mod_3']

#xgb_features_7_high.remove('saldo_var42')
xgb_features_7_high.remove('var38')
#xgb_features_7_high.remove('var38_modified')
#xgb_features_7_high.remove('saldo_medio_var5_hace2')
#xgb_features_7_high.remove('saldo_medio_var5_ult3')
#xgb_features_7_high.remove('saldo_medio_var5_ult1')
#xgb_features_7_high.remove('saldo_medio_var5_hace3')
#xgb_features_7_high.remove('num_var45_hace3')
#xgb_features_7_high.remove('num_var22_ult3')


params_7_high = {'learning_rate': 0.02,
              'subsample': 0.9,
              'reg_alpha': 0.5,
#              'lambda': 0.99,
              'gamma': 0.5,
              'seed': 7,
              'colsample_bytree': 0.5,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 1.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 4,
#              'min_child_weight': 3,
              }
num_rounds = 10000
num_boost_rounds = 150

train_v15_high = train.loc[train.var15 > 23]
test_v15_high = test.loc[test.var15 > 23]

#result_xgb_7_high = fit_xgb_model(train_v15_high,test_v15_high,params_7_high,xgb_features_7_high,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
#result_xgb_7_high = fit_xgb_model(train_v15_high,test_v15_high,params_7_high,xgb_features_7_high,
#                              num_rounds = num_rounds, num_boost_rounds = 900,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
xgb_features_7 = []
xgb_features_7 = xgb_features_7 + base_features
#xgb_features_7 = [x for x in xgb_features_7 if x not in low_count_features]

xgb_features_7 = xgb_features_7 + ['sum_bool_cols']
xgb_features_7 = xgb_features_7 + ['zero_values']
#xgb_features_7 = xgb_features_7 + ['large_max_values']
#xgb_features_7 = xgb_features_7 + ['unique_values']
#xgb_features_7 = xgb_features_7 + ['negative_values']
#xgb_features_7 = xgb_features_7 + freq_1_columns



#xgb_features_7 = xgb_features_7 + [
#                                'var15_freq',
##                               'saldo_medio_var5_ult3_freq',
#                                'saldo_medio_var5_hace3_freq',
#                                'saldo_medio_var5_hace2_freq',
##                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
##                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
#                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
#                               'num_var22_ult3_freq','num_var22_ult1_freq',
#                               ]
xgb_features_7 = xgb_features_7 + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
xgb_features_7 = xgb_features_7 + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
xgb_features_7 = xgb_features_7 + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
xgb_features_7 = xgb_features_7 + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

xgb_features_7 = xgb_features_7 + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
xgb_features_7 = xgb_features_7 + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
xgb_features_7 = xgb_features_7 + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
xgb_features_7 = xgb_features_7 + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']

#xgb_features_7 = xgb_features_7 + ['saldo_var30_num_digits']
#xgb_features_7 = xgb_features_7 + ['saldo_var30_decimal_count']
#xgb_features_7 = xgb_features_7 + ['saldo_var30_num_zeros']
#xgb_features_7 = xgb_features_7 + ['saldo_var30_num_consec_zeros']

xgb_features_7 = xgb_features_7 + ['num_duplicate_rows']
xgb_features_7 = xgb_features_7 + ['num_duplicate_bool_rows']

xgb_features_7 = xgb_features_7 + ['var38_modified']
#xgb_features_7 = xgb_features_7 + ['id']
xgb_features_7 = xgb_features_7 + ind_diff_list
xgb_features_7 = xgb_features_7 + saldo_diff_list
#xgb_features_7 = xgb_features_7 + ['saldo_var42_var30_diff']
#xgb_features_7 = xgb_features_7 + ['saldo_var30_var5_diff']
#xgb_features_7 = xgb_features_7 + ['saldo_var5_saldo_medio_var5_ult3_diff']

#xgb_features_7.remove('saldo_var42')
xgb_features_7.remove('var38')
#xgb_features_7.remove('var38_modified')
#xgb_features_7.remove('zero_values')
#xgb_features_7.remove('saldo_medio_var5_hace2')
#xgb_features_7.remove('saldo_medio_var5_ult3')
#xgb_features_7.remove('saldo_medio_var5_ult1')
#xgb_features_7.remove('saldo_medio_var5_hace3')
#xgb_features_7.remove('saldo_var42')
#xgb_features_7.remove('saldo_var30')
#xgb_features_7.remove('saldo_var5')
#xgb_features_7.remove('sum_bool_cols')
#xgb_features_7.remove('saldo_medio_var5_hace2')
#xgb_features_7.remove('saldo_medio_var5_ult3')
#xgb_features_7.remove('saldo_medio_var5_ult1')
#xgb_features_7.remove('saldo_medio_var5_hace3')
#xgb_features_7.remove('num_var45_hace3')
#xgb_features_7.remove('num_var22_ult3')


params_7 = {'learning_rate': 0.02,
              'subsample': 0.7,
#              'reg_alpha': 0.7,
#              'lambda': 0.3,
#              'gamma': 0.7,
              'seed': 7,
              'colsample_bytree': 0.6,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 0.9,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 5,
#              'min_child_weight': 2,
              }
num_rounds = 10000
num_boost_rounds = 150

train_v15_23 = train.loc[(train.var15 <= 23) | (train.target == 1)]
#train_v15_23 = train.loc[(train.var15 <= 23)]
test_v15_23 = test.loc[test.var15 <= 23]

#this is used to find early stopping rounds, then scale up when using all of dataset
#as number of rounds should be proportional to size of dataset

#result_xgb_7 = fit_xgb_model(train_v15_23,test_v15_23,params_7,xgb_features_7,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
#result_xgb_7 = fit_xgb_model(train_v15_23,test_v15_23,params_7,xgb_features_7,
#                              num_rounds = num_rounds, num_boost_rounds = 350,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
#test_result_xgb_7 = pd.merge(result_xgb_7[['id','pred']],test_v15_23[['id','target','zero_values','freq_1_mean','num_duplicate_bool_rows','num_duplicate_rows'] + base_features]
##test_result_xgb_7 = pd.merge(result_ens.loc[result_ens.var15 <= 23][['id','pred']],test_v15_23[['id','target','zero_values','freq_1_mean','num_duplicate_bool_rows','num_duplicate_rows'] + base_features]
##test_result_xgb_7 = pd.merge(result_xgb_7[['id','pred']],test_v15_23[ float_cols]
#             ,left_on = ['id'],right_on = ['id'],how='left')
#test_result_xgb_7['target'] = test_result_xgb_7['target'].astype(int)
#test_result_xgb_7['error'] = np.abs(test_result_xgb_7['target'] - test_result_xgb_7['pred'])
#print('ens auc',round(skm.roc_auc_score(test_result_xgb_7.target.values,test_result_xgb_7.pred.values),5))
#%%
#zero values 277,278
#var36
#%%
xgb_features_2 = []
xgb_features_2 = xgb_features_2 + base_features

xgb_features_2 = xgb_features_2 + ['sum_bool_cols']
#xgb_features_2 = xgb_features_2 + ['zero_values']
#xgb_features_2 = xgb_features_2 + ['large_max_values']
xgb_features_2 = xgb_features_2 + ['unique_values']
#xgb_features_2 = xgb_features_2 + ['negative_values']

#xgb_features_2 = xgb_features_2 + freq_1_columns
#xgb_features_2 = xgb_features_2 + freq_2_columns
#xgb_features_2 = xgb_features_2 + freq_3_columns
#xgb_features_2 = xgb_features_2 + freq_4_columns



xgb_features_2 = xgb_features_2 + [
                                'var15_freq',
                               'saldo_medio_var5_hace3_freq','saldo_medio_var5_hace3_freq',
#                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
#                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
#                               'num_var22_ult3_freq','num_var22_ult1_freq',
                               ]
#xgb_features_2 = xgb_features_2 + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
xgb_features_2 = xgb_features_2 + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
xgb_features_2 = xgb_features_2 + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
xgb_features_2 = xgb_features_2 + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

#xgb_features_2 = xgb_features_2 + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
#xgb_features_2 = xgb_features_2 + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
#xgb_features_2 = xgb_features_2 + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
#xgb_features_2 = xgb_features_2 + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']
#xgb_features_2 = xgb_features_2 + ['cumulative_4_mean','cumulative_4_std']


xgb_features_2 = xgb_features_2 + ['num_duplicate_rows']
xgb_features_2 = xgb_features_2 + ['num_duplicate_bool_rows']

xgb_features_2 = xgb_features_2 + ['var38_modified']
xgb_features_2 = xgb_features_2 + ['saldo_var42_var30_diff']
xgb_features_2 = xgb_features_2 + ['saldo_var30_var5_diff']
xgb_features_2 = xgb_features_2 + ['saldo_var5_saldo_medio_var5_ult3_diff']

#xgb_features_2 = xgb_features_2 + ['saldo_var30_num_digits']
xgb_features_2 = xgb_features_2 + ['saldo_var30_decimal_count']
#xgb_features_2 = xgb_features_2 + ['saldo_var30_num_zeros']
xgb_features_2 = xgb_features_2 + ['saldo_var30_num_consec_zeros']

xgb_features_2 = xgb_features_2 + ind_diff_list
xgb_features_2 = xgb_features_2 + saldo_diff_list

#xgb_features_2.remove('saldo_var42')
xgb_features_2.remove('var38')


params_2 = {'learning_rate': 0.01,
              'subsample': 0.9,
              'reg_alpha': 0.5,
#              'lambda': 0.99,
#              'gamma': 0.5,
              'seed': 2,
              'colsample_bytree': 0.4,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 0.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 3,
#              'min_child_weight': 2,
              }
num_rounds_2 = 10000
num_boost_rounds_2 = 150

train_rare = train.loc[train['freq_1_min'] <= 0.01]
test_rare = test.loc[test['freq_1_min'] <= 0.01]

#result_xgb_2 = fit_xgb_model(train_rare,test_rare,params_2,xgb_features_2,
#                                               num_rounds = num_rounds_2, num_boost_rounds = num_boost_rounds_2,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
result_xgb_2 = fit_xgb_model(train_rare,test_rare,params_2,xgb_features_2,
                              num_rounds = num_rounds_2, num_boost_rounds = 1150,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
xgb_features_2_common = []
xgb_features_2_common = xgb_features_2_common + base_features

xgb_features_2_common = xgb_features_2_common + ['sum_bool_cols']
#xgb_features_2_common = xgb_features_2_common + ['zero_values']
xgb_features_2_common = xgb_features_2_common + ['large_max_values']
xgb_features_2_common = xgb_features_2_common + ['unique_values']
xgb_features_2_common = xgb_features_2_common + ['negative_values']

xgb_features_2_common = xgb_features_2_common + freq_1_columns
xgb_features_2_common = xgb_features_2_common + freq_2_columns
xgb_features_2_common = xgb_features_2_common + freq_3_columns
#xgb_features_2_common = xgb_features_2_common + freq_4_columns



xgb_features_2_common = xgb_features_2_common + [
                                'var15_freq',
                               'saldo_medio_var5_hace3_freq','saldo_medio_var5_hace3_freq',
#                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
#                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
#                               'num_var22_ult3_freq','num_var22_ult1_freq',
                               ]
#xgb_features_2_common = xgb_features_2_common + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
xgb_features_2_common = xgb_features_2_common + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
xgb_features_2_common = xgb_features_2_common + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
xgb_features_2_common = xgb_features_2_common + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

#xgb_features_2_common = xgb_features_2_common + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
#xgb_features_2_common = xgb_features_2_common + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
#xgb_features_2_common = xgb_features_2_common + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
#xgb_features_2_common = xgb_features_2_common + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']
#xgb_features_2_common = xgb_features_2_common + ['cumulative_4_mean','cumulative_4_std']


xgb_features_2_common = xgb_features_2_common + ['num_duplicate_rows']
xgb_features_2_common = xgb_features_2_common + ['num_duplicate_bool_rows']

xgb_features_2_common = xgb_features_2_common + ['var38_modified']
xgb_features_2_common = xgb_features_2_common + ['saldo_var42_var30_diff']
xgb_features_2_common = xgb_features_2_common + ['saldo_var30_var5_diff']
xgb_features_2_common = xgb_features_2_common + ['saldo_var5_saldo_medio_var5_ult3_diff']

xgb_features_2_common = xgb_features_2_common + saldo_diff_list
#xgb_features_2_common = xgb_features_2_common + ind_diff_list

#xgb_features_2_common = xgb_features_2_common + ['saldo_var30_num_digits']
#xgb_features_2_common = xgb_features_2_common + ['saldo_var30_decimal_count']
#xgb_features_2_common = xgb_features_2_common + ['saldo_var30_num_zeros']
#xgb_features_2_common = xgb_features_2_common + ['saldo_var30_num_consec_zeros']

#xgb_features_2_common.remove('saldo_var42')
xgb_features_2_common.remove('var38')


params_2_common = {'learning_rate': 0.02,
              'subsample': 0.8,
              'reg_alpha': 0.9,
#              'lambda': 0.99,
              'gamma': 1.5,
              'seed': 2,
              'colsample_bytree': 0.75,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 1.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 3,
              'min_child_weight': 4,
              }
num_rounds_2 = 10000
num_boost_rounds_2 = 150

train_common = train.loc[train['freq_1_min'] > 0.01]
test_common = test.loc[test['freq_1_min'] > 0.01]

#result_xgb_2_common = fit_xgb_model(train_common,test_common,params_2_common,xgb_features_2_common,
#                                               num_rounds = num_rounds_2, num_boost_rounds = num_boost_rounds_2,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
result_xgb_2_common = fit_xgb_model(train_common,test_common,params_2_common,xgb_features_2_common,
                              num_rounds = num_rounds_2, num_boost_rounds = 950,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
xgb_features_5_ind_var5_0 = []
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + base_features
#xgb_features_5_ind_var5_0 = [x for x in xgb_features_5_ind_var5_0 if x not in low_count_features]


#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['sum_bool_cols']
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['zero_values']
#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['laxrge_max_values']
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['unique_values']
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['negative_values']

xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + freq_1_columns
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + freq_2_columns
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + freq_3_columns
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + freq_4_columns



xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + [
#                                'var15_freq',
                               'saldo_medio_var5_hace3_freq','saldo_medio_var5_hace3_freq',
#                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
#                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
#                               'num_var22_ult3_freq','num_var22_ult1_freq',
                               ]
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']
#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['cumulative_4_mean','cumulative_4_std']


xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['num_duplicate_rows']
xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['num_duplicate_bool_rows']

xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ind_diff_list

xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['var38_modified']
#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['saldo_var42_var30_diff']
#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['saldo_var30_var5_diff']
#xgb_features_5_ind_var5_0 = xgb_features_5_ind_var5_0 + ['saldo_var5_saldo_medio_var5_ult3_diff']

#xgb_features_5_ind_var5_0.remove('saldo_var42')
xgb_features_5_ind_var5_0.remove('var38')


params_5_ind_var5_0 = {'learning_rate': 0.01,
              'subsample': 0.98,
              'reg_alpha': 0.6,
#              'lambda': 0.99,
#              'gamma': 0.0,
              'seed': 5,
              'colsample_bytree': 0.9,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 1.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 2,
              'min_child_weight': 2,
              }
num_rounds_2 = 10000
num_boost_rounds_2 = 150

train_ind_var5_0_0 = train.loc[train['ind_var5_0'] == 0]
test_ind_var5_0_0 = test.loc[test['ind_var5_0'] == 0]

#result_xgb_5_ind_var5_0 = fit_xgb_model(train_ind_var5_0_0,test_ind_var5_0_0,params_5_ind_var5_0,xgb_features_5_ind_var5_0,
#                                               num_rounds = num_rounds_2, num_boost_rounds = num_boost_rounds_2,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
result_xgb_5_ind_var5_0 = fit_xgb_model(train_ind_var5_0_0,test_ind_var5_0_0,params_5_ind_var5_0,xgb_features_5_ind_var5_0,
                              num_rounds = num_rounds_2, num_boost_rounds = 750,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
#test_result_xgb_5 = pd.merge(result_xgb_5_ind_var5_0[['id','pred']],test_ind_var5_0_0[['id','target','zero_values','freq_1_mean','num_duplicate_bool_rows','num_duplicate_rows'] + base_features]
##test_result_xgb_5 = pd.merge(result_xgb_5[['id','pred']],test_v15_23[ float_cols]
#             ,left_on = ['id'],right_on = ['id'],how='left')
#test_result_xgb_5['target'] = test_result_xgb_5['target'].astype(int)
#test_result_xgb_5['error'] = np.abs(test_result_xgb_5['target'] - test_result_xgb_5['pred'])
#print('ens auc',round(skm.roc_auc_score(test_result_xgb_5.target.values,test_result_xgb_5.pred.values),5))

#%%
xgb_features_5_ind_var5_1 = []
xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + base_features
#xgb_features_5_ind_var5_1 = [x for x in xgb_features_5_ind_var5_1 if x not in low_count_features]

#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['var38','var15','saldo_var30','saldo_medio_var5_hace2',
#                               'saldo_medio_var5_ult3','num_var22_ult3','num_var45_hace3',
#                               'num_var45_hace2']

xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['sum_bool_cols']
xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['zero_values']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['laxrge_max_values']
xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['unique_values']
xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['negative_values']

#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + freq_1_columns
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + freq_2_columns
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + freq_3_columns
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + freq_4_columns



#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + [
##                                'var15_freq',
#                               'saldo_medio_var5_hace3_freq','saldo_medio_var5_hace3_freq',
##                               'num_var22_ult3_freq','saldo_medio_var5_hace3_freq',
##                               'var38_freq','saldo_var30_freq','num_var22_ult3_freq',
#                               'num_var45_hace3_freq','num_var45_hace2_freq','num_var45_ult3_freq',
##                               'num_var22_ult3_freq','num_var22_ult1_freq',
#                               ]
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['freq_1_min','freq_1_mean','freq_1_std','freq_1_max']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['freq_2_min','freq_2_mean','freq_2_std','freq_2_max']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['freq_3_min','freq_3_mean','freq_3_std','freq_3_max']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['freq_4_min','freq_4_mean','freq_4_std','freq_4_max']

#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['cumulative_1_min','cumulative_1_mean','cumulative_1_std','cumulative_1_max']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['cumulative_2_min','cumulative_2_mean','cumulative_2_std','cumulative_2_max']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['cumulative_3_min','cumulative_3_mean','cumulative_3_std','cumulative_3_max']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['cumulative_4_min','cumulative_4_mean','cumulative_4_std','cumulative_4_max']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['cumulative_4_mean','cumulative_4_std']


xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['num_duplicate_rows']
xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['num_duplicate_bool_rows']

xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ind_diff_list

xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['var38_modified']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['saldo_var42_var30_diff']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['saldo_var30_var5_diff']
#xgb_features_5_ind_var5_1 = xgb_features_5_ind_var5_1 + ['saldo_var5_saldo_medio_var5_ult3_diff']

#xgb_features_5_ind_var5_1.remove('saldo_var42')
xgb_features_5_ind_var5_1.remove('var38')


params_5_ind_var5_1 = {'learning_rate': 0.02,
              'subsample': 0.85,
              'reg_alpha': 0.5,
#              'lambda': 0.99,
              'gamma': 1.5,
              'seed': 5,
              'colsample_bytree': 0.6,
#              'colsample_bylevel': 0.3,
#              'max_delta_step': 1.5,
              'n_estimators': 100,
              'objective': 'binary:logistic',
              'eval_metric':'logloss',
              'max_depth': 5,
              'min_child_weight': 2,
              }
num_rounds_2 = 10000
num_boost_rounds_2 = 150

train_ind_var5_0_1 = train.loc[train['ind_var5_0'] == 1]
test_ind_var5_0_1 = test.loc[test['ind_var5_0'] == 1]

#result_xgb_5_ind_var5_1 = fit_xgb_model(train_ind_var5_0_1,test_ind_var5_0_1,params_5_ind_var5_1,xgb_features_5_ind_var5_1,
#                                               num_rounds = num_rounds_2, num_boost_rounds = num_boost_rounds_2,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 6)
result_xgb_5_ind_var5_1 = fit_xgb_model(train_ind_var5_0_1,test_ind_var5_0_1,params_5_ind_var5_1,xgb_features_5_ind_var5_1,
                              num_rounds = num_rounds_2, num_boost_rounds = 500,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 7)
#%%
#train_rare = train.loc[train['ind_var5_0_freq'] < 0.05]
#test_rare = train.loc[train['ind_var5_0_freq'] < 0.05]

#print('auc',round(skm.roc_auc_score(result_xgb_df.target.values,result_xgb_df.pred.values),5))
#%%
#if(is_sub_run):
#    result_xgb_df.reset_index('id',inplace=True)
#    result_xgb_2.reset_index('id',inplace=True)
#%%
#result_xgb_2['pred_2'] = result_xgb_2['pred']
#result_2_df = pd.merge(result_xgb_df,result_xgb_2[['id','pred_2']],
#              left_on = ['id'],
#              right_on = ['id'],how='left')
#result_2_df['pred_2'].fillna(-1,inplace=True)
#result_2_df['pred_2'] = result_2_df.apply(lambda row: row['pred'] if row['pred_2'] == -1 else row['pred_2'],axis=1)
#result_2_df['pred'] = result_2_df['pred_2']
#result_xgb_df.sort('id',inplace=True)
#result_2_df.sort('id',inplace=True)
#test.sort('id',inplace=True)
#if(not is_sub_run):
#    result_xgb_df.reset_index('id',inplace=True)
#    result_2_df.reset_index('id',inplace=True)
#%%
result_xgb_2_combo = pd.concat([result_xgb_2,result_xgb_2_common])
result_xgb_5_combo = pd.concat([result_xgb_5_ind_var5_0,result_xgb_5_ind_var5_1])
#result_xgb_7_combo = pd.concat([result_xgb_7,result_xgb_7_high])
#%%
#result_xgb_df.sort('id',inplace=True)
#result_xgb_2_combo.sort('id',inplace=True)
#test.sort('id',inplace=True)
#%%
#if(not is_sub_run):
#    result_xgb_df.reset_index('id',inplace=True)
#    result_xgb_2_combo.reset_index('id',inplace=True)
#%%

#saldo_var20 - no nonzero are target 1
#saldo_medio_var13_largo_hace2 - no nonzero are target 1
#saldo_medio_var13_largo_hace3 - no nonzero are target 1
#saldo_medio_var13_largo_ult1 - no nonzero are target 1
#saldo_medio_var13_largo_ult3 - no nonzero are target 1
#num_meses_var13_largo_ult3 - no nonzero are target 1
#ind_var20_0 - no nonzero are target 1
#ind_var20 - no nonzero are target 1
#ind_var33_0 - no nonzero are target 1
#ind_var33 - no nonzero are target 1
#num_var20_0 - no nonzero are target 1
#num_var20 - no nonzero are target 1
#num_var33_0 - no nonzero are target 1
#num_var33 - no nonzero are target 1
#saldo_var33 - no nonzero are target 1
#delta_imp_venta_var44_1y3 - no nonzero are target 1
#delta_num_venta_var44_1y3 - no nonzero are target 1
#imp_venta_var44_ult1 - no nonzero are target 1
#num_aport_var17_hace3 - no nonzero are target 1
#num_aport_var33_hace3 - no nonzero are target 1
#num_compra_var44_hace3 - no nonzero are target 1
#num_meses_var33_ult3 - no nonzero are target 1
#num_venta_var44_ult1 - no nonzero are target 1
#saldo_medio_var17_hace2 - no nonzero are target 1
#saldo_medio_var33_hace2 - no nonzero are target 1
#saldo_medio_var33_hace3 - no nonzero are target 1
#saldo_medio_var33_ult1 - no nonzero are target 1
#saldo_medio_var33_ult3 - no nonzero are target 1
#saldo_medio_var44_hace2 - no nonzero are target 1
#saldo_medio_var44_hace3 - no nonzero are target 1
#%%
#num_med_var45_ult3
#imp_op_var40_efect_ult1
#imp_op_var40_efect_ult3
#num_op_var40_efect_ult1
#num_op_var40_efect_ult3
#num_sal_var16_ult1

#num_med_var45_ult3
#num_med_var45_hace2
#imp_trans_var37_ult1

#num_var13_largo_0
#ind_var13_largo_0
#ind_var_20_0
#num_var20_0
#ind_var33_0
#num_var33_0
#num_var1_0
#saldo_medio_var44_hace3

#num_meses_var13_largo_ult3

#var36 - 0


#num_var13_0
#num_var26
#num_op_var40_ult3
#num_op_var39_ult1
#num_var30
#num_var40_0

#delta_imp_aport_var13_1y3 -1
#delta_num_aport_var13_1y3
#num_aport_var13_hace3
#deltas,cte

#num_var22_ult1
#num_var22_ult3

#num_meses_var12_ult3
#num_meses_var13_corto_ult3
#%%
#result_ens = 0.5*result_xgb_df + 0.5*result_2_df
#result_xgb_df['pred_mod'] = result_xgb_df['pred'] / result_xgb_df['pred'].mean()
#result_xgb_2_combo['pred_mod'] = result_xgb_2_combo['pred'] / result_xgb_2_combo['pred'].mean()
#result_xgb_3['pred_mod'] = result_xgb_3['pred'] / result_xgb_3['pred'].mean()
#result_xgb_4['pred_mod'] = result_xgb_4['pred'] / result_xgb_4['pred'].mean()
#result_xgb_5_combo['pred_mod'] = result_xgb_5_combo['pred'] / result_xgb_5_combo['pred'].mean()
#result_xgb_6['pred_mod'] = result_xgb_6['pred'] / result_xgb_6['pred'].mean()
#result_xgb_7_combo['pred_mod'] = result_xgb_7_combo['pred'] / result_xgb_7_combo['pred'].mean()



#result_ens = (1.0*result_xgb_df + 1.0*result_xgb_2_combo + 1.0*result_xgb_3 + 1.0*result_xgb_4
#            + 1.0*result_xgb_5_combo + 1.0*result_xgb_6 + 1.0*result_xgb_7_combo)
result_ens = result_xgb_df.copy()
#result_ens['pred'] = np.exp(((0.0*np.log(result_xgb_df['pred']) + 2.0*np.log(result_xgb_2_combo['pred']) + 0.0*np.log(result_xgb_3['pred']) + 0.0*np.log(result_xgb_4['pred'])
#            + 3.0*np.log(result_xgb_5_combo['pred']) + 2.0*np.log(result_xgb_6['pred']) + 0.0*np.log(result_xgb_7_combo['pred'])))/7)
#result_ens['pred'] = np.exp(((1.0*np.log(result_xgb_df['pred']) + 1.0*np.log(result_xgb_2_combo['pred']) + 1.0*np.log(result_xgb_3['pred']) + 1.0*np.log(result_xgb_4['pred'])
#            + 1.0*np.log(result_xgb_5_combo['pred']) + 1.0*np.log(result_xgb_6['pred']) + 1.0*np.log(result_xgb_7_combo['pred'])))/7)
result_ens['pred'] = np.exp(((1.0*np.log(result_xgb_df['pred']) + 1.0*np.log(result_xgb_2_combo['pred']) + 1.0*np.log(result_xgb_3['pred']) + 0.0*np.log(result_xgb_4['pred'])
            + 1.0*np.log(result_xgb_5_combo['pred']) + 1.0*np.log(result_xgb_6['pred']) ))/5)
#result_ens = ( 1.0*result_xgb_7_combo)
#result_ens = ( 1.0*result_xgb_4)
#result_ens = 1.0*result_xgb_5_combo
#result_ens = result_xgb_df
#result_ens = 0.5*result_xgb_df
#result_ens = result_xgb_6

#result_ens['pred'] = result_ens['pred_mod'] / 7.0
#result_ens['pred'] = result_ens['pred'] / 9.0
#result_ens['pred'] = result_ens['pred'] / 10.0
result_ens['id'] = result_xgb_df['id']
result_ens = result_ens[['id','pred']]

result_ens['pred1'] = result_xgb_df['pred']
result_ens['pred2'] = result_xgb_2_combo['pred']
result_ens['pred3'] = result_xgb_3['pred']
result_ens['pred4'] = result_xgb_4['pred']
result_ens['pred5'] = result_xgb_5_combo['pred']
result_ens['pred6'] = result_xgb_6['pred']
#result_ens['pred7'] = result_xgb_7_combo['pred']

result_ens['std'] = result_ens[['pred1','pred2','pred3','pred4','pred5','pred6']].std(axis=1)
result_ens['min'] = result_ens[['pred1','pred2','pred3','pred4','pred5','pred6']].min(axis=1)
result_ens['max'] = result_ens[['pred1','pred2','pred3','pred4','pred5','pred6']].max(axis=1)
result_ens['range'] = result_ens['max'] - result_ens['min']
result_ens['range_center'] = result_ens['range'] - result_ens['pred']
result_ens['range_norm'] = result_ens['range'] / result_ens['pred']
result_ens['pred_4_diff'] = result_ens['pred4'] - result_ens['pred']
result_ens['pred_4_ratio'] = result_ens['pred4'] / result_ens['pred']
#result_ens['pred'] = result_ens['pred'] + 0*(result_ens['std'] * result_ens['pred'])

#result_ens['pred'] = result_ens['pred'] / (result_ens['pred'].max() + 0.001)

result_ens = pd.merge(result_ens,test[['id','target','var15','saldo_var30','var38','saldo_medio_var5_hace3','num_med_var45_ult3','imp_trans_var37_ult1','ind_var5_0',
                                        'ind_var8_diff',
                                       'delta_imp_aport_var13_1y3','num_meses_var13_largo_ult3',
                                       'imp_op_var40_efect_ult3','num_sal_var16_ult1',
                                       'saldo_var33','saldo_var8'] +
             ['saldo_var20','saldo_medio_var13_largo_hace2','saldo_medio_var13_largo_ult3','ind_var20_0','ind_var33_0','num_var20_0','saldo_medio_var33_ult3','saldo_medio_var44_hace3']
             ],left_on = ['id'],right_on = ['id'],how='left')
#result_ens = result_ens[['pred']]



result_ens_var15_low_cond = result_ens['var15'] <= 22
result_ens_var15_23_cond = result_ens['var15'] == 23
result_ens_saldo_var33_nonzero = result_ens['saldo_var33'] > 0
result_ens_saldo_var20_nonzero = result_ens['saldo_var20'] > 0
result_ens_saldo_medio_var13_largo_hace2_nonzero = result_ens['saldo_medio_var13_largo_hace2'] > 0
result_ens_saldo_medio_var13_largo_ult3_nonzero = result_ens['saldo_medio_var13_largo_ult3'] > 0
result_ens_ind_var20_0_nonzero = result_ens['ind_var20_0'] > 0
result_ens_ind_var33_0_nonzero = result_ens['ind_var33_0'] > 0
result_ens_num_var20_0_nonzero = result_ens['num_var20_0'] > 0
result_ens_saldo_medio_var33_ult3_nonzero = result_ens['saldo_medio_var33_ult3'] > 0
result_ens_saldo_medio_var44_hace3_nonzero = result_ens['saldo_medio_var44_hace3'] > 0
result_ens_num_meses_var13_largo_ult3_nonzero = result_ens['num_meses_var13_largo_ult3'] != 0


result_ens_pred_above_0_8 = result_ens['pred'] >= 0.08

result_ens_range_center_low = result_ens['range_center'] <= -0.05
result_ens_range_norm_high = result_ens['range_norm'] >= 4.0

#result_ens_pred4_low = result_ens['pred_4_ratio'] <= 0.6

result_ens_ind_var5_0_zero = result_ens['ind_var5_0'] == 0
result_ens_delta_imp_aport_var13_1y3_neg1 = result_ens['delta_imp_aport_var13_1y3'] == -1

result_ens['pred'][result_ens_var15_low_cond] = result_ens['pred'][result_ens_var15_low_cond] / 3
result_ens['pred'][result_ens_saldo_var33_nonzero] = result_ens['pred'][result_ens_saldo_var33_nonzero] / 1.5
result_ens['pred'][result_ens_saldo_var20_nonzero] = result_ens['pred'][result_ens_saldo_var20_nonzero] / 1.25
result_ens['pred'][result_ens_saldo_medio_var13_largo_hace2_nonzero] = result_ens['pred'][result_ens_saldo_medio_var13_largo_hace2_nonzero] / 1.25
result_ens['pred'][result_ens_saldo_medio_var13_largo_ult3_nonzero] = result_ens['pred'][result_ens_saldo_medio_var13_largo_ult3_nonzero] / 1.25
result_ens['pred'][result_ens_ind_var20_0_nonzero] = result_ens['pred'][result_ens_ind_var20_0_nonzero] / 1.25
result_ens['pred'][result_ens_num_var20_0_nonzero] = result_ens['pred'][result_ens_num_var20_0_nonzero] / 1.25
result_ens['pred'][result_ens_saldo_medio_var33_ult3_nonzero] = result_ens['pred'][result_ens_saldo_medio_var33_ult3_nonzero] / 1.25
result_ens['pred'][result_ens_saldo_medio_var44_hace3_nonzero] = result_ens['pred'][result_ens_saldo_medio_var44_hace3_nonzero] / 1.5
result_ens['pred'][result_ens_num_meses_var13_largo_ult3_nonzero] = result_ens['pred'][result_ens_num_meses_var13_largo_ult3_nonzero] / 1.5

result_ens['pred'][result_ens_delta_imp_aport_var13_1y3_neg1] = result_ens['pred'][result_ens_delta_imp_aport_var13_1y3_neg1] / 1.2

#result_ens['pred'][result_ens_pred_above_0_8 & result_ens_var15_23_cond] = result_ens['pred'][result_ens_pred_above_0_8 & result_ens_var15_23_cond] / 1.3


#result_ens['pred'][result_ens_pred4_low] = result_ens['pred'][result_ens_pred4_low] * result_ens['pred_4_ratio'][result_ens_pred4_low]

#result_ens['pred'][result_ens_ind_var5_0_zero] = result_ens['pred'][result_ens_ind_var5_0_zero] * 1.0

#result_ens['pred'][result_ens_range_center_low] = result_ens['pred'][result_ens_range_center_low] * 1.2
#result_ens['pred'][result_ens_range_norm_high] = result_ens['pred'][result_ens_range_norm_high] * 1.2

max_pred = result_ens['pred'].max()
if (max_pred >=1.0):
    result_ens['pred'] = result_ens['pred'] / (max_pred + 0.001)

if(is_sub_run):
    print('is a submission run')
else:

#    result_ens = pd.merge(result_ens,test[['id','target']],left_on = ['id'],right_on = ['id'],how='left')
#    result_ens = pd.merge(result_ens,test[['id','target','below_min_values','above_max_values']],left_on = ['id'],right_on = ['id'],how='left')
#    result_ens = pd.merge(result_ens,test[['id','target','var15','saldo_var33','saldo_var8']],left_on = ['id'],right_on = ['id'],how='left')
    result_ens['target'] = result_ens['target'].astype(int)
    result_ens['error'] = np.abs(result_ens['target'] - result_ens['pred'])
    print('ens auc',round(skm.roc_auc_score(result_ens.target.values,result_ens.pred.values),5))

if(is_sub_run):
#    submission = result_xgb_df.copy()
    submission = result_ens.copy()
    submission = submission[['id','pred']]
#    submission.reset_index('id',inplace=True)
#    submission = submission[['id','pred']]
    submission['id'] = submission['id'].astype(int)
    submission = submission.rename(columns={'id':'ID','pred':'TARGET'})
    submission.to_csv('santander.csv', index=False)
    print('xgb submission created')
#%%
#result_ens = 0.5*result_xgb_df + 0.3*result_xgb_2
#
#if(is_sub_run):
#    print('is a submission run')
#else:
#    result_ens = result_ens[['pred']]
#    result_ens['id'] = result_xgb_df['id']
#    result_ens = pd.merge(result_ens,test[['id','target']],left_on = ['id'],right_on = ['id'],how='left')
#    result_ens['target'] = result_ens['target'].astype(int)
#    print('ens auc',round(skm.roc_auc_score(result_ens.target.values,result_ens.pred.values),5))
#%%
if(not is_sub_run):
#    wrong_df = result_xgb_df.copy()
    wrong_df = result_ens.copy()
    wrong_df['error'] = np.abs(wrong_df['target'] - wrong_df['pred'])
    wrong_df_2 = pd.merge(wrong_df[['id','error','pred']],combined,left_on = ['id'], right_on = ['id'], how='left')
    wrong_df = pd.merge(wrong_df,combined[['id','var38','freq_1_mean','freq_1_std','freq_1_min',
                                           'freq_4_mean','freq_4_std','freq_4_min',
                                           'negative_values',
                                           'var15_freq','ind_var5_0_freq',
                                           'saldo_medio_var5_hace2','saldo_medio_var5_ult3','saldo_var30',
                                           'num_var22_ult3','saldo_var42','num_var45_hace3','zero_values',
                                           'num_duplicate_rows','num_duplicate_bool_rows']],
                  left_on = ['id'],
                  right_on = ['id'],how='left')
    bad_df = wrong_df[wrong_df['error'] >= 0.5]
#    wrong_rare = wrong_df.loc[wrong_df['ind_var5_0_freq'] < 0.05]
#    wrong_common = wrong_df.loc[wrong_df['ind_var5_0_freq'] > 0.05]
    wrong_rare = wrong_df.loc[wrong_df['var15'] <= 23]
    wrong_common = wrong_df.loc[wrong_df['var15'] > 23]
#    wrong_rare = wrong_df.loc[wrong_df['freq_1_min'] <= 0.01]
#    wrong_common = wrong_df.loc[wrong_df['freq_1_min'] > 0.01]
    print('wrong rare auc',round(skm.roc_auc_score(wrong_rare.target.values,wrong_rare.pred.values),5))
    print('wrong common auc',round(skm.roc_auc_score(wrong_common.target.values,wrong_common.pred.values),5))

    bad_df2 = wrong_df_2[wrong_df_2['error'] >= 0.75]
    bad_df2 = bad_df2.loc[bad_df2['target'] == 1]
    bad_df2 = bad_df2[['id','target','error','var38','var15','saldo_medio_var5_hace3','saldo_medio_var5_ult3','saldo_var5','saldo_var30','saldo_var37'] + freq_1_columns + freq_2_columns + freq_3_columns + freq_4_columns]



    train_subset = train[['id','target','var38','var15','freq_1_mean','freq_1_std','freq_1_min',
                          'freq_4_mean','freq_4_std','freq_4_min',
                            'var15_freq','negative_values',
                                           'saldo_medio_var5_hace2','saldo_medio_var5_ult3','saldo_var30',
                                           'num_var22_ult3','saldo_var42','num_var45_hace3','zero_values',
                                           'num_duplicate_rows','num_duplicate_bool_rows']]
    train_subset_float_cols = train[float_cols]
    train_ind_var5_0_0 = train.loc[train['ind_var5_0'] == 0][['id','target'] + base_features]
    train_target1_base = train.loc[train.target == 1][['id','target'] + base_features]
    train_target1_float_cols = train.loc[train.target == 1][['id','target','var38','var15','saldo_medio_var5_hace3','saldo_medio_var5_ult3','saldo_var5','saldo_var30','saldo_var37'] + freq_1_columns + freq_2_columns + freq_3_columns + freq_4_columns]
    train_target0_float_cols = train.loc[train.target == 0][['id','target','var38','var15','saldo_medio_var5_hace3','saldo_medio_var5_ult3','saldo_var5','saldo_var30','saldo_var37'] + freq_1_columns + freq_2_columns + freq_3_columns + freq_4_columns]
#%%
#ind_var5_0 - when this is not 0, the model fails badly for target 0
#var15
#saldo_medio_var5_ult3,saldo_medio_var5_hace3
#%%
#saldo_var30_dict = combined.saldo_var30.value_counts().to_dict()
#for val in combined.saldo_var30.unique():
#    if (saldo_var30_dict[val] <= 10):
#        continue
##    print(val)
#    print_value_counts_spec(combined,'saldo_var30',val)
#%%
#tic = timeit.default_timer()
#train_data = train[xgb_features].values
#train_target = train['target'].astype(int).values
#test_data = test[xgb_features].values
#
#extc = ExtraTreesClassifier(
#                            n_estimators=500,
#                            max_features = None,
#                            criterion= 'entropy',
#                            min_samples_split= 1,
#                            max_depth= None,
#                            min_samples_leaf= 100,
#                            random_state = 1,
#                            n_jobs = 2
#                            )
#extc = extc.fit(train_data, train_target)
#
#y_pred = extc.predict_proba(test_data)
#columns = ['pred']
#result_extc_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred[:,1])
##result_extc_df.replace([0],[0.00001],inplace=True)
##result_extc_df = norm_rows(result_extc_df)
#result_extc_df['id'] = result_extc_df.index
#if(is_sub_run):
#    print('is a sub run')
#else:
#    result_extc_df = pd.merge(result_extc_df,test[['id','target']],left_on = ['id'],
#                           right_on = ['id'],how='left')
#    result_extc_df.set_index('id',drop=False,inplace=True)
#    result_extc_df['target'] = result_extc_df['target'].astype(int)
#    print('auc',round(skm.roc_auc_score(result_extc_df.target.values,result_extc_df.pred.values),5))
#
#toc=timeit.default_timer()
#print('extc Time',toc - tic)

#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)