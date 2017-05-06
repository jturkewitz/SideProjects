#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:20:14 2016

@author: Jared
"""
## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import random
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import MaxNLocator
import pylab as p


import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import cross_validation
    import xgboost as xgb

#import xgboost as xgb
import operator
from sklearn.neighbors import KNeighborsClassifier
import timeit
import scipy.stats as stats


import subprocess
import theano
#theano.config.openmp = True
#theano.config.cxx = '/usr/local/opt/llvm/bin/clang++'
#OMP_NUM_THREADS=2
#import tensorflow as tf

#tf.python.control_flow_ops = tf

from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
#from keras.optimizers import Adadelta, Adam, rmsprop, SGD, RMSprop
import keras.optimizers
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.regularizers import l2, activity_l2
import keras.regularizers
from keras.callbacks import EarlyStopping


tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'
#%%
## Batch generators ###########################################################

def batch_generator(X, y, batch_size, shuffle):
#    np.random.seed(7)
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0
#%%
#random_seed = 5
#random.seed(random_seed)
#np.random.seed(random_seed)

tic=timeit.default_timer()
combined = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/combined_ints.csv', header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)
combined['loss_orig'] = combined['loss']


c1_to_72 = []
for i in range(1,73):
    c1_to_72.append('c'+str(i))

combined['sum_c1to72'] = combined[c1_to_72].sum(axis=1)
## read data
#%%
#%%
#is_sub_run = False
is_sub_run = True
if (is_sub_run):
    train = combined.loc[combined['loss'] != -1 ]
    test = combined.loc[combined['loss'] == -1 ]
else:
    train = combined.loc[(combined['loss'] != -1) & (combined['id'] > 200000)]
    test = combined.loc[(combined['loss'] != -1) & (combined['id'] <= 200000)]
#    train = train.sample(frac=0.2,random_state = 3)
#%%
## set test loss to NaN

## response and IDs
y = train['loss_orig'].values
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
combined_2 = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []
#cat_nums = ['c90','c91','c92','c101','c103','c104',
#            'c105','c106','c107','c111','c113','c114','c115','c116']
cat_nums = []
f_cat = [f for f in combined_2.columns if f.startswith('c') and f[1].isdigit() and f not in cat_nums]
for f in f_cat:
    if combined_2[f].nunique() == 2:
        dummy = combined_2[f].astype(int).to_frame()
    else:
        dummy = pd.get_dummies(combined_2[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in combined_2.columns if f.startswith('cont')]
f_num.append('sum_c1to72')
f_num += cat_nums

scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(combined_2[f_num]))
sparse_data.append(tmp)

#del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

#del(xtr_te, sparse_data, tmp)


#%%
tic=timeit.default_timer()

np.random.seed(123)
#np.random.seed(126)

#np.random.seed(7)
#random.seed(7)
## neural net
w_reg_weight_l1 = 0.002
w_reg_weight_l2 = 0.001
init_string = 'glorot_normal'
#act_reg_weight_l1 = 0.0001
#act_reg_weight_l2 = 0.0001
def nn_model(xtrain = xtrain):
    model = Sequential()
    model.add(Dense(200, input_dim = xtrain.shape[1],
                    W_regularizer=keras.regularizers.WeightRegularizer(l1 = w_reg_weight_l1 , l2 = w_reg_weight_l2 ),
#                    activity_regularizer = keras.regularizers.ActivityRegularizer(l1=0.0002, l2=0.0),
                    init = init_string))

#    model.add(PReLU())

    model.add(ELU())
#    model.add(LeakyReLU())
#    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(100,
                    W_regularizer=keras.regularizers.WeightRegularizer(l1 = w_reg_weight_l1 , l2 = w_reg_weight_l2 ),
                    init = init_string))

    model.add(ELU())
#    model.add(LeakyReLU())
#    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

#    model.add(Dense(50,
##                    W_regularizer=l2(w_reg_weight_l2),
#                    init = init_string))
#    model.add(ELU())
#    model.add(LeakyReLU())
#    model.add(BatchNormalization())
#    model.add(Dropout(0.1))


    model.add(Dense(1,
#                    W_regularizer=l2(w_reg_weight_l2),
                    init = init_string))
#    model.add(PReLU())
#    model.add(BatchNormalization())
#    model.add(LeakyReLU())

#    model.add(Dense(1, W_regularizer=l2(w_reg_weight), init = 'he_normal'))

#    model.compile(loss = 'mae', optimizer = 'RMSprop')
#    model.compile(loss = 'mae', optimizer = 'adadelta')
#    model.compile(loss = 'mae', optimizer = 'rmsprop')
    sgd = keras.optimizers.SGD(lr=0.005, decay=1e-3, momentum=0.2, nesterov=False)
    model.compile(loss='mae', optimizer=sgd)
#    rms_prop = keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
#    model.compile(loss='mae', optimizer=rms_prop)
    return(model)





#xtrain = xtr_te[:ntrain, :]
#xtest = xtr_te[ntrain:, :]
#xtr, xwatch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
#use_early_stopping = True
use_early_stopping = False
use_log_transform = False
log_const = 0
if (use_early_stopping):
    xtr_len = int(ntrain * 0.8)
    xtr = xtrain[:xtr_len, :]
    ytr = y[:xtr_len]

    xwatch_2 = xtrain[xtr_len:, :]
    ywatch_2 = y[xtr_len:]
    ids_train = id_train[:xtr_len]

#    xtr = xtrain
#    ytr = y
#    ids_train = id_train

#    xwatch_2 = xtest
#    ywatch_2 = test.loss.values

    nepochs = 200
else:
#    xtr = xtrain[:xtr_len, :]
#    ytr = y[:xtr_len]
    xtr = xtrain
    ytr = y
    ids_train = id_train
#    nepochs = 75
#    nepochs = 50
    nepochs = 450
#    nepochs = 80

#xte = xtrain[inTe]
#yte = y[inTe]
#pred = np.zeros(xte.shape[0])


early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])


## cv-folds
nfolds = 5
#nfolds = 1
if nfolds > 1:
    folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)
else:
    folds = [(slice(None, None),slice(None,None))]
## train models
i = 0
nbags = 5
#nbags = 1
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

scores = []
for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        print(i,j)
        model = nn_model(xtr)
    #    fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
    #                              nb_epoch = nepochs,
    #                              samples_per_epoch = xtr.shape[0],
    #                              verbose = 2)
        batches = 600
#        batches = 1200
#        batches = 150
    #    batches = 75
    #    batches = 64
        if use_early_stopping:
            fit = model.fit_generator(generator = batch_generator(xtr, ytr, batches, True),
                                      validation_data = batch_generator(xwatch_2, ywatch_2, xwatch_2.shape[0], False),
                                      nb_val_samples = xwatch_2.shape[0],
                                      callbacks=[early_stopping],
                                      nb_epoch = nepochs,
                                      samples_per_epoch = xtr.shape[0],
                                      verbose = 2)
        else:
            fit = model.fit_generator(generator = batch_generator(xtr, ytr, batches, True),
                                      nb_epoch = nepochs,
                                      samples_per_epoch = xtr.shape[0],
                                      verbose = 2)

        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0]
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0]
    #    pred_test += model.predict_generator(generator = batch_generatorp(xtest, 1000, False), val_samples = xtest.shape[0])[:,0]
    pred /= nbags
    pred_oob[inTe] = pred
    scores.append(mean_absolute_error(yte, pred))
    i += 1
    print('Fold ', i, '- MAE:', mean_absolute_error(yte, pred))
#
#print('Total - MAE:', mean_absolute_error(y, pred_oob))

## train predictions
result_nn_train = pd.DataFrame({'id': id_train, 'pred': pred_oob})
result_nn_train.to_csv('preds_oob.csv', index = False)
## test predictions

pred_test /= (nfolds*nbags)

result_nn_test = pd.DataFrame({'id': id_test, 'pred': pred_test})


if(is_sub_run):
    print('creating xgb output')
else:
    result_nn_test_2 = pd.merge(result_nn_test,test[['id','loss']],left_on = ['id'],
                           right_on = ['id'],how='left')
    if use_log_transform:
        result_nn_test_2['loss'] = np.exp(result_nn_test_2['loss']) - log_const
    else:
        pass

    result_nn_test_2['error_0'] = result_nn_test_2['loss'] - result_nn_test_2['pred']
    result_nn_test_2['error'] = np.abs(result_nn_test_2['loss'] - result_nn_test_2['pred'])
    print('mae',round(result_nn_test_2['error'].mean(),5))
toc=timeit.default_timer()
print('Training Time',toc - tic)
#%%
#result_nn_test_2.to_csv('train_keras_6_1137.csv', index = False)
#%%
#res_1 = pd.read_csv('Train_Results_Id_2e5/train_nn_oob_1.csv',header=0)
#res_2 = pd.read_csv('Train_Results_Id_2e5/train_nn_oob_2.csv',header=0)
#res_3 = pd.read_csv('Train_Results_Id_2e5/train_nn_oob_3.csv',header=0)
#res_1 = pd.read_csv('Train_Results_Id_2e5/train_nn_oob_4.csv',header=0)
#
#result_1 = pd.merge(res_1,train[['id','loss']],left_on = ['id'],
#                       right_on = ['id'],how='left')
#
#result_1['error_0'] = result_1['loss'] - result_1['pred']
#result_1['error'] = np.abs(result_1['loss'] - result_1['pred'])
#print('mae',round(result_1['error'].mean(),5))
#%%
#%%
#sns.distplot(np.log(train['loss'] + 1),kde=False,fit=stats.norm)

#data = np.log(train['loss'] + 1)
## best fit of data
#(mu, sigma) = stats.norm.fit(data)
#
## the histogram of the data
#n, bins, patches = plt.hist(data, 100, normed=1, facecolor='green')
#
## add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)
#l = plt.plot(bins, y, 'r--', linewidth=2)
#
##plot
#plt.xlabel('ln(loss + 200)')
#plt.ylabel('Normalized Count')
##plt.yscale('log')
#plt.ylim(1e-5,0.6)
#plt.xlim(4,10)
#plt.title(r'$\mathrm{\ln(loss + 200),}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
#plt.grid(True)
#
#plt.show()

#plt.yscale('log')
#%%
if(is_sub_run):
    submission_nn = result_nn_test.copy()
    submission_nn.reset_index('id',inplace=True)
    submission_nn.rename(columns = {'pred':'loss'},inplace=True)
    submission_nn = submission_nn[['id','loss']]
    print('nn submission created')
    submission_nn.to_csv('test_keras_sub3.csv', index = False)
#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)