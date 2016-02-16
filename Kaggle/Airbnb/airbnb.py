# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 02:49:17 2015

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
import datetime as datetime
import os
import xgboost as xgb
import operator
import rank_metrics as rm

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
tic=timeit.default_timer()
age_gender_bkts = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Airbnb/age_gender_bkts.csv', header=0)
countries = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Airbnb/countries.csv', header=0)
#%%
sessions = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Airbnb/sessions_new.csv', header=0)
#%%
test_users = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Airbnb/test_users_new.csv', header=0)
train_users = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Airbnb/train_users_2.csv', header=0)
toc=timeit.default_timer()
print('Loading time',toc-tic)
#%%
#is_sub_run = False
is_sub_run = True
#%%
tic=timeit.default_timer()
test_users['country_destination'] = 'dummy'
#if(is_sub_run):
#    combined = pd.concat([train_users, test_users], axis=0,ignore_index=True)
#else:
#    combined = train_users
combined = pd.concat([train_users, test_users], axis=0,ignore_index=True)
user_id_dict = dict(zip(combined.id, combined.index))
user_id_rev_dict = {v: k for k, v in user_id_dict.items()}
combined['user_id'] = combined['id'].map(user_id_dict)
combined['date_account_created'] = pd.to_datetime(combined['date_account_created'])

combined['timestamp_first_active'] = pd.to_datetime(combined['timestamp_first_active'],
                                                        format='%Y%m%d%H%M%S')
#combined = combined.drop('date_first_booking',axis=1)
combined['date_first_booking'] = combined['date_first_booking'].fillna('2050-01-01')
combined['date_first_booking'] = pd.to_datetime(combined['date_first_booking'])

combined['time_to_first_booking'] = combined['date_first_booking'] - combined['date_account_created']
combined['time_to_first_booking'] = combined['time_to_first_booking'].map(lambda x: x.astype('timedelta64[D]').astype(int))

combined = combined.drop('date_first_booking',axis=1)

#%%
#dangerous, testing reason for different distribution
#long_gap = combined.time_to_first_booking >= 180
#no_dummies = combined.country_destination != 'dummy'
#
#combined['country_destination'][long_gap & no_dummies] = 'NDF'

toc=timeit.default_timer()
print('Date times time',toc-tic)
#%%
#combined_reduced = combined.loc[(combined.country_destination != 'NDF') & (combined.country_destination != 'dummy')]
#combined_far = combined_reduced.loc[combined_reduced.time_to_first_booking >= 180]
#%%
tic=timeit.default_timer()

cond_age_null = combined['age'].isnull()
combined['age'][cond_age_null] = -1
bins = [-1.0,0,20,25,30,40,50,60,75,100,150,1000000000]
combined['age_binned'] = np.digitize(combined['age'], bins, right=True)

cond_aff_tracked_null = combined['first_affiliate_tracked'].isnull()
combined['first_affiliate_tracked'][cond_aff_tracked_null] = 'unknown'

combined['timestamp_hour'] = combined['timestamp_first_active'].map(lambda x: x.hour)
combined['timestamp_minute'] = combined['timestamp_first_active'].map(lambda x: x.minute)
combined['timestamp_second'] = combined['timestamp_first_active'].map(lambda x: x.second)
combined['timestamp_year'] = combined['timestamp_first_active'].map(lambda x: x.year)
combined['timestamp_month'] = combined['timestamp_first_active'].map(lambda x: x.month)
combined['timestamp_day'] = combined['timestamp_first_active'].map(lambda x: x.day)
combined['timestamp_weekday'] = combined['timestamp_first_active'].map(lambda x: x.weekday())
#%%
beginning_date = pd.to_datetime('2010-01-01 00:00:00')
combined['timestamp_dateint'] = combined['timestamp_first_active'].map(lambda x: (x - beginning_date).days)
bins = [0,725,850,1000,1150,1300,1450,1600,1750]
combined['timestamp_dateint_binned'] = np.digitize(combined['timestamp_dateint'], bins, right=True)

dateint_group = combined.groupby('timestamp_dateint')
num_sameday_users = dateint_group['user_id'].count().to_dict()
combined['num_sameday_users'] = combined['timestamp_dateint'].map(num_sameday_users)

bins = [-1,0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,900,1000,1200]
combined['num_sameday_users_binned'] = np.digitize(combined['num_sameday_users'], bins, right=True)
#%%

combined['gender_unknown'] = combined['gender'].map(lambda x: 1 if x == '-unknown-' else 0)
combined['age_unknown'] = combined['age'].map(lambda x: 1 if x == -1 else 0)
combined['first_affiliate_tracked_untracked'] = combined['first_affiliate_tracked'].map(lambda x: 1 if x == 'untracked' else 0)
combined['first_device_type_unknown'] = combined['first_device_type'].map(lambda x: 1 if x == 'Other/Unknown' else 0)
combined['first_browser_unknown'] = combined['first_browser'].map(lambda x: 1 if x == '-unknown-' else 0)

combined['unknowns_summed'] = (combined['gender_unknown'] + combined['age_unknown'] +
                               combined['first_affiliate_tracked_untracked'] +
                               combined['first_device_type_unknown'] + combined['first_browser_unknown'])

#TODO testing dangerous
combined = combined.loc[combined['timestamp_year'] >= 2012]

def convert_strings_to_ints(input_df,col_name,output_col_name):
    labels, levels = pd.factorize(input_df[col_name])
    input_df[output_col_name] = labels
    output_dict = dict(zip(input_df[col_name],input_df[output_col_name]))
    return (output_dict,input_df)
(gender_dict,combined) = convert_strings_to_ints(combined,'gender','gender_hash')
(signup_method_dict,combined) = convert_strings_to_ints(combined,'signup_method','signup_method_hash')
(language_method_dict,combined) = convert_strings_to_ints(combined,'language','language_hash')
(affiliate_channel_dict,combined) = convert_strings_to_ints(combined,'affiliate_channel','affiliate_channel_hash')
(affiliate_provider_dict,combined) = convert_strings_to_ints(combined,'affiliate_provider','affiliate_provider_hash')
(first_affiliate_tracked_dict,combined) = convert_strings_to_ints(combined,'first_affiliate_tracked','first_affiliate_tracked_hash')
(signup_app_dict,combined) = convert_strings_to_ints(combined,'signup_app','signup_app_hash')
(first_device_type_dict,combined) = convert_strings_to_ints(combined,'first_device_type','first_device_type_hash')
(first_browser_dict,combined) = convert_strings_to_ints(combined,'first_browser','first_browser_hash')
(country_destination_dict,combined) = convert_strings_to_ints(combined,'country_destination','country_destination_hash')

toc=timeit.default_timer()
print('Converting strings time',toc-tic)
#%%
combined['is_basic'] = combined['signup_method'].map(lambda x: 1 if x == 'basic' else 0)
combined['is_google'] = combined['signup_method'].map(lambda x: 1 if x == 'google' else 0)
is_age_known = combined['age_unknown'] == 0
is_basic = combined['is_basic'] == 1
is_google = combined['is_google'] == 1
combined['basic_with_age'] = 0
combined['google_with_age'] = 0
combined['basic_with_age'][is_age_known & is_basic] = 1
combined['google_with_age'][is_age_known & is_google] = 1
#%%
combined['timestamp_shifted'] = combined['timestamp_first_active'].shift(-1)
combined['timestamp_shifted'].fillna(pd.to_datetime('2014-09-30 23:59:41'),inplace=True)
combined['time_to_next_user'] = combined['timestamp_shifted'] - combined['timestamp_first_active']
combined['time_to_next_user'] = combined['time_to_next_user'].map(lambda x: x.astype('timedelta64[s]').astype(int))
bins = [-1.0,0,60,120,180,300,600,1000,1500,2000,3000,5000,10000,20000,1000000000000]
combined['time_to_next_user_binned'] = np.digitize(combined['time_to_next_user'], bins, right=True)
#%%
tic=timeit.default_timer()
users = set(combined['user_id'].unique())
sessions.rename(columns={'action':'action_done'},inplace=True)
sessions['id'] = sessions['user_id']
sessions['user_id'] = sessions['user_id'].map(user_id_dict)
sessions['has_user'] = sessions['user_id'].map(lambda x: x in users)
sessions = sessions.loc[sessions['has_user']]
sessions.drop('has_user',axis=1,inplace=True)

sessions['device_type'] = sessions['device_type'].map(lambda x: x.replace(" ", "_"))
sessions_users = set(sessions['user_id'].unique())
combined['has_session'] = combined['user_id'].map(lambda x: x in sessions_users).astype(int)

year_2014 = combined['timestamp_year'] == 2014
has_no_session = combined['has_session'] == 0
combined['no_session_2014'] = 0
combined['no_session_2014'][has_no_session & year_2014] = 1
sessions['secs_elapsed'].fillna(-1,inplace=True)
sessions.fillna('null_value',inplace=True)

bins = [-1.0,0,8,20,50,100,200,1000,10000,100000,1000000000000]
sessions['secs_elapsed_binned'] = np.digitize(sessions['secs_elapsed'], bins, right=True)

sessions = pd.merge(sessions,combined[['user_id','country_destination']],on='user_id',how='left')
sessions['action_set'] = sessions['action_done'] + sessions['action_detail'] + sessions['action_type']

sessions_old = sessions.copy()
toc=timeit.default_timer()
print('Reducing sessions size time',toc-tic)
#%%
#null_users = set(sessions_null_secs['user_id'].unique())
#sessions_null_secs_all = sessions.copy()
#sessions_null_secs_all['has_user'] = sessions_null_secs_all['user_id'].map(lambda x: x in null_users)
#sessions_null_secs_all = sessions_null_secs_all.loc[sessions_null_secs_all['has_user']]
#bins = [-1.0,0,4,10,20,30,50,100,200,500,1000,10000,100000,1000000000000]
#sessions_old['secs_elapsed_binned'] = np.digitize(sessions_old['secs_elapsed'], bins, right=True)
#%%
combined_age_null = combined.loc[combined.age == -1]
combined_age_notnull = combined.loc[combined.age != -1]
combined_age_notnull_signup_not_fb = combined.loc[(combined.age != -1) & (combined.signup_method != 'facebook')]
#%%
#combined_age_notnull_signup_not_fb_2014 = combined_2014.loc[(combined_2014.age != -1)
#                                            & (combined_2014.timestamp_month >= 1)]
#%%
tic=timeit.default_timer()
sessions_first = sessions.drop_duplicates(subset = 'user_id')
sessions_last = sessions.sort_index(ascending=False).drop_duplicates(subset = 'user_id')

sessions_first.rename(columns={'action_done':'first_action_done','action_type':'first_action_type',
                               'action_detail':'first_action_detail','device_type':'first_device_type',
                               'secs_elapsed_binned':'first_secs_elapsed_binned',
                               'action_set':'first_action_set'},inplace=True)
sessions_last.rename(columns={'action_done':'last_action_done','action_type':'last_action_type',
                               'action_detail':'last_action_detail','device_type':'last_device_type',
                               'secs_elapsed_binned':'last_secs_elapsed_binned',
                               'action_set':'last_action_set'},inplace=True)
toc=timeit.default_timer()
print('Sessions first time',toc-tic)
#%%
tic=timeit.default_timer()
def add_feature(input_df,feature,column_name):
    feature_name = column_name + '_' + feature
    input_df[feature_name] = input_df[column_name].map(lambda x: 1 if x == feature else 0)
    return input_df
#action_list = ['change_currency','decision_tree','recent_reservation','multi',
#               'faq_experiment_ids','unavailabilities','active','other_hosting_reviews',
#               'login','my','track_page_view','social_connections','15','collections',
#               'complete_status','edit_verification',
#               '12','jumio_redirect','jumio_token','login_modal','search','pending','delete',
#               'connect','signup_modal','10','this_hosting_reviews','receipt',
#               'request_new_confirm_email','reviews_new','ajax_photo_widget_form_iframe',
#               'privacy','countries','status','authorize',
#               'languages_multiselect','image_order',
#               'ajax_google_translate_description','ajax_google_translate',
#               'ajax_google_translate_reviews','update_cached',
#               'itinerary','add_guests','profile_pic','apply_reservation','signature',
#               'glob','review_page','host_summary','tell_a_friend']

action_done_list = ['verify','pending','requested','concierge','clear_reservation',
               'cancellation_policies','qt2','request_new_confirm_email',
               'ajax_photo_widget_form_iframe','identity',
               'qt_reply_v2','travel_plans_current',
               'complete_status','populate_from_facebook','kba_update','kba','15','12',
               'jumio_redirect','jumio_token','domains','edit_verification',
               'connect','10','pay','at_checkpoint','push_notification_callback',
               'this_hosting_reviews','widget','apply_reservation','itinerary',
               'receipt','add_guests','ajax_image_upload','qt_with','webcam_upload',
               'ajax_google_translate_description','status','upload','cancel',
               'phone_verification_modal','change','email_itinerary_colorbox',
               'terms','multi_message','reputation','complete_redirect',
               'image_order','review_page','place_worth','change_availability',
               'hospitality','ajax_price_and_availability','guest_booked_elsewhere',
               'why_host','payoneer_signup_complete','onenight','respond',
               'ajax_photo_widget','invalid_action','slideshow','ajax_google_translate_reviews',
               'spoken_languages','message_to_host_focus','cancellation_policy_click',
               'message_to_host_change','agree_terms_check','read_policy_click','phone_verification_success',
               'phone_verification_number_sucessfully_submitted','phone_verification_number_submitted_for_sms',
               'endpoint_error','apply_coupon_error_type','apply_coupon_click',
               'coupon_field_focus','phone_verification_call_taking_too_long']

action_detail_list = ['pending','p5','create_phone_numbers','cancellation_policies',
                      'message_thread','request_new_confirm_email','send_message',
                      'your_trips','profile_verifications','post_checkout_action',
                      'at_checkpoint','manage_listing','your_listings','update_listing',
                      'change_availability','apply_coupon','guest_itinerary',
                      'guest_receipt','lookup_message_thread','remove_dashboard_alert',
                      'delete_phone_numbers','change_or_alter','terms_and_privacy',
                      'alteration_field,alteration_request','create_alteration_request',
                      'translate_listing_reviews','complete_booking','message_to_host_focus','message_to_host_change']

#device_type_list = [#iPodtouch,Android_App_Unknown_Phone/Tablet,Mac_Desktop,Tablet,Linux_Desktop,Chromebook]

#sessions.sort_values(by='action_done',inplace=True)
#action_list_names = []
#for value in action_done_list:
#    action_list_names.append('action_done'+ '_'+ value)
#    sessions = add_feature(sessions,value,'action_done')
#sessions.sort_values(by='device_type',inplace=True)
#device_type_list_names = []
#for value in sessions.device_type.unique():
#    device_type_list_names.append('device_type'+ '_'+ value)
#    sessions = add_feature(sessions,value,'device_type')
#sessions.sort_values(by='action_detail',inplace=True)
#action_detail_list_names = []
##for value in action_detail_list:
#for value in sessions.action_detail.unique():
#    action_detail_list_names.append('action_detail'+ '_'+ value)
#    sessions = add_feature(sessions,value,'action_detail')
#sessions.sort_values(by='action_type',inplace=True)
#action_type_list_names = []
#for value in sessions.action_type.unique():
#    action_type_list_names.append('action_type'+ '_'+ value)
#    sessions = add_feature(sessions,value,'action_type')

#sessions_dummies = pd.get_dummies(sessions,columns=['device_type','action_detail'],sparse=True)
common_action_done_set = set(['show','index','search_results','personalize','search','ajax_refresh_subtotal','similar_listings',
                         'update','social_connections','reviews','active','similar_listings_v2','lookup','create','dashboard',
                         'header_userpic','collections','edit','campaigns'])
common_action_detail_set = set(['view_search_results','p3','null_value','-unknown-',
                         'wishlist_content_update','user_profile','change_trip_characteristics',
                         'similar_listings','user_social_connections','listing_reviews',
                         'update_listing','dashboard','user_wishlists','header_userpic',
                         'message_thread','edit_profile'])

ad_series = sessions.action_done.value_counts()
ad_series_small = ad_series.loc[ad_series <= 15]
rare_action_done_set = set(ad_series_small.index.values)
adet_series = sessions.action_detail.value_counts()
adet_series_small = adet_series.loc[adet_series <= 15]
rare_action_detail_set = set(adet_series_small.index.values)

sessions['common_action_done'] = sessions['action_done'].map(lambda x: 1 if x in common_action_done_set else 0)
sessions['rare_action_done'] = sessions['action_done'].map(lambda x: 1 if x in rare_action_done_set else 0)
sessions['common_action_detail'] = sessions['action_detail'].map(lambda x: 1 if x in common_action_detail_set else 0)
sessions['rare_action_detail'] = sessions['action_detail'].map(lambda x: 1 if x in rare_action_detail_set else 0)

sessions_small_action_done = sessions.loc[(sessions['common_action_done'] == 0) & (sessions['rare_action_done'] == 0)]
sessions_small_action_detail = sessions.loc[(sessions['common_action_detail'] == 0) & (sessions['rare_action_detail'] == 0)]

sessions_large_action_done = sessions.loc[(sessions['common_action_done'] == 1) & (sessions['rare_action_done'] == 0)]
sessions_large_action_detail = sessions.loc[(sessions['common_action_detail'] == 1) & (sessions['rare_action_detail'] == 0)]

sessions_dummies = pd.get_dummies(sessions,columns=['action_type','device_type'])
sessions_action_done_small_dummies = pd.get_dummies(sessions_small_action_done,columns=['action_done'])
sessions_action_detail_small_dummies = pd.get_dummies(sessions_small_action_detail,columns=['action_detail'])

sessions_action_done_large_dummies = pd.get_dummies(sessions_large_action_done,columns=['action_done'])
sessions_action_detail_large_dummies = pd.get_dummies(sessions_large_action_detail,columns=['action_detail'])

#sessions['device_type'] = sessions['device_type'].astype('category')
#sessions_action_detail = sessions[['user_id','action_detail']]
#sessions_action_detail['action_detail'] = sessions_action_detail['action_detail'].astype('category')
#sessions_dummies = pd.get_dummies(sessions,columns=['device_type'])
#sessions_action_detail_dummies = pd.get_dummies(sessions_action_detail,columns=['action_detail'])
#sessions_dummies_action_done = pd.get_dummies(sessions,columns=['action_done'])
toc=timeit.default_timer()
print('Sessions Has Action Time',toc - tic)
#%%
tic=timeit.default_timer()
sessions_old['action_per_time'] = sessions_old['action_done'] + sessions_old['action_detail'] + sessions_old['action_type'] + sessions_old['secs_elapsed_binned'].astype(str)
(action_set_dict,sessions_old) = convert_strings_to_ints(sessions_old,'action_set','action_set_hash')
sessions_old['action_set'] = sessions_old['action_set_hash']
sessions_old = sessions_old.drop('action_set_hash',axis=1)

def get_second_most_common(x):
    try:
        return x.value_counts().index[1]
    except IndexError:
        return -2
most_common_action_set_dict = sessions_old.groupby(['user_id'])['action_set'].agg(lambda x: x.value_counts().index[0]).to_dict()
second_most_common_action_set_dict = sessions_old.groupby(['user_id'])['action_set'].agg(lambda x: get_second_most_common(x)).to_dict()

#%%
sessions_old['next_action_set'] = sessions_old['action_set'].shift(1)
sessions_old['prev_action_set'] = sessions_old['action_set'].shift(-1)
sessions_old['next_user_id'] = sessions_old['user_id'].shift(1)
sessions_old.fillna(-1,inplace=True)
sessions_old['is_action_set_repeat'] = 0
sessions_old_same_user = sessions_old['next_user_id'] == sessions_old['user_id']
sessions_old_same_action_set = sessions_old['next_action_set'] == sessions_old['action_set']
sessions_old['is_action_set_repeat'][sessions_old_same_user & sessions_old_same_action_set] = 1
#%%


ad_combo_series = sessions_old.action_set.value_counts()
ad_combo_series_large = ad_combo_series.loc[ad_combo_series >= 100000]
ad_combo_series_med = ad_combo_series.loc[(ad_combo_series < 100000) & (ad_combo_series >= 5000)]
ad_combo_series_small = ad_combo_series.loc[(ad_combo_series < 5000) & (ad_combo_series >= 20)]
#ad_combo_series_small = ad_combo_series.loc[(ad_combo_series < 5000)]
#ad_combo_series_small = ad_combo_series.loc[(ad_combo_series < 10000) & (ad_combo_series >= 20)]
common_action_combo_large_set = set(ad_combo_series_large.index.values)
common_action_combo_med_set = set(ad_combo_series_med.index.values)
common_action_combo_small_set = set(ad_combo_series_small.index.values)

sessions_old['common_action_large_set'] = sessions_old['action_set'].map(lambda x: 1 if x in common_action_combo_large_set else 0)
sessions_old['common_action_med_set'] = sessions_old['action_set'].map(lambda x: 1 if x in common_action_combo_med_set else 0)
sessions_old['common_action_small_set'] = sessions_old['action_set'].map(lambda x: 1 if x in common_action_combo_small_set else 0)
sessions_small_set = sessions_old.loc[(sessions_old['common_action_small_set'] == 1)]
sessions_large_set = sessions_old.loc[(sessions_old['common_action_large_set'] == 1)]
sessions_med_set = sessions_old.loc[(sessions_old['common_action_med_set'] == 1)]
print('large dummies')
sessions_large_action_set_dummies = pd.get_dummies(sessions_large_set,columns=['action_set'])
print('small dummies')
sessions_small_action_set_dummies = pd.get_dummies(sessions_small_set,columns=['action_set'])
print('med dummies')
sessions_med_action_set_dummies = pd.get_dummies(sessions_med_set,columns=['action_set'])

toc=timeit.default_timer()
print('Sessions Has Set Time',toc - tic)
#%%
tic=timeit.default_timer()
sessions_old_smaller = sessions_old[['user_id','action_set']].copy()
sessions_old_smaller['position_of_set'] = 1
sessions_old_smaller['position_of_set'] = sessions_old_smaller.groupby(['user_id'])['position_of_set'].cumsum()
SESSIONS_POS_DICT = {}
for i in range(1,2):
    temp_df = sessions_old_smaller.loc[sessions_old_smaller['position_of_set'] == i]
    temp_df = temp_df[['user_id','action_set']]
    sessions_pos_name = 'position_set_' + str(i)
    temp_df.rename(columns = {'action_set':sessions_pos_name},inplace=True)
    SESSIONS_POS_DICT[i] = temp_df
toc=timeit.default_timer()
print('Sessinos position Time',toc - tic)
#%%
sessions_old_impressions = sessions_old.loc[sessions_old.action_done == 'impressions']
sessions_old_impressions.rename(columns={'prev_action_set':'impressions_prev_action_set','next_action_set':'impressions_next_action_set'},inplace=True)
sessions_old_impressions.drop_duplicates('user_id',inplace=True)
#%%
#action_per_time_list = []
#for value in sessions_old['action_per_time'].unique():
#    new_df = sessions_old.loc[sessions_old['action_per_time'] == value]
#    new_df = new_df.drop_duplicates(subset = 'user_id')
#    if(new_df.index.shape[0] <= 100):
#        continue
#    us_number = new_df.loc[new_df.country_destination == 'US'].index.shape[0]
#    ndf_number = new_df.loc[new_df.country_destination == 'NDF'].index.shape[0]
#    dummies_number = new_df.loc[new_df.country_destination == 'dummy'].index.shape[0]
##    print(us_number,ndf_number)
#    if (dummies_number <= 20):
#        continue
#    if ((ndf_number + 20) >= us_number):
#        continue
#    print(value)
##    print(new_df.country_destination.value_counts(normalize=False))
#    action_per_time_list.append(value)
#print(end)
#complete_status-unknown--unknown-7,requestedp5view7,requestedpost_checkout_actionsubmit9
#%%
action_per_time_list = ['verify-unknown--unknown-9',
 'pendingpendingbooking_request7',
 'requestedp5view8',
 'createcreate_phone_numberssubmit10',
 'cancellation_policiescancellation_policiesview10',
 'null_valuemessage_postmessage_post1',
 'null_valuemessage_postmessage_post8',
 'qt2message_threadview8',
 'indexmessage_threadview10',
 'qt2message_threadview9',
 'ajax_photo_widget_form_iframe-unknown--unknown-7',
 'identity-unknown--unknown-8',
 'qt2message_threadview7',
 'qt_reply_v2send_messagesubmit8',
 'travel_plans_currentyour_tripsview8',
 'complete_status-unknown--unknown-7',
 'createcreate_phone_numberssubmit8',
 'ajax_photo_widget_form_iframe-unknown--unknown-8',
 'identity-unknown--unknown-7',
 'populate_from_facebook-unknown--unknown-9',
 'kba_update-unknown--unknown-9',
 'kba-unknown--unknown-9',
 'identity-unknown--unknown-10',
 '15message_postmessage_post10',
 'update-unknown--unknown-7',
 'null_valuemessage_postmessage_post7',
 '12message_postmessage_post10',
 '12message_postmessage_post9',
 'createcreate_phone_numberssubmit9',
 'jumio_redirect-unknown--unknown-9',
 'jumio_token-unknown--unknown-9',
 'requestedp5view7',
 'requestedp5view9',
 'travel_plans_currentyour_tripsview7',
 'pendingpendingbooking_request1',
 'edit_verificationprofile_verificationsview7',
 'connectoauth_loginsubmit9',
 'edit_verificationprofile_verificationsview8',
 'requestedpost_checkout_actionsubmit9',
 'identity-unknown--unknown-9',
 'updateupdate_user_profilesubmit10',
 'delete-unknown--unknown-8',
 'edit_verificationprofile_verificationsview9',
 'callbackoauth_responsepartner_callback9',
 '10message_postmessage_post10',
 'pendingpendingbooking_request10',
 'qt_reply_v2send_messagesubmit7',
 'payment_methods-unknown--unknown-9',
 'pay-unknown--unknown-8',
 'pay-unknown--unknown-7',
 'at_checkpointat_checkpointbooking_request8',
 '10message_postmessage_post9',
 'complete_status-unknown--unknown-8',
 'jumio_redirect-unknown--unknown-10',
 'cancellation_policiescancellation_policiesview8',
 'at_checkpointat_checkpointbooking_request9',
 'verify-unknown--unknown-10',
 'populate_from_facebook-unknown--unknown-8',
 'travel_plans_currentyour_tripsview9',
 'requestedpost_checkout_actionsubmit10',
 'requestedpost_checkout_actionsubmit7',
 'request_new_confirm_emailrequest_new_confirm_emailclick7',
 'this_hosting_reviewslisting_reviews_pageclick9',
 'phone_number_widget-unknown--unknown-7',
 'pending-unknown--unknown-8',
 'signature-unknown--unknown-9',
 'editedit_profileview10',
 'pendingpendingbooking_request9',
 'payment_methods-unknown--unknown-10',
 'apply_reservationapply_couponsubmit9',
 'pendingpendingbooking_request8',
 'cancellation_policiescancellation_policiesview9',
 'updateupdate_user_profilesubmit8',
 'itineraryguest_itineraryview9',
 'travel_plans_currentyour_tripsview10',
 'confirm_emailconfirm_email_linkclick7',
 'complete_status-unknown--unknown-9',
 'ajax_image_upload-unknown--unknown-9',
 'profile_pic-unknown--unknown-7',
 'kba-unknown--unknown-8',
 'jumio_token-unknown--unknown-8',
 'ajax_photo_widget_form_iframe-unknown--unknown-2',
 'qt_withlookup_message_threaddata8',
 'ajax_photo_widget_form_iframe-unknown--unknown-4',
 'ajax_photo_widget_form_iframe-unknown--unknown-3',
 'ajax_photo_widget_form_iframe-unknown--unknown-5',
 'profile_pic-unknown--unknown-8',
 'ajax_image_upload-unknown--unknown-10',
 'request_new_confirm_emailrequest_new_confirm_emailclick9',
 'ajax_photo_widget_form_iframe-unknown--unknown-9',
 'deletedelete_phone_numberssubmit9',
 'kba_update-unknown--unknown-8',
 'callbackoauth_responsepartner_callback8',
 'connectoauth_loginsubmit8',
 'jumio_token-unknown--unknown-10',
 'request_new_confirm_emailrequest_new_confirm_emailclick8',
 'at_checkpointat_checkpointbooking_request10',
 'populate_from_facebook-unknown--unknown-10',
 'qt2message_threadview10',
 'apply_reservationapply_couponsubmit10',
 'edit_verificationprofile_verificationsview10',
 'null_valuemessage_postmessage_post6',
 'requestedpost_checkout_actionsubmit8',
 'apply_reservationapply_couponsubmit8',
 'requestedp5view10',
 'verify-unknown--unknown-8',
 'complete_status-unknown--unknown-6',
 'qt_withlookup_message_threaddata9',
 'kba-unknown--unknown-10',
 'ajax_image_upload-unknown--unknown-8',
 'itineraryguest_itineraryview10',
 'delete-unknown--unknown-9',
 'itineraryguest_itineraryview8',
 'changechange_or_alterview9',
 'webcam_upload-unknown--unknown-9',
 'identity-unknown--unknown-5',
 'push_notification_callback-unknown--unknown-7',
 'termsterms_and_privacyview10',
 'signature-unknown--unknown-10',
 'changechange_or_alterview8',
 'complete-unknown--unknown-8',
 'ajax_photo_widget_form_iframe-unknown--unknown-10',
 'populate_from_facebook-unknown--unknown-7',
 'complete_redirect-unknown--unknown-8',
 'cancellation_policiescancellation_policiesview7',
 'createcreate_phone_numberssubmit7',
 'image_order-unknown--unknown-9',
 'ajax_photo_widget_form_iframe-unknown--unknown-6',
 'image_order-unknown--unknown-8',
 'deletedelete_phone_numberssubmit8',
 'qt_withlookup_message_threaddata10',
 'complete_redirect-unknown--unknown-9',
 'editedit_profileview6',
 'push_notification_callback-unknown--unknown-10',
 'complete_status-unknown--unknown-5',
 'push_notification_callback-unknown--unknown-8',
 'termsterms_and_privacyview9',
 'push_notification_callback-unknown--unknown-9',
 'receiptguest_receiptview7',
 'identity-unknown--unknown-6',
 'ajax_photo_widget_form_iframe-unknown--unknown-1',
 'changechange_or_alterview7',
 'editedit_profileview5',
 'receiptguest_receiptview8',
 'editedit_profileview3',
 'handle_vanity_url-unknown--unknown-8',
 'handle_vanity_url-unknown--unknown-7',
 'handle_vanity_url-unknown--unknown-9',
 'impressionsp4view0',
 'impressionsp4view8',
 'message_to_host_focusmessage_to_host_focusclick9',
 'message_to_host_focusmessage_to_host_focusclick10',
 'message_to_host_changemessage_to_host_changeclick10',
 'agree_terms_check-unknown--unknown-8',
 'message_to_host_focusmessage_to_host_focusclick8',
 'impressionsp4view9',
 'message_to_host_changemessage_to_host_changeclick9',
 'read_policy_clickread_policy_clickclick9',
 'impressionsp4view10',
 'cancellation_policy_clickcancellation_policy_clickclick8',
 'agree_terms_check-unknown--unknown-6',
 'agree_terms_check-unknown--unknown-9',
 'phone_verification_number_submitted_for_sms-unknown--unknown-8',
 'phone_verification_number_submitted_for_sms-unknown--unknown-9',
 'endpoint_error-unknown--unknown-9',
 'agree_terms_check-unknown--unknown-5',
 'agree_terms_check-unknown--unknown-10',
 'phone_verification_number_sucessfully_submitted-unknown--unknown-7',
 'phone_verification_successphone_verification_successclick9',
 'read_policy_clickread_policy_clickclick8',
 'phone_verification_successphone_verification_successclick8',
 'message_to_host_changemessage_to_host_changeclick8',
 'phone_verification_number_sucessfully_submitted-unknown--unknown-8',
 'endpoint_error-unknown--unknown-8',
 'message_to_host_focusmessage_to_host_focusclick7',
 'endpoint_error-unknown--unknown-7',
 'phone_verification_successphone_verification_successclick7',
 'agree_terms_check-unknown--unknown-7']
action_per_time_set = set(action_per_time_list)
#%%
sessions_old['action_per_time_set'] = sessions_old['action_per_time'].map(lambda x: 1 if x in action_per_time_set else 0)
print('action_per_time dummies')
sessions_action_per_time_set = sessions_old.loc[(sessions_old['action_per_time_set'] == 1)]
sessions_action_per_time_dummies = pd.get_dummies(sessions_action_per_time_set,columns=['action_per_time'])
#%%
session_dummmies_cols = sessions_dummies.columns.values
action_set_small_names = [col for col in list(sessions_small_action_set_dummies.columns.values) if col.startswith('action_set_')]
action_set_large_names = [col for col in list(sessions_large_action_set_dummies.columns.values) if col.startswith('action_set_')]
action_set_med_names = [col for col in list(sessions_med_action_set_dummies.columns.values) if col.startswith('action_set_')]

action_per_time_names = [col for col in list(sessions_action_per_time_dummies.columns.values) if col.startswith('action_per_time_')]

action_list_names = [col for col in list(sessions_action_done_small_dummies.columns.values) if col.startswith('action_done_')]
action_detail_list_names = [col for col in list(sessions_action_detail_small_dummies.columns.values) if col.startswith('action_detail_')]

action_list_large_names = [col for col in list(sessions_action_done_large_dummies.columns.values) if col.startswith('action_done_')]
action_detail_list_large_names = [col for col in list(sessions_action_detail_large_dummies.columns.values) if col.startswith('action_detail_')]

action_type_list_names = [col for col in list(session_dummmies_cols) if col.startswith('action_type_')]
device_type_list_names = [col for col in list(session_dummmies_cols) if col.startswith('device_type_')]
#%%
#sessions_dummies_dense = sessions_dummies.to_dense()
#%%
#%%
tic=timeit.default_timer()
#sessions_grouped = sessions.groupby('user_id')
sessions_grouped_orig = sessions.groupby('user_id')

sessions_grouped_old = sessions_old.groupby('user_id')

sessions_grouped = sessions_dummies.groupby('user_id')
sessions_null_secs = sessions.loc[sessions.secs_elapsed == 0]
sessions_grouped_null_secs = sessions_null_secs.groupby('user_id')

sessions_small_action_done_grouped = sessions_action_done_small_dummies.groupby('user_id')

sessions_small_action_set_dummies_grouped = sessions_small_action_set_dummies.groupby('user_id')
sessions_large_action_set_dummies_grouped = sessions_large_action_set_dummies.groupby('user_id')
sessions_med_action_set_dummies_grouped = sessions_med_action_set_dummies.groupby('user_id')

sessions_small_action_detail_grouped = sessions_action_detail_small_dummies.groupby('user_id')

sessions_large_action_done_grouped = sessions_action_done_large_dummies.groupby('user_id')

sessions_action_per_time_grouped = sessions_action_per_time_dummies.groupby('user_id')

sessions_large_action_detail_grouped = sessions_action_detail_large_dummies.groupby('user_id')
transactions = sessions_grouped['user_id'].count()
transactions.name = 'transactions_count'
transactions_df = transactions.to_frame()

secs_elapsed_sum = sessions_grouped['secs_elapsed'].sum()
secs_elapsed_sum.name = 'secs_elapsed_sum'
secs_elapsed_sum_df = secs_elapsed_sum.to_frame()

action_set_repeat_sum = sessions_grouped_old['is_action_set_repeat'].sum()
action_set_repeat_sum.name = 'is_action_set_repeat'
action_set_repeat_sum_df = action_set_repeat_sum.to_frame()

secs_elapsed_null_count = sessions_grouped_null_secs['user_id'].count()
secs_elapsed_null_count.name = 'secs_elapsed_null_count'
secs_elapsed_null_count_df = secs_elapsed_null_count.to_frame()

rare_actions_sum = sessions_grouped['rare_action_done'].sum()
rare_actions_sum.name = 'rare_actions_sum'
rare_actions_sum_df = rare_actions_sum.to_frame()

common_actions_sum = sessions_grouped['common_action_done'].sum()
common_actions_sum.name = 'common_actions_sum'
common_actions_sum_df = common_actions_sum.to_frame()

secs_elapsed_mean = sessions_grouped['secs_elapsed'].mean()
secs_elapsed_mean.name = 'secs_elapsed_mean'
secs_elapsed_mean_df = secs_elapsed_mean.to_frame()

unique_actions = sessions_grouped_orig[['action_done','action_type','action_detail','device_type']].aggregate(lambda x: np.unique(x).shape[0])
unique_actions.rename(columns={'action_done':'unique_actions','action_type':'unique_action_types',
                               'action_detail':'unique_action_details','device_type':'unique_device_types'},inplace=True)


actions_set = sessions_small_action_set_dummies_grouped[action_set_small_names].max()
actions_set_large = sessions_large_action_set_dummies_grouped[action_set_large_names].max()
actions_set_med = sessions_med_action_set_dummies_grouped[action_set_med_names].max()

actions_per_time = sessions_action_per_time_grouped[action_per_time_names].max()

actions_done = sessions_small_action_done_grouped[action_list_names].max()
actions_done_large = sessions_large_action_done_grouped[action_list_large_names].max()
action_types_done = sessions_grouped[action_type_list_names].max()
action_details_done = sessions_small_action_detail_grouped[action_detail_list_names].max()
action_details_done_large = sessions_large_action_detail_grouped[action_detail_list_large_names].max()
device_types_used = sessions_grouped[device_type_list_names].max()


actions_set_summed = sessions_small_action_set_dummies_grouped[action_set_small_names].sum()
actions_set_large_summed = sessions_large_action_set_dummies_grouped[action_set_large_names].sum()
actions_set_med_summed = sessions_med_action_set_dummies_grouped[action_set_med_names].sum()

actions_per_time_summed = sessions_action_per_time_grouped[action_per_time_names].sum()

actions_done_summed = sessions_small_action_done_grouped[action_list_names].sum()
actions_done_large_summed = sessions_large_action_done_grouped[action_list_large_names].sum()
action_types_done_summed = sessions_grouped[action_type_list_names].sum()
action_details_done_summed = sessions_small_action_detail_grouped[action_detail_list_names].sum()
action_details_done_large_summed = sessions_large_action_detail_grouped[action_detail_list_large_names].sum()
device_types_used_summed = sessions_grouped[device_type_list_names].sum()

actions_per_time_summed = actions_per_time_summed.rename(columns=lambda x: 'summed_'+x)
actions_set_summed = actions_set_summed.rename(columns=lambda x: 'summed_'+x)
actions_set_large_summed = actions_set_large_summed.rename(columns=lambda x: 'summed_'+x)
actions_set_med_summed = actions_set_med_summed.rename(columns=lambda x: 'summed_'+x)
actions_done_summed = actions_done_summed.rename(columns=lambda x: 'summed_'+x)
actions_done_large_summed = actions_done_large_summed.rename(columns=lambda x: 'summed_'+x)
action_types_done_summed = action_types_done_summed.rename(columns=lambda x: 'summed_'+x)
action_details_done_summed = action_details_done_summed.rename(columns=lambda x: 'summed_'+x)
action_details_done_large_summed = action_details_done_large_summed.rename(columns=lambda x: 'summed_'+x)
device_types_used_summed = device_types_used_summed.rename(columns=lambda x: 'summed_'+x)

#sessions_cat = ['action','action_type','action_detail','device_type']
#sessions_dummies = generate_dummies(sessions,sessions_cat)
toc=timeit.default_timer()
print('Sessions Group feature Time',toc - tic)
#%%
del (sessions_dummies,sessions_grouped,sessions_grouped_orig,sessions_grouped_old,sessions_action_detail_small_dummies,sessions_action_done_small_dummies,
     sessions_small_action_done_grouped,sessions_small_action_detail_grouped,
     sessions_large_action_done_grouped,sessions_large_action_detail_grouped,
     sessions_action_detail_large_dummies,sessions_action_done_large_dummies,
     sessions_small_action_set_dummies,sessions_large_action_set_dummies,
     sessions_small_action_set_dummies_grouped,sessions_large_action_set_dummies_grouped,
     sessions_med_action_set_dummies,sessions_med_action_set_dummies_grouped,
     sessions_action_per_time_dummies,sessions_action_per_time_grouped)

#%%
tic=timeit.default_timer()

combined.fillna(0,inplace=True)
for key in SESSIONS_POS_DICT:
    print(key)
    temp_df = SESSIONS_POS_DICT[key]
    combined = pd.merge(combined,temp_df,left_on = ['user_id'],
                               right_on = ['user_id'],how='left')
    combined.fillna(-1,inplace=True)

def get_most_common_action_set(x):
    try:
        return most_common_action_set_dict[x]
    except KeyError:
        return -1
def get_second_most_common_action_set(x):
    try:
        return second_most_common_action_set_dict[x]
    except KeyError:
        return -1

combined['most_common_action_set'] = combined['user_id'].map(lambda x: get_most_common_action_set(x))
combined['second_most_common_action_set'] = combined['user_id'].map(lambda x: get_second_most_common_action_set(x))
combined.fillna(0,inplace=True)

combined = pd.merge(combined,transactions_df,left_on = ['user_id'],
                               right_on = transactions_df.index.values,how='left')
combined['transactions_count'].fillna(-1,inplace=True)

#%%
combined = pd.merge(combined,sessions_old_impressions[['user_id','impressions_next_action_set','impressions_prev_action_set']],left_on = ['user_id'],
                               right_on = 'user_id',how='left')
combined.fillna(-1,inplace=True)
#%%

combined = pd.merge(combined,secs_elapsed_null_count_df,left_on = ['user_id'],
                               right_on = secs_elapsed_null_count_df.index.values,how='left')
combined['secs_elapsed_null_count'].fillna(0,inplace=True)

combined = pd.merge(combined,actions_set,left_on = ['user_id'],
                               right_on = actions_set.index.values,how='left')
combined = pd.merge(combined,actions_set_large,left_on = ['user_id'],
                               right_on = actions_set_large.index.values,how='left')
combined = pd.merge(combined,actions_set_med,left_on = ['user_id'],
                               right_on = actions_set_med.index.values,how='left')

combined = pd.merge(combined,actions_per_time,left_on = ['user_id'],
                               right_on = actions_per_time.index.values,how='left')

combined = pd.merge(combined,actions_set_summed,left_on = ['user_id'],
                               right_on = actions_set_summed.index.values,how='left')
combined = pd.merge(combined,actions_set_large_summed,left_on = ['user_id'],
                               right_on = actions_set_large_summed.index.values,how='left')
combined = pd.merge(combined,actions_set_med_summed,left_on = ['user_id'],
                               right_on = actions_set_med_summed.index.values,how='left')

combined = pd.merge(combined,actions_per_time_summed,left_on = ['user_id'],
                               right_on = actions_per_time_summed.index.values,how='left')

combined.fillna(0,inplace=True)


combined = pd.merge(combined,actions_done,left_on = ['user_id'],
                               right_on = actions_done.index.values,how='left')
combined = pd.merge(combined,action_details_done,left_on = ['user_id'],
                               right_on = action_details_done.index.values,how='left')
combined = pd.merge(combined,actions_done_large,left_on = ['user_id'],
                               right_on = actions_done_large.index.values,how='left')
combined = pd.merge(combined,action_details_done_large,left_on = ['user_id'],
                               right_on = action_details_done_large.index.values,how='left')
combined = pd.merge(combined,action_types_done,left_on = ['user_id'],
                               right_on = action_types_done.index.values,how='left')
combined = pd.merge(combined,device_types_used,left_on = ['user_id'],
                               right_on = device_types_used.index.values,how='left')

combined = pd.merge(combined,unique_actions,left_on = ['user_id'],
                               right_on = unique_actions.index.values,how='left')

combined = pd.merge(combined,actions_done_summed,left_on = ['user_id'],
                               right_on = actions_done_summed.index.values,how='left')
combined = pd.merge(combined,action_details_done_summed,left_on = ['user_id'],
                               right_on = action_details_done_summed.index.values,how='left')
combined = pd.merge(combined,actions_done_large_summed,left_on = ['user_id'],
                               right_on = actions_done_large_summed.index.values,how='left')
combined = pd.merge(combined,action_details_done_large_summed,left_on = ['user_id'],
                               right_on = action_details_done_large_summed.index.values,how='left')
combined = pd.merge(combined,action_types_done_summed,left_on = ['user_id'],
                               right_on = action_types_done_summed.index.values,how='left')
combined = pd.merge(combined,device_types_used_summed,left_on = ['user_id'],
                               right_on = device_types_used_summed.index.values,how='left')

combined.fillna(0,inplace=True)

combined = pd.merge(combined,secs_elapsed_sum_df,left_on = ['user_id'],
                               right_on = secs_elapsed_sum_df.index.values,how='left')

combined = pd.merge(combined,action_set_repeat_sum_df,left_on = ['user_id'],
                               right_on = action_set_repeat_sum_df.index.values,how='left')

combined = pd.merge(combined,secs_elapsed_mean_df,left_on = ['user_id'],
                               right_on = secs_elapsed_mean_df.index.values,how='left')
combined = pd.merge(combined,rare_actions_sum_df,left_on = ['user_id'],
                               right_on = rare_actions_sum_df.index.values,how='left')
combined = pd.merge(combined,common_actions_sum_df,left_on = ['user_id'],
                               right_on = common_actions_sum_df.index.values,how='left')
combined.fillna(0,inplace=True)


combined = pd.merge(combined,sessions_first[['first_secs_elapsed_binned']],left_on = ['user_id'],
                               right_on = sessions_first.user_id,how='left')
combined.fillna(-1,inplace=True)
combined = pd.merge(combined,sessions_last[['last_action_done']],left_on = ['user_id'],
                               right_on = sessions_last.user_id,how='left')
combined = pd.merge(combined,sessions_first[['first_action_done']],left_on = ['user_id'],
                               right_on = sessions_first.user_id,how='left')

combined = pd.merge(combined,sessions_last[['last_action_set']],left_on = ['user_id'],
                               right_on = sessions_last.user_id,how='left')
combined = pd.merge(combined,sessions_first[['first_action_set']],left_on = ['user_id'],
                               right_on = sessions_first.user_id,how='left')

#%%
combined.last_action_done.fillna('no_actions_done',inplace=True)
combined.first_action_done.fillna('no_actions_done',inplace=True)
(first_action_done_dict,combined) = convert_strings_to_ints(combined,'first_action_done','first_action_done_hash')
(last_action_done_dict,combined) = convert_strings_to_ints(combined,'last_action_done','last_action_done_hash')

combined.last_action_set.fillna('no_actions_set',inplace=True)
combined.first_action_set.fillna('no_actions_set',inplace=True)
(first_action_set_dict,combined) = convert_strings_to_ints(combined,'first_action_set','first_action_set_hash')
(last_action_set_dict,combined) = convert_strings_to_ints(combined,'last_action_set','last_action_set_hash')

bins = [-1.0,4,10,20,30,50,100,200,500,1000000000000]
combined['transactions_count_binned'] = np.digitize(combined['transactions_count'], bins, right=True)

bins = [-1000,-2,0,500,1500,3000,6000,12000,18000,25000,40000,80000,150000,10000000]
combined['secs_elapsed_mean_binned'] = np.digitize(combined['secs_elapsed_mean'], bins, right=True)
bins = [-1000,-2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]
combined['secs_elapsed_sum_binned'] = np.digitize(combined['secs_elapsed_sum'], bins, right=True)

toc=timeit.default_timer()
print('Merging time',toc-tic)
#%%
del(actions_done_summed,
    actions_done_large,actions_done,actions_done_large_summed,
    actions_per_time,actions_per_time_summed,
    actions_set,actions_set_large,actions_set_med,
    actions_set_summed,actions_set_large_summed,actions_set_med_summed,
    action_details_done,action_details_done_large,
    action_details_done_summed,action_details_done_large_summed,
    device_types_used,device_types_used_summed,action_types_done,action_types_done_summed)
#%%
del (sessions_first,sessions_large_action_detail,sessions_large_action_done,sessions_large_set,
     sessions_last,sessions_med_set,
#     sessions,
     sessions_small_action_done,
     sessions_small_action_detail,sessions_small_set)
#%%
has_agree_terms_uncheck = combined['action_done_agree_terms_uncheck'] == 1
combined['agree_terms_check_and_not_uncheck'] = combined['action_done_agree_terms_check']
combined['agree_terms_check_and_not_uncheck'][has_agree_terms_uncheck] = 0

language_en = combined['language'] == 'en'
combined['action_done_goog_trans_reviews_and_english'] = combined['action_done_ajax_google_translate_reviews']
combined['action_done_goog_trans_reviews_and_english'][~language_en] = 0
#%%
combined['repeat_action_sets_per_transaction'] = combined['is_action_set_repeat'] / combined['transactions_count']
#%%
basic_with_age_dateint_dict = combined.groupby(['timestamp_dateint'])['basic_with_age'].agg(lambda x: x.mean()).to_dict()
combined['dateint_by_basic_with_age_mean'] = combined['timestamp_dateint'].map(basic_with_age_dateint_dict)
combined['agree_terms_check_possible'] = combined['timestamp_dateint'].map(lambda x: 1 if x >= 1617 else 0)
combined['action_types_by_transactions_count'] = combined['unique_action_types'] / combined['transactions_count']
users_booking = set(combined.loc[combined['action_done_requested'] == 1].user_id.unique())
sessions_old['has_user_booking'] = sessions_old['user_id'].map(lambda x: x in users_booking)
sessions_post_checkout_request = sessions_old.loc[sessions_old['has_user_booking']]
sessions_post_checkout_request = sessions_post_checkout_request.loc[sessions_post_checkout_request['country_destination'] != 'dummy']
sessions_old.drop('has_user_booking',axis=1,inplace=True)
sessions_bookings_small = sessions_old.loc[sessions_old.action_type == 'booking_request']
# book -> investigate this, only in dummy action_done but could be significant 243 dummies
#my_reservations 553
# view 13088 dummy
#braintree_client_token 120 dummy
#%%
tic=timeit.default_timer()

def generate_dummies(input_df,cat_list):
    categoricals_df = input_df[cat_list]
    categoricals_df = categoricals_df.applymap(str)
    dummies = pd.get_dummies(categoricals_df,dummy_na = False)
#    input_df = pd.get_dummies(input_df,columns = cat_list,dummy_na = False)

    input_df = pd.concat([input_df,dummies],axis=1,join='inner')
    return input_df
cat_list = [
#        'gender_hash', 'signup_method_hash', 'signup_flow',
       'language_hash',
#       'affiliate_channel_hash', 'affiliate_provider_hash',
#       'first_affiliate_tracked_hash', 'signup_app_hash', 'first_device_type_hash',
#       'first_browser_hash',
#       'timestamp_year','timestamp_month',
#       'timestamp_day','timestamp_weekday','timestamp_hour',
#       'age_binned','transactions_count_binned',
#       'unique_actions', 'unique_action_types', 'unique_action_details',
#       'unique_device_types','first_secs_elapsed_binned',
#       'unknowns_summed',
#       'last_action_done',
#       'secs_elapsed_mean_binned','secs_elapsed_sum_binned'
       ]
#starting_date = pd.to_datetime('2014-06-20 00:00:00')
#starting_date = pd.to_datetime('2014-06-18 00:00:00')
starting_date = pd.to_datetime('2014-06-18 00:00:00')
#starting_date = pd.to_datetime('2014-05-15 00:00:00')
#ending_date = pd.to_datetime('2014-03-15 00:00:00')
#starting_date = pd.to_datetime('2014-01-01 00:00:00')
#ending_date = pd.to_datetime('2014-02-28 00:00:00')

#ending_date = pd.to_datetime('2012-07-01 12:00:00')
#combined_2 = generate_dummies(combined,cat_list)
#combined_2 = combined.copy()
#is_sub_run = True
#is_sub_run = False
if (is_sub_run):
    train = combined.loc[combined.country_destination != 'dummy' ]
    test = combined.loc[combined.country_destination == 'dummy' ]
else:
    train = combined.loc[(combined['country_destination'] != 'dummy') &
                         (combined['timestamp_first_active'] < starting_date)]
    test = combined.loc[(combined['country_destination'] != 'dummy') &
                         (combined['timestamp_first_active'] >= starting_date)]
#    train = combined_2.loc[(combined['country_destination'] != 'dummy') &
#                         ((combined['timestamp_first_active'] < starting_date) |
#                         (combined['timestamp_first_active'] > ending_date))]
#    test = combined_2.loc[((combined['country_destination'] != 'dummy') &
#                         ((combined['timestamp_first_active'] >= starting_date) &
#                         (combined['timestamp_first_active'] <= ending_date)))]
toc=timeit.default_timer()
print('Dummies Time',toc - tic)

tic=timeit.default_timer()
country_groups = train.groupby('country_destination')
country_means = country_groups.mean()
country_sum = country_groups.sum()
#error prone way, TODO think of better way to do this
action_set_cols = [col for col in list(country_sum) if col.startswith('action_set_')]
action_per_time_cols = [col for col in list(country_sum) if col.startswith('action_per_time_')]
action_count_cols = [col for col in list(country_sum) if col.startswith('action_done_')]
action_type_count_cols = [col for col in list(country_sum) if col.startswith('action_type_')]
detail_action_count_cols = [col for col in list(country_sum) if col.startswith('action_detail_')]
device_type_count_cols = [col for col in list(country_sum) if col.startswith('device_type_')]

summed_action_set_cols = [col for col in list(country_sum) if col.startswith('summed_action_set_')]
summed_action_per_time_cols = [col for col in list(country_sum) if col.startswith('summed_action_per_time_')]
summed_action_count_cols = [col for col in list(country_sum) if col.startswith('summed_action_done_')]
summed_action_type_count_cols = [col for col in list(country_sum) if col.startswith('summed_action_type_')]
summed_detail_action_count_cols = [col for col in list(country_sum) if col.startswith('summed_action_detail_')]
summed_device_type_count_cols = [col for col in list(country_sum) if col.startswith('summed_device_type_')]

#country_profiles = country_sum[cat_hashes]
#country_profiles_dict = dict(zip(range(country_profiles.index.shape[0]),country_profiles.index.values))
toc=timeit.default_timer()
print('Grouping Time',toc - tic)
#%%
def norm_rows(df):
    with np.errstate(invalid='ignore'):
        return df.div(df.sum(axis=1), axis=0).fillna(0)
def norm_cols(df):
    with np.errstate(invalid='ignore'):
        return df.div(df.sum(axis=0), axis=1).fillna(0)
#    return df.div(df.sum(axis=1), axis=0)
# dict lookup helper
def find_appropriate_weight(weights_dict, colname):
    for col, weight in weights_dict.items():
        if col == colname:
            return weight
    for col, weight in weights_dict.items():
        if colname.startswith(col):
            return weight
    print(colname)
    raise ValueError
### obtain string of top10 hashes according to similarity scores for every user
def get_top5_country_hashes_string(row):
#    row.sort_values(inplace=True)
    row.sort(inplace=True)
    return row.index[-5:][::-1].tolist()
def get_truth_values(input_list,truth):
    return [ int(x == truth) for x in input_list ]
def get_truth_row(row):
    return get_truth_values(row['pred'],row['country_destination'])
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
#%%
def binomial_error(num, den):
    """ Return the uncertainty on the ratio M/N assuming that there are N
    independent trials drawing from a binomial distribution with with
    probability p = num/den.
    For more information, see:
    http://www.pp.rhul.ac.uk/~cowan/stat/notes/efferr.pdf
    Args:
        num (int): The number of successful trials.
        dem (int): The total number of trials.
    Returns:
        float: The uncertainty on the ratio num/den.
    """
    return np.sqrt(num * (1 - (num/den))) / den
#%%
test_lb_dict = {}
test_lb_dict['NDF'] = 0.67909
test_lb_dict['US'] = 0.23470
test_lb_dict['other'] = 0.03403
test_lb_dict['FR'] = 0.01283
test_lb_dict['IT'] = 0.01004
test_lb_dict['GB'] = 0.00730
test_lb_dict['CA'] = 0.00730
test_lb_dict['ES'] = 0.00725
test_lb_dict['DE'] = 0.00344
test_lb_dict['NL'] = 0.00215
test_lb_dict['AU'] = 0.00113
test_lb_dict['PT'] = 0.00075

def rebalance_df(input_df):
    country_norm_dict = input_df.country_destination.value_counts(normalize=True)
    return country_norm_dict

def print_value_counts(input_df,col_name,is_normalized=False):
    for value in input_df[col_name].unique():
        print(value)
        print(input_df.loc[input_df[col_name] == value].country_destination.value_counts(normalize=is_normalized))

def print_value_counts_spec(input_df,col_name,col_value,is_normalized=False):
    print(col_value)
    print(input_df.loc[input_df[col_name] == col_value].country_destination.value_counts(normalize=is_normalized))


def customized_eval(preds, dtrain):
    labels = dtrain.get_label()
    top = []
    for i in range(preds.shape[0]):
        top.append(np.argsort(preds[i])[::-1][:5])
    mat = np.reshape(np.repeat(labels,np.shape(top)[1]) == np.array(top).ravel(),np.array(top).shape).astype(int)
    score = np.mean(np.sum(mat/np.log2(np.arange(2, mat.shape[1] + 2)),axis = 1))
    return 'ndcg5', score
#%%
def fit_xgb_model(train, test, params, xgb_features, num_rounds = 10, num_boost_rounds = 10,
                  do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
                  random_seed = 5, reweight_probs = True):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)
#    return 1
#    num_boost_rounds = 10
#if __name__ == "__main__":
#    try:
#        from multiprocessing import set_start_method
#    except ImportError:
#        raise ImportError("Unable to import multiprocessing.set_start_method."
#                          " This example only runs on Python 3.4")
#    set_start_method("forkserver")
#    os.environ["OMP_NUM_THREADS"] = "2"  # or to whatever you want
    X_train, X_watch = cross_validation.train_test_split(train, test_size=0.1,random_state=5)
    train_data = X_train[xgb_features].values
    train_data_full = train[xgb_features].values
    watch_data = X_watch[xgb_features].values
    train_country_destination = X_train['country_destination_hash'].values
    train_country_destination_full = train['country_destination_hash'].values
    watch_country_destination = X_watch['country_destination_hash'].values
    test_data = test[xgb_features].values
    dtrain = xgb.DMatrix(train_data, train_country_destination)
    dtrain_full = xgb.DMatrix(train_data_full, train_country_destination_full)
    dwatch = xgb.DMatrix(watch_data, watch_country_destination)
    dtest = xgb.DMatrix(test_data)
    watchlist = [(dtrain, 'train'),(dwatch, 'watch')]
    if(do_grid_search):
        print('Random search cv')
        gbm_search = xgb.XGBClassifier()
#        num_features = len(xgb_features)
        clf = RandomizedSearchCV(gbm_search,
                                 {'max_depth': sp_randint(1,20), 'learning_rate':sp_rand(0,0.9),
                                  'objective':['multi:softprob'],
                                  'subsample':sp_rand(0.1,0.9),
                                  'colsample_bytree':sp_rand(0.1,0.9),'seed':[random_seed],
                                  'gamma':sp_rand(0,3),'min_child_weight':sp_randint(1,3000),
                                  'max_delta_step':sp_rand(0,20),
                                  'n_estimators': [10,20,30,50,60,75,100,200]},
                                  verbose=10, n_jobs=1, cv = 4, scoring='log_loss', n_iter = 150,
                                  refit=False)
        clf.fit(train_data_full, train_country_destination_full)
        print('best clf score',clf.best_score_)
        print('best params:', clf.best_params_)
        toc=timeit.default_timer()
        print('Grid search time',toc - tic)
    if (use_early_stopping):
        xgb_classifier = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=watchlist,
                            early_stopping_rounds=100, verbose_eval=10,feval = customized_eval, maximize = True)
        y_pred = xgb_classifier.predict(dtest,ntree_limit=xgb_classifier.best_iteration)
    else:
        xgb_classifier = xgb.train(params, dtrain_full, num_boost_round=num_boost_rounds, evals=[(dtrain_full,'train')],
                            verbose_eval=10,feval = customized_eval, maximize = True)
        y_pred = xgb_classifier.predict(dtest)

    if(print_feature_imp):
        create_feature_map(xgb_features)
        imp_dict = xgb_classifier.get_fscore(fmap='xgb.fmap')
        imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1),reverse=True)
        print('{0:<20} {1:>5}'.format('Feature','Imp'))
        print("--------------------------------------")
        for i in imp_dict:
            print ('{0:20} {1:5.0f}'.format(i[0], i[1]))
    #columns = [country_profiles_dict[i] for i in range(0, y_pred.shape[1])]
    country_dest_rev_dict = {v: k for k, v in country_destination_dict.items()}
    columns = [country_dest_rev_dict[i] for i in range(0, y_pred.shape[1])]
    result_xgb_df = pd.DataFrame(index=test.index, columns=columns,data=y_pred)
    #NDF - 0.67909
    #US - 0.23470
    #other - 0.03403
    #FR - 0.01283
    #IT - 0.01004
    #GB - 0.00730
    #CA - 0.00730
    #ES - 0.00725
    #DE - 0.00344
    #NL - 0.00215
    #PT - under 0.00075
    #AU - under 0.00113

    result_xgb_df = norm_rows(result_xgb_df)
    if(reweight_probs):
        if(is_sub_run):
            for col in result_xgb_df.columns.values:
        #        ratio = test_lb_dict[col] / train_2014_country_norm_dict[col]
        #        result_xgb_df[col] = result_xgb_df[col] * ratio / (1 - (1 - ratio) * result_xgb_df[col])
                ratio = test_lb_dict[col] / result_xgb_df[col].mean()
                result_xgb_df[col] = result_xgb_df[col] * ratio
        else:
            test_country_norm_dict = test.country_destination.value_counts(normalize=True)
            for col in result_xgb_df.columns.values:
                ratio = test_country_norm_dict[col] / result_xgb_df[col].mean()
                result_xgb_df[col] = result_xgb_df[col] * ratio

    result_xgb_df = norm_rows(result_xgb_df)

    output_xgb = result_xgb_df.apply(get_top5_country_hashes_string, axis=1)
    output_xgb.name = 'pred'

    if(is_sub_run):
        print('creating xgb output')
        output_xgb_df = pd.concat([output_xgb,test['id']],axis=1)
        output_xgb_df['pred0'] = output_xgb_df['pred'].map(lambda x: x[0])
        output_xgb_df['pred1'] = output_xgb_df['pred'].map(lambda x: x[1])
        output_xgb_df['pred2'] = output_xgb_df['pred'].map(lambda x: x[2])
        output_xgb_df['pred3'] = output_xgb_df['pred'].map(lambda x: x[3])
        output_xgb_df['pred4'] = output_xgb_df['pred'].map(lambda x: x[4])
    else:
        output_xgb_df = pd.concat([output_xgb,test['id']],axis=1)
        output_xgb_df = pd.concat([output_xgb,test['country_destination']],axis=1)
        output_xgb_df['ndcg'] = output_xgb_df.apply(lambda row: get_truth_row(row),axis=1)
        output_xgb_df['ndcg_val'] = output_xgb_df['ndcg'].map(lambda x: rm.ndcg_at_k(x,5,method=1))
        print('xgb ndcg',output_xgb_df['ndcg_val'].mean())

        output_xgb_df['pred0'] = output_xgb_df['pred'].map(lambda x: x[0])
        output_xgb_df['pred1'] = output_xgb_df['pred'].map(lambda x: x[1])
        output_xgb_df['pred2'] = output_xgb_df['pred'].map(lambda x: x[2])
        output_xgb_df['pred3'] = output_xgb_df['pred'].map(lambda x: x[3])
        output_xgb_df['pred4'] = output_xgb_df['pred'].map(lambda x: x[4])

        output_truth = output_xgb_df.groupby('country_destination')
        for name,group in output_truth:
            print(name,'mean','{0:.3f}'.format(group['ndcg_val'].mean()),'count',group['ndcg_val'].count())
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_xgb_df
#%%
combined_2014 = combined.loc[combined.timestamp_year == 2014]
train_2014 = train.loc[(train.timestamp_year == 2014)]
#%%
#28,25,47,44,49,
#%%
xgb_features = ['gender_hash','age_binned','signup_method_hash','signup_flow','language_hash',
                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
                'signup_app_hash','first_device_type_hash','first_browser_hash',
                'first_secs_elapsed_binned','secs_elapsed_mean_binned','secs_elapsed_sum_binned',
                'unique_action_types','unique_device_types',
#                'num_sameday_users_binned',
                'timestamp_dateint_binned',
                'agree_terms_check_possible',
                'time_to_next_user_binned','agree_terms_check_and_not_uncheck',
                'timestamp_hour',
                'timestamp_year','unknowns_summed','no_session_2014','transactions_count_binned',
                'first_action_done_hash','last_action_done_hash',
                'common_actions_sum','rare_actions_sum',
                'impressions_next_action_set','impressions_prev_action_set'

                ]
xgb_features = (xgb_features + action_type_count_cols + device_type_count_cols + detail_action_count_cols +
                action_count_cols)
xgb_features = (xgb_features + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
                summed_action_count_cols)

params = {'learning_rate': 0.02, 'subsample': 0.5,
              'gamma': 0.03,
              'seed': 5,
              'colsample_bytree': 0.5, 'n_estimators': 100,
              'objective': 'multi:softprob',
#              'max_delta_step': 1.15,
              'max_depth': 6,
              'min_child_weight': 5,
              'num_class':12}
num_rounds = 1000
num_boost_rounds = 150

#result_xgb_df = fit_xgb_model(train_2014,test,params,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 5)
result_xgb_df = fit_xgb_model(train_2014,test,params,xgb_features,
                                               num_rounds = num_rounds, num_boost_rounds = 650,
                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 5)
#%%
select_action_done_list = ['action_done_10','action_done_11','action_done_12',
    'action_done_15','action_done_add_guests','action_done_agree_terms_check',
    'action_done_agree_terms_uncheck','action_done_ajax_google_translate',
    'action_done_ajax_google_translate_reviews','action_done_ajax_image_upload',
    'action_done_ajax_photo_widget_form_iframe','action_done_ajax_price_and_availability',
    'action_done_apply_coupon_click','action_done_apply_coupon_click_success',
    'action_done_apply_coupon_error','action_done_apply_coupon_error_type',
    'action_done_apply_reservation','action_done_at_checkpoint','action_done_cancel',
    'action_done_cancellation_policies','action_done_cancellation_policy_click',
    'action_done_change','action_done_change_availability','action_done_clear_reservation',
    'action_done_complete','action_done_complete_redirect','action_done_complete_status',
    'action_done_connect','action_done_coupon_code_click','action_done_coupon_field_focus',
    'action_done_delete','action_done_edit_verification','action_done_email_itinerary_colorbox',
    'action_done_endpoint_error','action_done_friends_new','action_done_guest_booked_elsewhere',
    'action_done_handle_vanity_url','action_done_identity','action_done_image_order',
    'action_done_impressions','action_done_itinerary','action_done_jumio',
    'action_done_jumio_redirect','action_done_jumio_token','action_done_kba',
    'action_done_kba_update','action_done_manage_listing','action_done_message_to_host_change',
    'action_done_message_to_host_focus','action_done_mobile_oauth_callback',
    'action_done_new_session','action_done_p4_refund_policy_terms','action_done_pay',
    'action_done_pending','action_done_phone_verification_call_taking_too_long',
    'action_done_phone_verification_error','action_done_phone_verification_number_submitted_for_call',
    'action_done_phone_verification_number_submitted_for_sms',
    'action_done_phone_verification_number_sucessfully_submitted','action_done_phone_verification_success',
    'action_done_populate_from_facebook','action_done_profile_pic','action_done_push_notification_callback',
    'action_done_qt2','action_done_qt_reply_v2','action_done_qt_with','action_done_read_policy_click',
    'action_done_receipt','action_done_request_new_confirm_email','action_done_requested',
    'action_done_set_user','action_done_slideshow','action_done_terms','action_done_toggle_starred_thread',
    'action_done_travel_plans_current','action_done_verify','action_done_webcam_upload',
    'action_done_goog_trans_reviews_and_english',
    ]

#%%
xgb_features_2 = ['gender_hash','signup_method_hash','signup_flow','language_hash',
                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
                'signup_app_hash','first_device_type_hash','first_browser_hash',
                'first_secs_elapsed_binned',
                'age',
#                'secs_elapsed_mean_binned','secs_elapsed_sum_binned',
#                'secs_elapsed_mean','secs_elapsed_sum',
                'unique_action_types','unique_device_types',
                'num_sameday_users_binned',
                'timestamp_dateint_binned',
#                'agree_terms_check_possible',
                'time_to_next_user_binned','agree_terms_check_and_not_uncheck',
                'dateint_by_basic_with_age_mean',
#                'timestamp_hour',
                'timestamp_year','unknowns_summed','no_session_2014','transactions_count_binned',
#                'first_action_done_hash','last_action_done_hash',
                'first_action_set_hash','last_action_set_hash',
                'secs_elapsed_null_count',
                'most_common_action_set',
                'impressions_next_action_set','impressions_prev_action_set'
#                'common_actions_sum',
#                'rare_actions_sum'

                ]
#xgb_features_2 = (xgb_features_2 + select_action_done_list)
xgb_features_2 = (xgb_features_2 + action_set_cols + summed_action_set_cols)
#xgb_features_2 = (xgb_features_2 + action_type_count_cols + device_type_count_cols + detail_action_count_cols +
#                action_count_cols)
#xgb_features_2 = (xgb_features_2 + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
#                summed_action_count_cols)
params_2 = {'learning_rate': 0.02, 'subsample': 0.7,
#              'gamma': 0.02,
          'reg_alpha':0.5,
          'seed': 7,
          'colsample_bytree': 0.6, 'n_estimators': 100,
          'objective': 'multi:softprob',
#          'max_delta_step': 1.5,
          'max_depth': 5,
          'min_child_weight': 1,
          'num_class':12}
num_rounds_2 = 1000
num_boost_rounds_2 = 150

#result_xgb_2_df = fit_xgb_model(train_2014,test,params_2,xgb_features_2,
#                                               num_rounds = num_rounds_2, num_boost_rounds = num_boost_rounds_2,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 7)
result_xgb_2_df = fit_xgb_model(train_2014,test,params_2,xgb_features_2,
                                               num_rounds = num_rounds_2, num_boost_rounds = 660,
                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 7)
#%%
xgb_features_3 = ['gender_hash',
#                  'age_binned',
                    'age','signup_method_hash','signup_flow','language_hash',
                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
                'signup_app_hash','first_device_type_hash','first_browser_hash',
#                'first_secs_elapsed_binned','secs_elapsed_mean_binned','secs_elapsed_sum_binned',
                'unique_action_types','unique_device_types',
#                'num_sameday_users_binned',
                'timestamp_dateint_binned',
#                'agree_terms_check_possible',
#                'time_to_next_user_binned','agree_terms_check_and_not_uncheck',
                'timestamp_hour','transactions_count',
#                'timestamp_year',
                'unknowns_summed',
                'no_session_2014','transactions_count_binned',
                'first_action_done_hash','last_action_done_hash',
                'first_action_set_hash','last_action_set_hash',
                'common_actions_sum',
                'rare_actions_sum',
                'action_types_by_transactions_count',
                'secs_elapsed_null_count',
                'most_common_action_set',
                'is_action_set_repeat','repeat_action_sets_per_transaction'


                ]
#xgb_features_3 = (xgb_features_3 + select_action_done_list)
xgb_features_3 = (xgb_features_3 + action_set_cols + summed_action_set_cols)
xgb_features_3 = (xgb_features_3 + device_type_count_cols + summed_device_type_count_cols)
#xgb_features_3 = (xgb_features_3 + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
#                summed_action_count_cols)
params_3 = {'learning_rate': 0.03, 'subsample': 0.5,
#              'gamma': 0.04,
              'reg_alpha': 0.2,
              'seed': 7,
              'colsample_bytree': 0.5, 'n_estimators': 100,
              'objective': 'multi:softprob',
#              'max_delta_step': 1.6,
              'max_depth': 8,
              'min_child_weight': 10,
              'num_class':12}
num_rounds_3 = 1000
num_boost_rounds_3 = 150

#result_xgb_3_df = fit_xgb_model(train_2014,test,params_3,xgb_features_3,
#                                               num_rounds = num_rounds_3, num_boost_rounds = num_boost_rounds_3,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 7)
result_xgb_3_df = fit_xgb_model(train_2014,test,params_3,xgb_features_3,
                                               num_rounds = num_rounds_3, num_boost_rounds = 225,
                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 7)
#%%
xgb_features_4 = [
#                 'gender_hash',
#                 'age_binned','signup_method_hash',
#                 'signup_flow','language_hash',
#                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
#                'signup_app_hash','first_device_type_hash','first_browser_hash',
#                'first_secs_elapsed_binned','secs_elapsed_mean_binned','secs_elapsed_sum_binned',
                'unique_action_types','unique_device_types',
#                'num_sameday_users_binned',
                'timestamp_dateint_binned',
                'agree_terms_check_possible',
#                'time_to_next_user_binned','agree_terms_check_and_not_uncheck',
#                'timestamp_hour',
#                'timestamp_year',
#                'unknowns_summed',
                'no_session_2014',
#                'transactions_count_binned',
#                'first_action_done_hash','last_action_done_hash',
#                'common_actions_sum',
#                'rare_actions_sum'

                ]
#xgb_features_4 = (xgb_features_4 + select_action_done_list)
xgb_features_4 = (xgb_features_4 + action_set_cols + summed_action_set_cols)
#xgb_features_4 = (xgb_features_4 + device_type_count_cols + summed_device_type_count_cols)
#xgb_features_4 = (xgb_features_4 + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
#                summed_action_count_cols)
params_4 = {'learning_rate': 0.3, 'subsample': 0.9,
              'gamma': 0.04,
              'seed': 8,
              'colsample_bytree': 0.9, 'n_estimators': 100,
              'objective': 'multi:softprob',
              'max_delta_step': 1.1,
              'max_depth': 4,
              'min_child_weight': 1,
              'num_class':12}
num_rounds_4 = 1000
num_boost_rounds_4 = 150

#result_xgb_4_df = fit_xgb_model(train_2014,test,params_4,xgb_features_4,
#                                               num_rounds = num_rounds_4, num_boost_rounds = num_boost_rounds_4,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 7)
result_xgb_4_df = fit_xgb_model(train_2014,test,params_4,xgb_features_4,
                                               num_rounds = num_rounds_4, num_boost_rounds = 50,
                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 7)
#%%
xgb_features_5 = ['gender_hash',
                  'age_binned',
                  'signup_method_hash','signup_flow','language_hash',
#                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
#                'signup_app_hash','first_device_type_hash','first_browser_hash',
#                'first_secs_elapsed_binned','secs_elapsed_mean_binned','secs_elapsed_sum_binned',
                'unique_action_types','unique_device_types',
#                'num_sameday_users_binned',
#                'timestamp_dateint_binned',
#                'agree_terms_check_possible',
#                'time_to_next_user_binned','agree_terms_check_and_not_uncheck',
#                'timestamp_hour',
                'timestamp_year','unknowns_summed','no_session_2014',
                'transactions_count_binned',
#                'first_action_done_hash','last_action_done_hash',
#                'common_actions_sum',
#                'rare_actions_sum'

                ]
#xgb_features_5 = (xgb_features_5 + action_set_cols + summed_action_set_cols)
xgb_features_5 = (xgb_features_5 + action_per_time_cols)
xgb_features_5 = (xgb_features_5 + device_type_count_cols + detail_action_count_cols)
#xgb_features_5 = (xgb_features_5 + action_type_count_cols + device_type_count_cols + detail_action_count_cols +
#                action_count_cols)
#xgb_features_5 = (xgb_features_5 + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
#                summed_action_count_cols)
params_5 = {'learning_rate': 0.02, 'subsample': 0.7,
#              'gamma': 0.02,
          'reg_alpha':0.2,
          'seed': 7,
          'colsample_bytree': 0.6, 'n_estimators': 100,
          'objective': 'multi:softprob',
#          'max_delta_step': 1.5,
          'max_depth': 7,
          'min_child_weight': 1,
          'num_class':12}
num_rounds_5 = 1000
num_boost_rounds_5 = 150

#result_xgb_5_df = fit_xgb_model(train_2014,test,params_5,xgb_features_5,
#                                               num_rounds = num_rounds_5, num_boost_rounds = num_boost_rounds_5,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                               random_seed = 8)
result_xgb_5_df = fit_xgb_model(train_2014,test,params_5,xgb_features_5,
                                               num_rounds = num_rounds_5, num_boost_rounds = 275,
                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
                                               random_seed = 8)
#%%
#xgb_features_6 = ['gender_hash',
#                  'age_binned',
#                  'signup_method_hash','signup_flow','language_hash',
##                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
##                'signup_app_hash','first_device_type_hash','first_browser_hash',
##                'first_secs_elapsed_binned','secs_elapsed_mean_binned','secs_elapsed_sum_binned',
#                'unique_action_types','unique_device_types',
##                'num_sameday_users_binned',
##                'timestamp_dateint_binned',
##                'agree_terms_check_possible',
##                'time_to_next_user_binned','agree_terms_check_and_not_uncheck',
##                'timestamp_hour',
#                'timestamp_year','unknowns_summed','no_session_2014',
#                'transactions_count_binned',
##                'first_action_done_hash','last_action_done_hash',
##                'common_actions_sum',
##                'rare_actions_sum'
#
#                ]
##xgb_features_5 = (xgb_features_5 + action_set_cols + summed_action_set_cols)
#xgb_features_6 = (xgb_features_6 + action_per_time_cols)
#xgb_features_6 = (xgb_features_6 + device_type_count_cols + detail_action_count_cols)
##xgb_features_5 = (xgb_features_5 + action_type_count_cols + device_type_count_cols + detail_action_count_cols +
##                action_count_cols)
##xgb_features_5 = (xgb_features_5 + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
##                summed_action_count_cols)
#params_6 = {'learning_rate': 0.1, 'subsample': 0.6,
##              'gamma': 0.02,
#          'reg_alpha':0.02,
#          'seed': 7,
#          'colsample_bytree': 0.6, 'n_estimators': 100,
#          'objective': 'multi:softprob',
##          'max_delta_step': 1.5,
#          'max_depth': 7,
#          'min_child_weight': 1,
#          'num_class':12}
#num_rounds_6 = 1000
#num_boost_rounds_6 = 150
#
#train_late = train_2014.loc[train_2014.timestamp_dateint >= 1600]
##result_xgb_6_df = fit_xgb_model(train_late,test,params_6,xgb_features_6,
##                                               num_rounds = num_rounds_6, num_boost_rounds = num_boost_rounds_6,
##                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
##                                               random_seed = 8)
#result_xgb_6_df = fit_xgb_model(train_late,test,params_6,xgb_features_6,
#                                               num_rounds = num_rounds_6, num_boost_rounds = 64,
#                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
#                                               random_seed = 8)
#%%

#position_action_set_cols = [col for col in list(train_2014) if col.startswith('position_set_')]
#xgb_features_6 = ['gender_hash','age_binned','signup_method_hash','signup_flow','language_hash',
#                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
#                'signup_app_hash','first_device_type_hash','first_browser_hash',
#                'first_secs_elapsed_binned',
##                'secs_elapsed_mean_binned','secs_elapsed_sum_binned',
##                'secs_elapsed_mean','secs_elapsed_sum',
#                'unique_action_types','unique_device_types',
#                'num_sameday_users_binned',
#                'timestamp_dateint_binned',
##                'agree_terms_check_possible',
#                'time_to_next_user_binned','agree_terms_check_and_not_uncheck',
##                'timestamp_hour',
#                'timestamp_year','unknowns_summed','no_session_2014','transactions_count',
##                'first_action_done_hash','last_action_done_hash',
#                'first_action_set_hash','last_action_set_hash',
##                'common_actions_sum',
##                'rare_actions_sum'
#
#                ]
##xgb_features_6 = (xgb_features_6 + select_action_done_list)
#xgb_features_6 = (xgb_features_6 + action_set_cols + summed_action_set_cols)
#xgb_features_6 = (xgb_features_6 + position_action_set_cols)
##xgb_features_6 = (xgb_features_6 + action_type_count_cols + device_type_count_cols + detail_action_count_cols +
##                action_count_cols)
##xgb_features_6 = (xgb_features_6 + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
##                summed_action_count_cols)
#params_6 = {'learning_rate': 0.02, 'subsample': 0.5,
##              'gamma': 0.02,
#          'reg_alpha':0.2,
#          'seed': 7,
#          'colsample_bytree': 0.5, 'n_estimators': 100,
#          'objective': 'multi:softprob',
##          'max_delta_step': 1.5,
#          'max_depth': 4,
#          'min_child_weight': 10,
#          'num_class':12}
#num_rounds_6 = 1000
#num_boost_rounds_6 = 150

#result_xgb_6_df = fit_xgb_model(train_2014,test,params_6,xgb_features_6,
#                                               num_rounds = num_rounds_6, num_boost_rounds = num_boost_rounds_6,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 7)
#result_xgb_6_df = fit_xgb_model(train_2014,test,params_6,xgb_features_6,
#                                               num_rounds = num_rounds_6, num_boost_rounds = 350,
#                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
#                                               random_seed = 7)
#%%
xgb_features_select = [
                'gender_hash',
                'age_binned','signup_method_hash',
                'signup_flow',
                'language_hash',
                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
                'signup_app_hash','first_device_type_hash','first_browser_hash',
                'first_secs_elapsed_binned','secs_elapsed_mean_binned','secs_elapsed_sum_binned',
                'unique_action_types','unique_device_types',
                'num_sameday_users_binned',
                'timestamp_dateint_binned',
                'agree_terms_check_possible',
                'time_to_next_user_binned',
                'agree_terms_check_and_not_uncheck',
#                'timestamp_hour',
                'timestamp_year','unknowns_summed','no_session_2014','transactions_count_binned',
                'first_action_done_hash','last_action_done_hash'
                ]
xgb_features_select = (xgb_features_select + action_type_count_cols + device_type_count_cols + detail_action_count_cols +
                action_count_cols)
xgb_features_select = (xgb_features_select + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
                summed_action_count_cols)
xgb_features_select = (xgb_features_select + action_set_cols + summed_action_set_cols)

#train_2014 = train.loc[(train.timestamp_year == 2014)]
train_2014_select = train_2014.loc[train_2014.age != -1]
test_select = test.loc[test.age != -1]

params_select = {'learning_rate': 0.02, 'subsample': 0.8,
#              'gamma': 0.5,
              'seed': 5,
              'reg_alpha':0.2,
              'colsample_bytree': 0.2, 'n_estimators': 200,
              'objective': 'multi:softprob',
#              'max_delta_step': 1,
              'max_depth': 4,
              'min_child_weight': 1,
              'num_class':12}
num_rounds_select = 1000
num_boost_rounds_select = 150
#
#result_xgb_df_select = fit_xgb_model(train_2014_select,test_select,params_select,xgb_features_select,
#                                               num_rounds = num_rounds_select, num_boost_rounds = num_boost_rounds_select,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 5, reweight_probs = False)
#result_xgb_df_select = fit_xgb_model(train_2014_select,test_select,params_select,xgb_features_select,
#                                               num_rounds = num_rounds_select, num_boost_rounds = 200,
#                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
#                                               random_seed = 5, reweight_probs = False)
#%%

#action_done_null_value
#%%

#print_value_counts(combined,'time_to_next_user_binned',is_normalized=False)
#print_value_counts(combined,'age_binned',is_normalized=False)
#print_value_counts(train,'timestamp_weekday')

#print_value_counts(combined_2014,'action_done_book')
#
#for col in action_set_cols:
#    print(col)
#    print_value_counts(combined_2014,col,is_normalized=False)

#action_type_modify (450 dummy none train)

#combined_2014_basic = combined.loc[combined.signup_method == 'basic']
#combined_2014_en = combined.loc[combined.language == 'en']
#combined_2014_en = combined.loc[combined.language != 'en']
#combined_2014_goog_trans = combined.loc[combined.action_done_ajax_google_translate_reviews == 1]
#combined_2014_action_p5 = combined.loc[combined.action_detail_p5 == 1]
#print_value_counts(combined_2014_basic,'age',is_normalized=False)
#print_value_counts(combined_2014_fr,'action_done_ajax_google_translate_description',is_normalized=False)
#print_value_counts(combined_2014_action_p5,'first_action_done',is_normalized=False)
#print_value_counts(combined_2014_goog_trans,'action_done_ajax_google_translate_description',is_normalized=False)


#----------
#action dones to investigate:
#action_done_agree_terms_check -- 0.06802 US, 0.02652 NDF (0.00075 from all PT)
#action_done_custom_recommended_destinations -- 	0.05320 US, 0.08455 NDF
#action_done_this_hosting_reviews_3000 -- 	0.00521 US, 0.00456 NDF


#action_done_my_reservations -- 2 to 1 NDF, not important
#-------
#action_done_ajax_photo_widget_form_iframe,
#action_done_message_to_host_change,action_done_message_to_host_focus --strongly correlated with agree_terms_check

#action_done_campaigns

#action_done_complete_redirect
#action_done_at_checkpoint
#action_done_agree_terms_uncheck -- important to check

#action_done_book,action_done_braintree_client_token
#action_done_print_confirmation

#action_done_apply_coupon_click,action_done_apply_coupon_click_success
#action_done_coupon_code_click,action_done_coupon_field_focus


#action_done_notifications
#action_done_impressions

#action_done_ajax_google_translate,action_done_ajax_google_translate_reviews,action_done_show_code

#action_done_email_itinerary_colorbox,action_done_guest_billing_receipt,action_done_guest_booked_elsewhere,action_done_add_guests
#action_done_receipt
#%%
params_no_sessions = {'learning_rate': 0.05, 'subsample': 0.3,
#              'gamma': 0.1,
              'reg_alpha': 0.1,
              'seed': 5,
              'colsample_bytree': 0.9, 'n_estimators': 100,
              'objective': 'multi:softprob',
              'max_delta_step': 2,
              'max_depth': 3,
              'min_child_weight': 1,
              'num_class':12}
xgb_features_no_sessions = [
#               'gender_hash',
                'age',
#                'age_binned',
                'signup_method_hash',
#                'signup_flow',
                'language_hash',
                'dateint_by_basic_with_age_mean',
#                'affiliate_channel_hash',
#                'affiliate_provider_hash',
#                'first_affiliate_tracked_hash',
#                'signup_app_hash',
#                'first_device_type_hash',
                 'first_browser_hash',
#                'timestamp_dateint_binned',
#                'agree_terms_check_and_not_uncheck',
#                'num_sameday_users_binned',
#                'timestamp_hour',
#                'timestamp_minute',
#                'timestamp_month',
#                'timestamp_second',
#                'timestamp_day',

#                'action_detail_p5',
#                'action_detail_book_it',
#                'action_done_requested',
#                'action_done_pending',
#                'action_done_at_checkpoint',
#                'action_done_agree_terms_check',
                'action_type_booking_request',
                'no_session_2014',
                'time_to_next_user_binned',
                'timestamp_year',
#                'unknowns_summed'
                ]

num_rounds_no_sessions = 1000
num_boost_rounds_no_sessions = 150
#result_xgb_no_sessions_df = fit_xgb_model(train,test,params_no_sessions,xgb_features_no_sessions,
#                                            num_rounds = num_rounds_no_sessions, num_boost_rounds = num_boost_rounds_no_sessions,
#                                            do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                            random_seed = 5)
result_xgb_no_sessions_df = fit_xgb_model(train,test,params_no_sessions,xgb_features_no_sessions,
                                            num_rounds = num_rounds_no_sessions, num_boost_rounds = 70,
                                            do_grid_search = False, use_early_stopping = False, print_feature_imp = True,
                                            random_seed = 5)
#%%
params_sessions = {'learning_rate': 0.2, 'subsample': 0.9,
#              'gamma': 0.15,
              'reg_alpha': 0.7,
              'seed': 5,
              'colsample_bytree': 0.9, 'n_estimators': 100,
              'objective': 'multi:softprob',
              'max_delta_step': 4,
              'max_depth': 5,
              'min_child_weight': 1,
              'num_class':12}
xgb_features_sessions = [
#                'gender_hash',
                'age_binned',
                'signup_method_hash',
#                'signup_flow',
#                'age',
                'language_hash',
#                'affiliate_channel_hash','affiliate_provider_hash','first_affiliate_tracked_hash',
#                'signup_app_hash','first_device_type_hash','first_browser_hash',
                'first_secs_elapsed_binned',
                'secs_elapsed_mean_binned','secs_elapsed_sum_binned',
#                'unique_action_types',
                'unique_device_types',
                'agree_terms_check_and_not_uncheck','agree_terms_check_possible',
                'timestamp_dateint_binned',
#                'num_sameday_users_binned',
#                'timestamp_second',
#                'timestamp_hour',
#                'timestamp_year',
#                'unknowns_summed',
                'no_session_2014',
                'transactions_count_binned',
                'first_action_done_hash',
#                'common_actions_sum','rare_actions_sum'
#                'last_action_done_hash'
                ]
xgb_features_sessions = (xgb_features_sessions +
#                 action_type_count_cols +
                 device_type_count_cols +
#                 detail_action_count_cols +
                action_count_cols
                )
#xgb_features_sessions = (xgb_features_sessions + summed_action_type_count_cols + summed_device_type_count_cols + summed_detail_action_count_cols +
#                summed_action_count_cols)
num_rounds_sessions = 1000
num_boost_rounds_sessions = 150
#result_xgb_sessions_df = fit_xgb_model(train_2014,test,params_sessions,xgb_features_sessions,
#                                            num_rounds = num_rounds_sessions, num_boost_rounds = num_boost_rounds_sessions,
#                                            do_grid_search = False, use_early_stopping = True, print_feature_imp = True,
#                                            random_seed = 6)
result_xgb_sessions_df = fit_xgb_model(train_2014,test,params_sessions,xgb_features_sessions,
                                            num_rounds = num_rounds_sessions, num_boost_rounds = 55,
                                            do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
                                            random_seed = 6)
#%%
tic=timeit.default_timer()
#result_ens_df = 0.1*result_xgb_no_sessions_df + 0.9*result_xgb_df + 0.1 * result_xgb_sessions_df
#result_ens_df = 0.15*result_xgb_no_sessions_df + 0.9*result_xgb_df + 0.2 * result_xgb_sessions_df
#result_ens_df = 0.0*result_xgb_no_sessions_df + 0.9*result_xgb_df + 0.2 * result_xgb_sessions_df
result_ens_df = (0.18*result_xgb_no_sessions_df + 0.8*result_xgb_df + 0.4 * result_xgb_sessions_df
                + 0.5 * result_xgb_2_df
                + 1.0 * result_xgb_3_df
                + 0.18 * result_xgb_4_df
                + 1.0 * result_xgb_5_df
#                + 0.05 * result_xgb_6_df
#                + 0.001 * result_xgb_6_df
                )
#result_ens_df = (1.0 * result_xgb_5_df)
#result_ens_df = 0.0*result_xgb_no_sessions_df + 1.0*result_xgb_df + 0.0 * result_xgb_sessions_df
result_ens_df = norm_rows(result_ens_df)

output_ens = result_ens_df.apply(get_top5_country_hashes_string, axis=1)
output_ens.name = 'pred'

if(is_sub_run):
    print('creating xgb output')
    output_ens_df = pd.concat([output_ens,test['id']],axis=1)
    output_ens_df['pred0'] = output_ens_df['pred'].map(lambda x: x[0])
    output_ens_df['pred1'] = output_ens_df['pred'].map(lambda x: x[1])
    output_ens_df['pred2'] = output_ens_df['pred'].map(lambda x: x[2])
    output_ens_df['pred3'] = output_ens_df['pred'].map(lambda x: x[3])
    output_ens_df['pred4'] = output_ens_df['pred'].map(lambda x: x[4])
else:
    output_ens_df = pd.concat([output_ens,test['id']],axis=1)
    output_ens_df = pd.concat([output_ens,test['country_destination']],axis=1)
    output_ens_df['ndcg'] = output_ens_df.apply(lambda row: get_truth_row(row),axis=1)
    output_ens_df['ndcg_val'] = output_ens_df['ndcg'].map(lambda x: rm.ndcg_at_k(x,5,method=1))
    print('xgb ens ndcg',output_ens_df['ndcg_val'].mean())

    output_ens_df['pred0'] = output_ens_df['pred'].map(lambda x: x[0])
    output_ens_df['pred1'] = output_ens_df['pred'].map(lambda x: x[1])
    output_ens_df['pred2'] = output_ens_df['pred'].map(lambda x: x[2])
    output_ens_df['pred3'] = output_ens_df['pred'].map(lambda x: x[3])
    output_ens_df['pred4'] = output_ens_df['pred'].map(lambda x: x[4])

    output_truth = output_ens_df.groupby('country_destination')
    for name,group in output_truth:
        print(name,'mean','{0:.3f}'.format(group['ndcg_val'].mean()),'count',group['ndcg_val'].count())
toc=timeit.default_timer()
print('Ensembling Time',toc - tic)

tic=timeit.default_timer()

def reweight_by_cond(result_df,test_df,train_df,col_name,col_value,do_reweight_ndf_us = False,reweight_factor = 1.0):
    output_df = result_df.copy()
    condition = test_df[col_name] == col_value
    train_subset = train_df.loc[train_df[col_name] == col_value]
    reweight_dict = train_subset.country_destination.value_counts(normalize=True)
    for col in output_df[condition].columns.values:
        try:
            ratio = reweight_dict[col] / output_df[col][condition].mean()
        except KeyError:
            ratio = 1.0
        ratio = ratio ** reweight_factor #be a bit more conservative
        if((not do_reweight_ndf_us) and ((col == 'US') | (col == 'NDF'))):
            ratio = 1
        output_df[col][condition] = output_df[col][condition] * ratio
    output_df = norm_rows(output_df)
    return output_df

def lower_prob(result_df,test_df,col_name,col_value,dest_to_lower='NDF',reweight_factor = 1.0):
    output_df = result_df.copy()
    condition = test_df[col_name] == col_value
    ratio = reweight_factor
    output_df[dest_to_lower][condition] = output_df[dest_to_lower][condition] * ratio
    output_df = norm_rows(output_df)
    return output_df

def lower_prob_if_greater(result_df,test_df,col_name,col_value,dest_to_lower='NDF',reweight_factor = 1.0):
    output_df = result_df.copy()
    condition = test_df[col_name] >= col_value
    ratio = reweight_factor
    output_df[dest_to_lower][condition] = output_df[dest_to_lower][condition] * ratio
    output_df = norm_rows(output_df)
    return output_df
result_ens3_df = result_ens_df
result_ens3_df = norm_rows(result_ens3_df)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','fr',reweight_factor = 0.4)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','de',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','nl',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','es',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','it',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','ko',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','zh',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','pt',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','no',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','sv',reweight_factor = 0.6)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','ru',reweight_factor = 0.6)

result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','de',reweight_factor = 0.7)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','nl',reweight_factor = 0.9)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','es',reweight_factor = 0.6)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','it',reweight_factor = 0.5)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','ko',reweight_factor = 0.2)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','zh',reweight_factor = 0.4)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','pt',reweight_factor = 0.4)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','no',reweight_factor = 0.2)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','sv',reweight_factor = 0.2)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'language','ru',reweight_factor = 0.2)

result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'age',105,do_reweight_ndf_us=True,reweight_factor = 0.7)

result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_done_impressions',1,do_reweight_ndf_us=True,reweight_factor = 0.8)

result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_done_receipt',1,do_reweight_ndf_us=True,reweight_factor = 0.7)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_done_add_guests',1,do_reweight_ndf_us=True,reweight_factor = 0.7)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_done_goog_trans_reviews_and_english',1,do_reweight_ndf_us=True,reweight_factor = 0.7)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'google_with_age',1,do_reweight_ndf_us=True,reweight_factor = 0.6)

result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'first_action_set','travel_plans_currentyour_tripsview',do_reweight_ndf_us=True,reweight_factor = 0.5)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'first_action_set','agree_terms_check-unknown--unknown-',do_reweight_ndf_us=True,reweight_factor = 0.5)

result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'last_action_set','travel_plans_currentyour_tripsview',do_reweight_ndf_us=True,reweight_factor = 0.5)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'last_action_set','createcreate_phone_numberssubmit',do_reweight_ndf_us=True,reweight_factor = 0.5)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'last_action_set','requestedpost_checkout_actionsubmit',do_reweight_ndf_us=True,reweight_factor = 0.5)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'last_action_set','phone_verification_number_sucessfully_submitted-unknown--unknown-',do_reweight_ndf_us=True,reweight_factor = 0.5)
result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'last_action_set','languages_multiselect-unknown--unknown-',do_reweight_ndf_us=True,reweight_factor = 0.5)


#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_set_'+str(action_set_dict['clear_reservation-unknown--unknown-']),1,do_reweight_ndf_us=True,reweight_factor = 0.7)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_set_'+str(action_set_dict['requestedpost_checkout_actionsubmit']),1,do_reweight_ndf_us=True,reweight_factor = 0.7)

#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_set_'+str(action_set_dict['itineraryguest_itineraryview']),1,do_reweight_ndf_us=True,reweight_factor = 0.5)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_set_'+str(action_set_dict['ajax_price_and_availabilityalteration_fieldclick']),1,do_reweight_ndf_us=True,reweight_factor = 0.5)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_set_'+str(action_set_dict['remove_dashboard_alertremove_dashboard_alertclick']),1,do_reweight_ndf_us=True,reweight_factor = 0.5)

#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_set_'+str(action_set_dict['pendingpendingbooking_request']),1,do_reweight_ndf_us=True,reweight_factor = 0.9)
#result_ens3_df = reweight_by_cond(result_ens3_df,test,train,'action_set_'+str(action_set_dict['requestedp5view']),1,do_reweight_ndf_us=True,reweight_factor = 0.5)

#action_set_135

#29,77,135,208,257
#24,25

result_ens3_df = norm_rows(result_ens3_df)

#result_ens3_df = lower_prob_if_greater(result_ens3_df,test,'summed_action_done_requested',6,'NDF',0.8)
#result_ens3_df = lower_prob_if_greater(result_ens3_df,test,'summed_action_set_77',2,'NDF',0.8)



#result_ens3_df = lower_prob(result_ens3_df,test,'age',29,'NDF',0.85)
#result_ens3_df = lower_prob(result_ens3_df,test,'age',105,'NDF',1.2)

result_ens3_df = lower_prob(result_ens3_df,test,'action_done_print_confirmation',1,'NDF',0.8)
#result_ens3_df = lower_prob(result_ens3_df,test,'action_done_add_guests',1,'NDF',0.65)
result_ens3_df = lower_prob(result_ens3_df,test,'action_done_email_itinerary_colorbox',1,'NDF',0.8)
#result_ens3_df = lower_prob(result_ens3_df,test,'action_done_this_hosting_reviews_3000',1,'NDF',0.65)

result_ens3_df['NDF'] = result_ens3_df['NDF']
result_ens3_df = norm_rows(result_ens3_df)

if(is_sub_run):
    for col in result_ens3_df.columns.values:
        ratio = test_lb_dict[col] / result_ens3_df[col].mean()
        result_ens3_df[col] = result_ens3_df[col] * ratio
else:
    test_country_norm_dict = test.country_destination.value_counts(normalize=True)
    for col in result_ens3_df.columns.values:
        ratio = test_country_norm_dict[col] / result_ens3_df[col].mean()
        result_ens3_df[col] = result_ens3_df[col] * ratio


output = result_ens3_df.apply(get_top5_country_hashes_string, axis=1)
output.name = 'pred'
if(is_sub_run):
    print('creating output')
    output_df = pd.concat([output,test['id']],axis=1)
    output_df['pred0'] = output_df['pred'].map(lambda x: x[0])
    output_df['pred1'] = output_df['pred'].map(lambda x: x[1])
    output_df['pred2'] = output_df['pred'].map(lambda x: x[2])
    output_df['pred3'] = output_df['pred'].map(lambda x: x[3])
    output_df['pred4'] = output_df['pred'].map(lambda x: x[4])

else:
    output_df = pd.concat([output,test['country_destination']],axis=1)


#    output_df = pd.concat([output,test['country_destination']],axis=1)
#    output_df = pd.concat([output_df,test['age']],axis=1)
#    output_df = output_df.loc[output_df.age != -1]

    output_df['ndcg'] = output_df.apply(lambda row: get_truth_row(row),axis=1)
    output_df['ndcg_val'] = output_df['ndcg'].map(lambda x: rm.ndcg_at_k(x,5,method=1))
    print('ndcg ens2',output_df['ndcg_val'].mean())
    output_df['pred0'] = output_df['pred'].map(lambda x: x[0])
    output_df['pred1'] = output_df['pred'].map(lambda x: x[1])
    output_df['pred2'] = output_df['pred'].map(lambda x: x[2])
    output_df['pred3'] = output_df['pred'].map(lambda x: x[3])
    output_df['pred4'] = output_df['pred'].map(lambda x: x[4])
    output_truth = output_df.groupby('country_destination')
    for name,group in output_truth:
        print(name,'mean','{0:.3f}'.format(group['ndcg_val'].mean()),'count',group['ndcg_val'].count())
    output_df = pd.concat([output_df,test['user_id']],axis=1)
toc=timeit.default_timer()
print('Reweight Time',toc - tic)
#%%
#if (not is_sub_run):
#    wrong_ndf = output_df.loc[(output_df.country_destination == 'US') & (output_df.ndcg_val != 1)]
#    wrong_ndf_ids = set(wrong_ndf.user_id.unique())
#    test_copy = test.copy()
#    age_ids = set(test_copy.loc[test_copy.age == -1].user_id.unique())
#    sessions_old_copy = sessions_old.copy()
#    sessions_old_copy['has_user'] = sessions_old_copy['user_id'].map(lambda x: x in wrong_ndf_ids)
#    sessions_old_copy['has_user_age'] = sessions_old_copy['user_id'].map(lambda x: x in age_ids)
#    sessions_old_copy = sessions_old_copy.loc[sessions_old_copy['has_user']]
#    sessions_old_copy_age_null = sessions_old_copy.loc[sessions_old_copy['has_user_age']]
#    test_copy['has_user'] = test_copy['user_id'].map(lambda x: x in wrong_ndf_ids)
#    test_copy = test_copy.loc[test_copy['has_user']]
#%%
combined_impressions = combined_2014.loc[combined_2014.action_done_impressions == 1]
#%%
#train_impressions = train.loc[train.action_done_impressions == 1]
#train_impressions_ids = set(train_impressions.user_id.unique())
#sessions_old_copy2 = sessions_old.copy()
#sessions_old_copy2['has_user'] = sessions_old_copy2['user_id'].map(lambda x: x in train_impressions_ids)
#sessions_old_copy2 = sessions_old_copy2.loc[sessions_old_copy2['has_user']]
#sessions_old_copy2['next_7'] = ((sessions_old_copy2['action_set'] == 361) & (sessions_old_copy2['next_action_set'] == 7)).astype(int)
#sessions_old_copy2['prev_7'] = ((sessions_old_copy2['action_set'] == 7) & (sessions_old_copy2['next_action_set'] == 361)).astype(int)
#sessions_old_copy2['sanwich_7'] = ((sessions_old_copy2['next_7'] == 1) & (sessions_old_copy2['prev_7'] == 1)).astype(int)
#sessions_old_copy2.sort('user_id',inplace=True)
#sessions_old_copy2['prev_as'] = sessions_old_copy2['action_set'].shift(1)
#sessions_old_copy2['prev_as'].fillna(0,inplace=True)
#sessions_old_copy2_imp = sessions_old_copy2.loc[sessions_old_copy2['action_done'] == 'impressions']
#%%
#check actino set 361
#combined_age_105 = combined.loc[combined.age == 105]
#combined_age_is_null = combined.loc[(combined.age == -1) & (combined.country_destination != 'dummy')]
#181
#%%
test_df_ens2 = pd.merge(test,result_ens3_df,left_on = test.index.values,right_on = result_ens3_df.index.values,how='left')
#temp_test = test_df_ens2.loc[(test_df_ens2['language'] == 'en') & (test_df_ens2['action_done_ajax_google_translate_reviews'])]
#temp_test = test_df_ens2.loc[(test_df_ens2['age'] != -1) & (test_df_ens2['signup_method'] == 'basic')]

#test_impressions = test_df_ens2.loc[test_df_ens2.action_done_impressions == 1]

temp_test = test_df_ens2.loc[(test_df_ens2['action_done_impressions'] == 1)]
possible_countries = ['NDF','US','other','FR','IT','ES','GB','CA']
for dest in possible_countries:
    dest_prob = temp_test[dest].mean()
    print(dest,dest_prob)

#action_sets
#29,77,135,208,257
#24,25

#last_action_set
#createcreate_phone_numberssubmit,travel_plans_currentyour_tripsview
#requestedpost_checkout_actionsubmit,languages_multiselect-unknown--unknown-
#read_policy_clickread_policy_clickclick
#phone_verification_number_sucessfully_submitted-unknown--unknown-
#custom_recommended_destinations-unknown--unknown- (all dummy 343)

#first_action_set
#travel_plans_currentyour_tripsview
#agree_terms_check-unknown--unknown-
#forgot_passwordforgot_passwordclick,spoken_languagesuser_languagesdata
#all dummy:
##custom_recommended_destinations-unknown--unknown-,indexview_reservationsview,
#indexview_locationsview,
#%%
#check action_set_165
#338,365,52
#%%
#action_set_rev_dict = {v: k for k, v in action_set_dict.items()}
#%%
#is_sub_run = True
if(is_sub_run):
#    output_df = test
#    agree_check_cond = (output_df['action_done_impressions'] == 1)
#
#    output_df['pred0'] = 'PT'
#    output_df['pred0'][agree_check_cond] = 'other'
#    output_df = output_df.loc[agree_check_cond]
#
#    output_df['pred1'] = 'other'
#    output_df['pred2'] = 'FR'
#    output_df['pred3'] = 'IT'
#    output_df['pred4'] = 'ES'
#    test = test[['id','pred0','pred1','pred2','pred3','pred4']]

#    output_df = output_df[['id','pred0','pred1','pred2','pred3','pred4']]
#    output_df = output_ens_df[['id','pred0','pred1','pred2','pred3','pred4']]

#    output_df = output_xgb_df[['id','pred0','pred1','pred2','pred3','pred4']]

    test_all = output_df[['id','pred0','pred1','pred2','pred3','pred4']]
#    test_all = output_df[['id','pred0']]
    test_long = test_all[['id','pred0']].rename(columns={'pred0':'country'})
    test_long = pd.concat([test_long,test_all[['id','pred1']].rename(columns={'pred1':'country'})],
                          axis=0,ignore_index=True)
    test_long = pd.concat([test_long,test_all[['id','pred2']].rename(columns={'pred2':'country'})],
                          axis=0,ignore_index=True)
    test_long = pd.concat([test_long,test_all[['id','pred3']].rename(columns={'pred3':'country'})],
                          axis=0,ignore_index=True)
    test_long = pd.concat([test_long,test_all[['id','pred4']].rename(columns={'pred4':'country'})],
                          axis=0,ignore_index=True)

    test_long.to_csv('basic_airbnb.csv', index=False)
    print('submission created')
#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)