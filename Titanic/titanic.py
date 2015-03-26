# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 04:46:11 2015

@author: Jared
"""
import csv as csv
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pylab as p
from sklearn.ensemble import RandomForestClassifier
import timeit

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('/Users/Jared/DataAnalysis/Titanic/train.csv', header=0)
# TEST DATA
test_df = pd.read_csv('/Users/Jared/DataAnalysis/Titanic/test.csv', header=0) 
# Load the test file into a dataframe
#%%
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df = df.drop(['Sex','Ticket',
              'PassengerId'], axis=1)

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
test_df = test_df.drop(['Sex','Ticket',
                        'PassengerId'], axis=1)

median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & 
        (df['Pclass'] == j+1)]['Age'].dropna().median()
 
##print(median_ages)

df['AgeFill'] = df['Age']
test_df['AgeFill'] = test_df['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),
        'AgeFill'] = median_ages[i,j]
        test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & 
                     (test_df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df = df.drop(['Age'], axis=1)
test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)
test_df = test_df.drop(['Age'], axis=1)

##experiment with binning Age to avoid mixing categorical / floats in Forest
df['Young'] = 0
df['MiddleAged'] = 0
df['Old'] = 0
test_df['Young'] = 0
test_df['MiddleAged'] = 0
test_df['Old'] = 0
max_young_age = 15
max_middle_age = 50

df.loc[df.AgeFill < max_young_age, 'Young'] = 1
df.loc[(max_young_age < df.AgeFill) & (df.AgeFill <= max_middle_age),
       'MiddleAged'] = 1
df.loc[df.AgeFill > max_middle_age,'Old'] = 1
test_df.loc[test_df.AgeFill < max_young_age, 'Young'] = 1
test_df.loc[(max_young_age < test_df.AgeFill) & (test_df.AgeFill <= max_middle_age),
       'MiddleAged'] = 1
test_df.loc[test_df.AgeFill > max_middle_age,'Old'] = 1




##hist1 = df['AgeFill'].hist(bins=16, range=(0,80), alpha = .5)
#P.show()
#%%
def getLastName(name):
    last_name = ''
    for c in name:
        if(c == ','):
            break
        last_name += c
    return last_name

def getTitle(name):
    title = ''
    after_comma = False
    for c in name:
        if (c == ' '):
            continue
        if(not after_comma):
            if(c ==','):
                after_comma = True
            continue
        else:
            if(c =='.'):
                break
            title += c
    return title

#maybe add male/female surviving family as well
#df['MaleFamSurv'] =
#%%
#Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
#Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
#train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int


df['Title'] = df.Name.map(lambda x: getTitle(x))
test_df['Title'] = test_df.Name.map(lambda x: getTitle(x))
def getIntFromTitle(title):
    if(title == 'Ms' or title == 'Miss'):
        return 0
    elif(title == 'Lady' or title == 'Mlle' or title == 'Dona' or
         title == 'Mme' or title == 'theCountess'):
        return 0
    elif(title == 'Mrs'):
        return 0
    elif(title == 'Mr'):
        return 1
    elif(title == 'Master' or 'Jonkheer'):
        return 2
    else:
        return 3
df['TitleInt'] = df.Title.map(lambda x: getIntFromTitle(x)).astype(int)
test_df['TitleInt'] = test_df.Title.map(lambda x: getIntFromTitle(x)).astype(int)
#%%
df['HaveFamOnShip'] = 0
test_df['HaveFamOnShip'] = 0
df.loc[(df.SibSp + df.Parch) > 0, 'HaveFamOnShip'] = 1
test_df.loc[(test_df.SibSp + test_df.Parch) > 0, 'HaveFamOnShip'] = 1

# All missing Embarked -> just make them embark from most common place
if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
    df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

#%%
df['HaveCabinInfo'] = 1
test_df['HaveCabinInfo'] = 1
df.loc[(df.Cabin.isnull()), 'HaveCabinInfo'] = 0
test_df.loc[test_df.Cabin.isnull(), 'HaveCabinInfo'] = 0

df['InCcabin'] = 0
test_df['InCcabin'] = 0
df['InCcabin'] = df[df.Cabin.isnull() != 1]['Cabin'].map(lambda x: 1 if x[0]=='C' else 0)
df.loc[df.InCcabin.isnull(), 'InCcabin'] = 0
test_df['InCcabin'] = test_df[test_df.Cabin.isnull() != 1]['Cabin'].map(lambda x: 1 if x[0]=='C' else 0)
test_df.loc[test_df.InCcabin.isnull(), 'InCcabin'] = 0
#test_df.loc[test_df.Cabin.isnull(), 'InCcabin'] = 1

#%%
#df = df.drop(['SibSp','Parch'],axis=1)
#test_df = test_df.drop(['SibSp','Parch'],axis=1)
#%%
#tic=timeit.default_timer()
df['FamSurv'] = 0
df['MaleFamSurv'] = 0
df['FemaleFamSurv'] = 0
last_name_field_list = []
for i in range(0, len(df['Name'])):
    last_name_field_list.append(getLastName(df['Name'][i]))
for i in range(0, len(df['Name'])):
    surviving_family = 0
    surviving_male_family = 0
    surviving_female_family = 0
    last_name_cand = getLastName(df['Name'][i])
    for j in range(0, len(df['Name'])):
        if(i == j):
            continue
        last_name_field = last_name_field_list[j]
        if((last_name_cand == last_name_field) & (df['Survived'][j] == 1)):
            surviving_family += 1
        elif((last_name_cand == last_name_field)):
            surviving_family -= 1
        if((last_name_cand == last_name_field) & (df['Survived'][j] == 1) &
           (df.Gender[j] == 0)):
            surviving_female_family += 1
        elif((last_name_cand == last_name_field) & (df.Gender[j] == 0)):
            surviving_female_family -= 1
        if((last_name_cand == last_name_field) & (df['Survived'][j] == 1) &
           (df.Gender[j] == 1)):
            surviving_male_family += 1
        elif((last_name_cand == last_name_field) & (df.Gender[j] == 1)):
            surviving_male_family -= 1
    df['FamSurv'][i] = surviving_family
    df['FemaleFamSurv'][i] = surviving_female_family
    df['MaleFamSurv'][i] = surviving_male_family
#%%
test_df['FamSurv'] = 0
test_df['MaleFamSurv'] = 0
test_df['FemaleFamSurv'] = 0
for i in range(0, len(test_df['Name'])):
    surviving_family = 0
    surviving_male_family = 0
    surviving_female_family = 0
    last_name_cand = getLastName(test_df['Name'][i])
    for j in range(0, len(df['Name'])):
        if(i == j):
            continue
        last_name_field = last_name_field_list[j]
        if((last_name_cand == last_name_field) & (df['Survived'][j] == 1)):
            surviving_family += 1
        elif((last_name_cand == last_name_field)):
            surviving_family -= 1
        if((last_name_cand == last_name_field) & (df['Survived'][j] == 1) &
           (df.Gender[j] == 0)):
            surviving_female_family += 1
        elif((last_name_cand == last_name_field) & (df.Gender[j] == 0)):
            surviving_female_family -= 1
        if((last_name_cand == last_name_field) & (df['Survived'][j] == 1) &
           (df.Gender[j] == 1)):
            surviving_male_family += 1
        elif((last_name_cand == last_name_field) & (df.Gender[j] == 1)):
            surviving_male_family -= 1
    test_df['FamSurv'][i] = surviving_family
    test_df['FemaleFamSurv'][i] = surviving_female_family
    test_df['MaleFamSurv'][i] = surviving_male_family
#toc=timeit.default_timer()
#print('Time',toc - tic)
    
#%%
    

#%%
#df = df.drop(['Name','Fare'],axis=1)
#test_df = test_df.drop(['Name','Fare'],axis=1)
#df = df.drop(['Name'],axis=1)
#test_df = test_df.drop(['Name'],axis=1)
#%%
#df['HighFare'] = 0
#fare_level = 45
#df.loc[(df.Fare > 45), 'HighFare'] = 1
#test_df['HighFare'] = 0
#test_df.loc[(test_df.Fare > 45), 'HighFare'] = 1
#
#df['LowFare'] = 0
#low_fare_level = 9
#df.loc[((df.Fare <= low_fare_level) & (df.Fare > 0) ) , 'LowFare'] = 1
#test_df['LowFare'] = 0
#test_df.loc[((test_df.Fare <= low_fare_level) & (test_df.Fare > 0) ) , 'LowFare'] = 1
#%%
p.scatter(df['FamSurv'],df['Survived'],alpha=0.1)
p.savefig('Images/scatter.png', bbox_inches='tight')
df['FamSurv'].hist(bins=12, range=(-6,6), alpha = .5)
p.savefig('Images/fam_surv_hist.png', bbox_inches='tight')
df.FamSurv[df.Survived == 1].hist(bins=12, range=(-6,6), alpha = .5)
df.FamSurv[df.Survived == 0].hist(bins=12, range=(-6,6), alpha = .5)
p.savefig('Images/fam_surv_success.png', bbox_inches='tight')

p.figure()
df.FamSurv[df.Gender == 1].hist(bins=12, range=(-6,6), alpha = .5)
df.FamSurv[(df.Survived == 1) & (df.Gender == 1)].hist(bins=12, range=(-6,6), alpha = .5)
df.FamSurv[(df.Survived == 0) & (df.Gender == 1)].hist(bins=12, range=(-6,6), alpha = .5)
p.savefig('Images/male_fam_surv_.png', bbox_inches='tight')
#p.show()
#%%
p.figure()
test_df.FamSurv.hist(bins=20, range=(-10,10), alpha = .5)
p.savefig('Images/test_fam_surv.png', bbox_inches='tight')
#%%
#df = df.drop(['FamSurv'],axis=1)
#test_df = test_df.drop(['FamSurv'],axis=1)
#df = df.drop(['Gender'],axis=1)
#test_df = test_df.drop(['Gender'],axis=1)
#df = df.drop(['AgeIsNull','AgeFill', 'Young', 'MiddleAged', 'Old'],axis=1)
#test_df = test_df.drop(['AgeIsNull','AgeFill','Young','MiddleAged',
     #                   'Old'],axis=1)
##TESTING
#df['Noise'] = 0
#test_df['Noise'] = 0
#df.loc[df.index % 2 == 0, 'Noise'] = 1
#test_df.loc[test_df.index % 2 == 0, 'Noise'] = 1
#%%
# The data is now ready to go. 
#So lets fit to the train, then predict to the test!
# Convert back to a numpy array

train_data_df = df.drop(['MaleFamSurv','AgeIsNull', 'Parch', 'SibSp', 'Fare',
                      'Embarked','HaveFamOnShip',
                      'AgeFill', 'MiddleAged', 'Old','Cabin', 'HaveCabinInfo',
                      'InCcabin', 'Title','Name'],
                      axis = 1)
test_data_hidden_df = test_df.drop(['MaleFamSurv', 'AgeIsNull', 'Parch', 'SibSp',
                                 'Fare','HaveCabinInfo','InCcabin',
                                 'Embarked','HaveFamOnShip', 
                                 'AgeFill', 'MiddleAged', 'Old','Cabin',
                                 'Title','Name'],
                                axis=1)                 

train_data = train_data_df[train_data_df.index > -1].values
test_data_hidden = test_data_hidden_df.values
test_data = train_data_df[train_data_df.index <= 200].values

tic=timeit.default_timer()
print ('Training...')
forest = RandomForestClassifier(n_estimators=5000, random_state=1)
#forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print('Training Score',forest.score(test_data[0::,1::], test_data[0::,0]))

print ('Predicting...')
output = forest.predict(test_data_hidden).astype(int)



predictions_file = open("mysecondforestTitanic.csv", "wt")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print ('Done.')
toc=timeit.default_timer()
print('Time',toc - tic)


#%%
