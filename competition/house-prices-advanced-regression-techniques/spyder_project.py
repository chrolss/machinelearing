# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:39:58 2018

@author: Christian Olsson
"""
#%% Import basic libraries and data
import pandas as pd

filepath = 'train.csv'
df = pd.read_csv(filepath)


#%% Label encoding and N/A-filling before we create new features

for i in range(0,81):
    if (df.iloc[:,i].dtype == 'O'):
        df.iloc[:,i] = df.iloc[:,i].fillna("0")
    else:
        df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].mean())


#%% Raw work with the data

""" Create a new column and drop unwanted ones, e.g. pure year columns should be dropped """

df['YearRenSold'] = df.YrSold - df.YearRemodAdd
df['GarageAge'] = df.YrSold - df.GarageYrBlt

df = df.drop(columns=['YearRemodAdd','GarageYrBlt','YrSold','MoSold'])

#%% Preprocessing

from sklearn import preprocessing
le = preprocessing.LabelEncoder() 
df = df.apply(preprocessing.LabelEncoder().fit_transform)

#%% Train-test-split

from sklearn.model_selection import train_test_split

y = df['SalePrice']
X = df.drop(['SalePrice'],axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size = .2)

#%% Train model

import xgboost as xgb

my_model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.35)
my_model.fit(train_X, train_y, early_stopping_rounds=500, eval_set=[(test_X, test_y)], verbose=False)

#%% Make predictions and evaluate model

# standard, no tuning -> 35.98 & 2334.3
# n_est = 1000, early_stopping_round = 5 -> 35.6 & 2290.3
# n_est = 1000, early_stopping_round = 5, learning_rate = 0.05 -> 35.3 & 2235.4
# n_est = 1000, early_stopping_round = 5, learning_rate = 0.05 (drop new version) -> 34.1 & 2148.0
# n_est = 2000, early_stopping_round = 100, learning_rate = 0.05 (drop new version) -> 32.8 & 2074.1
# n_est = 2000, early_stopping_round = 300, learning_rate = 0.03 (drop new version) -> 34.0 & 1901.4
# n_est = 2000, early_stopping_round = 500, learning_rate = 0.025 (drop new version) -> 33.4 & 1848.9
predictions = my_model.predict(test_X)
# mean absolute error
from sklearn.metrics import mean_absolute_error
print('Mean Absolute Error : ' + str(mean_absolute_error(predictions, test_y)))
# mean squared error
from sklearn.metrics import mean_squared_error
print('Mean Squared Error: ' + str(mean_squared_error(predictions, test_y)))
percentErr = 0
for j in range(0,len(test_y)):
    percentErr = percentErr + (test_y[j]-predictions[j])/test_y[j]
    
percentErr = percentErr / len(test_y)
print('Percentage error: ' + str(percentErr*100) + '%')

#%% Test where the largest error is


err_marg = 5
for i in range (0,292):
    if (abs((test_y[i] - predictions[i])/test_y[i])>err_marg):
        print(i)

#%% Print some stuff

import matplotlib.pyplot  as plt
plt.plot(test_y)
plt.plot(predictions)
plt.show()     
        
plt.plot((test_y - predictions)/test_y)
plt.show()

#%% Start prediction the competition data

testFile = "test.csv"

test_data = pd.read_csv(testFile)
#%%
# Do the same data-manipulation as with the training data

for i in range(0,80):
    if (test_data.iloc[:,i].dtype == 'O'):
        test_data.iloc[:,i] = test_data.iloc[:,i].fillna("0")
    else:
        test_data.iloc[:,i] = test_data.iloc[:,i].fillna(test_data.iloc[:,i].mean())

#%%
test_data['YearRenSold'] = test_data.YrSold - test_data.YearRemodAdd
test_data['GarageAge'] = test_data.YrSold - test_data.GarageYrBlt

test_data = test_data.drop(columns=['YearRemodAdd','GarageYrBlt','YrSold','MoSold'])

#%%

test_data = test_data.apply(preprocessing.LabelEncoder().fit_transform)
#%%
test_data = test_data.reindex(X.columns, axis=1)

test_preds = my_model.predict(test_data.values)
output = pd.DataFrame({'Id': test_data['Id'],
                       'SalePrice': test_preds*1000})

output.to_csv('submission.csv', index=False)



