##Code Created by: Thiyagarajan Ramanathan
#USC ID: 4973341255
##EE660 Final Project PUBG Finish Placement Prediction
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import keras
from keras.layers import Dense
from __future__ import division
import os
import gc, sys
gc.enable()

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
return df

def pre_process(train, is_train=True):
    if is_train:
        train=train[train['maxPlace']>1] ##Remove games which have only one player
        y=train.groupby(['matchId','groupId'])['winPlacePerc'].agg('mean')
        train=train.drop(['winPlacePerc'], axis=1)
    train['totalDistance']=train['swimDistance']+train['walkDistance']+train['rideDistance']
    train['headshotrate'] = train['kills']/train['headshotKills']
    train['killStreakrate'] = train['killStreaks']/train['kills']
    train['healthitems'] = train['heals'] + train['boosts']
    train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
    train['killPlace_over_maxPlace'] = train['killPlace'] / train['maxPlace']
    train['headshotKills_over_kills'] = train['headshotKills'] / train['kills']
    train['distance_over_weapons'] = train['totalDistance'] / train['weaponsAcquired']
    train['walkDistance_over_heals'] = train['walkDistance'] / train['heals']
    train['walkDistance_over_kills'] = train['walkDistance'] / train['kills']
    train['skill'] = train['headshotKills'] + train['roadKills']
    train[train == np.Inf] = np.NaN
    train[train == np.NINF] = np.NaN
    train.fillna(0, inplace=True)
    train=train.drop(['swimDistance', 'walkDistance','rideDistance'], axis=1)
    train['teamSize']=train.groupby('groupId')['groupId'].transform('count')
    features=list(train.columns)
    features.remove("Id")
    features.remove("numGroups")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")
    train_out = train.groupby(['matchId','groupId'])[features].agg({'avg':'mean','max': 'max', 'min': 'min','sum':'sum'})
    train_out.columns = ["_".join(x) for x in train_out.columns.ravel()]
    train_out=train_out.replace([np.inf, -np.inf], 0)
    train_out=train_out.drop(['max_teamSize', 'min_teamSize','sum_teamSize'], axis=1)
    features=list(train_out.columns)
    train_rank = train_out.groupby('matchId')[features].rank(pct=True, na_option= 'top')
    train_final = train_out.reset_index()[['matchId','groupId']]
    train_final = train_final.merge(train_out.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    train_final = train_final.merge(train_rank, suffixes=["","_rank"], how='left', on=['matchId', 'groupId'])
    train_final['matchSize']=train_final.groupby('matchId')['matchId'].transform('count')
    if is_train:
        train_final=train_final.drop(['matchId', 'groupId'], axis=1)
    train_final=reduce_mem_usage(train_final)
    del train_rank, train_out, train
        gc.collect()
        if is_train:
            return train_final, y
    return train_final

Train=reduce_mem_usage(pd.read_csv("../input/train_V2.csv"))
##doing the split between the training set and the custom test set
main_train_match, test_match = model_selection.train_test_split(Train.matchId.unique(), test_size=0.2)
main_train=Train.loc[Train['matchId'].isin(main_train_match)]
test=Train.loc[Train['matchId'].isin(test_match)]

##Creating the pre-train and train set
sub_train_match, pre_train_match = model_selection.train_test_split(main_train.matchId.unique(), test_size=0.1)
sub_train=main_train.loc[main_train['matchId'].isin(sub_train_match)]
pre_train=main_train.loc[main_train['matchId'].isin(pre_train_match)]

##Creating the validation and sub-train set
train_match, val_match = model_selection.train_test_split(sub_train.matchId.unique(), test_size=0.2)
train = sub_train.loc[sub_train['matchId'].isin(train_match)]
val = sub_train.loc[sub_train['matchId'].isin(val_match)]

print("Final dataset splits")
print("Total training set size: ", len(Train.matchId.unique()))
print("Main train set size: ", len(main_train.matchId.unique()))
print("Main test set size: ", len(test.matchId.unique()))
print("Pre train set size: ", len(pre_train.matchId.unique()))
print("Sub-Training set size: ", len(sub_train.matchId.unique()))
print("Training set size: ",len(train.matchId.unique()))
print("Validation set size: ", len(val.matchId.unique()))
del Train, main_train, sub_train
gc.collect()

X_train, y_train = pre_process(train, is_train=True)
X_val, y_val = pre_process(val, is_train=True)

##Linear Regression
lr=LinearRegression(fit_intercept=True, normalize=False)
lr.fit(X_train, y_train )
print("Training error: ", mean_absolute_error(y_train, lr.predict(X_train)))
print("Test error: ", mean_absolute_error(y_val, lr.predict(X_val)))

##Ridge Regression
lr=Ridge(fit_intercept=True, normalize=False, alpha=0.0001)
lr.fit(X_train, y_train )
print("Training error: ", mean_absolute_error(y_train, lr.predict(X_train)))
print("Test error: ", mean_absolute_error(y_val, lr.predict(X_val)))

##MLP Regressor
model = keras.Sequential()
model.add(Dense(1000, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(500, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
model.summary()
model.fit(X_train, y_train, batch_size=10000, epochs=200,validation_data=(X_val, y_val), shuffle=True)

##Random Forest Regressor
rf = RandomForestRegressor(max_depth=None, n_estimators=40, criterion='mae', n_jobs=-1)
##For training on subset of data
_, train, _, y = model_selection.train_test_split(X_train, y_train, test_size=0.01)
rf.fit(train,y)
print("Training done")
err_train=mean_absolute_error(y_train, rf.predict(X_train))
err_val=mean_absolute_error(y_val, rf.predict(X_val))
print("Train Prediction error: ", err_train)
print("Val Prediction error: ", err_val)
error.append([err_train, err_val, depth, estimator])

##XGBoost
xgbreg = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=10000, silent=True, objective='reg:linear', booster='gbtree')
xgbreg.fit(X_train,y_train,verbose = 50)
print("Training done")
err_train=mean_absolute_error(y_train, xgbreg.predict(X_train))
err_val=mean_absolute_error(y_val, xgbreg.predict(X_val))
print("Train Prediction error: ", err_train)
print("Val Prediction error: ", err_val)

##Light GBM
params = {"objective" : "regression", "metric" : "mae", 'n_estimators':10000, 'early_stopping_rounds':200,
    "num_leaves" : 30, "learning_rate" : 0.05, "bagging_fraction" : 0.7,"colsample_bytree" : 0.7}
lgtrain = lgb.Dataset(X_train, label=y_train)
lgval = lgb.Dataset(X_val, label=y_val)
model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)
pred_train_y = model.predict(X_train, num_iteration=model.best_iteration)
pred_val_y = model.predict(X_val, num_iteration=model.best_iteration)

test=pre_process(test, is_train=False)
pred_test_y=model.predict(test.drop(['matchId','groupId'], axis=1), num_iteration=model.best_iteration)

##Only for LGBM
test=pd.read_csv("../input/test_V2.csv")
final_out=test[['Id', 'groupId']]
test=pre_process(test, is_train=False)
values=zip(list(test['groupId'].values), list(pred_test_y))
df = pd.DataFrame(data = list(values), columns=['groupId', 'winPlacePerc'])
final_out=final_out.merge(df, on='groupId', how='left')

final_out=final_out.drop(['groupId'], axis=1)
submission = final_out
submission.loc[submission['winPlacePerc']>1,'winPlacePerc']=1
submission.loc[submission['winPlacePerc']<0,'winPlacePerc']=0
submission.to_csv('submission.csv', index=False)


##Reference:
##the mem reduce function was used from https://www.kaggle.com/gemartin
##to prevent the kaggle kernel from crashing
