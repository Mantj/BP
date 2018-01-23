# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:33:20 2017

@author: Administrator
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
import numpy as np


l_model = []
params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'eta': 0.1,
            'max_depth': 7,
            'seed': 2017,
            'silent': 0,
            'eval_metric': 'rmse'
        }
for day in range(1,6):
    for i in range(3,21):
        print('starting...%d_'%day + str(i))
        if(i == 3):
            df_in =  pd.read_csv(r'./Training_%d/Training_%d_%d.csv' %(day,day,i))
            feature=[x for x in df_in.columns if x not in ['wind','hour','label']]
            train_preds = np.zeros(df_in.shape[0])
            kf = KFold(len(df_in), n_folds = 5, shuffle=True, random_state=520)
            for kf_num, (train_index, test_index) in enumerate(kf):
                print('第{}次训练...'.format(kf_num+1))
                train_feat1 = df_in.iloc[train_index]
                train_feat2 = df_in.iloc[test_index]
                xgbtrain = xgb.DMatrix(train_feat1[feature],train_feat1['wind'])
                xgbtest = xgb.DMatrix(train_feat2[feature],train_feat2['wind'])
                
                watchlist = [ (xgbtrain,'train'), (xgbtest, 'test') ]
                num_rounds=1000
                print('Start training %s...' %str(i))
                model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=25,verbose_eval=False)
                xgb.plot_importance(model)
                plt.savefig(r'./new_model/pic_Training_%d_%d.png' %(day,i))
                print('Start predicting %s...' %str(i))
                train_preds[test_index] += model.predict(xgbtest)
                model.save_model(r'./new_model/Training_%d_%d_KF%d.model' %(day,i,kf_num+1))
            y_1 = list(train_preds)
        else:
            df_in =  pd.read_csv(r'./Training_%d/Training_%d_%d.csv' %(day,day,i))
            df_in['before'] = y_1
            feature=[x for x in df_in.columns if x not in ['wind','hour','label']]
            train_preds = np.zeros(df_in.shape[0])
            kf = KFold(len(df_in), n_folds = 5, shuffle=True, random_state=520)
            
            for kf_num, (train_index, test_index) in enumerate(kf):
                print('第{}次训练...'.format(kf_num))
                train_feat1 = df_in.iloc[train_index]
                train_feat2 = df_in.iloc[test_index]
                xgbtrain = xgb.DMatrix(train_feat1[feature],train_feat1['wind'])
                xgbtest = xgb.DMatrix(train_feat2[feature],train_feat2['wind'])
                
                watchlist = [ (xgbtrain,'train'), (xgbtest, 'test') ]
                num_rounds=1000
                print('Start training %s...' %str(i))
                model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=25,verbose_eval=False)
                xgb.plot_importance(model)
                plt.savefig(r'./new_model/pic_Training_%d_%d.png' %(day,i))
                print('Start predicting %s...' %str(i))
                train_preds[test_index] += model.predict(xgbtest)
                model.save_model(r'./new_model/Training_%d_%d_KF%d.model' %(day,i,kf_num+1))
            y_1 = list(train_preds)
        print('finish...' + str(i))
