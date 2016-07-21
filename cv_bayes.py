# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:02:25 2016

@author: babou
"""

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
from bayes_opt import BayesianOptimization


def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
#              gamma,
#              min_child_weight,
#              max_delta_step,
              subsample,
              colsample_bytree,
              ratio=131.708,
              silent =True,
              nthread = -1,
              seed = 42):
    return cross_val_score(XGBClassifier(max_depth = int(max_depth),
                                         learning_rate = learning_rate,
                                         n_estimators = int(n_estimators),
                                         silent = silent,
                                         nthread = nthread,
#                                         gamma = gamma,
#                                         min_child_weight = min_child_weight,
#                                         max_delta_step = max_delta_step,
                                         subsample = subsample,
                                         colsample_bytree = colsample_bytree,
                                         scale_pos_weight = ratio,
                                         seed = seed),
                           X,
                           y,
                           scoring='f1',
                           cv=5).mean()



if __name__ == "__main__":
    data = pd.read_csv('data/data.csv', encoding='utf-8')#, dtype=dtype)

    useful_col = [col for col in data.columns if col not in ['word', 'doc_name', 'paragraph_nb', 'firstname_is_french']]
    
    word_save = data['word']
    doc_name_save = data['doc_name']
    paragraph_nb_save = data['paragraph_nb']
    firstname_is_french_save = data['firstname_is_french']
    
    data = data[useful_col]
    y = data['is_target']
    data = data.drop('is_target', axis=1)
    X = data
    
    ratio = float(np.sum(y == 0)) / np.sum(y==1) # 131.708

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (5, 8),
                                      'learning_rate': (0.01, 0.3),
                                      'n_estimators': (150, 300),
#                                      'gamma': (1., 0.01),
#                                      'min_child_weight': (1, 10),
#                                      'max_delta_step': (0, 0.1),
                                      'subsample': (0.85, 1),
                                      'colsample_bytree' :(0.5, 1),
#                                      'scale_pos_weight' : ratio
                                     })
                                     
    xgboostBO.maximize(init_points=10, n_iter=40)
    print('-'*53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
    
    print('-'*53)
    print xgboostBO.res['max']