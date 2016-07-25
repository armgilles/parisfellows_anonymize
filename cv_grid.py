# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:46:55 2016

@author: babou
"""

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score

from scipy.stats import sem

def display_scores(params, scores, append_star=False):
    """Format the mean score +/- std error for params"""
    params = ", ".join("{0}={1}".format(k, v)
                      for k, v in params.items())
    line = "{0}:\t{1:.3f} (+/-{2:.3f})".format(
        params, np.mean(scores), sem(scores))
    if append_star:
        line += " *"
    return line

def display_grid_scores(grid_scores, top=None):
    """Helper function to format a report on a grid of scores"""
    
    grid_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
    if top is not None:
        grid_scores = grid_scores[:top]
        
    # Compute a threshold for staring models with overlapping
    # stderr:
    _, best_mean, best_scores = grid_scores[0]
    threshold = best_mean - 2 * sem(best_scores)
    
    for params, mean_score, scores in grid_scores:
        append_star = mean_score + 2 * sem(scores) > threshold
        print(display_scores(params, scores, append_star=append_star))



def xg_f1(yhat, y):
    y = y.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in yhat] # binaryzing your output
    return 'f1',f1_score(y, y_bin)

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

ratio = float(np.sum(y == 0)) / np.sum(y==1)

#clf = xgb.XGBClassifier(learning_rate=0.1, max_depth=8, n_estimators=220,
#                        nthread=-1, scale_pos_weight=ratio, seed=42)
                        
skf = StratifiedKFold(y, n_folds=5, random_state=2016)

#grid = GridSearchCV(clf, param_grid=params, cv=skf, scoring='f1', early_stopping_rounds=25)
                    #, early_stopping_rounds=25)

#grid.fit(X, y)
#xgtrain = xgb.DMatrix(X.values, y.values)
#cvresult = xgb.cv(params, xgtrain, 300, nfold=5, feval=xg_f1, maximize=True, early_stopping_rounds=30)

#score = cross_val_score(clf, X, y, cv=skf, scoring='f1')

params = {
    'max_depth':[6, 8]
#    'min_child_weight':range(1,6,2)
}


gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.1, n_estimators=350, max_depth=8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=ratio, seed=42), 
                       param_grid = params, scoring='f1',n_jobs=4, cv=skf)
gsearch1.fit(X,y)

