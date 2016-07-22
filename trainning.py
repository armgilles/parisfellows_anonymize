# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:07:26 2016

@author: babou
"""

import pandas as pd
import numpy as np
import operator

import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


DUMMY = False #To dummys caterogical features

#################################################
###                 FUNCTIONS                 ###
#################################################

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    
    
def xgboost_feature_importance(model, train, return_df=False):
    
        
    features = train.columns
    create_feature_map(features)
    
    importance = model.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    
    sns.barplot(x="fscore", y="feature", data=df)
#    plt.xticks(range(len(df)), df.feature.tolist(), rotation=60)
    plt.title('Feature Importances')
    plt.ylabel('Relative Importance')
    
    print df
    
    if return_df is True:
        return df

def get_metric(y_test, y_pred, plot=False):
    """
    Calcul metrics.
    In : y_test, y_pred
    Return : 
    If plot == True, then plot CM normalize
    """
    # Metrics
    metrics_classification = classification_report(y_test, y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    auc = roc_auc_score(y_test, y_pred)
    
    recall = recall_score(y_test, y_pred)
    
    print "Metrics classification : " 
    print metrics_classification
    print "Accuracy score : "
    print accuracy 
    print "Roc auc score : "
    print auc
    print "Recall score : "
    print recall


def get_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    label_unique = y_test.unique()
#    #Graph Confusion Matrix
    tick_marks = np.arange(len(label_unique))
#    plt.figure(figsize=(8,6))
    sns.heatmap(cm_normalized, cmap='Greens',annot=True,linewidths=.5)
#    plt.title('confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(tick_marks + 0.5, list(label_unique))
    plt.yticks(tick_marks + 0.5,list(reversed(list(label_unique))) , rotation=0)
#    
#    plt.imshow(cm_normalized, interpolation='nearest', cmap='Greens')
#    plt.title('confusion matrix')
#    plt.colorbar()
#    tick_marks = np.arange(len(label_unique))
#    plt.xticks(tick_marks + 0.5, list(reversed(list(label_unique))))
#    plt.yticks(tick_marks + 0.5,list(label_unique) , rotation=0)
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')

def recall(yhat, y):
    y = y.get_label()
    y_pred = [1 if pred > 0.5 else 0 for pred in yhat]
    return "recall", recall_score(y, y_pred)

def xg_f1(yhat, y):
    y = y.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in yhat] # binaryzing your output
    return 'f1',f1_score(y, y_bin)


#################################################
###                 GENERAL                   ###
#################################################

#dtype = {'word_encoded' : 'str',
#         'word_encoded_shift_1b' : 'str',
#         'word_encoded_shift_2b' : 'str',
#         'word_encoded_shift_1a' : 'str',
#         'word_encoded_shift_2a' : 'str'}
         
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

if DUMMY == True:
    X = pd.get_dummies(X)
    

# Split data to get an unknow dataset (valide): 
X_trainning, X_valide, y_trainning, y_valide = train_test_split(X, y, stratify=y, 
                                                                   test_size=0.20, random_state=20)
                                                                   

# Split data to get X_train / X_test :
X_train, X_test, y_train, y_test = train_test_split(X_trainning, y_trainning, stratify=y_trainning, 
                                                                   test_size=0.33, random_state=21)


dtrain = xgb.DMatrix(X_train, y_train, missing=-1)
dtest = xgb.DMatrix(X_test, y_test, missing=-1)
evallist = [(dtrain, 'train'), (dtest, 'test')]

params = {'max_depth':8,#12,
         'eta':0.1,#0.01,
#         'subsample':0.85,#0.8,
#         'colsample_bytree':0.95,#0.7,
         'silent':1,
         'scale_pos_weight' : ratio,
#         'min_child_weight': 6,
        # 'max_delta_step': 0.086,
         'objective':'binary:logistic',
         'nthread':8,
         'seed':42}
         
num_round = 400         

bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=25,
                feval=xg_f1, maximize=True)

# Validation on X_test
y_pred = bst.predict(xgb.DMatrix(X_test), ntree_limit=bst.best_ntree_limit)
y_pred_b = [1. if y_cont > 0.5 else 0. for y_cont in y_pred] # binaryzing your output
auc_test = recall_score(y_test, y_pred_b)
print "RECALL on Test dataset: "+ str(auc_test) 

# Validation on Valide dataset
y_pred_valide = bst.predict(xgb.DMatrix(X_valide), ntree_limit=bst.best_ntree_limit)
y_pred_valide_b = [1. if y_cont > 0.5 else 0. for y_cont in y_pred_valide] # binaryzing your output
auc_valide = recall_score(y_valide, y_pred_valide_b)
print "RECALL on unknow dataset: "+ str(auc_valide) 


X_valide = X_valide.join(word_save)
X_valide = X_valide.join(doc_name_save)
X_valide = X_valide.join(paragraph_nb_save)
X_valide = X_valide.join(firstname_is_french_save)
X_valide['is_target'] = y_valide
X_valide['y_pred'] = y_pred_valide_b
X_valide['y_pred_proba'] = y_pred_valide
X_valide['error'] =0
X_valide.loc[X_valide['is_target'] != X_valide['y_pred'], 'error'] = 1


# False Positive selector
fp = X_valide[(X_valide.error == 1) & (X_valide.is_target == 1)]
# False Negative selector
fn = X_valide[(X_valide.error == 1) & (X_valide.is_target == 0)]

print "_"*54
print "Some metrics : "
get_metric(y_valide, y_pred_valide_b)

print "_"*54
print "Confusion matrix : "
cm = confusion_matrix(y_valide, y_pred_valide_b)
print cm