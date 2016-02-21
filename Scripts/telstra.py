import os
import numpy as np
from scipy.stats import mode, skew, kurtosis, entropy
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import sys
my_dir = os.getcwd()
sys.path.append(my_dir+'/Telstra/Scripts')

from feat_eng import *

(X_all, y, num_class, n_train, n_feat) = feat_eng()

X = X_all[:n_train, :n_feat]
X_cat = X_all[:n_train, :]

## specify parameters for xgb
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
param['eval_metric'] = 'mlogloss'
param['num_class'] = num_class
param['nthread'] = 10
param['silent'] = 1

param['eta'] = 0.02
param['colsample_bytree'] = 0.5
param['subsample'] = 0.9
param['max_depth'] = 8

num_round = 10000

X_test = X_all[n_train:, :n_feat]
X_cat_test = X_all[n_train:, :]

# set random seed
np.random.seed(0)

LR = LogisticRegression(solver='lbfgs', multi_class='multinomial')
LR_fit = LR.fit(X_cat, y)
y_pred_test_LR = np.reshape(LR_fit.predict(X_cat_test), (X_test.shape[0], 1))
X_test = np.concatenate((X_test, y_pred_test_LR), axis=1)
xg_test = xgb.DMatrix(X_test)

best_score = []
y_pred_sum = np.zeros((X_test.shape[0], num_class))
# k-fold
R = 1
for r in range(R):
  k = 5
  kf = KFold(X.shape[0], n_folds=k, shuffle=True)
  i = 0
  for train, val in kf:
    i += 1
    print(i)
    X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
    X_cat_train, X_cat_val = X_cat[train], X_cat[val]
    LR_fit = LR.fit(X_cat_train, y_train)
    y_pred_train_LR = np.reshape(LR_fit.predict(X_cat_train), (X_train.shape[0], 1))
    y_pred_val_LR = np.reshape(LR_fit.predict(X_cat_val), (X_val.shape[0], 1))
    X_train = np.concatenate((X_train, y_pred_train_LR), axis=1)
    X_val = np.concatenate((X_val, y_pred_val_LR), axis=1)
    xg_train = xgb.DMatrix(X_train, y_train)
    xg_val = xgb.DMatrix(X_val, y_val)
    evallist  = [(xg_train,'train'), (xg_val,'eval')]
    # train
    bst = xgb.train(param, xg_train, num_round, evallist, early_stopping_rounds=100)
    best_score += [bst.best_score]
    # predict
    y_pred = bst.predict(xg_test, ntree_limit=bst.best_iteration)
    y_pred_sum = y_pred_sum+y_pred

# average
y_pred = y_pred_sum/(k*R)
print(np.mean(best_score))

# save pred
sub = pd.DataFrame(data={'id':ids, 'predict_0':y_pred[:, 0], 'predict_1':y_pred[:, 1],
	'predict_2':y_pred[:, 2]}, columns=['id', 'predict_0', 'predict_1', 'predict_2'])
my_dir = os.getcwd()+'/Telstra/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)
