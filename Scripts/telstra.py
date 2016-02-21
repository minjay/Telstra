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

# load data
my_dir = os.getcwd()
df_train = pd.read_csv(my_dir+'/Telstra/Data/train.csv')
df_test = pd.read_csv(my_dir+'/Telstra/Data/test.csv')
df_eve = pd.read_csv(my_dir+'/Telstra/Data/event_type.csv')
df_log = pd.read_csv(my_dir+'/Telstra/Data/log_feature.csv')
df_res = pd.read_csv(my_dir+'/Telstra/Data/resource_type.csv')
df_sev = pd.read_csv(my_dir+'/Telstra/Data/severity_type.csv')

y = df_train['fault_severity'].values
num_class = max(y)+1
df_train.drop('fault_severity', axis=1, inplace=True)
ids = df_test['id'].values

n_train = df_train.shape[0]
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)


# combine
df_all_cb = df_all.join(df_loc_log_vol_sum, on='location')
df_all_cb = df_all_cb.join(df_loc_log_feat_num, on='location')
df_all_cb = df_all_cb.join(df_loc_log_feat_max, on='location')
df_all_cb = df_all_cb.join(df_loc_eve_max, on='location')
df_all_cb = df_all_cb.join(df_loc_res_num, on='location')
df_all_cb = df_all_cb.join(df_loc_lfn_num, on='location')
df_all_cb = df_all_cb.join(df_eve_max, on='id')
df_all_cb = df_all_cb.join(df_eve_min, on='id')
df_all_cb = df_all_cb.join(df_eve_std, on='id')
df_all_cb = df_all_cb.join(df_eve_skew, on='id')
df_all_cb = df_all_cb.join(df_log_vol_sum, on='id')
df_all_cb = df_all_cb.join(df_log_vol_num, on='id')
df_all_cb = df_all_cb.join(df_log_vol_min, on='id')
df_all_cb = df_all_cb.join(df_log_feat_num, on='id')
df_all_cb = df_all_cb.join(df_log_feat_max, on='id')
df_all_cb = df_all_cb.join(df_log_feat_min, on='id')
df_all_cb = df_all_cb.join(df_log_feat_std, on='id')
df_all_cb = df_all_cb.join(df_log_feat_skew, on='id')
df_all_cb = df_all_cb.join(df_log_feat_freq_max, on='id')
df_all_cb = df_all_cb.join(df_log_feat_freq_min, on='id')
df_all_cb = df_all_cb.join(df_res_num, on='id')
df_all_cb = df_all_cb.join(df_res_std, on='id')
df_all_cb = df_all_cb.merge(df_loc_lfm_freq, on='id')
df_all_cb = df_all_cb.merge(df_loc_lfn_freq, on='id')
df_all_cb = df_all_cb.merge(df_time, on='id')
df_all_cb = df_all_cb.join(df_eve_table, on='id')
df_all_cb = df_all_cb.join(df_log_table, on='id')
df_all_cb = df_all_cb.join(df_res_table, on='id')
df_all_cb = df_all_cb.join(df_sev_table, on='id')
n_feat = df_all_cb.shape[1]-1
df_all_cb = df_all_cb.join(df_loc_table, on='id')

# check NaN
df_all_cb.isnull().any().any()

# drop id
df_all_no_id = df_all_cb.drop('id', axis=1, inplace=False)

X_all = df_all_no_id.values
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
