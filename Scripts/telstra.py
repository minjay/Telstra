import os
import numpy as np
from scipy.stats import mode
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

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

## feature engineering
# df_loc
# contain information
loc_diff = np.setdiff1d(df_test['location'].values, df_train['location'].values)
df_loc_table = pd.pivot_table(df_all, index='id', columns='location', aggfunc=len,
  fill_value=0)
df_loc_table.drop(loc_diff, axis=1, inplace=True)

loc_freq = df_all['location'].value_counts()
loc_freq = pd.DataFrame({'loc_freq': loc_freq})
loc_freq.index.name = 'location'
df_all = df_all.join(loc_freq, on='location')

# df_all
myfun = lambda x: int(x.strip('location '))
df_all['location'] = df_all['location'].apply(myfun)

# df_eve
df_eve_table = pd.pivot_table(df_eve, index='id', columns='event_type', aggfunc=len,
	fill_value=0)
grouped = df_eve[['id', 'event_type']].groupby('id')
# df_eve_num is not useful!
myfun = lambda x: len(np.unique(x))
df_eve_num = grouped.aggregate(myfun)
myfun = lambda x: max(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
df_eve_max = grouped.aggregate(myfun)
df_eve_max.rename(columns={'event_type': 'eve_max'}, inplace=True)
myfun = lambda x: min(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
df_eve_min = grouped.aggregate(myfun)
df_eve_min.rename(columns={'event_type': 'eve_min'}, inplace=True)
myfun = lambda x: np.std(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
df_eve_std = grouped.aggregate(myfun)
df_eve_std.rename(columns={'event_type': 'eve_std'}, inplace=True)

## df_log
df_log_table = pd.pivot_table(df_log, values='volume', index='id', columns='log_feature', aggfunc=np.sum,
	fill_value=0)
df_log_vol_sum = pd.DataFrame(data={'log_vol_sum': df_log_table.sum(axis=1)})
grouped = df_log[['id', 'log_feature']].groupby('id')
myfun = lambda x: len(np.unique(x))
df_log_feat_num = grouped.aggregate(myfun)
df_log_feat_num.rename(columns={'log_feature': 'log_feat_num'}, inplace=True)
myfun = lambda x: max(x.apply(lambda x: x.strip('feature ')).astype(int))
df_log_feat_max = grouped.aggregate(myfun)
df_log_feat_max.rename(columns={'log_feature': 'log_feat_max'}, inplace=True)
myfun = lambda x: min(x.apply(lambda x: x.strip('feature ')).astype(int))
df_log_feat_min = grouped.aggregate(myfun)
df_log_feat_min.rename(columns={'log_feature': 'log_feat_min'}, inplace=True)
myfun = lambda x: np.std(x.apply(lambda x: x.strip('feature ')).astype(int))
df_log_feat_std = grouped.aggregate(myfun)
df_log_feat_std.rename(columns={'log_feature': 'log_feat_std'}, inplace=True)
grouped = df_log[['id', 'volume']].groupby('id')
myfun = lambda x: len(np.unique(x))
df_log_vol_num = grouped.aggregate(myfun)
df_log_vol_num.rename(columns={'volume': 'vol_num'}, inplace=True)
# df_log_vol_max is controlled by sum and num
myfun = lambda x: min(x)
df_log_vol_min = grouped.aggregate(myfun)
df_log_vol_min.rename(columns={'volume': 'vol_min'}, inplace=True)

grouped = df_log[['log_feature', 'volume']].groupby('log_feature')
myfun = lambda x: np.sum(x)
log_feat_freq = grouped.aggregate(myfun)
log_feat_freq.rename(columns={'volume': 'log_feat_freq'}, inplace=True)
df_log_feat_freq = df_log.join(log_feat_freq, on='log_feature')
grouped = df_log_feat_freq[['id', 'log_feat_freq']].groupby('id')
myfun = lambda x: max(x)
df_log_feat_freq_max = grouped.aggregate(myfun)
df_log_feat_freq_max.rename(columns={'log_feat_freq': 'log_feat_freq_max'}, inplace=True)
myfun = lambda x: min(x)
df_log_feat_freq_min = grouped.aggregate(myfun)
df_log_feat_freq_min.rename(columns={'log_feat_freq': 'log_feat_freq_min'}, inplace=True)

# df_res
df_res_table = pd.pivot_table(df_res, index='id', columns='resource_type', aggfunc=len,
	fill_value=0)
grouped = df_res[['id', 'resource_type']].groupby('id')
myfun = lambda x: len(np.unique(x))
df_res_num = grouped.aggregate(myfun)
df_res_num.rename(columns={'resource_type': 'res_num'}, inplace=True)
myfun = lambda x: np.std(x.apply(lambda x: x.strip('resource_type ')).astype(int))
df_res_std = grouped.aggregate(myfun)
df_res_std.rename(columns={'resource_type': 'res_std'}, inplace=True)

# df_sev
# It is categorical. It does not have an ordering.
df_sev_table = pd.pivot_table(df_sev, index='id', columns='severity_type', aggfunc=len,
	fill_value=0)

# combine
df_all_cb = df_all.join(df_loc_table, on='id')
df_all_cb = df_all_cb.join(df_eve_table, on='id')
df_all_cb = df_all_cb.join(df_eve_max, on='id')
df_all_cb = df_all_cb.join(df_eve_min, on='id')
df_all_cb = df_all_cb.join(df_eve_std, on='id')
df_all_cb = df_all_cb.join(df_log_table, on='id')
df_all_cb = df_all_cb.join(df_log_vol_sum, on='id')
df_all_cb = df_all_cb.join(df_log_vol_num, on='id')
df_all_cb = df_all_cb.join(df_log_vol_min, on='id')
df_all_cb = df_all_cb.join(df_log_feat_num, on='id')
df_all_cb = df_all_cb.join(df_log_feat_max, on='id')
df_all_cb = df_all_cb.join(df_log_feat_min, on='id')
df_all_cb = df_all_cb.join(df_log_feat_std, on='id')
df_all_cb = df_all_cb.join(df_log_feat_freq_max, on='id')
df_all_cb = df_all_cb.join(df_log_feat_freq_min, on='id')
df_all_cb = df_all_cb.join(df_res_table, on='id')
df_all_cb = df_all_cb.join(df_res_num, on='id')
df_all_cb = df_all_cb.join(df_res_std, on='id')
df_all_cb = df_all_cb.join(df_sev_table, on='id')

# check NaN
df_all_cb.isnull().any().any()

# drop id
df_all_no_id = df_all_cb.drop('id', axis=1, inplace=False)

X_all = df_all_no_id.values
X = X_all[:n_train, :]

## specify parameters for xgb
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
param['eval_metric'] = 'mlogloss'
param['num_class'] = num_class
param['nthread'] = 1
param['silent'] = 1

param['eta'] = 0.02
param['colsample_bytree'] = 0.3
param['subsample'] = 0.9
param['max_depth'] = 8

num_round = 10000

X_test = X_all[n_train:, :]
xg_test = xgb.DMatrix(X_test)

# set random seed
np.random.seed(0)

# k-fold
k = 5
y_pred_sum = np.zeros((X_test.shape[0], num_class))
best_score = []
kf = KFold(X.shape[0], n_folds=k, shuffle=True)
i = 0
for train, val in kf:
  i += 1
  print(i)
  X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
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
y_pred = y_pred_sum/k
print(np.mean(best_score))

# save pred
sub = pd.DataFrame(data={'id':ids, 'predict_0':y_pred[:, 0], 'predict_1':y_pred[:, 1],
	'predict_2':y_pred[:, 2]}, columns=['id', 'predict_0', 'predict_1', 'predict_2'])
my_dir = os.getcwd()+'/Telstra/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)
