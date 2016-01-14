import os
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
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
# df_all
# we assume that location is numerical
# it makes sense since given a specific class, the locations show the pattern
# of a number of segments with consecutive numbers
myfun = lambda x: int(x.strip('location '))
df_all['location'] = df_all['location'].apply(myfun)

# df_eve
df_eve = pd.pivot_table(df_eve, index='id', columns='event_type', aggfunc=len,
	fill_value=0)

# df_log
df_log_table = pd.pivot_table(df_log, values='volume', index='id', columns='log_feature', aggfunc=np.sum,
	fill_value=0)
df_log_vol_sum = pd.DataFrame(data={'log_vol_sum': df_log_table.sum(axis=1)})
grouped = df_log[['id', 'log_feature']].groupby('id')
myfun = lambda x: len(np.unique(x))
df_log_feat_num = grouped.aggregate(myfun)

# df_res
df_res_table = pd.pivot_table(df_res, index='id', columns='resource_type', aggfunc=len,
	fill_value=0)
grouped = df_res[['id', 'resource_type']].groupby('id')
myfun = lambda x: len(np.unique(x))
df_res_num = grouped.aggregate(myfun)

df_sev = pd.pivot_table(df_sev, index='id', columns='severity_type', aggfunc=len,
	fill_value=0)

# combine
df_all = df_all.join(df_eve, on='id')
df_all = df_all.join(df_log_table, on='id')
df_all = df_all.join(df_log_vol_sum, on='id')
df_all = df_all.join(df_log_feat_num, on='id')
df_all = df_all.join(df_res_table, on='id')
df_all = df_all.join(df_res_num, on='id')
df_all = df_all.join(df_sev, on='id')

# check NaN
df_all.isnull().any().any()

# drop id
df_all_no_id = df_all.drop('id', axis=1, inplace=False)

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

param['eta'] = 0.1
param['colsample_bytree'] = 0.3

num_round = 1000

X_test = X_all[n_train:, :]
xg_test = xgb.DMatrix(X_test)

# k-fold
k = 10
y_pred_sum = np.zeros((X_test.shape[0], num_class))
best_score = []
kf = KFold(X.shape[0], n_folds=k, shuffle=True, random_state=0)
i = 0
for train, val in kf:
  i += 1
  print(i)
  X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
  xg_train = xgb.DMatrix(X_train, y_train)
  xg_val = xgb.DMatrix(X_val, y_val)
  evallist  = [(xg_train,'train'), (xg_val,'eval')]
  # train
  bst = xgb.train(param, xg_train, num_round, evallist, early_stopping_rounds=30)
  best_score += [bst.best_score]
  # predict
  xg_test = xgb.DMatrix(X_test)
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
