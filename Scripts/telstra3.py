import os
import numpy as np
from scipy.stats import mode, skew, kurtosis, entropy
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

import sys
my_dir = os.getcwd()
sys.path.append(my_dir+'/Telstra/Scripts')

import get_raw_feat3
import xgb_clf
import gen_clf

seed = 0

print('Getting raw features...')
(X_all, y, num_class, n_train, n_feat, n_feat2, ids, X_loc_all) = get_raw_feat3.feat_eng()

X = X_all[:n_train, :]
X_numeric = X_all[:n_train, :n_feat]
X_categ = X_all[:n_train, n_feat:]

X_test = X_all[n_train:, :]
X_numeric_test = X_all[n_train:, :n_feat]
X_categ_test = X_all[n_train:, n_feat:]

# super bagging
y_pred_sum = np.zeros((X_test.shape[0], num_class))
set_colsample_bytree = [0.5, 0.6]
set_subsample = [0.9]
set_max_depth = [8]
for colsample_bytree in set_colsample_bytree:
	for subsample in set_subsample:
		for max_depth in set_max_depth:
			seed += 1
			print(seed)
			my_xgb = xgb_clf.my_xgb(obj='multi:softprob', eval_metric='mlogloss', num_class=num_class, 
    			nthread=20, silent=1, eta=0.01, colsample_bytree=colsample_bytree, subsample=subsample, 
    			max_depth=max_depth, max_delta_step=1, gamma=0.1, alpha=0, param_lambda=1, n_fold=35, seed=seed)

			clf1 = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial', verbose=1, n_jobs=20)
			my_clf = gen_clf.my_clf(num_class=num_class, n_fold=35, seed=seed)
			meta_feat1 = my_clf.predict(clf1, X_categ, y, X_categ_test, 'base') 
			meta_feat1_1 = np.reshape(np.apply_along_axis(np.argmax, 1, meta_feat1), (-1, 1))

			X_meta = np.concatenate([X_numeric, meta_feat1_1[:n_train, :]], axis=1)
			X_meta_test = np.concatenate([X_numeric_test, meta_feat1_1[n_train:, :]], axis=1)

			y_pred = my_xgb.predict(X_meta, y, X_meta_test, 'meta')
			y_pred_sum = y_pred_sum+y_pred

y_pred = y_pred_sum/seed
# save pred
sub = pd.DataFrame(data={'id':ids, 'predict_0':y_pred[:, 0], 'predict_1':y_pred[:, 1],
	'predict_2':y_pred[:, 2]}, columns=['id', 'predict_0', 'predict_1', 'predict_2'])
my_dir = os.getcwd()+'/Telstra/Subs/'
sub.to_csv(my_dir+'sub3.csv', index=False)
