import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold

class my_xgb(object):
	'''My xgboost classifier.'''
	# init
	def __init__(self, obj, eval_metric, num_class, nthread, silent, eta, colsample_bytree, subsample, max_depth, n_fold, seed):
		self.obj = obj
		self.eval_metric = eval_metric
		self.num_class = num_class
		self.nthread = nthread
		self.silent = silent
		self.eta = eta
		self.colsample_bytree = colsample_bytree
		self.subsample = subsample
		self.max_depth = max_depth
		self.n_fold = n_fold
		self.seed = seed
	# predict
	def predict(self, X, y, X_test, stage):
		np.random.seed(self.seed)
		n_train = X.shape[0]
		kf = KFold(n_train, n_folds=self.n_fold, shuffle=True)
		param = {}
		param['objective'] = self.obj
		param['eval_metric'] = self.eval_metric
		param['num_class'] = self.num_class
		param['nthread'] = self.nthread
		param['silent'] = self.silent
		param['eta'] = self.eta
		param['colsample_bytree'] = self.colsample_bytree
		param['subsample'] = self.subsample
		param['max_depth'] = self.max_depth
		num_round = 10000
		best_score = []
		best_iter = []
		y_pred_sum = np.zeros((X_test.shape[0], self.num_class))
		if stage=='base':
			meta_feat = np.zeros((n_train+X_test.shape[0], 3))
		xg_test = xgb.DMatrix(X_test)
		i = 0
		for train, val in kf:
			i += 1
			print(i)
			X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
			xg_train = xgb.DMatrix(X_train, y_train)
			xg_val = xgb.DMatrix(X_val, y_val)
			evallist  = [(xg_train,'train'), (xg_val,'eval')]
			## CV sets
			# train
			bst = xgb.train(param, xg_train, num_round, evallist, early_stopping_rounds=100)
			best_score += [bst.best_score]
			best_iter += [bst.best_iteration]
			# predict
			if stage=='base':
				meta_feat[val, :] = bst.predict(xg_val, ntree_limit=bst.best_iteration)
			else:
				y_pred = bst.predict(xg_test, ntree_limit=bst.best_iteration)
				y_pred_sum = y_pred_sum+y_pred
		print(np.mean(best_score), np.std(best_score))
		## test set
		if stage=='base':
			# train
			xg_train = xgb.DMatrix(X, y)
			evallist  = [(xg_train,'train')]
			bst = xgb.train(param, xg_train, int(np.mean(best_iter)), evallist)
			# predict
			meta_feat[n_train:, :] = bst.predict(xg_test)
			return meta_feat
		else:
			y_pred = y_pred_sum/self.n_fold
			return y_pred