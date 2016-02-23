import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

class my_clf(object):
	'''My classifier.'''
	# init
	def __init__(self, num_class, n_fold, seed):
		self.num_class = num_class
		self.n_fold = n_fold
		self.seed = seed
	# predict
	def predict(self, clf, X, y, X_test, stage):
		np.random.seed(self.seed)
		n_train = X.shape[0]
		kf = KFold(n_train, n_folds=self.n_fold, shuffle=True)
		best_score = []
		y_pred_sum = np.zeros((X_test.shape[0], self.num_class))
		if stage=='base':
			meta_feat = np.zeros((n_train+X_test.shape[0], self.num_class))
		i = 0
		for train, val in kf:
			i += 1
			print(i)
			X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
			## CV sets
			# train
			clf.fit(X_train, y_train)
			curr_pred = clf.predict_proba(X_val)
			curr_best_score = log_loss(y_val, curr_pred)
			print(curr_best_score)
			best_score += [curr_best_score]
			# predict
			if stage=='base':
				meta_feat[val, :] = curr_pred
			else:
				y_pred = clf.predict_proba(X_test)
				y_pred_sum = y_pred_sum+y_pred
		print(np.mean(best_score), np.std(best_score))
		## test set
		if stage=='base':
			# train
			clf.fit(X, y)
			# predict
			meta_feat[n_train:, :] = clf.predict_proba(X_test)
			return meta_feat
		else:
			y_pred = y_pred_sum/self.n_fold
			return y_pred