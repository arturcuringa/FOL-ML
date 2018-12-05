df = pd.read_csv("data/all-data-ready.csv", header=None)
pca = PCA(0.96)
pca.fit(X)
print(pca.explained_variance_ratio_)
base_classifiers(pca.transform(X), y, pca.transform(X_val), y_val)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_random_seed = 42

def get_data():
	df = pd.read_csv("data/all-data-ready.csv", header=None)
	X, y = df.drop(['heuristic'], axis=1).astype('float64'), df['heuristic']
	return X, y

def get_raw():
	X, y = get_data()

	X, X_test, y, y_test = train_test_split(
	    X, y, stratify=y, test_size=0.1, random_state=_random_seed)

	scaler = StandardScaler()
	X.update(scaler.fit_transform(X))
	X_test.update(scaler.transform(X_test))
	return X, X_test, y, y_test

def get_pca():
	X, X_test, y, y_test = get_raw()
	pca_ = PCA(0.96)
	pca_.fit(X)
	return pca_.transform(X), pca_.transform(X_test), y, y_test
