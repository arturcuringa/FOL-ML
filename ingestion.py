import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_random_seed = 42

def get_data():
	df = pd.read_csv("data/all-data-ready.csv", index_col=0)
	X, y = df.drop(['heuristic'], axis=1).astype('float64'), df['heuristic']
	return X, y

def get_raw():
	X, y = get_data()

	X, X_test, y, y_test = train_test_split(
	    X, y, stratify=y, test_size=0.1, random_state=_random_seed)

	return X, X_test, y, y_test

def get_pca():
	X, X_test, y, y_test = get_raw()

	scaler = StandardScaler()
	X = pd.DataFrame(scaler.fit_transform(X))
	X_test = pd.DataFrame(scaler.transform(X_test))

	pca_ = PCA(0.96)
	pca_.fit(X)
	X_pca = pd.DataFrame(pca_.transform(X))
	X_test_pca = pd.DataFrame(pca_.transform(X_test))
	return X_pca, X_test_pca, y, y_test

def get_full_pca():
	X, y = get_data()

	scaler = StandardScaler()
	X = pd.DataFrame(scaler.fit_transform(X))

	pca_ = PCA(0.96)
	X_pca = pd.DataFrame(pca_.fit_transform(X))

	return X_pca, y
