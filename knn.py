from ingestion import get_raw, get_pca
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

random_seed = 42

def knn(ingest, scale):
	X, X_test, y, y_test = ingest()

	params = {
        'classifier__n_neighbors': list(range(10, 21)),
        'classifier__weights' : ['distance', 'uniform']
	}

	pipe_steps = []
	if scale:
		pipe_steps.append(('scale', StandardScaler()))
	pipe_steps.append(('classifier', KNeighborsClassifier()))
	pipe = Pipeline(pipe_steps)

	kfold = StratifiedKFold(10, shuffle=True, random_state=random_seed)
	cv_grid = GridSearchCV(pipe, params, scoring='accuracy', 
	    return_train_score=True, cv=kfold, verbose=1, n_jobs=-1)
	cv_grid.fit(X, y)

	return pd.DataFrame(cv_grid.cv_results_)

datasets = [('raw', get_raw), ('pca', get_pca)]
for data_name, data_func in datasets:
	for scale in (True, False):
		result = knn(data_func, scale)
		filename = "knn_result_" + data_name + "_"
		filename += "scaled" if scale else "not_scaled"
		result.to_csv("results/"+filename+".csv")
		print(filename, "done")
