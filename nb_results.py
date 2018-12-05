from ingestion import get_raw, get_pca
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

random_seed = 42

def naive_bayes(ingest, scale):
    X, X_test, y, y_test = ingest()

    pipe_steps = []
    if scale:
        pipe_steps.append(('scale', StandardScaler()))
    pipe_steps.append(('classifier', GaussianNB()))
    pipe = Pipeline(pipe_steps)

    kfold = StratifiedKFold(10, shuffle=True, random_state=random_seed)
    cv = cross_validate(pipe, X, y, scoring='accuracy', 
        return_train_score=True, cv=kfold, verbose=1, n_jobs=-1)

    cv_result = {
	    'mean_test_score': np.mean(cv['test_score']),
	    'std_test_score': np.std(cv['test_score']),
	    'mean_train_score': np.mean(cv['train_score']),
	    'std_train_score': np.std(cv['train_score'])
    }

    pipe.fit(X, y)
    y_pred = pipe.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    cv_result['holdout_test_score'] = test_score

    return pd.DataFrame(cv_result, index=[0])

datasets = [('raw', get_raw), ('pca', get_pca)]
for data_name, data_func in datasets:
    for scale in (True, False):
        result = naive_bayes(data_func, scale)
        filename = "nb_result_" + data_name + "_"
        filename += "scaled" if scale else "not_scaled"
        result.to_csv("results/nb/"+filename+".csv")
        print(filename, "done")
