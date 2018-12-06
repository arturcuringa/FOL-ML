import os
import pandas as pd
import numpy as np
from ingestion import get_raw, get_pca
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


os.makedirs("results/mlp", exist_ok=True)

def mlp(ingest, scale):

    X, X_test, y, y_test = ingest()

    params = {
            'classifier__hidden_layer_sizes' : [(x,) for x in range(X.shape[1]//2, X.shape[1]) ],        
            'classifier__max_iter' : [100, 1000, 10000],
            'classifier__learning_rate_init' : [0.1 , 0.01, 0.001 ]
    }
    pipe_steps = []
    if scale:
        pipe_steps.append(('scale', StandardScaler()))
    pipe_steps.append(('classifier', MLPClassifier(solver='sgd',momentum= 0.8,nesterovs_momentum= False)))
    pipe = Pipeline(pipe_steps)
    
    kfold = StratifiedKFold(2, shuffle=True)
    cv_grid = GridSearchCV(pipe, params, scoring='accuracy',return_train_score=True, cv=kfold, verbose=1, n_jobs=-1)
    cv_grid.fit(X,y)

    cv_result = pd.DataFrame(cv_grid.cv_results_)

    test_scores = []
    for params in cv_result["params"]:
        pipe = pipe.set_params(**params).fit(X, y)
        y_pred = pipe.predict(X_test)

        score = accuracy_score(y_test, y_pred)
        test_scores.append(score)

    cv_result['holdout_test_score'] = pd.Series(
        test_scores, index=cv_result.index)

    return cv_result

datasets = [('raw', get_raw), ('pca', get_pca)]
for data_name, data_func in datasets:
    for scale in (True, False):
        result = mlp(data_func, scale)
        filename = "mlp_result_" + data_name + "_"
        filename += "scaled" if scale else "not_scaled"
        result.to_csv("results/mlp/" + filename + ".csv")
        print(filename, "done")

