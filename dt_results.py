import numpy as np
import pandas as pd
from ingestion import get_pca, get_raw
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from prunable_tree import PrunableTree

random_seed = 42

def decision_tree(ingest, scale, post_prune):
    X, X_test, y, y_test = ingest()

    pipe_steps = []
    if scale:
        pipe_steps.append(('scale', StandardScaler()))
    pipe_steps.append(('classifier', PrunableTree(
        criterion='entropy', prune=post_prune)))
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
    cv_result['inner_nodes_prev'] = \
        pipe.named_steps['classifier'].inner_nodes_prev
    cv_result['inner_nodes_post'] = \
        pipe.named_steps['classifier'].inner_nodes_post

    y_pred = pipe.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    cv_result['holdout_test_score'] = test_score

    return cv_result

results = []
datasets = [('raw', get_raw), ('pca', get_pca)]
for data_name, data_func in datasets:
    for scale in (True, False):
        for prune in (True, False):
            result = decision_tree(data_func, scale, prune)
            result['data'] = data_name
            result['scaled'] = scale
            result['pruned'] = prune
            results.append(result)

df = pd.DataFrame(results)
df.to_csv("results/dt/dt_results.csv")
