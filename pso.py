import pandas as pd
import numpy as np
import random
from itertools import compress
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
import pyswarms as ps

def best_heuristic(row, time_cols):
    n_heuristics = 5
    h_times = row[time_cols].reset_index(drop=True)
    h_times.replace({-100.0 : np.nan}, inplace=True)
    idx, min_time = h_times.idxmin(), h_times.min()
    if np.isnan(min_time):
       return 0
    else:
       return idx+1

def f_per_particle_corr(m, alpha):
    if np.count_nonzero(m) == 0:
        return 1
    else:
        X_subset = X.loc[:, m==1]
    P = abs(X_subset.corrwith(y)).sum()
    P /= np.count_nonzero(m) 
    return (alpha * (1.0 - P))

def f_corr(x, alpha = 1.0):
    n_particles = x.shape[0]
    j = [f_per_particle_corr(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def f_per_particle_knn(m, alpha = 1.0):
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X.loc[:, m==1]
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, stratify=y, test_size=0.3, random_state=random_seed)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return alpha * (1 - accuracy_score(y_test, y_pred))

def f_knn(x, alpha = 1.0):
    n_particles = x.shape[0]
    j = [f_per_particle_knn(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

random_seed = 42
df = pd.read_csv("data/all-data-raw.csv", header=None)
time_cols = list(range(53, 58))
df['heuristic'] = df.apply(lambda r : best_heuristic(r, time_cols), axis=1)
df.drop(time_cols, axis=1, inplace=True)
X, y = df.drop(['heuristic'], axis=1).astype('float64'), df['heuristic']

X, X_val, y, y_val = train_test_split(
    X, y, stratify=y, test_size=0.1, random_state=random_seed)

scaler = StandardScaler()
X.update(scaler.fit_transform(X))
X_val.update(scaler.transform(X_val))


options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}
dimensions = 53 # dimensions should be the number of features

optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

cost, pos = optimizer.optimize(f_corr, print_step=100, iters=1000, verbose=2)


selected_attrs = list(compress(X.columns, pos))
print(cost, len(selected_attrs), selected_attrs)
print()

X = X[selected_attrs]
X_val = X_val[selected_attrs]

knn_params = {
    'n_neighbors': range(1,15),
    'weights': ['uniform', 'distance']
}
knn = KNeighborsClassifier()
kfold = StratifiedKFold(10, shuffle=True, random_state=random_seed)
cv_grid = GridSearchCV(knn, knn_params, scoring='accuracy', 
    cv=kfold, verbose=1, n_jobs=-1)
cv_grid.fit(X, y)

print(cv_grid.best_params_)
print("best knn score: {:.4f}".format(cv_grid.best_score_))
print()

best_knn = cv_grid.best_estimator_
best_knn.fit(X, y)

y_pred = best_knn.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("final validation score:", accuracy)
