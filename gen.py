import pandas as pd
import numpy as np
import random
from pyeasyga import pyeasyga
from itertools import compress
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

def best_heuristic(row, time_cols):
    n_heuristics = 5
    h_times = row[time_cols].reset_index(drop=True)
    h_times.replace({-100.0 : np.nan}, inplace=True)
    idx, min_time = h_times.idxmin(), h_times.min()
    if np.isnan(min_time):
       return 0
    else:
       return idx+1

def fitness_corr(individual, idx_corr):
    fitness = 0
    n = individual.count(1)
    if n > 0:
        fitness = sum(corr for idx, corr in compress(idx_corr, individual))
        fitness /= n
    return fitness

def fitness_knn(individual, X_y):
    X, y = X_y
    X = X[individual]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=random_seed)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

def create_individual(X_y):
    X, y = X_y
    features = X.columns
    selected = [random.randint(0, 1) for _ in features]
    return list(compress(features, selected))

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

# abs_correlations = abs(X.corrwith(y)).fillna(0)
# idx_corr = list(zip(abs_correlations.index, abs_correlations))
# ga = pyeasyga.GeneticAlgorithm(idx_corr)

ga = pyeasyga.GeneticAlgorithm((X, y), generations=20)

ga.fitness_function = fitness_knn
ga.create_individual = create_individual
ga.run()

score, individual = ga.best_individual()
selected_attrs = list(compress(X.columns, individual))
print(score, len(selected_attrs), selected_attrs)
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
