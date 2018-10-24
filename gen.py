import pandas as pd
import numpy as np
import random
from pyeasyga import pyeasyga
from itertools import compress
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

df = pd.read_csv("data/all-data-raw.csv", header=None)
time_cols = list(range(53, 58))
df['heuristic'] = df.apply(lambda r : best_heuristic(r, time_cols), axis=1)
df.drop(time_cols, axis=1, inplace=True)
X, y = df.drop(['heuristic'], axis=1).astype('float64'), df['heuristic']

abs_correlations = abs(X.corrwith(y)).fillna(0)
idx_corr = list(zip(abs_correlations.index, abs_correlations))

# ga = pyeasyga.GeneticAlgorithm(idx_corr)
ga = pyeasyga.GeneticAlgorithm(df)

def fitness_corr(individual, data):
    fitness = 0
    n = individual.count(1)
    if n > 0:
        attrs = zip(individual, data)
        fitness = sum(corr for idx, corr in compress(data, individual))
        fitness /= n
    return fitness

def fitness_knn(individual, data):
    X, y = df.drop(['heuristic'], axis=1).astype('float64'), df['heuristic']
    X = X[individual]

    knn = KNeighborsClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

def create_individual(df):
    features = df.columns.drop('heuristic')
    selected = [random.randint(0, 1) for _ in features]
    return list(compress(features, selected))

ga.fitness_function = fitness_knn
ga.create_individual = create_individual
ga.run()

score, individual = ga.best_individual()
selected_attrs = list(compress(abs_correlations.index, individual))
print(score, selected_attrs)