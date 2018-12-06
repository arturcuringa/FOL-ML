from ingestion import get_raw, get_pca
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import normaltest, shapiro

alpha = 0.01

def test_normality(ingest, scale):
    X, X_test, y, y_test = ingest()

    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X))
        X_test = pd.DataFrame(scaler.transform(X_test))

    results = []
    total = len(X.columns)
    for col in X.columns:
        k2, p_n = normaltest(X[col])
        stat, p_s = normaltest(X[col])

        results.append({
            "feature" : col,
            "k2_stat" : k2,
            "k2_p" : p_n,
            "k2_rejected" : p_n < alpha,
            "shapiro_stat" : stat,
            "shapiro_p" : p_s,
            "shapiro_rejected" : p_s < alpha
        })

    return pd.DataFrame(results)

datasets = [('raw', get_raw), ('pca', get_pca)]
for data_name, data_func in datasets:
    for scale in (True, False):
        result = test_normality(data_func, scale)
        filename = "normality_" + data_name + "_"
        filename += "scaled" if scale else "not_scaled"
        result.to_csv("results/normality/"+filename+".csv")
        