import os
import pandas as pd
import numpy as np
from ingestion import get_raw, get_pca
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
warnings.filterwarnings('ignore')
os.makedirs("results/hierac", exist_ok=True)

X, X_test, y, y_test = get_raw()
attr_sz = len(np.unique(y))
print(attr_sz)
k_dict = {}
for k in range(2,13):
	hierarc = AgglomerativeClustering(n_clusters=k)
	hierarc.fit(X)
	labels = hierarc.labels_
	print("k = %d" % k, davies_bouldin_score(X,labels))
	k_dict[k] = davies_bouldin_score(X,labels)

df = pd.DataFrame.from_dict(k_dict, orient='index')
df.to_csv('results/hierac/hierac.csv')

ax = sns.lineplot(x=list(k_dict.keys()), y=list(k_dict.values()))
ax.set(xlabel = 'k', ylabel = 'DB')
plt.savefig('results/hierac/hierac.png')
