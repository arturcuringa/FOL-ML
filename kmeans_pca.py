import os
import pandas as pd
import numpy as np
from ingestion import get_raw, get_pca, get_full_pca
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
warnings.filterwarnings('ignore')
os.makedirs("results/kmeans", exist_ok=True)

X, y = get_full_pca()
attr_sz = len(np.unique(y))
print(attr_sz)
k_dict = {}
for k in range(2,13):
	db_values = []
	for i in range(5):
		kmeans = KMeans(n_clusters=k)
		kmeans.fit(X)
		labels = kmeans.labels_
		db_values.append(davies_bouldin_score(X,labels))
	print("k = %d" % k, np.array(db_values).sum()/5)
	k_dict[k] = np.array(db_values).sum()/5

df = pd.DataFrame.from_dict(k_dict, orient='index')
df.to_csv('results/kmeans/kmeans_PCA.csv')
ax = sns.lineplot(x=list(k_dict.keys()), y=list(k_dict.values()))
ax.set(xlabel = 'k', ylabel = 'DB')
plt.savefig('results/kmeans/kmeans_PCA.png')
