from ingestion import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# X, y = get_data()
# corr = X.corr()
# ax = sns.heatmap(corr.abs())
# plt.savefig("corr_raw.png")

X, y = get_full_pca()
corr = X.corr()
ax = sns.heatmap(corr.abs())
plt.savefig("corr_pca.png")