from ingestion import get_raw, get_pca
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

X, X_test, y, y_test = get_pca()

# scaler = StandardScaler()
# X.update(scaler.fit_transform(X))
# X_test.update(scaler.transform(X_test))

print(X.head())

f, axes = plt.subplots(1, 2, figsize=(7, 7), sharex=True)
sns.distplot(X[0], kde=False, ax=axes[0], color="skyblue")
sns.distplot(X[1], kde=False, ax=axes[1], color="olive")
plt.show()