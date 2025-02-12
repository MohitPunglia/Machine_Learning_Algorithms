# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(
    "/Users/mohit/Developer/Machine Learning/Machine_Learning_Algorithms/Random Forest/train.csv"
)

# %%
df.head()

# %%
df.shape

# %%
df.sample(5)

# %%
plt.imshow(df.iloc[11111, 1:].values.reshape(28, 28), cmap="gray")

# %%
X = df.iloc[:, 1:]

# %%
y = df.iloc[:, 0]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
X_train.shape

# %% [markdown]
# Applying KNN to check the score

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# %%
y_pred = knn.predict(X_test)

# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

# %% [markdown]
# Performing Standard Scaler

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# %%
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)


# %% [markdown]
# Applying PCA

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=100)

# %%
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# %%
x_train_pca.shape

# %%
knn_pca = KNeighborsClassifier()
knn_pca.fit(x_train_pca, y_train)
y_pred_pca = knn_pca.predict(x_test_pca)

# %%
accuracy_score(y_test, y_pred_pca)

# %%
for i in range(1, 785):
    pca = PCA(n_components=i)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    knn_pca = KNeighborsClassifier()
    knn_pca.fit(x_train_pca, y_train)
    y_pred_pca = knn_pca.predict(x_test_pca)
    print(i, accuracy_score(y_test, y_pred_pca))

# %% [markdown]
# Transforming data into 2D coordinates

# %%
pca = PCA(n_components=2)

# %%
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# %%
x_train_pca.shape
