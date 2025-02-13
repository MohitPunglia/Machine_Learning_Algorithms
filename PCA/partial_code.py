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


# %%
import plotly.express as px

y_train_pca = y_train.astype(str)
fig = px.scatter(x=x_train_pca[:, 0], y=x_train_pca[:, 1], color=y_train_pca)
fig.show()

# %% [markdown]
# Transforming same data into 3D coordinate

# %%
pca = PCA(n_components=3)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# %%
x_train_pca.shape

# %%
x_train_pca

# %%
import plotly.express as px

y_train_pca = y_train.astype(str)
fig = px.scatter_3d(
    x=x_train_pca[:, 0], y=x_train_pca[:, 1], z=x_train_pca[:, 2], color=y_train_pca
)
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="LightSteelBlue")
fig.show()

# %%
# Eigen Values
pca.explained_variance_

# %%
# Eigen Vectors
pca.components_.shape

# %%
