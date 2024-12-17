# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("Social_Netwoks_Ads.csv")
data.head()

# %%
X = data.iloc[:, 2:4].values

# %%
X.shape

# %%
y = data.iloc[:, -1].values

# %%
y.shape

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# %%
X_train.shape

# %%
X_test.shape

# %% [markdown]
# As we are using KNN 2 input colums have very large difference we need to scale it dowm to age's value
#
# we will use class standardscaler inside sklearn.preprocessor

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# %%
X_train = scaler.fit_transform(X_train)

# %%
X_train

# %%
X_test = scaler.transform(X_test)

# %%
X_test

# %% [markdown]
# Applying KNN algo :
#
#  We need to find K by applying square root

# %%
np.sqrt(X_train.shape[0])

# %%
k = 17

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=k)

# %%
# for training our model

knn.fit(X_train, y_train)

# %%
y_pred = knn.predict(X_test)

# %%
y_pred.shape

# %%
# need to check y_test shape to ensure both are same
y_test.shape
