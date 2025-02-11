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
