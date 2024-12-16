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
