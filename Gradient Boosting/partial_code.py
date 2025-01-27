# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# %%
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

# %%
df = pd.DataFrame()

# %%
df["X"] = X.reshape(100)
df["y"] = y

# %%
df

# %%
plt.scatter(df["X"], df["y"])
plt.title("Scatter plot of X and y")
