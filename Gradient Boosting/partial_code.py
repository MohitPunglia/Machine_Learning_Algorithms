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

# %% [markdown]
# Adding new column as mean

# %%
df["pred1"] = df["y"].mean()

# %%
df

# %% [markdown]
# Calculating Resilence

# %%
df["res1"] = df["y"] - df["pred1"]

# %%
df

# %% [markdown]
# Plotting the graph

# %%
plt.scatter(df["X"], df["y"])
plt.plot(df["X"], df["pred1"], color="red")

# %% [markdown]
# Creating object of Decision tree and max nodes as 8(generally we keep it from 8 to 32)
