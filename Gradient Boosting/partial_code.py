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
# %%
tree1 = DecisionTreeRegressor(max_leaf_nodes=8)

# %%
tree1.fit(df["X"].values.reshape(100, 1), df["res1"].values)

# %%
from sklearn.tree import plot_tree

plot_tree(tree1)
plt.show()

# %%
# generating X_test
X_test = np.linspace(-0.5, 0.5, 500).reshape(500)

# %% [markdown]
# Calculating y_pred by first model output + 2nd model output

# %%
y_pred = 0.265458 + tree1.predict(X_test.reshape(500, 1))

# %%
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(X_test, y_pred, color="red")
plt.scatter(df["X"], df["y"])

# %%
df["pred2"] = 0.265458 + tree1.predict(df["X"].values.reshape(100, 1))

# %%
df

# %%
df["res2"] = df["y"] - df["pred2"]

# %%
df

# %%
tree2 = DecisionTreeRegressor(max_leaf_nodes=8)

# %%
tree2.fit(df["X"].values.reshape(100, 1), df["res2"].values)

# %%
y_pred = (
    0.265458
    + tree1.predict(X_test.reshape(500, 1))
    + tree2.predict(X_test.reshape(500, 1))
)

# %%
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(X_test, y_pred, color="red")
plt.scatter(df["X"], df["y"])
plt.title("Prediction of y using 2 trees")

# %%
