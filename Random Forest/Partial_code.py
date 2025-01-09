# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# %%
np.random.seed(42)
X, y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=42)
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
X.shape

# %%
plt.scatter(X[:, 0], X[:, 1], c=y)

# %% [markdown]
# Using Decision Tree
#
#
#

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = dtree.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.title("Decision tree")
plt.show()

# %% [markdown]
# Now applying Random Forest on the same data

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
