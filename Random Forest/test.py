# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# %%
df = pd.read_csv("heart.csv")

# %%
df.head()

# %%
df.shape

# %%
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
print(X_train.shape)
print(X_test.shape)

# %% [markdown]
# Creating object of all the algorithm to test the behaviour
#

# %%
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
svc = SVC()
lr = LogisticRegression()

# %%
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
accuracy_score(y_test, y_pred)

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
rf = RandomForestClassifier(max_samples=0.75, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
from sklearn.model_selection import cross_val_score

np.mean(
    cross_val_score(
        RandomForestClassifier(max_samples=0.75), X, y, cv=10, scoring="accuracy"
    )
)

# %% [markdown]
# Hyper parameter tunening of Random forest by Grid Serach CV

# %%
# Number of trees in random forest
n_estimators = [20, 60, 100, 120]

# Number of features to consider at every split
max_features = [0.2, 0.6, 1.0]

# Maximum number of levels in tree
max_depth = [2, 8, None]

# Number of samples
max_samples = [0.5, 0.75, 1.0]

# 108 diff random forest train

# %%
param_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "max_samples": max_samples,
}
print(param_grid)

rf = RandomForestClassifier()

# %%
from sklearn.model_selection import GridSearchCV

rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2)

# %%
rf_grid.fit(X_train, y_train)


# %%
# Random Search CV
#
# It can be used for large dataset and large parameters
