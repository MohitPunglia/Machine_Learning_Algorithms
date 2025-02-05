# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("titanic.csv")

# %%
df.head()

# %%
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# %%
df

# %% [markdown]
# 1. Missing values in Age and Embarked to fill
# 2. one Hot Endocing in Sex and Embarked

# %%
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# %%
X

# %%
y

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# %%
X_train.head()
