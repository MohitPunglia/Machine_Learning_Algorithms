 %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


# %%
X, y = make_classification(
    n_features=5, n_redundant=0, n_informative=5, n_clusters_per_class=1
)

# %%
df = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4", "col5"])
df["target"] = y


# %%
df.shape

# %%
df.head()

# %% [markdown]
# Function for row sampling
# 

# %%
def sample_rows(df, percent):
    return df.sample(int(percent * df.shape[0]), replace=True)

# %% [markdown]
# Function for column sampling

# %%
import random


def sample_features(df, percent):
    cols = random.sample(df.columns.tolist()[:-1], int(percent * (df.shape[1] - 1)))
    new_df = df[cols]
    new_df["target"] = df["target"]
    return new_df

# %% [markdown]
# Function for combined sampling
# 

# %%
def combined_sampling(df, row_percent, col_percent):
    new_df = sample_rows(df, row_percent)
    return sample_features(new_df, col_percent)


# %%
# df1 = sample_rows(df, 0.1)
# df2= sample_rows(df, 0.1)
# df3 = sample_rows(df, 0.1)

# %%
df1 = sample_features(df, 0.8)
df2 = sample_features(df, 0.8)
df3 = sample_features(df, 0.8)

# %%
df1.shape

# %%
df2.shape

# %%
df3

# %%
from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()
clf3 = DecisionTreeClassifier()

# %%
# random row
# clf1.fit(df1.iloc[:, 0:5], df1.iloc[:, -1])
# clf2.fit(df2.iloc[:, 0:5], df2.iloc[:, -1])
# clf3.fit(df3.iloc[:, 0:5], df3.iloc[:, -1])

# random column
clf1.fit(df1.iloc[:, 0:4], df1.iloc[:, -1])
clf2.fit(df2.iloc[:, 0:4], df2.iloc[:, -1])
clf3.fit(df3.iloc[:, 0:4], df3.iloc[:, -1])

# %%
from sklearn.tree import plot_tree

plot_tree(clf1)

# %%
plot_tree(clf2)

# %%
plot_tree(clf3)


# %%
# Row predict
# clf1.predict(
#     np.array([-1.149951, -2.264081, 1.522788, -0.347368, 0.313946]).reshape(1, 5)
# )
# clf2.predict(
#     np.array([-1.149951, -2.264081, 1.522788, -0.347368, 0.313946]).reshape(1, 5)
# )
# clf3.predict(
#     np.array([-1.149951, -2.264081, 1.522788, -0.347368, 0.313946]).reshape(1, 5)
# )

# %%
# Column predict
clf1.predict(np.array([-0.357941, 1.577110, 0.823404, 0.337607]).reshape(1, 4))


# %%
clf2.predict(np.array([-0.357941, 1.577110, 0.823404, 0.337607]).reshape(1, 4))
# clf3.predict(np.array([-0.357941, 1.577110, 0.823404, 0.337607]).reshape(1, 4))

# %%
clf3.predict(np.array([-0.357941, 1.577110, 0.823404, 0.337607]).reshape(1, 4))

# %%

