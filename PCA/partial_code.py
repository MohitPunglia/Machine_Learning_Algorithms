# %% [markdown]
# Generating Random data

# %%
import numpy as np
import pandas as pd

np.random.seed(23)
mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)


# %%
df = pd.DataFrame(class1_sample, columns=["feature1", "feature2", "feature3"])
df["target"] = 1

mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

df1 = pd.DataFrame(class2_sample, columns=["feature1", "feature2", "feature3"])

df1["target"] = 0

# df = df.append(df1,ignore_index=True)
df = pd.concat([df, df1], ignore_index=True)

df = df.sample(40)

# %%
df.head()
